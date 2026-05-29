import subprocess
import time
import signal
import os
import sys
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone

import torch


def env_int(name, default_value):
    value = os.environ.get(name)
    if value is None or value == "":
        return default_value
    try:
        return int(value)
    except ValueError:
        print(f"Invalid integer env {name}={value}; using {default_value}")
        return default_value

ROOT = Path(__file__).resolve().parent
PY_API = ROOT / "py_api"
VENV_PYTHON = PY_API / ".venv/bin/python"

FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8000
PARALLEL_JOBS = max(1, env_int("TRIBES_PARALLEL_JOBS", 10))
FASTAPI_WORKERS = max(1, env_int("TRIBES_FASTAPI_WORKERS", 1))
INFERENCE_CONCURRENCY_PER_WORKER = max(1, env_int("TRIBES_INFERENCE_CONCURRENCY_PER_WORKER", 1))
INFERENCE_THREADS_PER_QUERY = max(1, env_int("TRIBES_INFERENCE_THREADS_PER_QUERY", 1))
CAPTURE_RATE_LOG_INTERVAL_SECONDS = max(1, env_int("TRIBES_CAPTURE_RATE_LOG_INTERVAL_SECONDS", 10))
SELFPLAY_SEED_BASE = env_int(
    "TRIBES_SELFPLAY_SEED_BASE",
    int(time.time() * 1000) % 9_000_000_000_000,
)

# Training loop knobs. These are intentionally hardcoded here instead of hidden
# in Run.java/train.py defaults, so an overnight run is reproducible from this file.
#
# Reasonable presets:
# - Bootstrap/debug: 2-5 games, 64-96 MCTS sims, 20k-30k captures, 1-2 epochs.
# - Early real training: 5-8 games, 96-160 MCTS sims, 40k-60k captures, 1-2 epochs.
# - Stronger later runs: 8-12 games, 192-400 MCTS sims, 80k-150k captures, 1 epoch.
#
# With roughly 600+ captured states per game, 50k captures is about 80 games at
# the current game length. If games get longer, increase MAX_CAPTURES before
# increasing TRAIN_EPOCHS, otherwise the model will overfit stale recent games.
SELFPLAY_GAMES_PER_LOOP = 10
MAX_CAPTURES = 50000
NUM_LOOPS = 1000  # effectively overnight/until stopped

TRAIN_EPOCHS = 2
TRAIN_BATCH_SIZE = 128
TRAIN_LEARNING_RATE = 3e-4
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 0.25

# MCTS budget. With 20-60 legal actions, 100 visits is only a rough bootstrap
# policy. 128 is still cheap enough to iterate, while giving PUCT more than one
# look at high-prior moves. Raise this as the value net becomes useful.
AZ_MCTS_SIMULATIONS = 128
AZ_MCTS_CPUCT = 1.5
AZ_DIRICHLET_ALPHA = 0.30
AZ_DIRICHLET_EPSILON = 0.25
AZ_FORCE_END_TURN_IN_SEARCH = False
AZ_DEBUG_DECISIONS = True
MASK_SEND_STARS = True
PLAY_WITH_FULL_OBS = True

EVALUATION_GAMES = 10
EVALUATION_WIN_THRESHOLD = 0.60
EVALUATION_AZ_MCTS_SIMULATIONS = 32
EVALUATION_OLD_PORT = 8001
EVALUATION_NEW_PORT = 8002

# If Ctrl+C interrupts a self-play game, any MCTS captures from that unfinished
# game will lack final value targets. The dataloader can skip value loss for
# those files, but for AlphaZero training it is cleaner to drop interrupted-game
# captures before each train step.
DROP_ORPHAN_CAPTURES_BEFORE_TRAIN = True
CHECKPOINT_INTERVAL_SECONDS = 30 * 60

SLEEP_AFTER_SERVER_START = 5

LOG_DIR = ROOT / "training_logs"
LOG_DIR.mkdir(exist_ok=True)
RUNTIME_CONFIG_DIR = LOG_DIR / "runtime_play_configs"

MODEL_PATH = PY_API / "model_weights.pth"
CHECKPOINT_DIR = PY_API / "checkpoints"

_ACTIVE_COMMANDS = {}
_ACTIVE_COMMANDS_LOCK = threading.Lock()


def validate_paths():
    missing = []
    if not VENV_PYTHON.exists():
        missing.append(str(VENV_PYTHON))
    if not (ROOT / "play.json").exists():
        missing.append(str(ROOT / "play.json"))
    if not (PY_API / "train.py").exists():
        missing.append(str(PY_API / "train.py"))

    if missing:
        raise FileNotFoundError("Missing required training file(s): " + ", ".join(missing))


def print_config():
    print(f"Root: {ROOT}")
    print(f"Python API: {PY_API}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(
        "Loop config: "
        f"games={SELFPLAY_GAMES_PER_LOOP}, "
        f"parallel_jobs={PARALLEL_JOBS}, "
        f"captures={MAX_CAPTURES}, "
        f"epochs={TRAIN_EPOCHS}, "
        f"batch={TRAIN_BATCH_SIZE}, "
        f"lr={TRAIN_LEARNING_RATE}"
    )
    print(
        "AZ MCTS config: "
        f"sims={AZ_MCTS_SIMULATIONS}, "
        f"cpuct={AZ_MCTS_CPUCT}, "
        f"dirichlet_alpha={AZ_DIRICHLET_ALPHA}, "
        f"dirichlet_epsilon={AZ_DIRICHLET_EPSILON}, "
        f"mask_send_stars={MASK_SEND_STARS}, "
        f"full_observability={PLAY_WITH_FULL_OBS}"
    )
    print(
        "Arena config: "
        f"games={EVALUATION_GAMES}, "
        f"win_threshold={EVALUATION_WIN_THRESHOLD:.0%}, "
        f"agent=az_mcts, "
        f"sims={EVALUATION_AZ_MCTS_SIMULATIONS}, "
        "paired_seeds=5"
    )
    print(f"Drop orphan captures before training: {DROP_ORPHAN_CAPTURES_BEFORE_TRAIN}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL_SECONDS // 60} minutes")
    print(
        "Parallel config: "
        f"jobs={PARALLEL_JOBS}, "
        f"fastapi_workers={FASTAPI_WORKERS}, "
        f"inference_concurrency_per_worker={INFERENCE_CONCURRENCY_PER_WORKER}, "
        f"inference_threads_per_query={INFERENCE_THREADS_PER_QUERY}, "
        f"policy_request_slots_per_server={FASTAPI_WORKERS * INFERENCE_CONCURRENCY_PER_WORKER}, "
        f"capture_rate_log_interval={CAPTURE_RATE_LOG_INTERVAL_SECONDS}s, "
        f"selfplay_seed_base={SELFPLAY_SEED_BASE}"
    )


def terminate_process(proc, name, timeout=5):
    if proc is None or proc.poll() is not None:
        return

    print(f"Stopping {name} (PID={proc.pid})...")
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=timeout)
        return
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        print(f"{name} did not exit after SIGINT; terminating...")

    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=timeout)
        return
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        print(f"{name} did not exit after SIGTERM; killing...")

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()


def register_active_command(proc, name):
    with _ACTIVE_COMMANDS_LOCK:
        _ACTIVE_COMMANDS[proc] = name


def unregister_active_command(proc):
    if proc is None:
        return
    with _ACTIVE_COMMANDS_LOCK:
        _ACTIVE_COMMANDS.pop(proc, None)


def terminate_active_commands(reason, timeout=3):
    with _ACTIVE_COMMANDS_LOCK:
        active = list(_ACTIVE_COMMANDS.items())

    if not active:
        return

    print(f"Stopping {len(active)} active command(s) ({reason})...")
    for proc, name in active:
        terminate_process(proc, name, timeout=timeout)


class FastAPIServer:
    def __init__(self, port=FASTAPI_PORT, model_path=MODEL_PATH, name="fastapi", workers=FASTAPI_WORKERS):
        self.proc = None
        self.log_file = None
        self.port = port
        self.model_path = Path(model_path)
        self.name = name
        self.workers = max(1, workers)

    def start(self):
        kill_port(self.port)
        time.sleep(1)
        print(f"\n=== STARTING FASTAPI SERVER ({self.name}, port {self.port}) ===")

        self.log_file = open(
            LOG_DIR / f"{self.name}_errors_{timestamp()}.log",
            "a",
            buffering=1,
        )

        cmd = [
            str(VENV_PYTHON),
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            FASTAPI_HOST,
            "--port",
            str(self.port),
            "--workers",
            str(self.workers),
            "--no-access-log",
            "--log-level",
            "warning",
        ]

        self.proc = subprocess.Popen(
            cmd,
            cwd=PY_API,
            env=python_training_env(model_path=self.model_path, inference_server=True),
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        time.sleep(SLEEP_AFTER_SERVER_START)

        if self.proc.poll() is not None:
            raise RuntimeError(f"FastAPI server failed to start: {self.name}")

        print(
            f"FastAPI running ({self.name}, PID={self.proc.pid}, "
            f"workers={self.workers}, model={self.model_path})"
        )

    def stop(self):
        if self.proc is None:
            return

        print(f"\n=== STOPPING FASTAPI SERVER ({self.name}) ===")

        terminate_process(self.proc, f"FastAPI {self.name}", timeout=5)

        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

        print("FastAPI stopped")
        self.proc = None

    def restart(self):
        print("\n=== RESTARTING FASTAPI SERVER ===")
        self.stop()
        time.sleep(2)
        self.start()


def ensure_initial_weights():
    """
    Create randomly initialized model weights if none exist yet.
    Prevents the system from running with an implicitly random in-memory model.
    """

    if MODEL_PATH.exists():
        print(f"Found existing weights: {MODEL_PATH}")
        return

    print("\n=== INITIALIZING RANDOM MODEL WEIGHTS ===")

    init_script = r"""
from pathlib import Path
import torch

from model import TribesTransformerModel, StateEncoder

MODEL_PATH = Path("model_weights.pth")

state_encoder = StateEncoder()

model = TribesTransformerModel(
    state_size=state_encoder.total_state_size
)

torch.save(model.state_dict(), MODEL_PATH)

print(f"Saved initial random weights to {MODEL_PATH}")
"""

    run_command(
        [
            str(VENV_PYTHON),
            "-c",
            init_script,
        ],
        cwd=PY_API,
        log_name=f"init_weights_{timestamp()}.log",
    )


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def utc_minute_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")


class CheckpointManager:
    def __init__(self, interval_seconds):
        self.interval_seconds = interval_seconds
        self.last_checkpoint_monotonic = time.monotonic()
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def maybe_save(self, reason):
        now = time.monotonic()
        if now - self.last_checkpoint_monotonic < self.interval_seconds:
            return False
        self.save(reason)
        return True

    def save(self, reason):
        if not MODEL_PATH.exists():
            print(f"Skipping checkpoint ({reason}); model weights do not exist yet: {MODEL_PATH}")
            return None

        output_path = unique_checkpoint_path()
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        shutil.copy2(MODEL_PATH, temp_path)
        temp_path.replace(output_path)
        self.last_checkpoint_monotonic = time.monotonic()
        print(f"Saved checkpoint ({reason}): {output_path}")
        return output_path


def unique_checkpoint_path():
    base = CHECKPOINT_DIR / f"checkpoint_{utc_minute_timestamp()}.pth"
    if not base.exists():
        return base

    idx = 2
    while True:
        candidate = CHECKPOINT_DIR / f"{base.stem}_{idx}{base.suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def training_parameter_snapshot():
    return {
        "parallel_jobs": PARALLEL_JOBS,
        "selfplay_games_per_loop": SELFPLAY_GAMES_PER_LOOP,
        "fastapi_workers_per_server": FASTAPI_WORKERS,
        "inference_concurrency_per_worker": INFERENCE_CONCURRENCY_PER_WORKER,
        "inference_threads_per_query": INFERENCE_THREADS_PER_QUERY,
        "policy_request_slots_per_server": FASTAPI_WORKERS * INFERENCE_CONCURRENCY_PER_WORKER,
        "selfplay_policy_model_copies": FASTAPI_WORKERS,
        "arena_policy_model_copies": 2 * FASTAPI_WORKERS,
        "torch_threads_per_policy_server": FASTAPI_WORKERS * INFERENCE_THREADS_PER_QUERY,
        "approx_selfplay_cpu_threads": (
            PARALLEL_JOBS
            + FASTAPI_WORKERS
            * INFERENCE_CONCURRENCY_PER_WORKER
            * INFERENCE_THREADS_PER_QUERY
        ),
        "az_mcts_simulations": AZ_MCTS_SIMULATIONS,
        "evaluation_games": EVALUATION_GAMES,
        "evaluation_az_mcts_simulations": EVALUATION_AZ_MCTS_SIMULATIONS,
        "mask_send_stars": MASK_SEND_STARS,
        "play_with_full_obs": PLAY_WITH_FULL_OBS,
        "capture_rate_log_interval_seconds": CAPTURE_RATE_LOG_INTERVAL_SECONDS,
        "selfplay_seed_base": SELFPLAY_SEED_BASE,
    }


def count_capture_json_files():
    captures_dir = PY_API / "captures"
    return len(list(captures_dir.glob("*.json")))


class CaptureRateLogger:
    def __init__(self, interval_seconds):
        self.interval_seconds = interval_seconds
        self.path = LOG_DIR / f"capture_rate_{timestamp()}.jsonl"
        self.stop_event = threading.Event()
        self.thread = None
        self.state_lock = threading.Lock()
        self.phase = "startup"
        self.loop_idx = None
        self.last_monotonic = None
        self.last_count = None

    def start(self):
        self._write(
            {
                "type": "config",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": training_parameter_snapshot(),
            }
        )
        print(f"Capture-rate log: {self.path}")
        self.thread = threading.Thread(target=self._run, name="capture_rate_logger", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval_seconds + 1)
            self.thread = None
        self.sample()

    def set_phase(self, phase, loop_idx=None):
        with self.state_lock:
            self.phase = phase
            self.loop_idx = loop_idx

    def _run(self):
        while not self.stop_event.wait(self.interval_seconds):
            self.sample()

    def sample(self):
        now = time.monotonic()
        capture_count = count_capture_json_files()
        mcts_count = count_capture_files()

        if self.last_monotonic is None:
            elapsed = 0.0
            delta = 0
            captures_per_minute = 0.0
        else:
            elapsed = max(0.0, now - self.last_monotonic)
            delta = capture_count - self.last_count
            captures_per_minute = (delta / elapsed * 60.0) if elapsed > 0 else 0.0

        self.last_monotonic = now
        self.last_count = capture_count

        with self.state_lock:
            phase = self.phase
            loop_idx = self.loop_idx

        self._write(
            {
                "type": "capture_rate",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "phase": phase,
                "loop": loop_idx,
                "captures_total": capture_count,
                "mcts_captures_total": mcts_count,
                "delta_captures": delta,
                "interval_seconds": elapsed,
                "captures_per_minute": captures_per_minute,
                "parameters": training_parameter_snapshot(),
            }
        )

    def _write(self, payload):
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")


def run_command(cmd, cwd=None, env=None, log_name=None, command_name=None):
    display_name = command_name or " ".join(cmd)
    print(f"\nRUNNING: {display_name}")

    stdout = None

    if log_name:
        stdout = open(LOG_DIR / log_name, "a", buffering=1)

    result = None
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=stdout if stdout else None,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        register_active_command(proc, display_name)
        result = proc.wait()
    except KeyboardInterrupt:
        terminate_process(proc, display_name, timeout=3)
        raise
    finally:
        unregister_active_command(proc)
        if stdout is not None:
            stdout.close()

    if result is not None and result != 0:
        raise RuntimeError(
            f"Command failed with code {result}: {display_name}"
        )


def compile_java():
    print("\n=== COMPILING JAVA ===")

    compile_cmd = (
        'javac -cp .:lib/json.jar $(find src -name "*.java")'
    )

    run_command(
        ["bash", "-c", compile_cmd],
        cwd=ROOT,
        log_name=f"compile_{timestamp()}.log",
    )


def read_base_play_config():
    with (ROOT / "play.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def selfplay_play_config(loop_idx, game_idx):
    game_seed = SELFPLAY_SEED_BASE + loop_idx * 100_000 + game_idx * 100
    config = read_base_play_config()
    config.update(
        {
            "Game Seed": str(game_seed),
            "Agents Seed": str(game_seed + 17),
            "Level Seed": str(game_seed + 33),
        }
    )
    return config


def run_java_game_with_config(config, env, log_name, config_label):
    RUNTIME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = RUNTIME_CONFIG_DIR / f"{config_label}.json"
    write_json_atomic(config_path, config)

    run_env = env.copy()
    run_env["TRIBES_PLAY_CONFIG"] = str(config_path)

    succeeded = False
    try:
        run_command(
            ["java", "-Djava.awt.headless=true", "-cp", ".:src:lib/json.jar", "Play"],
            cwd=ROOT,
            env=run_env,
            log_name=log_name,
            command_name=config_label,
        )
        succeeded = True
    finally:
        if succeeded:
            config_path.unlink(missing_ok=True)


def run_selfplay_game(game_idx, loop_idx):
    print(f"\n=== SELF-PLAY GAME {game_idx} (LOOP {loop_idx}) ===")

    config = selfplay_play_config(loop_idx, game_idx)
    run_java_game_with_config(
        config,
        env=java_training_env(),
        log_name=f"selfplay_loop{loop_idx}_game{game_idx}.log",
        config_label=f"selfplay_loop{loop_idx}_game{game_idx}",
    )
    return {
        "game_idx": game_idx,
        "game_seed": int(config["Game Seed"]),
    }


def run_parallel_jobs(jobs, description, on_complete=None):
    if not jobs:
        return []

    workers = min(PARALLEL_JOBS, len(jobs))
    print(f"\n=== {description.upper()} ({len(jobs)} total, {workers} parallel jobs) ===")

    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=description.lower().replace(" ", "_"),
    )
    future_to_label = {
        executor.submit(job_fn): label
        for label, job_fn in jobs
    }
    results = []
    completed = 0

    try:
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"{description} failed ({label}): {exc}")
                raise

            completed += 1
            results.append(result)
            if on_complete is not None:
                on_complete(result)
            print(f"{description}: {completed}/{len(jobs)} complete ({label})")
    except BaseException:
        for future in future_to_label:
            future.cancel()
        terminate_active_commands(f"{description} shutdown", timeout=3)
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    executor.shutdown(wait=True)
    return results


def run_selfplay_games(loop_idx, checkpoint_manager):
    jobs = [
        (
            f"selfplay loop {loop_idx} game {game_idx}",
            lambda game_idx=game_idx: run_selfplay_game(game_idx, loop_idx),
        )
        for game_idx in range(1, SELFPLAY_GAMES_PER_LOOP + 1)
    ]
    return run_parallel_jobs(
        jobs,
        "Self-play games",
        on_complete=lambda _result: checkpoint_manager.maybe_save("after self-play game"),
    )


def train_model(loop_idx):
    print(f"\n=== TRAINING MODEL (LOOP {loop_idx}) ===")
    if DROP_ORPHAN_CAPTURES_BEFORE_TRAIN:
        deleted = delete_orphan_captures()
        if deleted > 0:
            print(f"Deleted {deleted} capture files without matching result files.")

    run_command(
        [
            str(VENV_PYTHON),
            "train.py",
            "--epochs",
            str(TRAIN_EPOCHS),
            "--batch-size",
            str(TRAIN_BATCH_SIZE),
            "--learning-rate",
            str(TRAIN_LEARNING_RATE),
            "--policy-loss-weight",
            str(POLICY_LOSS_WEIGHT),
            "--value-loss-weight",
            str(VALUE_LOSS_WEIGHT),
            "--max-captures",
            str(MAX_CAPTURES),
            "--device",
            'cuda' if torch.cuda.is_available() else 'cpu',
        ],
        cwd=PY_API,
        env=python_training_env(),
        log_name=f"train_loop{loop_idx}.log",
    )


def apply_thread_limits(env, threads):
    thread_count = str(max(1, threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = thread_count
    env["TRIBES_TORCH_THREADS"] = thread_count


def python_training_env(model_path=None, inference_server=False):
    env = os.environ.copy()
    env.update(
        {
            "TRIBES_MASK_SEND_STARS": str(MASK_SEND_STARS).lower(),
            "TRIBES_PLAY_WITH_FULL_OBS": str(PLAY_WITH_FULL_OBS).lower(),
            "TRIBES_REQUIRE_FULL_OBS_CAPTURES": str(PLAY_WITH_FULL_OBS).lower(),
        }
    )
    if model_path is not None:
        env["TRIBES_MODEL_PATH"] = str(Path(model_path))
    if inference_server:
        apply_thread_limits(env, INFERENCE_THREADS_PER_QUERY)
        env["TRIBES_INFERENCE_CONCURRENCY"] = str(INFERENCE_CONCURRENCY_PER_WORKER)
    return env


def java_training_env():
    env = os.environ.copy()
    env.update(
        {
            "TRIBES_AZ_MCTS_SIMULATIONS": str(AZ_MCTS_SIMULATIONS),
            "TRIBES_AZ_MCTS_CPUCT": str(AZ_MCTS_CPUCT),
            "TRIBES_AZ_DIRICHLET_ALPHA": str(AZ_DIRICHLET_ALPHA),
            "TRIBES_AZ_DIRICHLET_EPSILON": str(AZ_DIRICHLET_EPSILON),
            "TRIBES_AZ_FORCE_END_TURN_IN_SEARCH": str(AZ_FORCE_END_TURN_IN_SEARCH).lower(),
            "TRIBES_AZ_DEBUG_DECISIONS": str(AZ_DEBUG_DECISIONS).lower(),
            "TRIBES_MASK_SEND_STARS": str(MASK_SEND_STARS).lower(),
            "TRIBES_PLAY_WITH_FULL_OBS": str(PLAY_WITH_FULL_OBS).lower(),
            "TRIBES_POLICY_URL": f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/query",
        }
    )
    return env


def count_capture_files():
    captures_dir = PY_API / "captures"
    return len(list(captures_dir.glob("mcts_*.json")))


def count_result_files():
    results_dir = PY_API / "results"
    return len(list(results_dir.glob("result_*.json")))


def result_keys():
    keys = set()
    for path in (PY_API / "results").glob("result_*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            keys.add((int(payload["game_seed"]), int(payload["player_id"])))
        except Exception as exc:
            print(f"Skipping unreadable result file {path}: {exc}")
    return keys


def delete_orphan_captures():
    keys = result_keys()
    if not keys:
        return 0

    deleted = 0
    for path in (PY_API / "captures").glob("mcts_*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            key = (int(payload["game_seed"]), int(payload["player_id"]))
        except Exception as exc:
            print(f"Deleting unreadable capture file {path}: {exc}")
            path.unlink(missing_ok=True)
            deleted += 1
            continue

        if key not in keys:
            path.unlink(missing_ok=True)
            deleted += 1

    return deleted


def print_status():
    print("\n=== DATASET STATUS ===")
    print(f"MCTS captures: {count_capture_files()}")
    print(f"Result files:  {count_result_files()}")


def copy_model_snapshot(label, loop_idx):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHECKPOINT_DIR / f"arena_{label}_loop{loop_idx}_{timestamp()}.pth"
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    shutil.copy2(MODEL_PATH, temp_path)
    temp_path.replace(output_path)
    return output_path


def write_json_atomic(path, payload):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temp_path.replace(path)


def arena_play_config(game_seed, level_seed):
    config = read_base_play_config()

    config.update(
        {
            "Run Mode": "PlayLG",
            "Players": ["AZ_MCTS", "AZ_MCTS"],
            "Tribes": ["Xin Xi", "Imperius"],
            "Verbose": False,
            "Game Seed": str(game_seed),
            "Agents Seed": str(game_seed + 17),
            "Level Seed": str(level_seed),
        }
    )
    return config


def read_arena_result(result_file, candidate_player_id):
    results = []
    if not result_file.exists():
        raise RuntimeError(f"Arena result file was not created: {result_file}")

    with result_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if len(results) < 2:
        raise RuntimeError(f"Arena result file is incomplete: {result_file}")

    candidate = None
    for result in results:
        if int(result.get("player_id", -1)) == candidate_player_id:
            candidate = result
            break

    if candidate is None:
        raise RuntimeError(f"Missing candidate result in {result_file}")

    return candidate.get("winner") == "WIN"


def run_arena_game(loop_idx, game_idx, new_as_player_zero):
    seed_pair_idx = (game_idx - 1) // 2
    game_seed = 900000000 + loop_idx * 10000 + seed_pair_idx
    level_seed = 910000000 + loop_idx * 10000 + seed_pair_idx
    result_file = LOG_DIR / f"arena_loop{loop_idx}_game{game_idx}.jsonl"
    result_file.unlink(missing_ok=True)

    candidate_player_id = 0 if new_as_player_zero else 1
    old_url = f"http://{FASTAPI_HOST}:{EVALUATION_OLD_PORT}/query"
    new_url = f"http://{FASTAPI_HOST}:{EVALUATION_NEW_PORT}/query"

    env = java_training_env()
    env.update(
        {
            "TRIBES_AZ_MCTS_SIMULATIONS": str(EVALUATION_AZ_MCTS_SIMULATIONS),
            "TRIBES_AZ_CAPTURE_MCTS": "false",
            "TRIBES_AZ_DIRICHLET_ROOT_NOISE": "false",
            "TRIBES_AZ_DEBUG_DECISIONS": "false",
            "TRIBES_POLICY_URL_PLAYER_0": new_url if new_as_player_zero else old_url,
            "TRIBES_POLICY_URL_PLAYER_1": old_url if new_as_player_zero else new_url,
            "TRIBES_EVAL_RESULT_FILE": str(result_file),
        }
    )

    run_java_game_with_config(
        arena_play_config(game_seed, level_seed),
        env=env,
        log_name=f"arena_loop{loop_idx}_game{game_idx}.log",
        config_label=f"arena_loop{loop_idx}_game{game_idx}",
    )

    return {
        "game_idx": game_idx,
        "candidate_won": read_arena_result(result_file, candidate_player_id),
    }


def evaluate_candidate(old_model_path, new_model_path, loop_idx):
    print(f"\n=== AZ MCTS ARENA EVALUATION (LOOP {loop_idx}) ===")
    old_server = FastAPIServer(
        port=EVALUATION_OLD_PORT,
        model_path=old_model_path,
        name=f"arena_old_loop{loop_idx}",
    )
    new_server = FastAPIServer(
        port=EVALUATION_NEW_PORT,
        model_path=new_model_path,
        name=f"arena_new_loop{loop_idx}",
    )

    wins = 0
    try:
        old_server.start()
        new_server.start()

        completed = 0

        def record_arena_result(result):
            nonlocal wins, completed
            completed += 1
            if result["candidate_won"]:
                wins += 1
            print(
                f"Arena game {result['game_idx']}/{EVALUATION_GAMES} complete: "
                f"candidate_wins={wins}, win_rate={wins / completed:.1%}"
            )

        jobs = [
            (
                f"arena loop {loop_idx} game {game_idx}",
                lambda game_idx=game_idx: run_arena_game(
                    loop_idx,
                    game_idx,
                    game_idx % 2 == 1,
                ),
            )
            for game_idx in range(1, EVALUATION_GAMES + 1)
        ]
        run_parallel_jobs(jobs, "Arena games", on_complete=record_arena_result)
    finally:
        new_server.stop()
        old_server.stop()

    win_rate = wins / EVALUATION_GAMES
    accepted = win_rate >= EVALUATION_WIN_THRESHOLD
    print(
        f"Arena result: candidate_wins={wins}/{EVALUATION_GAMES} "
        f"({win_rate:.1%}), accepted={accepted}"
    )
    return accepted, win_rate


def kill_port(port):
    try:
        result = subprocess.check_output(["lsof", "-t", f"-i:{port}"]).decode().strip()
        if result:
            for pid in result.split("\n"):
                print(f"Killing stale process on port {port}: {pid}")
                subprocess.run(["kill", "-9", pid])
    except Exception:
        pass


def main():
    print("=" * 60)
    print("ALPHAZERO TRAINING LOOP")
    print("=" * 60)
    validate_paths()
    print_config()

    server = FastAPIServer()
    checkpoint_manager = CheckpointManager(CHECKPOINT_INTERVAL_SECONDS)
    capture_rate_logger = CaptureRateLogger(CAPTURE_RATE_LOG_INTERVAL_SECONDS)
    capture_rate_logger.start()

    try:
        capture_rate_logger.set_phase("compile")
        compile_java()

        capture_rate_logger.set_phase("ensure_initial_weights")
        ensure_initial_weights()

        capture_rate_logger.set_phase("start_policy_server")
        server.start()

        for loop_idx in range(1, NUM_LOOPS + 1):
            print("\n" + "#" * 60)
            print(f"STARTING TRAINING LOOP {loop_idx}")
            print("#" * 60)

            # --------------------------------------------------
            # SELF-PLAY
            # --------------------------------------------------
            capture_rate_logger.set_phase("selfplay", loop_idx)
            run_selfplay_games(loop_idx, checkpoint_manager)

            print_status()

            # --------------------------------------------------
            # TRAIN
            # --------------------------------------------------
            capture_rate_logger.set_phase("train", loop_idx)
            old_model_path = copy_model_snapshot("old", loop_idx)
            train_model(loop_idx)
            candidate_model_path = copy_model_snapshot("candidate", loop_idx)
            checkpoint_manager.maybe_save("after training")

            # --------------------------------------------------
            # MCTS ARENA GATE
            # --------------------------------------------------
            capture_rate_logger.set_phase("arena", loop_idx)
            server.stop()
            accepted, _win_rate = evaluate_candidate(old_model_path, candidate_model_path, loop_idx)
            if accepted:
                print("Candidate model accepted.")
            else:
                print(f"Candidate model rejected; restoring previous weights from {old_model_path}")
                temp_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".tmp")
                shutil.copy2(old_model_path, temp_path)
                temp_path.replace(MODEL_PATH)

            capture_rate_logger.set_phase("restart_policy_server", loop_idx)
            server.start()
            checkpoint_manager.maybe_save("after arena gate")

            print(f"\nLOOP {loop_idx} COMPLETE")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        checkpoint_manager.save("interrupt")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        raise

    finally:
        capture_rate_logger.set_phase("shutdown")
        server.stop()
        capture_rate_logger.stop()


if __name__ == "__main__":
    main()
