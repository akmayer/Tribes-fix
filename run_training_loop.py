import subprocess
import time
import signal
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

import torch

ROOT = Path(__file__).resolve().parent
PY_API = ROOT / "py_api"
VENV_PYTHON = PY_API / ".venv/bin/python"

FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8000

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
SELFPLAY_GAMES_PER_LOOP = 5
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

MODEL_PATH = PY_API / "model_weights.pth"
CHECKPOINT_DIR = PY_API / "checkpoints"


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


class FastAPIServer:
    def __init__(self, port=FASTAPI_PORT, model_path=MODEL_PATH, name="fastapi"):
        self.proc = None
        self.log_file = None
        self.port = port
        self.model_path = Path(model_path)
        self.name = name

    def start(self):
        kill_port(self.port)
        time.sleep(1)
        print(f"\n=== STARTING FASTAPI SERVER ({self.name}, port {self.port}) ===")

        self.log_file = open(
            LOG_DIR / f"{self.name}_errors_{timestamp()}.log",
            "a",
            buffering=1,
        )

        self.proc = subprocess.Popen(
            [
                str(VENV_PYTHON),
                "-m",
                "uvicorn",
                "app:app",
                "--host",
                FASTAPI_HOST,
                "--port",
                str(self.port),
                "--no-access-log",
                "--log-level",
                "warning",
            ],
            cwd=PY_API,
            env=python_training_env(model_path=self.model_path),
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        time.sleep(SLEEP_AFTER_SERVER_START)

        if self.proc.poll() is not None:
            raise RuntimeError(f"FastAPI server failed to start: {self.name}")

        print(f"FastAPI running ({self.name}, PID={self.proc.pid}, model={self.model_path})")

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


def run_command(cmd, cwd=None, env=None, log_name=None):
    print(f"\nRUNNING: {' '.join(cmd)}")

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
        result = proc.wait()
    except KeyboardInterrupt:
        terminate_process(proc, "active command", timeout=3)
        raise
    finally:
        if stdout is not None:
            stdout.close()

    if result is not None and result != 0:
        raise RuntimeError(
            f"Command failed with code {result}: {' '.join(cmd)}"
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


def run_selfplay_game(game_idx, loop_idx):
    print(f"\n=== SELF-PLAY GAME {game_idx} (LOOP {loop_idx}) ===")

    run_command(
        ["java", "-Djava.awt.headless=true", "-cp", ".:src:lib/json.jar", "Play"],
        cwd=ROOT,
        env=java_training_env(),
        log_name=f"selfplay_loop{loop_idx}_game{game_idx}.log",
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


def python_training_env(model_path=None):
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
    play_config_path = ROOT / "play.json"
    with play_config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

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

    play_config_path = ROOT / "play.json"
    with play_config_path.open("r", encoding="utf-8") as handle:
        original_config = json.load(handle)

    try:
        write_json_atomic(play_config_path, arena_play_config(game_seed, level_seed))
        run_command(
            ["java", "-Djava.awt.headless=true", "-cp", ".:src:lib/json.jar", "Play"],
            cwd=ROOT,
            env=env,
            log_name=f"arena_loop{loop_idx}_game{game_idx}.log",
        )
    finally:
        write_json_atomic(play_config_path, original_config)

    return read_arena_result(result_file, candidate_player_id)


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

        for game_idx in range(1, EVALUATION_GAMES + 1):
            new_as_player_zero = game_idx % 2 == 1
            if run_arena_game(loop_idx, game_idx, new_as_player_zero):
                wins += 1
            print(
                f"Arena game {game_idx}/{EVALUATION_GAMES}: "
                f"candidate_wins={wins}, win_rate={wins / game_idx:.1%}"
            )
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

    try:
        compile_java()

        ensure_initial_weights()

        server.start()

        for loop_idx in range(1, NUM_LOOPS + 1):
            print("\n" + "#" * 60)
            print(f"STARTING TRAINING LOOP {loop_idx}")
            print("#" * 60)

            # --------------------------------------------------
            # SELF-PLAY
            # --------------------------------------------------
            for game_idx in range(1, SELFPLAY_GAMES_PER_LOOP + 1):
                run_selfplay_game(game_idx, loop_idx)
                checkpoint_manager.maybe_save("after self-play game")

            print_status()

            # --------------------------------------------------
            # TRAIN
            # --------------------------------------------------
            old_model_path = copy_model_snapshot("old", loop_idx)
            train_model(loop_idx)
            candidate_model_path = copy_model_snapshot("candidate", loop_idx)
            checkpoint_manager.maybe_save("after training")

            # --------------------------------------------------
            # MCTS ARENA GATE
            # --------------------------------------------------
            server.stop()
            accepted, _win_rate = evaluate_candidate(old_model_path, candidate_model_path, loop_idx)
            if accepted:
                print("Candidate model accepted.")
            else:
                print(f"Candidate model rejected; restoring previous weights from {old_model_path}")
                temp_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".tmp")
                shutil.copy2(old_model_path, temp_path)
                temp_path.replace(MODEL_PATH)

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
        server.stop()


if __name__ == "__main__":
    main()
