import subprocess
import time
import signal
import os
import sys
from pathlib import Path
from datetime import datetime

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
AZ_UNIFORM_PRIOR_WEIGHT = 0.10
AZ_DIRICHLET_ALPHA = 0.30
AZ_DIRICHLET_EPSILON = 0.25
AZ_FORCE_END_TURN_IN_SEARCH = False

SLEEP_AFTER_SERVER_START = 5

LOG_DIR = ROOT / "training_logs"
LOG_DIR.mkdir(exist_ok=True)

MODEL_PATH = PY_API / "model_weights.pth"


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
        f"uniform_prior={AZ_UNIFORM_PRIOR_WEIGHT}, "
        f"dirichlet_alpha={AZ_DIRICHLET_ALPHA}, "
        f"dirichlet_epsilon={AZ_DIRICHLET_EPSILON}"
    )


class FastAPIServer:
    def __init__(self):
        self.proc = None
        self.log_file = None

    def start(self):
        kill_port(FASTAPI_PORT)
        time.sleep(1)
        print("\n=== STARTING FASTAPI SERVER ===")

        self.log_file = open(
            LOG_DIR / f"fastapi_errors_{timestamp()}.log",
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
                str(FASTAPI_PORT),
                "--no-access-log",
                "--log-level",
                "warning",
            ],
            cwd=PY_API,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )

        time.sleep(SLEEP_AFTER_SERVER_START)

        if self.proc.poll() is not None:
            raise RuntimeError("FastAPI server failed to start")

        print(f"FastAPI running (PID={self.proc.pid})")

    def stop(self):
        if self.proc is None:
            return

        print("\n=== STOPPING FASTAPI SERVER ===")

        try:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("FastAPI did not exit cleanly, killing...")
            self.proc.kill()
            self.proc.wait()

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


def run_command(cmd, cwd=None, env=None, log_name=None):
    print(f"\nRUNNING: {' '.join(cmd)}")

    stdout = None

    if log_name:
        stdout = open(LOG_DIR / log_name, "a", buffering=1)

    result = None
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=stdout if stdout else None,
            stderr=subprocess.STDOUT,
        )
    finally:
        if stdout is not None:
            stdout.close()

    if result is not None and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {result.returncode}: {' '.join(cmd)}"
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
        ["java", "-cp", ".:src:lib/json.jar", "Play"],
        cwd=ROOT,
        env=java_training_env(),
        log_name=f"selfplay_loop{loop_idx}_game{game_idx}.log",
    )


def train_model(loop_idx):
    print(f"\n=== TRAINING MODEL (LOOP {loop_idx}) ===")

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
        log_name=f"train_loop{loop_idx}.log",
    )


def java_training_env():
    env = os.environ.copy()
    env.update(
        {
            "TRIBES_AZ_MCTS_SIMULATIONS": str(AZ_MCTS_SIMULATIONS),
            "TRIBES_AZ_MCTS_CPUCT": str(AZ_MCTS_CPUCT),
            "TRIBES_AZ_UNIFORM_PRIOR_WEIGHT": str(AZ_UNIFORM_PRIOR_WEIGHT),
            "TRIBES_AZ_DIRICHLET_ALPHA": str(AZ_DIRICHLET_ALPHA),
            "TRIBES_AZ_DIRICHLET_EPSILON": str(AZ_DIRICHLET_EPSILON),
            "TRIBES_AZ_FORCE_END_TURN_IN_SEARCH": str(AZ_FORCE_END_TURN_IN_SEARCH).lower(),
        }
    )
    return env


def count_capture_files():
    captures_dir = PY_API / "captures"
    return len(list(captures_dir.glob("mcts_*.json")))


def count_result_files():
    results_dir = PY_API / "results"
    return len(list(results_dir.glob("result_*.json")))


def print_status():
    print("\n=== DATASET STATUS ===")
    print(f"MCTS captures: {count_capture_files()}")
    print(f"Result files:  {count_result_files()}")

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

            print_status()

            # --------------------------------------------------
            # TRAIN
            # --------------------------------------------------
            train_model(loop_idx)

            # --------------------------------------------------
            # RESTART SERVER TO LOAD NEW WEIGHTS
            # --------------------------------------------------
            server.restart()

            print(f"\nLOOP {loop_idx} COMPLETE")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        raise

    finally:
        server.stop()


if __name__ == "__main__":
    main()
