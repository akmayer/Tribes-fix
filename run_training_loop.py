import subprocess
import time
import signal
import os
import sys
from pathlib import Path
from datetime import datetime

import torch

ROOT = Path("/home/akmayer/Tribes")
PY_API = ROOT / "py_api"
VENV_PYTHON = PY_API / ".venv/bin/python"

FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8000

SELFPLAY_GAMES_PER_LOOP = 5
TRAIN_EPOCHS = 2
MAX_CAPTURES = 20000
NUM_LOOPS = 1000  # effectively overnight/until stopped

SLEEP_AFTER_SERVER_START = 5

LOG_DIR = ROOT / "training_logs"
LOG_DIR.mkdir(exist_ok=True)

MODEL_PATH = PY_API / "model_weights.pth"


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

    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=stdout if stdout else None,
        stderr=subprocess.STDOUT,
    )

    if result.returncode != 0:
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
            "--max-captures",
            str(MAX_CAPTURES),
            "--device",
            'cuda' if torch.cuda.is_available() else 'cpu',
        ],
        cwd=PY_API,
        log_name=f"train_loop{loop_idx}.log",
    )


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
