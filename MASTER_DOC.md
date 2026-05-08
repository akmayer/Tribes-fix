Master Documentation - Tribes AI Training Pipeline

Overview
--------
This repository contains the Tribes game, a Python FastAPI policy server, and Java agents used to run playouts and integrate with neural networks for AlphaZero-style self-play training.

Top-level components
--------------------
- `src/` - Java game engine and agents.
  - `src/players/RandomAgent.java` - Random agent updated to sample from masked `action_type` logits returned by the Python policy server.
  - `src/players/PythonBridge.java` - Serializes game state and posts to Python FastAPI `/query` endpoint.
- `py_api/` - Python policy server and action-space utilities.
  - `py_api/app.py` - FastAPI server; `/query` endpoint returns masked logits and masked post-softmax probabilities for each factorized component: `action_type`, `source`, `target`, `param`.
  - `py_api/action_encoding.py` - `ActionSpaceEncoder` implementing encode/decode, masks, and helpers.
  - `py_api/test_action_encoding.py` - Test script validating encoder and masked-softmax behavior.
  - `py_api/action_space_schema.json` - Canonical factorized action-space schema.
  - `py_api/captures/` - Saved captures from Java playouts (JSON snapshots).

Key Design Notes
----------------
- Factorized action space: NN outputs separate logits for `action_type`, `source_actor`, `target_actor`, and `param`.
- Masking: illegal entries are masked out (zero probability after softmax). The Python server computes masked logits and masked softmax probabilities and returns both logits and probs for consumer convenience.
- Java-side sampling: currently `RandomAgent` samples an `action_type` from `action_type_logits` (masked softmax), then selects uniformly among available actions of that type. This is an incremental step towards composing full per-action probabilities from all components.

Running the server and tests
---------------------------
1. Activate the Python virtualenv in `py_api`:

```bash
cd py_api
source .venv/bin/activate
```

2. Install dependencies (if needed):

```bash
pip install -r requirements.txt
```

3. Run unit tests for the action-encoding and masked-softmax:

```bash
python test_action_encoding.py
```

4. Start the FastAPI server:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

5. Run the Java game (from repository root):

```bash
javac -cp ".:lib/*" $(find src -name "*.java")
java -cp "src:lib/*" Play
```

What's next (high level)
------------------------
- Extend Java-side selection to combine `source`, `target`, and `param` logits into a single per-action probability (requires mapping from mixed-radix components to concrete actions).
- Implement MCTS+NN Java agents that call Python for evaluation and record training samples (state, MCTS visit counts, rewards).
- Implement Python training scripts that consume captured data and train NN models (PyTorch), producing weights that the FastAPI server can load.
- Add automated tests for the full end-to-end loop (small playouts with a trivial NN).

If you want, I can now:
- Modify `py_api/app.py` to support loading a PyTorch model and returning live NN outputs (next step).
- Add Java unit tests to validate that masked entries are never sampled.
- Implement composing per-action probabilities from full factorized logits in Java.

Where would you like me to continue? (I recommend adding Java unit tests next.)
