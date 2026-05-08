ALPHAZERO REFERENCE - Tribes
=============================

Purpose
-------
A single reference for implementing AlphaZero-style self-play (MCTS + NN) for Tribes. Designed as a hands-on runbook for the Java and Python pieces so future sessions can pick up where this one left off.

High-level flow
---------------
1. Java `MCTS+NN` agent runs in-play, performing MCTS per move.
2. The agent queries the Python FastAPI model for policy logits and value estimates.
3. MCTS uses the returned priors (composed from factorized heads) to guide simulations and stores the root visit counts for each chosen move.
4. After each game, the agent writes self-play records (captures) to `py_api/captures/` with sparse root visit counts and final outcome.
5. Python `train.py` consumes captures, converts sparse visit-counts to per-head marginal policy targets and a value target, and trains `TribesModel`.

Design and key decisions
------------------------
- Use factorized action-space heads (action_type, source, target, param). Do not enumerate full joint-space.
- Store sparse root visit counts per move as lists of joint-action tuples with counts; reconstruct per-head marginals during training.
- Add Dirichlet noise to the root priors and a temperature schedule for exploration during self-play.
- Persist metadata with every capture (model version, seed, hyperparams) for reproducibility.

Java MCTS+NN Agent (high level)
-------------------------------
Responsibilities:
- Run MCTS to select moves during self-play using NN priors and values.
- Query FastAPI for priors and values when expanding nodes.
- Record for each root position: the set of children expanded and their visit counts.
- After game end, write the per-move record including the final result (value from perspective of root player).

Core API (Java ↔ FastAPI)
-------------------------
Endpoint: POST /query
Payload: same JSON used by existing `PythonBridge` (contains fog-of-war filtered state)
Response (JSON):
- action_type_logits: [32]
- source_logits: [151]
- target_logits: [163]
- param_logits: [80]
- masks: {action_type_mask, source_mask, target_mask, param_mask}
- value: float

Java usage pattern:
- Call `/query` for a given state to get logits and masks.
- Apply mask and softmax on Java side (or accept probabilities from Python if endpoint returns them).
- Compose joint priors from factorized probabilities when initializing root children priors.

Pseudocode (root initialization)
```
logits = query_nn(state)
probs_heads = softmax_with_mask(logits.heads, masks)
// When creating candidate children, you can compute joint prior for a child action (t,s,r,p):
joint_prior = probs_heads.action_type[t] * probs_heads.source[s] * probs_heads.target[r] * probs_heads.param[p]
// Add Dirichlet noise at root
prior = (1 - epsilon) * joint_prior + epsilon * Dir(alpha)
```

MCTS specifics
--------------
- Use standard PUCT formula: U(s,a) = c_puct * P(s,a) * sqrt(sum_N)/ (1 + N(s,a))
- Virtual loss helps parallel threads; if single-threaded, skip it.
- Root visit counts: after N simulations, for each child a record (t,s,r,p, visit_count)
- When selecting a move for self-play, sample from root visit counts with temperature tau:
  - If tau > 0: sample proportionally to visit_count^(1/tau)
  - If tau -> 0: pick argmax

Capture format (recommended JSON record per move)
-------------------------------------------------
Store one JSON object per move (append to file or separate file per game). Example:
```
{
  "game_id": "uuid",
  "move_index": 12,
  "player_id": 1,
  "state": { ... },        // same JSON as /query uses (FOW filtered)
  "root_children": [
    {"action": [type, source, target, param], "visits": 123},
    {"action": [type, source, target, param], "visits": 45},
    ...
  ],
  "temperature": 1.0,
  "model_version": "model_weights_2026-05-07.pth",
  "seed": 12345,
  "final_result": 1  // +1 win, -1 loss, 0 draw from perspective of player_id
}
```
Notes:
- Keep `state` identical to what was sent to the NN (including masks).
- `root_children` should be sparse: only the explored children (top-K may suffice).
- Write records as you go (flush after each move) to avoid data loss.

Encoding utilities (Java & Python)
----------------------------------
Implement shared utilities to encode/decode factorized actions and to map joint tuples to a canonical representation:
- `encode_action_tuple(type, source, target, param)` → tuple or string key
- `decode_action_tuple(key)` → components
- Use the same mapping in Java and Python. Prefer JSON tuple representation to avoid index mismatches.

Training pipeline (Python)
--------------------------
1. `collect`: run many self-play games with MCTS+NN agent; writes captures into `py_api/captures/`.
2. `prepare`: read capture files and convert `root_children` visit counts into per-head marginal targets.
   - For each move: build arrays `policy_type`, `policy_source`, `policy_target`, `policy_param` by summing visit counts across joint actions where the component matches.
   - Normalize each head target: target_head = counts_head / sum_counts
   - Value target = `final_result` (from perspective of player at that state)
3. `train`: feed inputs (states), masks, per-head targets and value targets into `TribesModel`.
   - Loss = sum(head_losses) + value_loss
   - Head loss: cross-entropy / KL between target distribution and model (apply mask to logits before softmax)
   - Value loss: MSE between predicted value and final_result
4. `checkpoint`: save `model_weights.pth` and metadata (hyperparams, step)

Replay buffer and sampling
--------------------------
- Keep a large replay buffer of move records (or just files on disk) and sample uniformly.
- Optionally implement prioritized sampling (e.g., weight recent games higher).
- Typical buffer size: 100k - 1M moves; for initial experiments, 10k–50k is fine.

Hyperparameters (suggested starting values)
-------------------------------------------
- MCTS simulations per move: 100 (dev) → 800 (production)
- c_puct: 1.5
- Dirichlet alpha (root): 0.3 (tune by action branching)
- Dirichlet epsilon: 0.25
- Temperature schedule: tau = 1 for first 30 moves, then 0.1, then argmax
- Replay buffer: 100000 moves
- Batch size: 64
- Learning rate: 1e-3 with Adam, decay schedule optional
- Value scaling: map game outcomes to ±1

Evaluation and promotion
------------------------
- Periodically (e.g., every N steps), run evaluation tournament:
  - New model vs champion for M games (e.g., 100)
  - If new model wins > threshold (e.g., 55%), promote it to champion
- Record all evaluation matches and seed values

Operational & performance tips
------------------------------
- Batch NN evaluations where possible: collect leaf expansions across parallel games and send batched queries to FastAPI / a batched inference endpoint.
- Run FastAPI with `uvicorn --workers 1` and let Java-side handle concurrency; for GPU, run a single process that manages batches.
- Consider a thin JNI bridge for lower-latency in-process inference if IPC is a bottleneck.

Testing checklist
-----------------
- [ ] `PythonBridge` serializes FOW properly (unit test for visibility)
- [ ] Action encoding round-trip (`encode_action_tuple` ↔ `decode_action_tuple`)
- [ ] Small MCTS smoke test: run 1 game with 10 sims per move and verify captures are written
- [ ] Training pipeline end-to-end on small dataset (10–100 moves)

Minimal run commands
--------------------
Start model server (py_api):
```bash
cd py_api
source .venv/bin/activate
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Run Java self-play with MCTSAgent (example):
```bash
# ensure classpath includes compiled classes and json lib
java -cp "src:lib/*:." Play MCTSAgent MCTSAgent --sims 100
```
Train on captures:
```bash
cd py_api
python train.py --captures-dir captures/ --batch-size 64 --epochs 10
```
Run evaluation (example):
```bash
# script that runs Play with two agent classes and reports win-rates
bash scripts/evaluate.sh --champion model_champion.pth --challenger model_new.pth --games 100
```

Metadata, reproducibility, and logging
-------------------------------------
- Every capture file and checkpoint should include: RNG seed, model version, NN git/commit hash, hyperparameters, and timestamp.
- Keep a `runs/` directory where each self-play run and its artifacts are grouped.

Open issues / TODOs
-------------------
- Implement batched inference endpoint to reduce IPC overhead.
- Decide on deterministic ordering for joint-action keys (for smaller storage format).
- Consider storing top-K joint actions only to reduce capture size (with K tuned per game complexity).

Contacts & next steps
---------------------
- Implement Java `MCTS+NN` agent (core priority)
- Modify `Play` to persist captures reliably
- Implement training pipeline upgrades (replay buffer & sampling)


End of document
