# Next Steps - Model Training & Deployment

## Current Status
✅ FOW fix implemented and verified
✅ FastAPI `/query` returns policy + value
✅ `AZ_MCTS` uses AlphaZero-style PUCT (NN priors + NN value)
✅ Self-play captures via `/capture` (`mcts_*.json`) + `/result` (`result_*.json`)
✅ `train.py` supports capture buffer pruning (default: keep newest 10k)

---

## Immediate Next Steps

### 1. Two-loop AlphaZero smoke test (self-play → train → self-play)

Optional: start from a clean slate (avoids shell glob limits if there are many files):
```bash
cd /home/akmayer/Tribes/py_api
find captures -maxdepth 1 -type f -name '*.json' -delete
find results -maxdepth 1 -type f -name '*.json' -delete
```

Terminal 1 (start FastAPI):
```bash
cd /home/akmayer/Tribes/py_api
/home/akmayer/Tribes/py_api/.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Terminal 2 (self-play; uses `play.json`):
```bash
cd /home/akmayer/Tribes
javac -cp .:lib/json.jar $(find src -name "*.java")
java -cp .:src:lib/json.jar Play
```

Train (writes `model_weights.pth`):
```bash
cd /home/akmayer/Tribes/py_api
/home/akmayer/Tribes/py_api/.venv/bin/python train.py --epochs 1 --max-captures 10000
```

Restart FastAPI (so it reloads the new `model_weights.pth`), then run `Play` again for loop 2.

---

### 2. Generate More Training Data (Optional but Recommended)
```bash
cd /home/akmayer/Tribes

# `Play` reads configuration from play.json (players, tribes, seeds, etc.).
# To generate MCTS-labeled training data for AlphaZero-style training, use AZ_MCTS in play.json.

# Compile once:
javac -cp .:lib/json.jar $(find src -name "*.java")

# Run several self-play games (each run generates new mcts_*.json + result_*.json):
for i in {1..3}; do
   echo "Self-play game $i"
   java -cp .:src:lib/json.jar Play
done
```

This creates more `mcts_*.json` files in `py_api/captures/` (and `result_*.json` in `py_api/results/`).

### 3. Train the Model
```bash
cd py_api
source .venv/bin/activate

# Basic training (10 epochs)
python train.py --epochs 10

# With more options
python train.py --epochs 50 --batch-size 32 --learning-rate 0.001

# Resume from checkpoint
python train.py --resume-from checkpoints/checkpoint_epoch_10.pth --epochs 50
```

**Output:**
- `model_weights.pth` - Final trained weights (auto-loaded by FastAPI)
- `checkpoints/checkpoint_epoch_*.pth` - Per-epoch snapshots

### 4. Test Trained Model
```bash
# Start FastAPI (uses updated model_weights.pth automatically)
cd py_api
python -m uvicorn app:app --host 127.0.0.1 --port 8000

# In another terminal, run a game (uses play.json players)
cd /home/akmayer/Tribes
java -cp .:src:lib/json.jar Play
```

---

## Quick Reference Commands

### Run Game with Current Model
```bash
# Terminal 1: Start FastAPI server
cd py_api && python -m uvicorn app:app --host 127.0.0.1 --port 8000

# Terminal 2: Run game (uses play.json)
cd /home/akmayer/Tribes && java -cp .:src:lib/json.jar Play
```

### Monitor Training
```bash
cd py_api
# Training will show:
# - Loss decreasing
# - Per-epoch metrics
# - Checkpoint savings
python train.py --epochs 20 --batch-size 32
```

### Generate Captures Only (No Training)
```bash
cd /home/akmayer/Tribes
for i in {1..5}; do
  echo "Generating game $i..."
   timeout 30 java -cp .:src:lib/json.jar Play
done
```

---

## Training Data Quality

### Current Status
- **Captures:** Produced via FastAPI `/capture` and stored in `py_api/captures/`
- **Policy Targets:** MCTS visit counts when available (`mcts_*.json`)
- **Value Targets:** Game outcomes via `/result` (stored in `py_api/results/`)

### To Improve:
1. **Better Policy Targets:**
   - Increase MCTS budget per decision (see hyperparameters below)
   
2. **More Diverse Data:**
   - Different starting positions
   - Different agent combinations
   - Different map seeds

3. **Value Labels:**
   - Track game outcomes (win/loss)
   - Use actual scores as targets

---

## Performance Expectations

### With Random (Untrained) Model
- Policy/value heads are effectively random
- Agent should still play legally (masking enforced)

### After Training on MCTS Targets
- Policy is trained to match MCTS visit counts
- Value is trained from game results

### With MCTS-Improved Targets
- Policy learns from better strategies
- Performance: significant improvement
- Still limited by quality of MCTS

### With Self-Play
- Model improves iteratively
- Performance: best results
- Requires loop: play → collect → train → repeat

---

## Important Notes

⚠️ **Model Initialization:** If no weights exist, FastAPI starts with random weights.

⚠️ **FOW Filtering:** Is now active. Only visible enemy units/cities in captures. This is correct behavior.

⚠️ **Inference logging:** `/query` can be called extremely frequently by MCTS. By default it does **not** dump per-query JSON files.
Enable request logging only when debugging payload contents:
```bash
export TRIBES_SAVE_INFERENCE=1
export TRIBES_INFERENCE_DIR=inference
```

✅ **AlphaZero-style loop is wired:** `AZ_MCTS` self-play → `train.py` → self-play.

---

## File Locations

```
/home/akmayer/Tribes/
├── py_api/
│   ├── model_weights.pth          ← Current model
│   ├── model.py                   ← Model definition
│   ├── train.py                   ← Training script
│   ├── app.py                     ← FastAPI server
│   ├── captures/                  ← Game data
│   │   └── mcts_*.json             ← MCTS-labeled samples (visit counts)
│   ├── results/                   ← Terminal values
│   │   └── result_*.json           ← Game outcomes/value targets
│   └── checkpoints/               ← Training checkpoints
├── src/
│   ├── players/PythonBridge.java  ← FOW filtering
│   ├── players/RandomAgent.java   ← Model inference
│   ├── players/mcts/SingleTreeNode.java ← PUCT + NN value integration
│   └── Play.java                  ← Game entry point
└── VERIFICATION_REPORT.md         ← Test results
```

---

## Testing Checklist

Before/after each training run:

- [ ] FastAPI server starts without errors
- [ ] Model loads with `model_weights.pth`
- [ ] `AZ_MCTS` self-play runs end-to-end
- [ ] `py_api/captures/` contains new `mcts_*.json`
- [ ] `py_api/results/` contains new `result_*.json`
- [ ] FOW filtering works (check captures for enemy visibility)

---

## Troubleshooting

### "Connection refused" when running game
→ Start FastAPI server first: `python -m uvicorn app:app --host 127.0.0.1 --port 8000`

### Training loss is NaN
→ Check learning rate (default 0.001 should be fine)
→ Verify captures exist and have actions

### Game runs very slowly
→ Model inference is happening 40+ times/turn
→ Normal delay for CPU inference
→ Try GPU: add `--device cuda` to train.py

### "No captures found" during training
→ Ensure `play.json` uses `AZ_MCTS` (it posts to `/capture`), then run a few self-play games:
`java -cp .:src:lib/json.jar Play`
→ Captures saved to `py_api/captures/` (look for `mcts_*.json`)

---

## Success Criteria

After training, model should be better than random:
- RandomAgent should make more diverse moves
- Fewer illegal action attempts (via masking)
- Game should complete without errors
- Model inference should be reasonably fast (~50ms/query on CPU)

---

## Next Major Milestones

1. ✅ FOW Fix - DONE
2. ✅ Model Integration - DONE
3. ⏳ Model Training - READY TO START
4. ✅ MCTS Integration (PUCT + priors/value) - DONE
5. ⏳ Performance Evaluation - NEXT
6. ⏳ Longer self-play runs - NEXT

---

## Key Hyperparameters (current defaults)

### Java (AZ_MCTS)
- `NEURAL_PRIORS=true`, `NEURAL_VALUE=true`, `CPUCT=1.5` (see `src/Run.java`)
- Budget: `stop_type=STOP_FMCALLS` (set in `src/Run.java`), `num_fmcalls=2000` (default in `src/players/heuristics/AlgParams.java`)
- Rollouts: controlled by `play.json` → `"Rollouts"` (typically `false` for NN value)
- Rollout length: `play.json` → `"Search Depth"` (wired to `Run.MAX_LENGTH`)

### Python training (`py_api/train.py`)
- `--epochs` (default 10), `--batch-size` (default 32), `--learning-rate` (default 1e-3)
- `--max-captures` (default 10000) prunes oldest `capture_*.json` + `mcts_*.json`

