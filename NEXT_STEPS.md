# Next Steps - Model Training & Deployment

## Current Status
✅ FOW fix implemented and verified
✅ Model loaded in FastAPI server
✅ Game successfully runs with model inference
✅ Training data being collected with FOW filtering

---

## Immediate Next Steps

### 1. Generate More Training Data (Optional but Recommended)
```bash
cd /home/akmayer/Tribes

# Run several games to generate diverse training data
java -cp "src:lib/*" Play Random Random   # 2-3 times
java -cp "src:lib/*" Play Random "Rule Based"
java -cp "src:lib/*" Play "Do Nothing" Random
```

This creates more captures in `py_api/captures/` for better training.

### 2. Train the Model
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

### 3. Test Trained Model
```bash
# Start FastAPI (uses updated model_weights.pth automatically)
cd py_api
python -m uvicorn app:app --host 127.0.0.1 --port 8000

# In another terminal, run game with RandomAgent
cd /home/akmayer/Tribes
java -cp "src:lib/*" Play Random Random
```

---

## Quick Reference Commands

### Run Game with Current Model
```bash
# Terminal 1: Start FastAPI server
cd py_api && python -m uvicorn app:app --host 127.0.0.1 --port 8000

# Terminal 2: Run game
cd /home/akmayer/Tribes && java -cp "src:lib/*" Play Random Random
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
  timeout 30 java -cp "src:lib/*" Play Random Random
done
```

---

## Training Data Quality

### Current Status
- **Captures:** 6+ already generated with FOW filtering
- **Policy Targets:** Currently using uniform over masked actions
- **Value Targets:** Random (placeholder)

### To Improve:
1. **Better Policy Targets:**
   - Run MCTS during capture generation
   - Use actual game outcomes
   
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
- Policy is random (uniform over masked actions)
- Value is garbage
- Agent plays poorly but legally

### After Training on Uniform Targets
- Policy learns to distinguish actions
- But based on weak targets (uniform)
- Performance: modest improvement

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

⚠️ **Model Initialization:** Currently using random weights. First training epoch will show high loss, then decreasing.

⚠️ **FOW Filtering:** Is now active. Only visible enemy units/cities in captures. This is correct behavior.

⚠️ **No AlphaZero Yet:** We're not running the full AlphaZero loop (MCTS + self-play). Current setup just trains on uniform policies.

✅ **Foundation Ready:** All infrastructure in place for full AlphaZero when needed.

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
│   └── checkpoints/               ← Training checkpoints
├── src/
│   ├── players/PythonBridge.java  ← FOW filtering
│   ├── players/RandomAgent.java   ← Model inference
│   └── Play.java                  ← Game entry point
└── VERIFICATION_REPORT.md         ← Test results
```

---

## Testing Checklist

Before/after each training run:

- [ ] FastAPI server starts without errors
- [ ] Model loads with `model_weights.pth`
- [ ] Game runs with RandomAgent
- [ ] Model inference happens 40+ times per game
- [ ] Captures are valid (no errors)
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
→ Run a few games first: `java -cp "src:lib/*" Play Random Random`
→ Captures saved to `py_api/captures/`

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
4. ⏳ Performance Evaluation - NEXT
5. ⏳ MCTS Integration - FUTURE
6. ⏳ Full AlphaZero - FUTURE

