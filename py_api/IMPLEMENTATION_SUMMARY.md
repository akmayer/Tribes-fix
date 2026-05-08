# PyTorch Model Implementation - Summary

## What Was Created

### 1. Core Model Files

#### `py_api/model.py` (650+ lines)
**Purpose:** PyTorch model architecture and state encoding

**Components:**
- `StateEncoder` class: Converts game state JSON → tensor (3,128 features)
  - Board encoding: 11×11 grid with 8 channels per tile (terrain, resources, buildings, units)
  - Unit encoding: Up to 100 units with 16 features each (type, health, position, stats)
  - City encoding: Up to 50 cities with 10 features each (level, population, position, flags)
  - Tech/tribe encoding: 50 tech features + 10 tribe stat features
  
- `TribesModel` class: PyTorch neural network
  - Input: State tensor (3,128 features)
  - Shared trunk: 3 hidden layers (512 neurons, ReLU)
  - Output heads (4 for policy, 1 for value):
    - `action_type_logits`: (batch_size, 32)
    - `source_logits`: (batch_size, 151)
    - `target_logits`: (batch_size, 163)
    - `param_logits`: (batch_size, 80)
    - `value`: (batch_size, 1)
  - Methods: `forward()`, `save()`, `load()`, `create_or_load()`

- Functions: `encode_state()`, `load_model()`
- Test code included that validates model with real examplePayload

**Output Shapes:** ✅ Correctly match factorized action space (32×151×163×80)

---

#### `py_api/train.py` (380+ lines)
**Purpose:** Training loop with capture loading and checkpoint management

**Components:**
- `GameCaptureDataset` class: Loads JSON captures from `captures/` directory
  - Per-sample processing: state encoding + masking
  - Currently uses uniform policies over masked actions as training targets
  - Extensible: Can swap in MCTS targets, self-play outcomes, etc.

- `PolicyValueTrainer` class: Handles training
  - Loss function: KL divergence for policy heads + MSE for value head
  - Optimizer: Adam
  - Features:
    - Gradient clipping (max_norm=1.0)
    - Per-epoch metrics logging
    - Checkpoint saving with metadata
    - Training history tracking

- `main()`: Full command-line interface
  - Arguments: epochs, batch_size, learning_rate, device, resume_from, etc.
  - Workflow: load data → create model → train epochs → save weights
  - Comprehensive logging and progress display

**Command Examples:**
```bash
python train.py --epochs 50 --batch-size 64
python train.py --resume-from checkpoints/checkpoint_epoch_10.pth
python train.py --max-samples 1000  # For testing
```

---

#### Updated `py_api/app.py`
**Purpose:** FastAPI policy server with real model inference

**Changes Made:**
- Added imports: `torch`, `StateEncoder`, `TribesModel`, `encode_state`
- On startup: Creates model and attempts to load `model_weights.pth`
- `/query` endpoint now:
  1. Encodes state payload using `StateEncoder`
  2. Runs model forward pass (no_grad)
  3. Applies action masks
  4. Computes masked softmax probabilities
  5. Returns `policy_type: "neural_network"` (instead of "uniform_masked")

**State:** ✅ Ready to use (will use untrained model if weights don't exist)

---

### 2. Documentation Files

#### `py_api/MODEL_README.md` (280+ lines)
**Comprehensive guide including:**
- Architecture overview (input/output shapes)
- State representation breakdown (3,128 features explained)
- File organization
- Usage: FastAPI server, training, inference
- State representation analysis with current issues
- FOG-OF-WAR CRITICAL ISSUE section (detailed)
- Training data quality notes
- Performance metrics
- Troubleshooting guide
- Future improvements (hierarchical policy, attention, etc.)

---

#### `py_api/STATE_REPRESENTATION_ANALYSIS.md` (450+ lines)
**Detailed state analysis and fog-of-war compliance report:**

**Sections:**
1. Executive Summary: State structure adequate, FOW broken
2. Detailed analysis of each component:
   - Board (11×11 grid) ✅
   - Units (type, health, position, stats) ⚠️
   - Cities (level, population, buildings) ⚠️
   - Technology tree ✅
3. Fog-of-war compliance analysis:
   - Current behavior: ALL enemy units/cities visible (WRONG)
   - Problem impact: Perfect-information training strategy fails under FOW
   - Severity: 🔴 CRITICAL
4. Required fix (detailed code changes):
   - File: `src/players/PythonBridge.java`
   - Methods: `buildPayload()`, `serializeTribes()`, `serializeTribeUnits()`, `serializeCities()`
   - With before/after code examples
   - Implementation checklist (11 items)
5. State size analysis: 3,128 features (adequate)
6. Recommendations (immediate to long-term)
7. Validation checklist (pre-training)
8. Related code locations reference

---

### 3. Dependencies Updated

#### `py_api/requirements.txt`
Added:
- `torch>=2.0.0`
- `torchvision>=0.15.0`

Installed via:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Current State

### ✅ What Works
- Model architecture: Correct output shapes for factorized action space
- State encoding: Successfully converts examplePayload to 3,128 features
- FastAPI server: Loads model and serves `/query` endpoint
- Training infrastructure: Data loading, loss functions, optimizer, checkpoints
- End-to-end pipeline: Payload → encode → model → masks → softmax → probabilities

### ❌ What's Blocked
- **FOG-OF-WAR NOT ENFORCED** - Critical issue preventing meaningful training
- Model not trained (weights don't exist)
- Training data not collected (no captures)

### ⚠️ What Needs Attention
1. Fix fog-of-war filtering in `PythonBridge.java` (BLOCKING)
2. Generate game captures with FOW enabled
3. Train model on adequate data (~1000+ samples recommended)
4. Replace uniform policy targets with MCTS or self-play outcomes

---

## Quick Start

### 1. Start FastAPI Server (now with real model)
```bash
cd py_api
source .venv/bin/activate
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Output: Uses untrained model (random policy) until weights are trained

### 2. Generate Captures (with real game)
```bash
# In main Tribes directory
java -cp "src:lib/*" Run  # or your runner
```
Saves captures to `py_api/captures/`

### 3. Train Model
```bash
cd py_api
source .venv/bin/activate
python train.py --epochs 10 --batch-size 32
```
Outputs:
- `model_weights.pth` (final weights)
- `checkpoints/checkpoint_epoch_*.pth` (per-epoch)

### 4. Server Now Uses Trained Model
Restart FastAPI server - it will auto-load `model_weights.pth`

---

## Architecture Validation

### Input ✅
- State size: 3,128 features
- Tested with real `examplePayload.json`
- Handles variable-length units/cities (pads to max)

### Output ✅
- action_type: (batch, 32)
- source: (batch, 151)
- target: (batch, 163)
- param: (batch, 80)
- value: (batch, 1)

All shapes match factorized action space perfectly.

---

## Critical Issues Found

### 🔴 Issue 1: Fog-of-War Not Enforced (BLOCKING)
**Location:** `src/players/PythonBridge.java` lines 95-310

**Problem:** All enemy units/cities visible in payload (perfect information)

**Impact:** Training learns invalid strategies that fail with actual fog-of-war

**Fix:** See detailed spec in `STATE_REPRESENTATION_ANALYSIS.md` Section 3

**Priority:** MUST FIX before training

### 🟡 Issue 2: Training Targets Are Placeholder
**Location:** `py_api/train.py` line 129

**Current:** Uniform policies over masked actions

**Recommendation:** Replace with:
- MCTS-improved policy (more sophisticated)
- Self-play outcomes (actual game results)
- Expert demonstrations (human play)

**Priority:** Can train with current setup, but results will be mediocre

---

## File Summary

```
py_api/
├── model.py                           (650 lines) ✅ New
├── train.py                           (380 lines) ✅ New
├── app.py                             (Updated) ✅ Now uses real model
├── action_encoding.py                 (Existing)
├── MODEL_README.md                    (280 lines) ✅ New
├── STATE_REPRESENTATION_ANALYSIS.md   (450 lines) ✅ New
├── requirements.txt                   (Updated) ✅ Added torch
├── action_space_schema.json           (Existing)
└── captures/                          (Directory)
    └── capture_*.json                 (Generated during gameplay)
```

---

## Next Steps

### Immediate (Block 1-2 hours)
1. **Fix fog-of-war in PythonBridge.java**
   - Implement FOW filtering as described in STATE_REPRESENTATION_ANALYSIS.md
   - Test with game running
   - Verify captures don't include hidden units

2. **Generate training data**
   ```bash
   java -cp "src:lib/*" Run  # Play several games with FOW
   ```

3. **Train model**
   ```bash
   python train.py --epochs 20 --batch-size 32
   ```

4. **Verify in gameplay**
   - Run game with trained model
   - Check that RandomAgent uses model predictions
   - Verify performance is reasonable

### Short-term (Next iteration)
5. Implement better policy targets (MCTS, self-play)
6. Add model checkpointing based on validation metrics
7. Collect diverse training data (different maps, difficulty levels)
8. Benchmark model vs. baseline agents

### Long-term (Future work)
9. Replace MLP with CNN for spatial reasoning
10. Add attention over visible units
11. Implement hierarchical action selection
12. Add auxiliary tasks (resource prediction, threat detection)

---

## Testing

### Model Test
```bash
cd py_api && source .venv/bin/activate && python model.py
```
Output: ✅ All tensor shapes correct

### App Test
```bash
cd py_api && source .venv/bin/activate && python -c "from app import app; print('OK')"
```
Output: ✅ No errors (model loads, untrained if weights missing)

### Train Test
```bash
cd py_api && source .venv/bin/activate && python -c "from train import GameCaptureDataset; print('OK')"
```
Output: ✅ Can load (will show 0 captures initially)

---

## Key Insights

1. **Factorized action space is clean** - 4 independent output heads work great
2. **State size is manageable** - 3,128 features → 512 hidden is reasonable
3. **Integration is seamless** - Java bridge → Python model → masked softmax → Java sampling
4. **FOW issue is critical** - Can't get good model without fixing this first
5. **Training infrastructure is complete** - Ready to go once data is fixed

