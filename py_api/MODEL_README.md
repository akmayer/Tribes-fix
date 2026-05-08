# PyTorch Model for Tribes

## Overview

This directory contains the PyTorch-based policy and value prediction model for the Tribes game. The model takes game state as input and outputs:

1. **Action logits** (4 factorized heads):
   - `action_type_logits` (32 dimensions): probability distribution over action types
   - `source_logits` (151 dimensions): probability distribution over source actors
   - `target_logits` (163 dimensions): probability distribution over target actors
   - `param_logits` (80 dimensions): probability distribution over action parameters

2. **Value estimate** (1 dimension): state value for training

## Architecture

### Model Class: `TribesModel` (model.py)

**Input:** State tensor of shape `(batch_size, 3128)`

**Output:**
- action_type_logits: `(batch_size, 32)`
- source_logits: `(batch_size, 151)`
- target_logits: `(batch_size, 163)`
- param_logits: `(batch_size, 80)`
- value: `(batch_size, 1)`

**Internals:**
- Shared trunk: 3 hidden layers (512 neurons, ReLU activation)
- Policy heads: 4 separate linear layers (one per action component)
- Value head: 2-layer MLP with final linear output

### State Encoder: `StateEncoder` (model.py)

Converts game state JSON payload to a tensor representation:

```
State Features (total 3128):
├── Board representation (968):
│   └── 11x11 grid × 8 channels per tile
│       ├── Terrain one-hot (8 types)
│       ├── Resource indicator
│       ├── Building indicator
│       └── Unit presence
├── Units (1600):
│   └── Up to 100 units × 16 features each
│       ├── Type (normalized)
│       ├── Health percentage
│       ├── X/Y position (normalized)
│       ├── Attack/Defense/Movement stats
│       └── Status flags (has_moved, has_attacked, is_veteran)
├── Cities (500):
│   └── Up to 50 cities × 10 features each
│       ├── Level (normalized)
│       ├── Population (normalized)
│       ├── X/Y position (normalized)
│       ├── Capital flag
│       └── Walls flag
└── Tribe stats (60):
    ├── Technology tree (50 features)
    └── Tribe stats (10 features): stars, score, etc.
```

**⚠️ IMPORTANT - FOG-OF-WAR:** The state encoder does NOT enforce fog-of-war filtering. It encodes whatever is provided in the payload. If the payload from `PythonBridge.java` includes enemy units/cities that should be hidden, the model will learn from perfect information instead of partial observability.

## Files

- **`model.py`**: PyTorch model class and state encoder
- **`app.py`**: FastAPI server that loads the model and serves `/query` endpoint
- **`train.py`**: Training script that loads game captures and trains the model
- **`model_weights.pth`**: Trained model weights (loaded by app.py on startup)
- **`action_space_schema.json`**: Factorized action space definition (in parent dir)

## Usage

### 1. Running the FastAPI Server

The model is automatically loaded when the FastAPI server starts:

```bash
cd py_api
source .venv/bin/activate
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

On startup, the server will:
- Load `ActionSpaceEncoder` for action masking
- Create a new `TribesModel` instance
- Attempt to load `model_weights.pth` if it exists
- If weights don't exist, start with an untrained model

### 2. Training the Model

First, generate game captures by running the Tribes game with `RandomAgent`:

```bash
# In the main Tribes directory
java -cp "src:lib/*" Run # or other runner
```

This creates JSON capture files in `py_api/captures/`.

Then train the model:

```bash
cd py_api
source .venv/bin/activate

# Basic training
python train.py

# With options
python train.py --epochs 50 --batch-size 64 --learning-rate 0.001 --max-samples 1000

# Resume from checkpoint
python train.py --resume-from checkpoints/checkpoint_epoch_10.pth --epochs 50
```

**Command-line arguments:**
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Adam learning rate (default: 1e-3)
- `--model-path`: Path to save final weights (default: model_weights.pth)
- `--checkpoint-dir`: Directory for epoch checkpoints (default: checkpoints)
- `--resume-from`: Checkpoint path to resume training
- `--max-samples`: Limit samples (useful for testing)
- `--device`: `cpu` or `cuda` (default: cpu)
- `--capture-dir`: Directory with captures (default: captures)

### 3. Model Inference

The `/query` endpoint in `app.py` performs inference:

```python
import torch
from model import TribesModel, StateEncoder, encode_state

# Load model
state_encoder = StateEncoder()
model = TribesModel(state_size=state_encoder.total_state_size)
model.load("model_weights.pth", device="cpu")
model.eval()

# Encode state from JSON payload
payload = {...}  # From Java bridge
state = encode_state(payload, state_encoder)

# Get predictions
with torch.no_grad():
    state_batch = state.unsqueeze(0)
    action_type_logits, source_logits, target_logits, param_logits, value = model(state_batch)
```

## State Representation Analysis

### Current State (`examplePayload.json`)

The current state representation includes:

**Board (11×11 grid):**
- Terrain type (one-hot encoded)
- Resource type (stars, custom, or none)
- Building type (monument, temple, road, or none)
- Unit present on tile

**Tribes Array (includes all tribes in game):**
- For each tribe:
  - Tribe ID, type, name
  - Stars, score, winner status
  - Technology tree (researched techs)
  - Monuments
  - Cities array:
    - Position, level, population
    - Buildings (temples, etc.)
    - Unit IDs
  - Units array:
    - Position, type, health
    - Attack/defense/movement stats
    - Status (veteran, moved, attacked)
  - Meta: wars declared, tribes met, etc.

### ⚠️ Critical Issue: FOG-OF-WAR NOT ENFORCED

**Problem:** The payload includes ALL tribes' units and cities, regardless of whether they should be visible to the active tribe.

**Impact:** The model learns optimal play with perfect information instead of under partial observability. This makes the learned policy unrealistic and potentially useless for actual gameplay where fog-of-war is enabled.

**Current Behavior:**
- Active tribe: All own units/cities visible (correct)
- Enemy tribes: ALL units/cities visible (WRONG - should only show what's in observability grid)

**Should Be:**
- Active tribe: All own units/cities visible
- Enemy tribes: Only units/cities within active tribe's `obsGrid` should be visible
- Hidden enemy units/cities: Should not appear in state at all (or be masked)

### Required Fix: Fog-of-War Filtering in PythonBridge.java

File: `src/players/PythonBridge.java`

**Current code:** `serializeTribes()` iterates all tribes and includes all units/cities

**Required change:**
1. Get active tribe's `obsGrid`:
   ```java
   Tribe activeTribe = gs.getActiveTribe();
   boolean[][] obsGrid = activeTribe.getObsGrid();
   ```

2. Filter `serializeTribeUnits()` to only include visible units:
   ```java
   // For enemy units, check if position is in obsGrid
   if (tribe.getTribeId() == activeTribeID) {
       // Own tribe: include all units
   } else {
       // Enemy tribe: only include if visible
       if (obsGrid[unit.getX()][unit.getY()]) {
           // include unit
       }
   }
   ```

3. Filter `serializeCities()` similarly

4. Update `serializeBoard()` to only show cities/buildings visible to active tribe

**Until this is fixed:** Any trained model will be learning with perfect information and will perform poorly in actual fog-of-war games.

## Training Data Quality

### Current Training Setup

The `train.py` script currently uses **uniform policies over masked actions** as training targets. This means:

```
Training target for each action component:
- For each legal action: probability = 1 / (number of legal actions for that component)
- For illegal actions: probability = 0
```

### Limitations

This is a placeholder approach. For meaningful training, you need better targets:

1. **MCTS-Improved Policy**: Run Monte Carlo Tree Search during game playback to get better action distributions
2. **Self-Play with Outcomes**: Train on actual game outcomes (win/loss/draw) not just uniform policies
3. **Expert Demonstrations**: Provide human expert play data
4. **Reinforcement Learning**: Train against other policies or self-play

### Recommended Next Steps

1. **Fix fog-of-war in PythonBridge.java** (critical!)
2. **Collect more diverse game data** (different starting positions, difficulty levels)
3. **Implement outcome labels** (which player won, score, etc.)
4. **Add MCTS policy targets** (run MCTS for each state, use result as training target)
5. **Use value from actual game outcome** (instead of random value target)

## Model Checkpoints

Training saves checkpoints in `checkpoints/` directory:

```
checkpoints/
├── checkpoint_epoch_1.pth
├── checkpoint_epoch_2.pth
└── ...
```

Each checkpoint contains:
- Model weights
- Optimizer state
- Training metrics
- Epoch number

## Integration with Java Bridge

The model is called through the FastAPI server:

1. **Java Agent** (`RandomAgent.java`) requests policy for available actions
2. **PythonBridge.java** serializes game state to JSON
3. **FastAPI `/query` endpoint** receives payload
4. **Model encoder** converts JSON → tensor
5. **Model forward** generates 4 logit heads
6. **Action masking** zeros out illegal actions
7. **Softmax** converts logits → probabilities
8. **RandomAgent** samples from factorized distribution
9. **Selected action** is executed in game

## Performance Notes

**Inference time per query:** ~10-50ms (CPU, depends on payload size)

**Memory footprint:** ~50MB for model weights + ~50MB for inference

**Typical game capture size:** 100-500KB JSON per game state

## Troubleshooting

### Model outputs NaN/Inf
- Check for empty masks (all-zero mask)
- Verify action encoding is correct
- Look at masked_softmax in app.py

### Policy doesn't improve during training
- Verify captures are being loaded correctly
- Check training loss - should decrease over epochs
- Increase training data (more game captures)
- Consider using better policy targets instead of uniform

### State representation looks wrong
- Inspect actual capture files in `captures/` directory
- Verify StateEncoder normalizations
- Check board size (should be 11×11)
- Verify unit/city counts

## Future Improvements

1. **Hierarchical policy**: Multi-level action selection (e.g., select unit first, then action)
2. **Attention mechanisms**: Better handling of variable-length state (units, cities, etc.)
3. **Convolutional layers**: Better spatial reasoning on 11×11 board
4. **Recurrent processing**: Handle long-term temporal dependencies
5. **Multi-task learning**: Predict multiple auxiliary tasks (resource locations, opponent strength, etc.)
6. **MCTS integration**: Use model as heuristic in actual MCTS during gameplay
