import json
import numpy as np
import torch
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from action_encoding import ActionSpaceEncoder
from model import TribesModel, StateEncoder, encode_state

app = FastAPI()

CAPTURE_DIR = Path("captures")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# Load the action space encoder
encoder = ActionSpaceEncoder()

# Load the PyTorch model
state_encoder = StateEncoder()
model = TribesModel(state_size=state_encoder.total_state_size)
model.eval()
device = torch.device("cpu")

# Try to load model weights if they exist
MODEL_PATH = Path("model_weights.pth")
if MODEL_PATH.exists():
    model.load(str(MODEL_PATH), device=str(device))
    print(f"Loaded model weights from {MODEL_PATH}")
else:
    print(f"No model weights found at {MODEL_PATH}, using untrained model")

model = model.to(device)


@app.get("/hello")
async def hello():
    return {"message": "hello from python"}


@app.post("/query")
async def query(req: Request):
    """
    Receive game state and available actions from Java.
    Return a policy (action logits) over available actions using the trained model.
    
    ⚠️ IMPORTANT: This endpoint assumes fog-of-war is correctly enforced in the payload.
    If PythonBridge.java sends unfiltered enemy unit/city data, the model will learn
    from perfect information instead of partial observability.
    """
    payload = await req.json()

    # Save the capture for analysis
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    tick = payload.get("tick", "unknown")
    action_count = payload.get("available_action_count", "na")
    filename = f"capture_tick{tick}_actions{action_count}_{timestamp}.json"
    output_path = CAPTURE_DIR / filename

    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)

    # Extract game state
    available_actions = payload.get("available_actions", [])
    active_tribe_id = payload.get("active_tribe_id", 0)

    # Create masks for legal actions
    try:
        masks = encoder.mask_available_actions(available_actions)
        
        # Encode the state and get model predictions
        state_tensor = encode_state(payload, state_encoder)
        
        with torch.no_grad():
            state_batch = state_tensor.unsqueeze(0).to(device)  # Add batch dimension
            action_type_logits, source_logits, target_logits, param_logits, _ = model(state_batch)
            
            # Remove batch dimension and convert to numpy
            action_type_logits = action_type_logits.squeeze(0).cpu().numpy()
            source_logits = source_logits.squeeze(0).cpu().numpy()
            target_logits = target_logits.squeeze(0).cpu().numpy()
            param_logits = param_logits.squeeze(0).cpu().numpy()
        
        # Apply masks (multiply by mask to zero out illegal actions)
        action_type_logits = action_type_logits * masks["action_type_mask"]
        source_logits = source_logits * masks["source_mask"]
        target_logits = target_logits * masks["target_mask"]
        param_logits = param_logits * masks["param_mask"]

        # Compute masked softmax probabilities (post-softmax masked entries will be 0)
        def masked_softmax(logits, mask):
            # logits and mask are numpy arrays
            masked_logits = np.where(np.array(mask) > 0, logits, -1e9)
            maxv = np.max(masked_logits)
            exps = np.exp(masked_logits - maxv)
            exps = exps * (np.array(mask) > 0)
            s = np.sum(exps)
            if s == 0:
                # no allowed entries -> uniform over mask==1 (or uniform over all if none)
                allowed = np.sum(np.array(mask) > 0)
                if allowed == 0:
                    return np.ones_like(exps) / len(exps)
                return (np.array(mask) > 0).astype(float) / allowed
            return exps / s

        action_type_probs = masked_softmax(action_type_logits, masks["action_type_mask"])
        source_probs = masked_softmax(source_logits, masks["source_mask"])
        target_probs = masked_softmax(target_logits, masks["target_mask"])
        param_probs = masked_softmax(param_logits, masks["param_mask"])
        
        policy_response = {
            "status": "success",
            "policy_type": "neural_network",
            "action_type_logits": action_type_logits.tolist(),
            "action_type_probs": action_type_probs.tolist(),
            "source_logits": source_logits.tolist(),
            "source_probs": source_probs.tolist(),
            "target_logits": target_logits.tolist(),
            "target_probs": target_probs.tolist(),
            "param_logits": param_logits.tolist(),
            "param_probs": param_probs.tolist(),
            "masks": {
                "action_type_mask": masks["action_type_mask"].tolist(),
                "source_mask": masks["source_mask"].tolist(),
                "target_mask": masks["target_mask"].tolist(),
                "param_mask": masks["param_mask"].tolist(),
            },
            "num_legal_actions": len(available_actions),
            "captured_path": str(output_path),
            "tick": tick,
        }
    except Exception as e:
        # If encoding fails, return error response
        import traceback
        policy_response = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "captured_path": str(output_path),
            "tick": tick,
        }

    return policy_response
