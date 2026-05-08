import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from action_encoding import ActionSpaceEncoder

app = FastAPI()

CAPTURE_DIR = Path("captures")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# Load the action space encoder
encoder = ActionSpaceEncoder()


@app.get("/hello")
async def hello():
    return {"message": "hello from python"}


@app.post("/query")
async def query(req: Request):
    """
    Receive game state and available actions from Java.
    Return a policy (action logits) over available actions.
    
    For now, returns a uniform distribution over masked legal actions as a dummy policy.
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
        
        # For dummy policy: uniform distribution over masked actions
        action_type_logits = np.ones(encoder.action_type_size, dtype=np.float32)
        source_logits = np.ones(encoder.source_actor_size, dtype=np.float32)
        target_logits = np.ones(encoder.target_actor_size, dtype=np.float32)
        param_logits = np.ones(encoder.param_size, dtype=np.float32)
        
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
            "policy_type": "uniform_masked",
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
        policy_response = {
            "status": "error",
            "error": str(e),
            "captured_path": str(output_path),
            "tick": tick,
        }

    return policy_response
