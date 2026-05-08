#!/usr/bin/env python3
"""
End-to-end test: verify that the action space encoding works with the FastAPI policy.

This script:
1. Tests the action_encoding module independently
2. Simulates a policy response with masks
3. Validates that masking works correctly
"""

import json
import numpy as np
from pathlib import Path
from action_encoding import ActionSpaceEncoder


def test_encoder_basics():
    """Test basic encoder functionality."""
    print("=" * 60)
    print("TEST 1: Encoder Basics")
    print("=" * 60)
    
    encoder = ActionSpaceEncoder()
    
    print(f"✓ Action types: {encoder.action_type_size}")
    print(f"✓ Source actors: {encoder.source_actor_size}")
    print(f"✓ Target actors: {encoder.target_actor_size}")
    print(f"✓ Params: {encoder.param_size}")
    
    # Test position encoding
    pos_idx = encoder.position_to_target_index(3, 5)
    print(f"✓ Position (3, 5) -> target_actor index {pos_idx}")
    x, y = encoder.target_index_to_position(pos_idx)
    print(f"✓ Target index {pos_idx} -> position ({x}, {y})")
    assert x == 3 and y == 5, "Position encoding failed"
    
    # Test action encoding
    move_enc = encoder.encode_action("MOVE", source_actor=5, target_actor=45)
    print(f"✓ MOVE action encoded: {move_enc}")
    
    # Test decoding
    decoded = encoder.decode_action(move_enc)
    print(f"✓ Decoded back: {decoded}")
    
    print()


def test_masking_with_mock_actions():
    """Test masking logic with mock available actions."""
    print("=" * 60)
    print("TEST 2: Masking with Mock Actions")
    print("=" * 60)
    
    encoder = ActionSpaceEncoder()
    
    # Mock available actions (as they would come from Java)
    mock_actions = [
        {
            "action_type": "END_TURN",
            "class_name": "EndTurn",
            "description": "END_TURN by tribe 0",
            "index": 0,
        },
        {
            "action_type": "MOVE",
            "class_name": "Move",
            "description": "MOVE by unit 2 to 3 : 8",
            "index": 1,
        },
        {
            "action_type": "BUILD_ROAD",
            "class_name": "BuildRoad",
            "description": "BUILD_ROAD by tribe 0 at location 1 : 9",
            "index": 2,
        },
    ]
    
    # Create masks
    masks = encoder.mask_available_actions(mock_actions)
    
    print(f"✓ Created masks for {len(mock_actions)} available actions")
    print(f"  - Legal action_types: {np.sum(masks['action_type_mask'])}")
    print(f"  - Legal sources: {np.sum(masks['source_mask'])}")
    print(f"  - Legal targets: {np.sum(masks['target_mask'])}")
    print(f"  - Legal params: {np.sum(masks['param_mask'])}")
    
    # Verify END_TURN is masked
    end_turn_idx = encoder.action_type_str_to_idx["END_TURN"]
    assert masks["action_type_mask"][end_turn_idx] > 0, "END_TURN should be legal"
    print(f"✓ END_TURN is legal (action_type index {end_turn_idx})")
    
    # Verify MOVE is masked
    move_idx = encoder.action_type_str_to_idx["MOVE"]
    assert masks["action_type_mask"][move_idx] > 0, "MOVE should be legal"
    print(f"✓ MOVE is legal (action_type index {move_idx})")
    
    print()


def test_policy_response_format():
    """Test that the policy response format is correct."""
    print("=" * 60)
    print("TEST 3: Policy Response Format")
    print("=" * 60)
    
    encoder = ActionSpaceEncoder()
    
    # Simulate a policy response
    action_type_logits = np.ones(encoder.action_type_size, dtype=np.float32)
    source_logits = np.ones(encoder.source_actor_size, dtype=np.float32)
    target_logits = np.ones(encoder.target_actor_size, dtype=np.float32)
    param_logits = np.ones(encoder.param_size, dtype=np.float32)
    
    policy_response = {
        "status": "success",
        "policy_type": "uniform_masked",
        "action_type_logits": action_type_logits.tolist(),
        "source_logits": source_logits.tolist(),
        "target_logits": target_logits.tolist(),
        "param_logits": param_logits.tolist(),
        "num_legal_actions": 3,
    }
    
    # Verify the response can be JSON serialized
    json_str = json.dumps(policy_response)
    print(f"✓ Policy response serialized to JSON ({len(json_str)} bytes)")
    
    # Verify it can be deserialized
    parsed = json.loads(json_str)
    print(f"✓ Policy response deserialized successfully")
    print(f"  - Status: {parsed['status']}")
    print(f"  - Policy type: {parsed['policy_type']}")
    print(f"  - Logit shapes: ({len(parsed['action_type_logits'])}, {len(parsed['source_logits'])}, {len(parsed['target_logits'])}, {len(parsed['param_logits'])})")
    
    print()


def test_masked_softmax_behavior():
    """Verify masked softmax sets zero probability for masked entries."""
    print("=" * 60)
    print("TEST 3b: Masked Softmax Behavior")
    print("=" * 60)

    encoder = ActionSpaceEncoder()

    # Create logits with distinct values
    logits = np.arange(encoder.action_type_size, dtype=np.float32)
    # Create a mask that allows only a subset (e.g., every 3rd entry)
    mask = np.zeros(encoder.action_type_size, dtype=np.float32)
    allowed_indices = list(range(0, encoder.action_type_size, 3))
    mask[allowed_indices] = 1.0

    # Recreate server-style masked softmax
    def masked_softmax_np(logits, mask):
        masked_logits = np.where(mask > 0, logits, -1e9)
        maxv = np.max(masked_logits)
        exps = np.exp(masked_logits - maxv)
        exps = exps * (mask > 0)
        s = np.sum(exps)
        if s == 0:
            allowed = np.sum(mask > 0)
            if allowed == 0:
                return np.ones_like(exps) / len(exps)
            return (mask > 0).astype(float) / allowed
        return exps / s

    probs = masked_softmax_np(logits, mask)

    # Assert masked entries have zero probability
    masked_zero = np.all(probs[mask == 0] == 0.0)
    assert masked_zero, "Masked entries must have zero probability"
    # Assert probabilities sum to 1
    assert abs(np.sum(probs) - 1.0) < 1e-6, "Probabilities must sum to 1"

    print("✓ Masked softmax behavior verified (masked entries zero, probs sum to 1)")
    print()


def test_load_real_capture():
    """Test loading and processing a real capture from py_api/captures."""
    print("=" * 60)
    print("TEST 4: Load Real Capture (if available)")
    print("=" * 60)
    
    encoder = ActionSpaceEncoder()
    captures_dir = Path(__file__).parent / "captures"
    
    if not captures_dir.exists():
        print(f"⚠ Captures directory not found: {captures_dir}")
        print("  (Skip this test - no captures yet)")
        print()
        return
    
    capture_files = sorted(captures_dir.glob("capture_*.json"))
    if not capture_files:
        print(f"⚠ No capture files found in {captures_dir}")
        print("  (Skip this test - run a game first)")
        print()
        return
    
    latest_capture = capture_files[-1]
    print(f"✓ Loading latest capture: {latest_capture.name}")
    
    with open(latest_capture, "r") as f:
        capture = json.load(f)
    
    available_actions = capture.get("available_actions", [])
    print(f"✓ Loaded {len(available_actions)} available actions")
    
    # Try to mask them
    try:
        masks = encoder.mask_available_actions(available_actions)
        print(f"✓ Successfully created masks")
        print(f"  - Legal action_types: {int(np.sum(masks['action_type_mask']))}")
        print(f"  - Legal sources: {int(np.sum(masks['source_mask']))}")
        print(f"  - Legal targets: {int(np.sum(masks['target_mask']))}")
        print(f"  - Legal params: {int(np.sum(masks['param_mask']))}")
    except Exception as e:
        print(f"✗ Error creating masks: {e}")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ACTION SPACE ENCODING END-TO-END TESTS")
    print("=" * 60 + "\n")
    
    test_encoder_basics()
    test_masking_with_mock_actions()
    test_policy_response_format()
    test_masked_softmax_behavior()
    test_load_real_capture()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Start the FastAPI server: uvicorn app:app --reload")
    print("2. Run the game with RandomAgent (configured to use PolicyAgent)")
    print("3. Check py_api/captures/ for the generated capture files")
    print("4. Verify that the policy responses include correct masks")
    print()


if __name__ == "__main__":
    main()
