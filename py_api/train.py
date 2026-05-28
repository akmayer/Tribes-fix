"""
Training script for Tribes policy/value model.

This script:
1. Loads game captures (state, available actions, game outcomes)
2. Trains the model using supervised learning on policy labels
3. Saves model weights
4. Logs training metrics

For now, this is a basic supervised learning setup. In the future, you can integrate
MCTS-based policy improvement or self-play to generate better training targets.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime
import argparse
import torch.nn.functional as F

from model import TribesModel, StateEncoder, encode_state, TribesTransformerModel

torch.backends.cudnn.benchmark = True

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prune_capture_files(capture_dir: Path, max_files: int, patterns: Optional[List[str]] = None) -> int:
    """Delete oldest capture files so at most `max_files` remain.

    Prunes only files matching `patterns` (defaults to training captures: capture_*.json + mcts_*.json).
    Returns the number of files deleted.
    """
    if max_files is None or max_files <= 0:
        return 0
    capture_dir = Path(capture_dir)
    if not capture_dir.exists():
        return 0

    if patterns is None:
        patterns = ["capture_*.json", "mcts_*.json"]

    files: List[Path] = []
    for pattern in patterns:
        files.extend(capture_dir.glob(pattern))

    # Deduplicate and sort by mtime (oldest first)
    unique_files = list({p.resolve(): p for p in files}.values())
    unique_files.sort(key=lambda p: p.stat().st_mtime)

    if len(unique_files) <= max_files:
        return 0

    to_delete = unique_files[: max(0, len(unique_files) - max_files)]
    deleted = 0
    for path in to_delete:
        try:
            path.unlink()
            deleted += 1
        except Exception as exc:
            print(f"Failed to delete {path}: {exc}")
    return deleted


class GameCaptureDataset(Dataset):
    """
    Dataset that loads game captures from files.
    Each capture contains: state, available actions, and their masks.
    """
    
    def __init__(
        self,
        capture_dir: Path = Path("captures"),
        max_samples: Optional[int] = None,
        mcts_only: bool = True,
    ):
        self.capture_dir = Path(capture_dir)
        self.state_encoder = StateEncoder()
        self.samples = []
        self.results = self._load_results(Path("results"))
        self.mcts_samples = 0
        self.value_samples = 0
        self.mcts_only = mcts_only
        
        # Load all capture files
        if self.capture_dir.exists():
            capture_files = sorted(self.capture_dir.glob("capture_*.json"))
            capture_files += sorted(self.capture_dir.glob("mcts_*.json"))
            for capture_file in capture_files[:max_samples] if max_samples else capture_files:
                try:
                    with open(capture_file, "r") as f:
                        payload = json.load(f)
                    if self.mcts_only and not self._has_mcts_policy(payload):
                        continue
                    self.samples.append(payload)
                except Exception as e:
                    print(f"Failed to load {capture_file}: {e}")
        
        print(f"Loaded {len(self.samples)} captures from {self.capture_dir}")
        self.mcts_samples = sum(1 for sample in self.samples if self._has_mcts_policy(sample))
        self.value_samples = sum(1 for sample in self.samples if self._value_target(sample)[1])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        payload = self.samples[idx]
        
        # Encode state
        state = encode_state(payload, self.state_encoder)
        
        available_actions = payload.get("available_actions", [])
        
        mcts_policy = payload.get("mcts", {})
        visit_counts = mcts_policy.get("visit_counts")

        if visit_counts:
            legal_action_policy = self._legal_action_policy(visit_counts, available_actions)
        else:
            legal_action_policy = self._uniform_legal_action_policy(available_actions)

        action_components, component_mask = self._legal_action_components(available_actions)

        value_target, has_value_target = self._value_target(payload)
        
        return {
            "state": state,
            "action_components": torch.from_numpy(action_components).long(),
            "action_component_mask": torch.from_numpy(component_mask).float(),
            "legal_action_policy": torch.from_numpy(legal_action_policy).float(),
            "value": torch.tensor(value_target, dtype=torch.float32),
            "value_weight": torch.tensor(1.0 if has_value_target else 0.0, dtype=torch.float32),
        }

    def _has_mcts_policy(self, payload: Dict) -> bool:
        return bool(payload.get("mcts", {}).get("visit_counts"))

    def _legal_action_policy(self, visit_counts, available_actions):
        counts = np.zeros(len(available_actions), dtype=np.float32)
        limit = min(len(visit_counts), len(available_actions))
        for idx in range(limit):
            count = visit_counts[idx]
            if count is not None and count > 0:
                counts[idx] = float(count)

        total = float(np.sum(counts))
        if total <= 0.0:
            return self._uniform_legal_action_policy(available_actions)
        return counts / total

    def _uniform_legal_action_policy(self, available_actions):
        n_actions = len(available_actions)
        if n_actions <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.ones(n_actions, dtype=np.float32) / n_actions

    def _legal_action_components(self, available_actions):
        components = np.zeros((len(available_actions), 4), dtype=np.int64)
        component_mask = np.zeros((len(available_actions), 4), dtype=np.float32)

        for idx, action in enumerate(available_actions):
            encoded = action.get("encoded_components", {})
            components[idx, 0] = encoded.get("action_type_index", encoded.get("action_type", 0))
            components[idx, 1] = encoded.get("source_actor_index", encoded.get("source_actor", 0))
            components[idx, 2] = encoded.get("target_actor_index", encoded.get("target_actor", 0))
            components[idx, 3] = encoded.get("param_index", encoded.get("param", 0))
            component_mask[idx] = self._component_mask_for_action(action.get("action_type"))

        return components, component_mask

    def _component_mask_for_action(self, action_type):
        mask = np.zeros(4, dtype=np.float32)
        mask[0] = 1.0

        if action_type in {"MOVE", "ATTACK", "CAPTURE", "CONVERT"}:
            mask[1] = 1.0
            mask[2] = 1.0
        elif action_type in {"BUILD_ROAD", "DECLARE_WAR"}:
            mask[2] = 1.0
        elif action_type == "SEND_STARS":
            mask[2] = 1.0
            mask[3] = 1.0
        elif action_type == "RESEARCH_TECH":
            mask[3] = 1.0
        elif action_type == "BUILD":
            mask[1] = 1.0
            mask[2] = 1.0
            mask[3] = 1.0
        elif action_type == "SPAWN":
            mask[1] = 1.0
            mask[3] = 1.0
        elif action_type in {"BURN_FOREST", "CLEAR_FOREST", "DESTROY", "GROW_FOREST"}:
            mask[1] = 1.0
            mask[2] = 1.0
        elif action_type == "LEVEL_UP":
            mask[1] = 1.0
            mask[3] = 1.0
        elif action_type == "RESOURCE_GATHERING":
            mask[1] = 1.0
        elif action_type in {
            "DISBAND", "EXAMINE", "HEAL_OTHERS", "MAKE_VETERAN", "RECOVER",
            "CLIMB_MOUNTAIN", "UPGRADE_BOAT", "UPGRADE_SHIP",
        }:
            mask[1] = 1.0

        return mask

    def _value_target(self, payload):
        game_seed = payload.get("game_seed")
        player_id = payload.get("player_id")
        if game_seed is not None and player_id is not None:
            key = self._result_key(game_seed, player_id)
            if key is not None and key in self.results:
                return self.results[key], True

        if "value_target" in payload:
            try:
                return float(payload["value_target"]), True
            except (TypeError, ValueError):
                pass

        return 0.0, False

    def _load_results(self, results_dir: Path) -> Dict[Tuple[int, int], float]:
        results: Dict[Tuple[int, int], float] = {}
        if not results_dir.exists():
            return results
        for result_file in sorted(results_dir.glob("result_*.json")):
            try:
                with open(result_file, "r") as handle:
                    data = json.load(handle)
                key = self._result_key(data.get("game_seed"), data.get("player_id"))
                if key is None:
                    continue
                results[key] = self._value_from_result(data)
            except Exception as e:
                print(f"Failed to load {result_file}: {e}")
        return results

    def _result_key(self, game_seed, player_id):
        try:
            return (int(game_seed), int(player_id))
        except (TypeError, ValueError):
            return None

    def _value_from_result(self, result: Dict) -> float:
        if "value" in result:
            try:
                return float(result["value"])
            except (TypeError, ValueError):
                pass

        winner = result.get("winner")
        if winner == "WIN":
            return 1.0
        if winner == "LOSS":
            return -1.0
        if winner == "DRAW":
            return 0.0

        score = result.get("reward")
        try:
            score_val = float(score)
            return float(np.tanh(score_val / 1000.0))
        except (TypeError, ValueError):
            return 0.0


def collate_game_captures(batch: List[Dict]) -> Dict:
    states = torch.stack([item["state"] for item in batch])
    values = torch.stack([item["value"] for item in batch])
    value_weights = torch.stack([item["value_weight"] for item in batch])

    max_actions = max(item["action_components"].shape[0] for item in batch)
    batch_size = len(batch)

    action_components = torch.zeros((batch_size, max_actions, 4), dtype=torch.long)
    action_component_mask = torch.zeros((batch_size, max_actions, 4), dtype=torch.float32)
    legal_action_policy = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    legal_action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)

    for idx, item in enumerate(batch):
        n_actions = item["action_components"].shape[0]
        if n_actions <= 0:
            continue
        action_components[idx, :n_actions] = item["action_components"]
        action_component_mask[idx, :n_actions] = item["action_component_mask"]
        legal_action_policy[idx, :n_actions] = item["legal_action_policy"]
        legal_action_mask[idx, :n_actions] = True

    return {
        "state": states,
        "action_components": action_components,
        "action_component_mask": action_component_mask,
        "legal_action_policy": legal_action_policy,
        "legal_action_mask": legal_action_mask,
        "value": values,
        "value_weight": value_weights,
    }


class PolicyValueTrainer:
    """Handles training loop for policy and value heads."""
    
    def __init__(
        self,
        model: TribesTransformerModel,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 0.1,
    ):
        self.model = model.to(device)
        self.device = device
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        self.use_amp = device.startswith("cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.train_log = []

    def _legal_action_scores(
        self,
        action_type_logits: torch.Tensor,
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
        param_logits: torch.Tensor,
        action_components: torch.Tensor,
        component_mask: torch.Tensor,
    ) -> torch.Tensor:
        action_type_idx = action_components[:, :, 0]
        source_idx = action_components[:, :, 1]
        target_idx = action_components[:, :, 2]
        param_idx = action_components[:, :, 3]

        score = torch.gather(action_type_logits, 1, action_type_idx)
        score = score + torch.gather(source_logits, 1, source_idx) * component_mask[:, :, 1]
        score = score + torch.gather(target_logits, 1, target_idx) * component_mask[:, :, 2]
        score = score + torch.gather(param_logits, 1, param_idx) * component_mask[:, :, 3]
        return score

    def _policy_loss(
        self,
        action_type_logits: torch.Tensor,
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
        param_logits: torch.Tensor,
        action_components: torch.Tensor,
        component_mask: torch.Tensor,
        legal_action_mask: torch.Tensor,
        target_policy: torch.Tensor,
    ) -> torch.Tensor:
        scores = self._legal_action_scores(
            action_type_logits,
            source_logits,
            target_logits,
            param_logits,
            action_components,
            component_mask,
        )
        mask_value = -1e4 if scores.dtype == torch.float16 else -1e9
        scores = scores.masked_fill(~legal_action_mask, mask_value)
        log_probs = F.log_softmax(scores, dim=-1)
        return -(target_policy * log_probs).sum(dim=-1).mean()
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            state = batch["state"].to(self.device)
            action_components = batch["action_components"].to(self.device)
            action_component_mask = batch["action_component_mask"].to(self.device)
            legal_action_policy = batch["legal_action_policy"].to(self.device)
            legal_action_mask = batch["legal_action_mask"].to(self.device)
            value_target = batch["value"].to(self.device)
            value_weight = batch["value_weight"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):

                # Forward pass
                action_type_logits, source_logits, target_logits, param_logits, value_pred = self.model(state)

                policy_loss = self._policy_loss(
                    action_type_logits,
                    source_logits,
                    target_logits,
                    param_logits,
                    action_components,
                    action_component_mask,
                    legal_action_mask,
                    legal_action_policy,
                )

                value_prediction = torch.tanh(value_pred.float().squeeze(-1))
                value_target = value_target.float()
                value_error = (value_prediction - value_target).pow(2)
                if torch.sum(value_weight) > 0:
                    value_loss = torch.sum(value_error * value_weight) / torch.sum(value_weight)
                else:
                    value_loss = torch.zeros((), device=self.device)

                loss = (
                    self.policy_loss_weight * policy_loss +
                    self.value_loss_weight * value_loss
                )

            # Backward pass
            self.optimizer.zero_grad()
            if not torch.isfinite(loss):
                print(f"Skipping non-finite loss at batch {batch_idx}: {loss.item()}")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Accumulate metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                print(
                    f"  Batch {batch_idx+1}/{len(train_loader)}: "
                    f"loss={loss.item():.4f}, policy_loss={policy_loss.item():.4f}, "
                    f"value_loss={value_loss.item():.4f}"
                )
        
        metrics = {
            "epoch": epoch,
            "avg_loss": total_loss / num_batches,
            "avg_policy_loss": total_policy_loss / num_batches,
            "avg_value_loss": total_value_loss / num_batches,
        }
        
        self.train_log.append(metrics)
        return metrics
    

def main():
    parser = argparse.ArgumentParser(description="Train Tribes policy/value model")
    parser.add_argument("--capture-dir", type=str, default="captures",
                        help="Directory containing game captures")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--policy-loss-weight", type=float, default=1.0,
                        help="Weight for policy loss")
    parser.add_argument("--value-loss-weight", type=float, default=0.1,
                        help="Weight for value loss")
    parser.add_argument("--model-path", type=str, default="model_weights.pth",
                        help="Path to save model weights")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of samples to load (for testing)")
    parser.add_argument("--include-unlabeled", action="store_true",
                        help="Also train on captures without MCTS visit counts using uniform policy targets")
    parser.add_argument("--max-captures", type=int, default=10000,
                        help="Max number of capture files to keep (oldest deleted) before training; set 0 to disable")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cpu or cuda)"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    
    print(f"Device: {device}")
    print(f"Loading data from: {args.capture_dir}")

    deleted = prune_capture_files(Path(args.capture_dir), max_files=args.max_captures)
    if deleted > 0:
        print(f"Pruned {deleted} old capture files (kept newest {args.max_captures}).")
    
    # Load dataset
    dataset = GameCaptureDataset(
        capture_dir=args.capture_dir,
        max_samples=args.max_samples,
        mcts_only=not args.include_unlabeled,
    )
    
    if len(dataset) == 0:
        print("No usable captures found. Run AZ_MCTS self-play to generate MCTS-labeled captures first.")
        if not args.include_unlabeled:
            print("Use --include-unlabeled only for debugging; uniform policy targets are not AlphaZero training data.")
        return
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_game_captures,
    )
    
    # Create model
    state_encoder = StateEncoder()
    model = TribesTransformerModel(state_size=state_encoder.total_state_size)
    
    # Initialize trainer
    trainer = PolicyValueTrainer(
        model=model,
        device=str(device),
        learning_rate=args.learning_rate,
        policy_loss_weight=args.policy_loss_weight,
        value_loss_weight=args.value_loss_weight,
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Dataset size: {len(dataset)}")
    print(f"MCTS-labeled samples: {dataset.mcts_samples}")
    print(f"Final-value samples: {dataset.value_samples}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        metrics = trainer.train_epoch(train_loader, epoch=epoch)
        
        print(f"  Avg loss: {metrics['avg_loss']:.4f}")
        print(f"  Avg policy loss: {metrics['avg_policy_loss']:.4f}")
        print(f"  Avg value loss: {metrics['avg_value_loss']:.4f}")
        
        model.save(args.model_path)
        print(f"Updated model weights at {args.model_path}")
    
    # Save final model weights
    print(f"\nSaving final model weights to {args.model_path}")
    model.save(args.model_path)
    
    # Print training summary
    print("\n--- Training Summary ---")
    print(f"Epochs trained: {args.epochs}")
    print(f"Final loss: {metrics['avg_loss']:.4f}")
    print(f"Model weights: {args.model_path}")
    
    # Note about training data quality
    if dataset.mcts_samples == 0:
        print("\nIMPORTANT: No MCTS visit counts were found.")
        print("Generate captures via AZ_MCTS to use AlphaZero-style policy targets.")
    else:
        print("\nUsing MCTS visit counts for policy targets when available.")
    if dataset.value_samples == 0:
        print("No completed-game value targets were found; value loss was skipped.")
    print("Ensure fog-of-war is enforced in PythonBridge.java for valid data.")


if __name__ == "__main__":
    main()
