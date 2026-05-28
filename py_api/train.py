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

from model import TribesModel, StateEncoder, encode_state, TribesTransformerModel, env_bool
from action_encoding import ActionSpaceEncoder

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
        mask_send_stars: bool = False,
    ):
        self.capture_dir = Path(capture_dir)
        self.state_encoder = StateEncoder()
        self.action_encoder = ActionSpaceEncoder()
        self.mask_send_stars = mask_send_stars
        self.samples = []
        self.results = self._load_results(self.capture_dir.parent / "results")
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
        
        all_available_actions = payload.get("available_actions", [])
        visit_counts = payload.get("mcts", {}).get("visit_counts")
        available_actions, visit_counts = self._effective_actions_and_visits(all_available_actions, visit_counts)
        masks = self.action_encoder.mask_available_actions(available_actions)
        
        if visit_counts:
            target_action_type_policy, target_source_policy, target_target_policy, target_param_policy = (
                self._policy_from_visit_counts(visit_counts, available_actions, masks)
            )
        else:
            target_action_type_policy = self._uniform_masked_policy(masks["action_type_mask"])
            target_source_policy = self._uniform_masked_policy(masks["source_mask"])
            target_target_policy = self._uniform_masked_policy(masks["target_mask"])
            target_param_policy = self._uniform_masked_policy(masks["param_mask"])

        value_target, has_value_target = self._value_target(payload)
        
        return {
            "state": state,
            "action_type_policy": torch.from_numpy(target_action_type_policy).float(),
            "source_policy": torch.from_numpy(target_source_policy).float(),
            "target_policy": torch.from_numpy(target_target_policy).float(),
            "param_policy": torch.from_numpy(target_param_policy).float(),
            "value": torch.tensor(value_target, dtype=torch.float32),
            "value_weight": torch.tensor(1.0 if has_value_target else 0.0, dtype=torch.float32),
            "masks": masks,
        }

    def _has_mcts_policy(self, payload: Dict) -> bool:
        return bool(payload.get("mcts", {}).get("visit_counts"))

    def _effective_actions_and_visits(self, available_actions, visit_counts):
        if not self.mask_send_stars:
            return available_actions, visit_counts

        filtered_actions = []
        filtered_visits = [] if visit_counts is not None else None
        for idx, action in enumerate(available_actions):
            if action.get("action_type") == "SEND_STARS":
                continue
            filtered_actions.append(action)
            if filtered_visits is not None and idx < len(visit_counts):
                filtered_visits.append(visit_counts[idx])

        return filtered_actions, filtered_visits

    def _uniform_masked_policy(self, mask):
        mask_array = np.array(mask, dtype=np.float32)
        allowed = np.sum(mask_array > 0)
        if allowed == 0:
            return np.ones_like(mask_array, dtype=np.float32) / len(mask_array)
        return mask_array / allowed

    def _policy_from_visit_counts(self, visit_counts, available_actions, masks):
        action_type_counts = np.zeros(self.action_encoder.action_type_size, dtype=np.float32)
        source_counts = np.zeros(self.action_encoder.source_actor_size, dtype=np.float32)
        target_counts = np.zeros(self.action_encoder.target_actor_size, dtype=np.float32)
        param_counts = np.zeros(self.action_encoder.param_size, dtype=np.float32)

        limit = min(len(visit_counts), len(available_actions))
        for idx in range(limit):
            count = visit_counts[idx]
            if count is None or count <= 0:
                continue

            action = available_actions[idx]
            components = action.get("encoded_components", {})
            action_type_idx = components.get("action_type_index", components.get("action_type", 0))
            source_idx = components.get("source_actor_index", components.get("source_actor", 0))
            target_idx = components.get("target_actor_index", components.get("target_actor", 0))
            param_idx = components.get("param_index", components.get("param", 0))

            if 0 <= action_type_idx < action_type_counts.shape[0]:
                action_type_counts[action_type_idx] += count
            if 0 <= source_idx < source_counts.shape[0]:
                source_counts[source_idx] += count
            if 0 <= target_idx < target_counts.shape[0]:
                target_counts[target_idx] += count
            if 0 <= param_idx < param_counts.shape[0]:
                param_counts[param_idx] += count

        return (
            self._normalize_counts(action_type_counts, masks["action_type_mask"]),
            self._normalize_counts(source_counts, masks["source_mask"]),
            self._normalize_counts(target_counts, masks["target_mask"]),
            self._normalize_counts(param_counts, masks["param_mask"]),
        )

    def _normalize_counts(self, counts, mask):
        masked = counts * np.array(mask, dtype=np.float32)
        total = float(np.sum(masked))
        if total <= 0.0:
            return self._uniform_masked_policy(mask)
        return masked / total

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

    def _masked_log_softmax(self, logits, mask):
        mask = mask.bool()
        valid_rows = mask.any(dim=-1, keepdim=True)
        safe_mask = torch.where(valid_rows, mask, torch.ones_like(mask))
        mask_value = -1e4 if logits.dtype == torch.float16 else -1e9
        masked_logits = logits.masked_fill(~safe_mask, mask_value)
        return F.log_softmax(masked_logits, dim=-1)

    def _policy_loss(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -(target * log_probs).sum(dim=-1).mean()
    
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
            action_type_policy_target = batch["action_type_policy"].to(self.device)
            source_policy_target = batch["source_policy"].to(self.device)
            target_policy_target = batch["target_policy"].to(self.device)
            param_policy_target = batch["param_policy"].to(self.device)
            value_target = batch["value"].to(self.device)
            value_weight = batch["value_weight"].to(self.device)
            masks = batch["masks"]
            action_type_mask = masks["action_type_mask"].to(self.device).bool()
            source_mask = masks["source_mask"].to(self.device).bool()
            target_mask = masks["target_mask"].to(self.device).bool()
            param_mask = masks["param_mask"].to(self.device).bool()

            with torch.amp.autocast("cuda", enabled=self.use_amp):

                # Forward pass
                action_type_logits, source_logits, target_logits, param_logits, value_pred = self.model(state)

                action_type_log_probs = self._masked_log_softmax(action_type_logits, action_type_mask)
                source_log_probs = self._masked_log_softmax(source_logits, source_mask)
                target_log_probs = self._masked_log_softmax(target_logits, target_mask)
                param_log_probs = self._masked_log_softmax(param_logits, param_mask)

                policy_loss = (
                    self._policy_loss(action_type_log_probs, action_type_policy_target) +
                    self._policy_loss(source_log_probs, source_policy_target) +
                    self._policy_loss(target_log_probs, target_policy_target) +
                    self._policy_loss(param_log_probs, param_policy_target)
                ) / 4.0

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
        
        if num_batches == 0:
            raise RuntimeError("No finite training batches completed in this epoch.")

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
    mask_send_stars = env_bool("TRIBES_MASK_SEND_STARS", False)
    print(f"Mask SEND_STARS policy head: {mask_send_stars}")

    deleted = prune_capture_files(Path(args.capture_dir), max_files=args.max_captures)
    if deleted > 0:
        print(f"Pruned {deleted} old capture files (kept newest {args.max_captures}).")
    
    # Load dataset
    dataset = GameCaptureDataset(
        capture_dir=args.capture_dir,
        max_samples=args.max_samples,
        mcts_only=not args.include_unlabeled,
        mask_send_stars=mask_send_stars,
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
    )
    
    # Resume from the current self-play model so each loop continues training
    # instead of starting from a fresh random initialization.
    state_encoder = StateEncoder()
    model = TribesTransformerModel(
        state_size=state_encoder.total_state_size,
        mask_send_stars=mask_send_stars,
    )
    model_path = Path(args.model_path)
    if model_path.exists():
        try:
            model.load(str(model_path), device=str(device))
            print(f"Loaded existing model weights from {model_path}")
        except Exception as exc:
            print(f"Could not load existing model weights from {model_path}: {exc}")
            print("Starting from a new model with the current architecture.")
    
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
