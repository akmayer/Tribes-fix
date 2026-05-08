"""
Training script for Tribes policy/value model.

This script:
1. Loads game captures (state, available actions, game outcomes)
2. Trains the model using supervised learning on policy labels
3. Saves checkpoints
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

from model import TribesModel, StateEncoder, encode_state
from action_encoding import ActionSpaceEncoder


class GameCaptureDataset(Dataset):
    """
    Dataset that loads game captures from files.
    Each capture contains: state, available actions, and their masks.
    """
    
    def __init__(self, capture_dir: Path = Path("captures"), max_samples: Optional[int] = None):
        self.capture_dir = Path(capture_dir)
        self.state_encoder = StateEncoder()
        self.action_encoder = ActionSpaceEncoder()
        self.samples = []
        
        # Load all capture files
        if self.capture_dir.exists():
            capture_files = sorted(self.capture_dir.glob("capture_*.json"))
            for capture_file in capture_files[:max_samples] if max_samples else capture_files:
                try:
                    with open(capture_file, "r") as f:
                        payload = json.load(f)
                    self.samples.append(payload)
                except Exception as e:
                    print(f"Failed to load {capture_file}: {e}")
        
        print(f"Loaded {len(self.samples)} captures from {self.capture_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        payload = self.samples[idx]
        
        # Encode state
        state = encode_state(payload, self.state_encoder)
        
        # Get masks for available actions
        available_actions = payload.get("available_actions", [])
        masks = self.action_encoder.mask_available_actions(available_actions)
        
        # For now, use uniform policy over masked actions as training target
        # In the future, you can replace this with:
        # - MCTS-improved policy targets
        # - Self-play outcomes
        # - Human expert demonstrations
        def uniform_masked_policy(mask):
            """Create uniform distribution over legal actions."""
            mask_array = np.array(mask, dtype=np.float32)
            allowed = np.sum(mask_array > 0)
            if allowed == 0:
                return np.ones_like(mask_array, dtype=np.float32) / len(mask_array)
            policy = mask_array.astype(np.float32) / allowed
            return policy
        
        target_action_type_policy = uniform_masked_policy(masks["action_type_mask"])
        target_source_policy = uniform_masked_policy(masks["source_mask"])
        target_target_policy = uniform_masked_policy(masks["target_mask"])
        target_param_policy = uniform_masked_policy(masks["param_mask"])
        
        # Dummy value target (for now, just random between -1 and 1)
        # In training, you'd use actual game outcomes
        value_target = np.random.uniform(-1, 1)
        
        return {
            "state": state,
            "action_type_policy": torch.from_numpy(target_action_type_policy).float(),
            "source_policy": torch.from_numpy(target_source_policy).float(),
            "target_policy": torch.from_numpy(target_target_policy).float(),
            "param_policy": torch.from_numpy(target_param_policy).float(),
            "value": torch.tensor(value_target, dtype=torch.float32),
            "masks": masks,
        }


class PolicyValueTrainer:
    """Handles training loop for policy and value heads."""
    
    def __init__(
        self,
        model: TribesModel,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 0.1,
    ):
        self.model = model.to(device)
        self.device = device
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.value_loss_fn = nn.MSELoss()
        
        self.train_log = []
    
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
            
            # Forward pass
            action_type_logits, source_logits, target_logits, param_logits, value_pred = self.model(state)
            
            # Compute policy loss (KL divergence)
            # Convert logits to log-probabilities for KL divergence
            action_type_log_probs = torch.log_softmax(action_type_logits, dim=-1)
            source_log_probs = torch.log_softmax(source_logits, dim=-1)
            target_log_probs = torch.log_softmax(target_logits, dim=-1)
            param_log_probs = torch.log_softmax(param_logits, dim=-1)
            
            policy_loss = (
                self.policy_loss_fn(action_type_log_probs, action_type_policy_target) +
                self.policy_loss_fn(source_log_probs, source_policy_target) +
                self.policy_loss_fn(target_log_probs, target_policy_target) +
                self.policy_loss_fn(param_log_probs, param_policy_target)
            ) / 4.0  # Average over 4 heads
            
            # Compute value loss
            value_loss = self.value_loss_fn(value_pred.squeeze(-1), value_target)
            
            # Combined loss
            loss = self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
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
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from {path}")
        return checkpoint["epoch"]


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
    parser.add_argument("--model-path", type=str, default="model_weights.pth",
                        help="Path to save model weights")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save training checkpoints")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of samples to load (for testing)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Loading data from: {args.capture_dir}")
    
    # Load dataset
    dataset = GameCaptureDataset(capture_dir=args.capture_dir, max_samples=args.max_samples)
    
    if len(dataset) == 0:
        print("No captures found. Run the game with RandomAgent to generate captures first.")
        return
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Create model
    state_encoder = StateEncoder()
    model = TribesModel(state_size=state_encoder.total_state_size)
    
    # Initialize trainer
    trainer = PolicyValueTrainer(
        model=model,
        device=str(device),
        learning_rate=args.learning_rate,
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume_from and Path(args.resume_from).exists():
        start_epoch = trainer.load_checkpoint(args.resume_from)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        metrics = trainer.train_epoch(train_loader, epoch=epoch)
        
        print(f"  Avg loss: {metrics['avg_loss']:.4f}")
        print(f"  Avg policy loss: {metrics['avg_policy_loss']:.4f}")
        print(f"  Avg value loss: {metrics['avg_value_loss']:.4f}")
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        trainer.save_checkpoint(str(checkpoint_path), epoch+1, metrics)
    
    # Save final model weights
    print(f"\nSaving final model weights to {args.model_path}")
    model.save(args.model_path)
    
    # Print training summary
    print("\n--- Training Summary ---")
    print(f"Epochs trained: {args.epochs}")
    print(f"Final loss: {metrics['avg_loss']:.4f}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Model weights: {args.model_path}")
    
    # Note about training data quality
    print("\n⚠️  IMPORTANT: This training used uniform policies over masked actions as targets.")
    print("For better results, integrate:")
    print("  - MCTS-based policy improvement")
    print("  - Self-play with outcome labels")
    print("  - Human expert demonstrations")
    print("  - Proper fog-of-war enforcement in PythonBridge.java")


if __name__ == "__main__":
    main()
