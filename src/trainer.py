"""
Training loop for the AlphaZero chess agent.

Trains the neural network on self-play data using:
  - Policy loss: Cross-entropy between MCTS visit distribution and predicted policy
  - Value loss: MSE between game outcome and predicted value
  - L2 regularization on all weights

Uses SGD with momentum and cosine annealing LR schedule, matching the
original AlphaZero paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from collections import deque
import time
import logging
import yaml

from .model import ChessNet, save_model, load_model, get_device

logger = logging.getLogger(__name__)


class ChessDataset(Dataset):
    """Dataset of (state, policy, value) training examples."""

    def __init__(self, states, policies, values):
        self.states = torch.from_numpy(states).float()
        self.policies = torch.from_numpy(policies).float()
        self.values = torch.from_numpy(values).float()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


class ReplayBuffer:
    """Sliding window replay buffer for training data."""

    def __init__(self, max_size: int = 500000):
        self.max_size = max_size
        self.states = deque(maxlen=max_size)
        self.policies = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)

    def add(self, examples: list):
        """Add training examples: list of (state, policy, value) tuples."""
        for state, policy, value in examples:
            self.states.append(state)
            self.policies.append(policy)
            self.values.append(value)

    def load_from_files(self, data_dir: str, max_files: int = None):
        """Load training data from saved .npz files."""
        data_path = Path(data_dir)
        files = sorted(data_path.glob("cycle_*.npz"))
        if max_files:
            files = files[-max_files:]

        total_loaded = 0
        for f in files:
            data = np.load(f)
            n = len(data["states"])
            for i in range(n):
                self.states.append(data["states"][i])
                self.policies.append(data["policies"][i])
                self.values.append(data["values"][i])
            total_loaded += n

        logger.info(f"Loaded {total_loaded} positions from {len(files)} files")

    def get_dataset(self) -> ChessDataset:
        """Create a PyTorch Dataset from the buffer contents."""
        return ChessDataset(
            np.array(self.states),
            np.array(self.policies),
            np.array(self.values),
        )

    def __len__(self):
        return len(self.states)


class Trainer:
    """Handles neural network training."""

    def __init__(self, model: ChessNet, config: dict, log_dir: str = "logs"):
        self.model = model
        self.config = config
        self.train_cfg = config["training"]
        self.device = next(model.parameters()).device

        # Optimizer: SGD with momentum (matching AlphaZero paper)
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=self.train_cfg["learning_rate"],
            momentum=self.train_cfg["momentum"],
            weight_decay=self.train_cfg["weight_decay"],
        )

        # LR scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.train_cfg["lr_milestones"],
            gamma=self.train_cfg["lr_gamma"],
        )

        # Mixed precision
        self.use_amp = self.train_cfg.get("use_mixed_precision", False)
        self.scaler = torch.amp.GradScaler() if self.use_amp and self.device.type == "cuda" else None

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.training_cycle = 0

    def train_on_examples(self, replay_buffer: ReplayBuffer) -> dict:
        """Train the model on data from the replay buffer.

        Returns dict of training metrics.
        """
        if len(replay_buffer) < self.train_cfg.get("min_positions_to_train", 1000):
            logger.warning(
                f"Not enough data to train: {len(replay_buffer)} < "
                f"{self.train_cfg['min_positions_to_train']}"
            )
            return {}

        dataset = replay_buffer.get_dataset()
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
            num_workers=0,  # Keep 0 for MPS compatibility
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )

        self.model.train()
        epochs = self.train_cfg.get("epochs_per_cycle", 1)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            for states, target_policies, target_values in dataloader:
                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        loss, p_loss, v_loss = self._compute_loss(
                            states, target_policies, target_values
                        )
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss, p_loss, v_loss = self._compute_loss(
                        states, target_policies, target_values
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += p_loss.item()
                total_value_loss += v_loss.item()
                num_batches += 1

                # TensorBoard logging
                self.writer.add_scalar("loss/total", loss.item(), self.global_step)
                self.writer.add_scalar("loss/policy", p_loss.item(), self.global_step)
                self.writer.add_scalar("loss/value", v_loss.item(), self.global_step)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], self.global_step)
                self.global_step += 1

        self.scheduler.step()
        self.training_cycle += 1

        metrics = {
            "total_loss": total_loss / max(num_batches, 1),
            "policy_loss": total_policy_loss / max(num_batches, 1),
            "value_loss": total_value_loss / max(num_batches, 1),
            "num_batches": num_batches,
            "num_positions": len(replay_buffer),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        logger.info(
            f"Training cycle {self.training_cycle}: "
            f"loss={metrics['total_loss']:.4f} "
            f"(policy={metrics['policy_loss']:.4f}, value={metrics['value_loss']:.4f}), "
            f"lr={metrics['lr']:.6f}, "
            f"{num_batches} batches, {len(replay_buffer)} positions"
        )

        self.writer.add_scalar("cycle/total_loss", metrics["total_loss"], self.training_cycle)
        self.writer.add_scalar("cycle/positions", len(replay_buffer), self.training_cycle)

        return metrics

    def _compute_loss(self, states, target_policies, target_values):
        """Compute combined policy + value loss."""
        policy_logits, predicted_values = self.model(states)

        # Policy loss: cross-entropy with MCTS visit distribution
        # Using log_softmax + manual cross-entropy for numerical stability
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(target_policies * log_probs, dim=1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(predicted_values.squeeze(-1), target_values)

        # Combined loss (no separate L2 term since we use weight_decay in optimizer)
        total_loss = policy_loss + value_loss

        return total_loss, policy_loss, value_loss

    def save_checkpoint(self, path: str, extra_metadata: dict = None):
        """Save training checkpoint."""
        metadata = {
            "training_cycle": self.training_cycle,
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        save_model(self.model, path, metadata)

    def load_checkpoint(self, path: str):
        """Load training state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "metadata" in checkpoint:
            meta = checkpoint["metadata"]
            if "optimizer_state" in meta:
                self.optimizer.load_state_dict(meta["optimizer_state"])
            if "scheduler_state" in meta:
                self.scheduler.load_state_dict(meta["scheduler_state"])
            self.training_cycle = meta.get("training_cycle", 0)
            self.global_step = meta.get("global_step", 0)
        logger.info(f"Loaded checkpoint from {path} (cycle {self.training_cycle})")
