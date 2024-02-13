"""
AlphaZero neural network for chess.

Architecture: Residual CNN with dual heads (policy + value).
  - Input: 18x8x8 board encoding
  - Body: Conv block + N residual blocks
  - Policy head: Conv -> FC -> 4672 (softmax over moves)
  - Value head: Conv -> FC -> 1 (tanh, game outcome prediction)

Optimized for Apple Silicon MPS backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Initial convolution block: Conv2d + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block: two conv+bn layers with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class ChessNet(nn.Module):
    """AlphaZero-style dual-headed residual network for chess."""

    def __init__(
        self,
        input_planes: int = 18,
        num_filters: int = 128,
        num_residual_blocks: int = 10,
        policy_output_size: int = 4672,
        l2_reg: float = 1e-4,
    ):
        super().__init__()
        self.input_planes = input_planes
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks
        self.policy_output_size = policy_output_size
        self.l2_reg = l2_reg

        # Initial convolution
        self.conv_block = ConvBlock(input_planes, num_filters)

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_output_size)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Tensor of shape (batch, 18, 8, 8)

        Returns:
            policy_logits: (batch, 4672) raw logits (apply softmax externally)
            value: (batch, 1) game value in [-1, 1]
        """
        # Shared body
        out = self.conv_block(x)
        out = self.residual_tower(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, board_tensor: torch.Tensor) -> tuple:
        """Single inference with no gradient tracking.

        Args:
            board_tensor: (18, 8, 8) or (batch, 18, 8, 8)

        Returns:
            policy: numpy array of move probabilities
            value: scalar value prediction
        """
        self.eval()
        with torch.no_grad():
            if board_tensor.dim() == 3:
                board_tensor = board_tensor.unsqueeze(0)

            device = next(self.parameters()).device
            board_tensor = board_tensor.to(device)

            policy_logits, value = self(board_tensor)

            policy = F.softmax(policy_logits, dim=1).cpu().numpy()
            value = value.cpu().numpy()

        return policy[0], float(value[0, 0])

    def get_l2_reg_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss over all parameters."""
        l2_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_reg * l2_loss

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def get_device(config_device: str = "auto") -> torch.device:
    """Determine the best available device."""
    if config_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(config_device)


def save_model(model: ChessNet, path: str, metadata: dict = None):
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "input_planes": model.input_planes,
            "num_filters": model.num_filters,
            "num_residual_blocks": model.num_residual_blocks,
            "policy_output_size": model.policy_output_size,
            "l2_reg": model.l2_reg,
        },
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str, device: torch.device = None) -> ChessNet:
    """Load model from checkpoint."""
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ChessNet(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    logger.info(f"Model loaded from {path} ({model.num_parameters:,} params)")
    return model


def create_model_from_config(config_path: str = "config.yaml") -> ChessNet:
    """Create a new model from YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    model = ChessNet(
        input_planes=model_cfg["input_planes"],
        num_filters=model_cfg["num_filters"],
        num_residual_blocks=model_cfg["num_residual_blocks"],
        policy_output_size=model_cfg["policy_output_size"],
        l2_reg=model_cfg["l2_reg"],
    )

    device = get_device(cfg["system"]["device"])
    model.to(device)
    logger.info(f"Created model with {model.num_parameters:,} params on {device}")
    return model
