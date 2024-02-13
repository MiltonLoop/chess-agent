#!/bin/bash
# ============================================================
# Chess Agent - Mac Mini Setup Script
# ============================================================
# Run this once to set up the environment on your Mac Mini:
#   chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

echo "=========================================="
echo "  Chess Agent Setup"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Install it via:"
    echo "  brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for Apple Silicon
echo ""
echo "Installing PyTorch (with MPS support for Apple Silicon)..."
pip install torch torchvision torchaudio

# Install other dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "Creating data directories..."
mkdir -p data/models data/self_play data/games logs

# Verify MPS (Apple GPU) support
echo ""
echo "Checking device support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('Apple Silicon GPU acceleration is available!')
else:
    print('Warning: MPS not available, will use CPU.')
    print('Make sure you are on macOS 12.3+ with Apple Silicon.')
"

# Verify imports
echo ""
echo "Verifying installation..."
python3 -c "
import chess
import torch
import berserk
import yaml
import numpy
print('All dependencies OK!')
"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo ""
echo "  # Train the agent (self-play learning):"
echo "  python train.py"
echo ""
echo "  # Play against it in the terminal:"
echo "  python play.py"
echo ""
echo "  # Run as Lichess bot:"
echo "  export LICHESS_TOKEN=your_token_here"
echo "  python bot.py"
echo ""
echo "  # Full daemon (train + play online 24/7):"
echo "  export LICHESS_TOKEN=your_token_here"
echo "  python run.py"
echo ""
echo "  # Monitor training with TensorBoard:"
echo "  tensorboard --logdir logs/tensorboard"
echo ""
