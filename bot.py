#!/usr/bin/env python3
"""
Lichess Bot Runner

Connects to Lichess and plays games autonomously using the trained model.

Usage:
  python bot.py                                # Use latest champion model
  python bot.py --model data/models/best.pt    # Use specific model
  python bot.py --token YOUR_TOKEN             # Override Lichess token

Setup:
  1. Create a Lichess Bot account: https://lichess.org/api#tag/Bot/operation/botAccountUpgrade
  2. Generate an API token: https://lichess.org/account/oauth/token/create
     - Check "Play games with the bot API" scope
  3. Set token in config.yaml or LICHESS_TOKEN environment variable
  4. Run this script!
"""

import argparse
import logging
import sys
import yaml
import os
from pathlib import Path

from src.model import load_model, create_model_from_config, get_device
from src.lichess_bot import LichessBot


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/bot.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Lichess Bot")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--model", default=None, help="Model checkpoint path")
    parser.add_argument("--token", default=None, help="Lichess API token")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    Path("logs").mkdir(exist_ok=True)
    setup_logging(config["system"].get("log_level", "INFO"))
    logger = logging.getLogger("bot")

    # Override token if provided
    if args.token:
        config["lichess"]["token"] = args.token
    elif os.environ.get("LICHESS_TOKEN"):
        config["lichess"]["token"] = os.environ["LICHESS_TOKEN"]

    # Load model
    device = get_device(config["system"]["device"])
    model_path = args.model or "data/models/champion.pt"

    if Path(model_path).exists():
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path, device)
    else:
        logger.warning(f"No model found at {model_path}, creating fresh model")
        model = create_model_from_config(args.config)

    model.eval()
    logger.info(f"Model loaded: {model.num_parameters:,} params on {device}")

    # Start bot
    bot = LichessBot(model, config)

    logger.info("=" * 50)
    logger.info(f"  Chess Agent Bot: {bot.bot_name}")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  MCTS sims: {config['lichess']['num_simulations']}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 50)
    logger.info("Waiting for challenges... (Ctrl+C to stop)")

    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        bot.stop()


if __name__ == "__main__":
    main()
