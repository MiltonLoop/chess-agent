#!/usr/bin/env python3
"""
AlphaZero Training Pipeline

Runs the full AlphaZero training loop:
  1. Self-play: Generate games using current best model
  2. Train: Update neural network on self-play data
  3. Evaluate: Pit new model against champion in arena
  4. Promote: If new model wins, it becomes the champion
  5. Repeat

Usage:
  python train.py                    # Start fresh training
  python train.py --resume           # Resume from latest checkpoint
  python train.py --config my.yaml   # Use custom config
"""

import argparse
import logging
import sys
import time
import yaml
import copy
import torch
from pathlib import Path

from src.model import ChessNet, save_model, load_model, get_device, create_model_from_config
from src.self_play import run_self_play_cycle
from src.trainer import Trainer, ReplayBuffer
from src.arena import evaluate_models


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/training.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Chess Training")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--cycles", type=int, default=1000, help="Number of training cycles")
    parser.add_argument("--skip-arena", action="store_true", help="Skip arena evaluation")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup
    Path("logs").mkdir(exist_ok=True)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("data/self_play").mkdir(parents=True, exist_ok=True)
    setup_logging(config["system"].get("log_level", "INFO"))
    logger = logging.getLogger("train")

    # Set seed
    seed = config["system"].get("seed", 42)
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)

    # Device
    device = get_device(config["system"]["device"])
    logger.info(f"Using device: {device}")

    # Create or load model
    champion_path = Path("data/models/champion.pt")
    if args.resume and champion_path.exists():
        logger.info("Resuming from champion checkpoint...")
        champion = load_model(str(champion_path), device)
    else:
        logger.info("Creating new model...")
        champion = create_model_from_config(args.config)
        save_model(champion, str(champion_path), {"cycle": 0, "elo_estimate": 0})

    logger.info(f"Model: {champion.num_parameters:,} parameters")

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=config["training"]["dataset_window"])

    # Load existing training data if resuming
    if args.resume:
        replay_buffer.load_from_files("data/self_play")

    # Initialize trainer
    trainer = Trainer(champion, config, log_dir="logs/tensorboard")

    if args.resume:
        trainer_ckpt = Path("data/models/trainer_state.pt")
        if trainer_ckpt.exists():
            trainer.load_checkpoint(str(trainer_ckpt))

    # ================================================================
    # Main Training Loop
    # ================================================================
    logger.info("=" * 60)
    logger.info("Starting AlphaZero training loop")
    logger.info("=" * 60)

    for cycle in range(trainer.training_cycle, trainer.training_cycle + args.cycles):
        cycle_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"CYCLE {cycle}")
        logger.info(f"{'='*60}")

        # ----------------------------------------------------------
        # Step 1: Self-Play
        # ----------------------------------------------------------
        logger.info("Step 1: Self-play game generation")
        examples = run_self_play_cycle(
            model=champion,
            config=config,
            cycle=cycle,
            save_dir="data/self_play",
        )
        replay_buffer.add(examples)
        logger.info(f"Replay buffer size: {len(replay_buffer)}")

        # ----------------------------------------------------------
        # Step 2: Training
        # ----------------------------------------------------------
        logger.info("Step 2: Neural network training")
        metrics = trainer.train_on_examples(replay_buffer)

        # ----------------------------------------------------------
        # Step 3: Arena Evaluation (optional)
        # ----------------------------------------------------------
        if not args.skip_arena and cycle > 0 and cycle % 5 == 0:
            logger.info("Step 3: Arena evaluation")

            # Save challenger
            challenger_path = "data/models/challenger.pt"
            save_model(champion, challenger_path)

            # Load previous champion for comparison
            prev_champion = load_model(str(champion_path), device)

            win_rate, arena_stats = evaluate_models(
                challenger=champion,
                champion=prev_champion,
                config=config,
            )

            if win_rate >= config["arena"]["win_threshold"]:
                logger.info(
                    f"New champion! Win rate: {win_rate:.1%} "
                    f"(threshold: {config['arena']['win_threshold']:.1%})"
                )
                save_model(champion, str(champion_path), {
                    "cycle": cycle,
                    "win_rate": win_rate,
                    "arena_stats": arena_stats,
                })
            else:
                logger.info(
                    f"Challenger rejected. Win rate: {win_rate:.1%} "
                    f"(needed: {config['arena']['win_threshold']:.1%})"
                )
        else:
            # Just save the model as champion (early training)
            if cycle % config["training"].get("checkpoint_interval", 10) == 0:
                save_model(champion, str(champion_path), {"cycle": cycle})

        # Save trainer state
        trainer.save_checkpoint("data/models/trainer_state.pt")

        # Save periodic checkpoint
        if cycle % 50 == 0:
            save_model(champion, f"data/models/checkpoint_cycle_{cycle:06d}.pt", {"cycle": cycle})

        cycle_time = time.time() - cycle_start
        logger.info(f"Cycle {cycle} complete in {cycle_time:.1f}s")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
