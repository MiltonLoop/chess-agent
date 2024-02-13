#!/usr/bin/env python3
"""
Main daemon: train and play simultaneously.

This is the "set it and forget it" script for your Mac Mini.
It runs the full AlphaZero loop AND plays on Lichess at the same time:

  1. Self-play to generate training data
  2. Train the neural network
  3. Evaluate new model vs champion
  4. Play games on Lichess with the latest champion
  5. Repeat forever

Usage:
  python run.py                         # Full daemon (train + play online)
  python run.py --train-only            # Just train, no Lichess
  python run.py --play-only             # Just play on Lichess, no training
  LICHESS_TOKEN=xxx python run.py       # Set token via environment
"""

import argparse
import logging
import sys
import threading
import time
import signal
import yaml
import os
from pathlib import Path

from src.model import load_model, create_model_from_config, get_device, save_model
from src.self_play import run_self_play_cycle
from src.trainer import Trainer, ReplayBuffer
from src.arena import evaluate_models


def setup_logging(level: str = "INFO"):
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/daemon.log"),
        ],
    )


class ChessAgentDaemon:
    """Main daemon that orchestrates training and online play."""

    def __init__(self, config: dict, train: bool = True, play: bool = True):
        self.config = config
        self.do_train = train
        self.do_play = play
        self.running = True
        self.logger = logging.getLogger("daemon")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Device
        self.device = get_device(config["system"]["device"])
        self.logger.info(f"Device: {self.device}")

        # Load or create model
        self.champion_path = Path("data/models/champion.pt")
        if self.champion_path.exists():
            self.model = load_model(str(self.champion_path), self.device)
            self.logger.info("Loaded existing champion model")
        else:
            self.model = create_model_from_config("config.yaml")
            Path("data/models").mkdir(parents=True, exist_ok=True)
            save_model(self.model, str(self.champion_path), {"cycle": 0})
            self.logger.info("Created new model")

        self.logger.info(f"Model: {self.model.num_parameters:,} parameters")

    def start(self):
        """Start the daemon."""
        self.logger.info("=" * 60)
        self.logger.info("  CHESS AGENT DAEMON")
        self.logger.info(f"  Training: {'ON' if self.do_train else 'OFF'}")
        self.logger.info(f"  Online play: {'ON' if self.do_play else 'OFF'}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info("=" * 60)

        threads = []

        if self.do_train:
            t = threading.Thread(target=self._training_loop, name="training", daemon=True)
            t.start()
            threads.append(t)

        if self.do_play:
            t = threading.Thread(target=self._play_loop, name="lichess", daemon=True)
            t.start()
            threads.append(t)

        # Status reporter
        t = threading.Thread(target=self._status_loop, name="status", daemon=True)
        t.start()
        threads.append(t)

        # Wait for shutdown
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        self.logger.info("Daemon shutting down...")
        self.running = False

        for t in threads:
            t.join(timeout=10)

        self.logger.info("Daemon stopped.")

    def _training_loop(self):
        """Continuous self-play + training loop."""
        replay_buffer = ReplayBuffer(max_size=self.config["training"]["dataset_window"])
        replay_buffer.load_from_files("data/self_play")

        trainer = Trainer(self.model, self.config, log_dir="logs/tensorboard")

        # Try to resume trainer state
        trainer_path = Path("data/models/trainer_state.pt")
        if trainer_path.exists():
            trainer.load_checkpoint(str(trainer_path))

        cycle = trainer.training_cycle

        while self.running:
            try:
                self.logger.info(f"\n--- Training Cycle {cycle} ---")

                # Self-play
                examples = run_self_play_cycle(
                    model=self.model,
                    config=self.config,
                    cycle=cycle,
                    save_dir="data/self_play",
                )
                replay_buffer.add(examples)

                # Train
                metrics = trainer.train_on_examples(replay_buffer)

                # Arena every 5 cycles
                if cycle > 0 and cycle % 5 == 0:
                    prev_champion = load_model(str(self.champion_path), self.device)
                    win_rate, stats = evaluate_models(self.model, prev_champion, self.config)

                    if win_rate >= self.config["arena"]["win_threshold"]:
                        self.logger.info(f"New champion at cycle {cycle}! ({win_rate:.1%})")
                        save_model(self.model, str(self.champion_path), {
                            "cycle": cycle, "win_rate": win_rate
                        })

                # Periodic saves
                trainer.save_checkpoint("data/models/trainer_state.pt")
                if cycle % 50 == 0:
                    save_model(
                        self.model,
                        f"data/models/checkpoint_{cycle:06d}.pt",
                        {"cycle": cycle}
                    )

                cycle += 1

            except Exception as e:
                self.logger.error(f"Training error: {e}", exc_info=True)
                time.sleep(10)

    def _play_loop(self):
        """Lichess bot loop."""
        try:
            from src.lichess_bot import LichessBot

            token = self.config["lichess"].get("token") or os.environ.get("LICHESS_TOKEN", "")
            if not token:
                self.logger.warning(
                    "No Lichess token configured. Set LICHESS_TOKEN env var or "
                    "add token to config.yaml to enable online play."
                )
                return

            self.config["lichess"]["token"] = token

            while self.running:
                try:
                    self.model.eval()
                    bot = LichessBot(self.model, self.config)
                    bot.start()
                except Exception as e:
                    self.logger.error(f"Lichess bot error: {e}", exc_info=True)
                    if self.running:
                        self.logger.info("Restarting bot in 30s...")
                        time.sleep(30)

        except ImportError as e:
            self.logger.warning(f"Lichess bot not available: {e}")

    def _status_loop(self):
        """Periodic status reporting."""
        while self.running:
            time.sleep(300)  # Every 5 minutes
            if self.running:
                sp_files = list(Path("data/self_play").glob("cycle_*.npz"))
                game_files = list(Path("data/games").glob("*.json"))
                self.logger.info(
                    f"Status: {len(sp_files)} self-play cycles, "
                    f"{len(game_files)} online games played"
                )

    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Chess Agent Daemon")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--play-only", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    setup_logging(config["system"].get("log_level", "INFO"))

    do_train = not args.play_only
    do_play = not args.train_only

    daemon = ChessAgentDaemon(config, train=do_train, play=do_play)
    daemon.start()


if __name__ == "__main__":
    main()
