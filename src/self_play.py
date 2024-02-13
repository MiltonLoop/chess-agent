"""
Self-play game generation for AlphaZero training.

Generates training data by having the neural network play against itself
using MCTS. Each game produces (board_state, policy_target, value_target) tuples.
"""

import chess
import chess.pgn
import numpy as np
import torch
import json
import time
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
import logging
import multiprocessing as mp

from .encoding import encode_board, encode_move
from .mcts import MCTS
from .model import ChessNet, get_device

logger = logging.getLogger(__name__)


@dataclass
class GameRecord:
    """Record of a single self-play game."""
    states: List[np.ndarray] = field(default_factory=list)        # Board encodings
    policies: List[np.ndarray] = field(default_factory=list)      # MCTS visit distributions
    players: List[bool] = field(default_factory=list)             # Who played (True=white)
    moves: List[str] = field(default_factory=list)                # UCI move strings
    result: float = 0.0                                            # 1=white wins, -1=black, 0=draw
    num_moves: int = 0


def play_game(
    model: ChessNet,
    config: dict,
    game_id: int = 0,
) -> GameRecord:
    """Play a single self-play game.

    Returns a GameRecord with training data.
    """
    device = next(model.parameters()).device
    sp_config = config["self_play"]

    mcts = MCTS(
        model=model,
        num_simulations=sp_config["num_simulations"],
        c_puct=sp_config["c_puct"],
        dirichlet_alpha=sp_config["dirichlet_alpha"],
        dirichlet_epsilon=sp_config["dirichlet_epsilon"],
        temperature=sp_config["temperature"],
        temp_threshold_move=sp_config["temp_threshold_move"],
        device=device,
        add_noise=True,
    )

    board = chess.Board()
    record = GameRecord()
    resign_threshold = sp_config.get("resign_threshold", -0.95)
    resign_min_move = sp_config.get("resign_check_min_move", 10)
    max_length = sp_config.get("max_game_length", 512)

    move_num = 0
    resigned = False

    while not board.is_game_over() and move_num < max_length:
        # Encode current state
        state = encode_board(board)

        # Run MCTS
        move, action_probs, value = mcts.get_best_move(board, move_num)

        # Store training data
        record.states.append(state)
        record.policies.append(action_probs)
        record.players.append(board.turn)
        record.moves.append(move.uci())

        # Check resignation
        if move_num >= resign_min_move and value < resign_threshold:
            resigned = True
            # The current player resigns -> opponent wins
            record.result = -1.0 if board.turn == chess.WHITE else 1.0
            break

        board.push(move)
        move_num += 1

    record.num_moves = move_num

    # Determine result if game ended naturally
    if not resigned:
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                record.result = 1.0
            elif result == "0-1":
                record.result = -1.0
            else:
                record.result = 0.0
        else:
            record.result = 0.0  # Max length draw

    logger.debug(
        f"Game {game_id}: {move_num} moves, result={record.result:.0f}, "
        f"{'resigned' if resigned else board.result() if board.is_game_over() else 'max_length'}"
    )

    return record


def game_record_to_training_data(record: GameRecord) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Convert a GameRecord into (state, policy, value) training examples.

    Value targets are from the perspective of the player who made the move:
      +1 if that player ultimately won, -1 if lost, 0 for draw.
    """
    examples = []
    for i in range(len(record.states)):
        state = record.states[i]
        policy = record.policies[i]
        player = record.players[i]

        # Value from the perspective of the player who moved
        if player == chess.WHITE:
            value = record.result
        else:
            value = -record.result

        examples.append((state, policy, value))

    return examples


def run_self_play_cycle(
    model: ChessNet,
    config: dict,
    cycle: int = 0,
    save_dir: str = "data/self_play",
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Run a full self-play cycle: generate games, extract training data.

    Args:
        model: Current best model
        config: Full config dict
        cycle: Cycle number for naming
        save_dir: Directory to save game records

    Returns:
        List of (state, policy, value) training examples
    """
    sp_config = config["self_play"]
    num_games = sp_config["num_games_per_cycle"]

    logger.info(f"Self-play cycle {cycle}: generating {num_games} games...")
    start_time = time.time()

    all_examples = []
    results = {"white": 0, "black": 0, "draw": 0}

    model.eval()

    for game_id in range(num_games):
        record = play_game(model, config, game_id)

        # Track results
        if record.result > 0:
            results["white"] += 1
        elif record.result < 0:
            results["black"] += 1
        else:
            results["draw"] += 1

        # Extract training examples
        examples = game_record_to_training_data(record)
        all_examples.extend(examples)

        if (game_id + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (game_id + 1) / elapsed
            logger.info(
                f"  Games: {game_id + 1}/{num_games} "
                f"({rate:.1f} games/s, {len(all_examples)} positions)"
            )

    elapsed = time.time() - start_time
    logger.info(
        f"Self-play cycle {cycle} complete: {num_games} games in {elapsed:.1f}s, "
        f"{len(all_examples)} positions. "
        f"W:{results['white']} B:{results['black']} D:{results['draw']}"
    )

    # Save training data
    save_path = Path(save_dir) / f"cycle_{cycle:06d}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        states=np.array([e[0] for e in all_examples]),
        policies=np.array([e[1] for e in all_examples]),
        values=np.array([e[2] for e in all_examples], dtype=np.float32),
    )
    logger.info(f"Training data saved to {save_path}")

    # Save PGN for review
    save_pgn_summary(config, cycle, results, save_dir)

    return all_examples


def save_pgn_summary(config: dict, cycle: int, results: dict, save_dir: str):
    """Save a summary of the self-play cycle."""
    summary_path = Path(save_dir) / f"cycle_{cycle:06d}_summary.json"
    summary = {
        "cycle": cycle,
        "num_games": config["self_play"]["num_games_per_cycle"],
        "results": results,
        "simulations_per_move": config["self_play"]["num_simulations"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
