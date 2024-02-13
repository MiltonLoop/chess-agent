"""
Arena: pit two models against each other to determine which is stronger.

After each training cycle, the new model plays a match against the current
champion. If it wins convincingly (>55% win rate), it becomes the new champion.
"""

import chess
import numpy as np
import torch
from typing import Tuple
import logging

from .mcts import MCTS
from .model import ChessNet
from .encoding import encode_board

logger = logging.getLogger(__name__)


def play_arena_game(
    white_model: ChessNet,
    black_model: ChessNet,
    config: dict,
) -> float:
    """Play a single game between two models.

    Returns:
        +1.0 if white wins, -1.0 if black wins, 0.0 for draw
    """
    arena_cfg = config["arena"]
    device_w = next(white_model.parameters()).device
    device_b = next(black_model.parameters()).device

    white_mcts = MCTS(
        model=white_model,
        num_simulations=arena_cfg["num_simulations"],
        c_puct=config["self_play"]["c_puct"],
        temperature=0.1,  # Near-deterministic for evaluation
        temp_threshold_move=0,  # Always deterministic
        device=device_w,
        add_noise=False,  # No exploration noise in evaluation
    )

    black_mcts = MCTS(
        model=black_model,
        num_simulations=arena_cfg["num_simulations"],
        c_puct=config["self_play"]["c_puct"],
        temperature=0.1,
        temp_threshold_move=0,
        device=device_b,
        add_noise=False,
    )

    board = chess.Board()
    move_num = 0
    max_moves = config["self_play"].get("max_game_length", 512)

    while not board.is_game_over() and move_num < max_moves:
        if board.turn == chess.WHITE:
            move, _, _ = white_mcts.get_best_move(board, move_num)
        else:
            move, _, _ = black_mcts.get_best_move(board, move_num)

        board.push(move)
        move_num += 1

    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
    return 0.0


def evaluate_models(
    challenger: ChessNet,
    champion: ChessNet,
    config: dict,
) -> Tuple[float, dict]:
    """Run an evaluation match between challenger and champion.

    Each model plays both colors for fairness.

    Returns:
        win_rate: Challenger's win rate (wins + 0.5*draws) / total
        stats: Dictionary with detailed results
    """
    arena_cfg = config["arena"]
    num_games = arena_cfg["num_games"]
    half = num_games // 2

    logger.info(f"Arena: {num_games} games ({half} as white, {half} as black)")

    challenger_wins = 0
    champion_wins = 0
    draws = 0

    # Challenger plays white
    for i in range(half):
        result = play_arena_game(challenger, champion, config)
        if result > 0:
            challenger_wins += 1
        elif result < 0:
            champion_wins += 1
        else:
            draws += 1

        if (i + 1) % 5 == 0:
            logger.info(
                f"  Progress: {i + 1}/{num_games} "
                f"(C:{challenger_wins} Ch:{champion_wins} D:{draws})"
            )

    # Challenger plays black
    for i in range(half):
        result = play_arena_game(champion, challenger, config)
        if result < 0:  # Black (challenger) wins
            challenger_wins += 1
        elif result > 0:  # White (champion) wins
            champion_wins += 1
        else:
            draws += 1

        if (half + i + 1) % 5 == 0:
            logger.info(
                f"  Progress: {half + i + 1}/{num_games} "
                f"(C:{challenger_wins} Ch:{champion_wins} D:{draws})"
            )

    total = challenger_wins + champion_wins + draws
    win_rate = (challenger_wins + 0.5 * draws) / total if total > 0 else 0.5

    stats = {
        "challenger_wins": challenger_wins,
        "champion_wins": champion_wins,
        "draws": draws,
        "total_games": total,
        "win_rate": win_rate,
    }

    logger.info(
        f"Arena result: Challenger {win_rate:.1%} win rate "
        f"(W:{challenger_wins} L:{champion_wins} D:{draws})"
    )

    return win_rate, stats
