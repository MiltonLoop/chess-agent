#!/usr/bin/env python3
"""
Play against the chess agent in the terminal.

Usage:
  python play.py                          # Play as white against the agent
  python play.py --color black            # Play as black
  python play.py --model path/to/model.pt # Use specific model
  python play.py --sims 400              # More MCTS simulations (stronger)
"""

import argparse
import chess
import yaml
import torch
from pathlib import Path

from src.model import load_model, create_model_from_config, get_device
from src.mcts import MCTS


PIECE_SYMBOLS = {
    'K': '\u2654', 'Q': '\u2655', 'R': '\u2656', 'B': '\u2657', 'N': '\u2658', 'P': '\u2659',
    'k': '\u265A', 'q': '\u265B', 'r': '\u265C', 'b': '\u265D', 'n': '\u265E', 'p': '\u265F',
}


def print_board(board: chess.Board, perspective: bool = chess.WHITE):
    """Pretty-print the board with Unicode pieces."""
    print()
    ranks = range(7, -1, -1) if perspective == chess.WHITE else range(8)
    files = range(8) if perspective == chess.WHITE else range(7, -1, -1)

    for rank in ranks:
        row = f"  {rank + 1} "
        for file in files:
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece:
                row += f" {PIECE_SYMBOLS.get(piece.symbol(), piece.symbol())} "
            else:
                bg = "\u00B7" if (rank + file) % 2 == 0 else " "
                row += f" {bg} "
        print(row)

    if perspective == chess.WHITE:
        print("     a  b  c  d  e  f  g  h")
    else:
        print("     h  g  f  e  d  c  b  a")
    print()


def get_human_move(board: chess.Board) -> chess.Move:
    """Get a move from the human player."""
    while True:
        try:
            uci = input("Your move (UCI, e.g. e2e4): ").strip()
            if uci in ("quit", "exit", "q"):
                raise KeyboardInterrupt
            if uci == "moves":
                print("Legal moves:", " ".join(m.uci() for m in board.legal_moves))
                continue
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                return move
            # Try SAN notation
            try:
                move = board.parse_san(uci)
                if move in board.legal_moves:
                    return move
            except ValueError:
                pass
            print(f"Illegal move: {uci}. Type 'moves' to see legal moves.")
        except ValueError:
            print("Invalid move format. Use UCI (e.g. e2e4) or SAN (e.g. Nf3).")


def main():
    parser = argparse.ArgumentParser(description="Play against the chess agent")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--color", default="white", choices=["white", "black"])
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(config["system"]["device"])
    model_path = args.model or "data/models/champion.pt"

    if Path(model_path).exists():
        model = load_model(model_path, device)
    else:
        print("No trained model found. Using random (untrained) model.")
        model = create_model_from_config(args.config)

    model.eval()

    mcts = MCTS(
        model=model,
        num_simulations=args.sims,
        c_puct=config["self_play"]["c_puct"],
        temperature=0.1,
        temp_threshold_move=0,
        device=device,
        add_noise=False,
    )

    human_color = chess.WHITE if args.color == "white" else chess.BLACK
    board = chess.Board()

    print("\n" + "=" * 40)
    print("  CHESS AGENT")
    print(f"  You: {'White' if human_color == chess.WHITE else 'Black'}")
    print(f"  MCTS sims: {args.sims}")
    print(f"  Type 'moves' for legal moves, 'quit' to exit")
    print("=" * 40)

    move_num = 0
    try:
        while not board.is_game_over():
            print_board(board, human_color)

            if board.turn == human_color:
                move = get_human_move(board)
            else:
                print("Agent thinking...")
                move, _, value = mcts.get_best_move(board, move_num)
                print(f"Agent plays: {board.san(move)} (eval: {value:+.3f})")

            board.push(move)
            move_num += 1

        print_board(board, human_color)
        result = board.result()
        print(f"\nGame over: {result}")
        if result == "1-0":
            print("White wins!" if human_color == chess.WHITE else "Agent wins!")
        elif result == "0-1":
            print("Black wins!" if human_color == chess.BLACK else "Agent wins!")
        else:
            print("Draw!")

    except KeyboardInterrupt:
        print("\nGame aborted.")


if __name__ == "__main__":
    main()
