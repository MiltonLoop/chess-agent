"""
Board and move encoding for AlphaZero-style chess agent.

Board encoding: 18 planes of 8x8
  Planes 0-5:   Current player's pieces (K, Q, R, B, N, P)
  Planes 6-11:  Opponent's pieces (K, Q, R, B, N, P)
  Plane 12:     Current player can castle kingside
  Plane 13:     Current player can castle queenside
  Plane 14:     Opponent can castle kingside
  Plane 15:     Opponent can castle queenside
  Plane 16:     En passant square
  Plane 17:     Halfmove clock / 100

Move encoding: 8x8x73 = 4672 action space
  For each from-square (8x8 = 64):
    56 queen-type moves: 7 distances x 8 directions
    8 knight moves
    9 underpromotions: 3 directions x 3 piece types (rook, bishop, knight)
"""

import chess
import numpy as np
from functools import lru_cache


# Piece type indices for encoding
PIECE_ORDER = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]

# Direction offsets for queen-type moves: (delta_rank, delta_file)
# Order: N, NE, E, SE, S, SW, W, NW
QUEEN_DIRECTIONS = [
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1)
]

# Knight move offsets: (delta_rank, delta_file)
KNIGHT_MOVES = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2)
]

# Underpromotion directions (from white's perspective, on rank 7->8)
# NW, N, NE for captures and straight pushes
UNDERPROMO_DIRECTIONS = [(-1, 0), (0, 0), (1, 0)]  # delta_file relative to from
UNDERPROMO_PIECES = [chess.ROOK, chess.BISHOP, chess.KNIGHT]


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a chess board as an 18x8x8 numpy array.

    The board is always encoded from the current player's perspective.
    If it's black's turn, the board is flipped vertically.
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    turn = board.turn  # True = white, False = black

    for piece_type_idx, piece_type in enumerate(PIECE_ORDER):
        # Current player's pieces
        for sq in board.pieces(piece_type, turn):
            rank, file = _square_to_pos(sq, turn)
            planes[piece_type_idx, rank, file] = 1.0

        # Opponent's pieces
        for sq in board.pieces(piece_type, not turn):
            rank, file = _square_to_pos(sq, turn)
            planes[6 + piece_type_idx, rank, file] = 1.0

    # Castling rights
    if turn == chess.WHITE:
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[12] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[13] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[14] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[15] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[12] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[13] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[14] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[15] = 1.0

    # En passant
    if board.ep_square is not None:
        rank, file = _square_to_pos(board.ep_square, turn)
        planes[16, rank, file] = 1.0

    # Halfmove clock (normalized)
    planes[17] = board.halfmove_clock / 100.0

    return planes


def _square_to_pos(square: int, perspective: bool) -> tuple:
    """Convert a chess square to (rank, file) from the given perspective.

    If perspective is WHITE, rank 0 = rank 1 (bottom).
    If perspective is BLACK, board is flipped so rank 0 = rank 8.
    """
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if not perspective:  # Black's perspective: flip the board
        rank = 7 - rank
        file = 7 - file
    return rank, file


def _pos_to_square(rank: int, file: int, perspective: bool) -> int:
    """Convert (rank, file) back to a chess square from the given perspective."""
    if not perspective:
        rank = 7 - rank
        file = 7 - file
    return chess.square(file, rank)


def encode_move(move: chess.Move, turn: bool) -> int:
    """Encode a chess move as an index into the 4672-dim action space.

    The move is encoded from the perspective of the current player.
    Returns an integer in [0, 4671].
    """
    from_sq = move.from_square
    to_sq = move.to_square

    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    to_rank = chess.square_rank(to_sq)
    to_file = chess.square_file(to_sq)

    # Flip for black's perspective
    if not turn:
        from_rank = 7 - from_rank
        from_file = 7 - from_file
        to_rank = 7 - to_rank
        to_file = 7 - to_file

    delta_rank = to_rank - from_rank
    delta_file = to_file - from_file

    from_plane_idx = from_rank * 8 + from_file  # 0..63

    # Check for underpromotion (not queen promotion - queen is default)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        # Underpromotion
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)  # 0, 1, 2
        if delta_file == -1:
            dir_idx = 0
        elif delta_file == 0:
            dir_idx = 1
        elif delta_file == 1:
            dir_idx = 2
        else:
            raise ValueError(f"Invalid promotion move: {move}")
        move_type = 64 + dir_idx * 3 + piece_idx  # 64..72
    else:
        # Check if it's a knight move
        knight_delta = (delta_rank, delta_file)
        if knight_delta in KNIGHT_MOVES:
            knight_idx = KNIGHT_MOVES.index(knight_delta)
            move_type = 56 + knight_idx  # 56..63
        else:
            # Queen-type move (includes queen promotions)
            distance = max(abs(delta_rank), abs(delta_file))
            if distance == 0:
                raise ValueError(f"Zero-distance move: {move}")

            # Normalize direction
            dr = 0 if delta_rank == 0 else delta_rank // abs(delta_rank)
            df = 0 if delta_file == 0 else delta_file // abs(delta_file)
            direction = (dr, df)

            try:
                dir_idx = QUEEN_DIRECTIONS.index(direction)
            except ValueError:
                raise ValueError(f"Invalid direction {direction} for move {move}")

            move_type = dir_idx * 7 + (distance - 1)  # 0..55

    return from_plane_idx * 73 + move_type


def decode_move(action_idx: int, board: chess.Board) -> chess.Move:
    """Decode an action index back to a chess.Move.

    The action is decoded from the current player's perspective.
    """
    turn = board.turn
    from_plane_idx = action_idx // 73
    move_type = action_idx % 73

    from_rank = from_plane_idx // 8
    from_file = from_plane_idx % 8

    if move_type < 56:
        # Queen-type move
        dir_idx = move_type // 7
        distance = (move_type % 7) + 1
        dr, df = QUEEN_DIRECTIONS[dir_idx]
        to_rank = from_rank + dr * distance
        to_file = from_file + df * distance
        promotion = None

        # Check if this is a pawn reaching the last rank (queen promotion)
        if not turn:
            actual_from_rank = 7 - from_rank
            actual_to_rank = 7 - to_rank
        else:
            actual_from_rank = from_rank
            actual_to_rank = to_rank

        if actual_to_rank == 7 or actual_to_rank == 0:
            actual_from_sq = _pos_to_square(from_rank, from_file, turn)
            piece = board.piece_at(actual_from_sq)
            if piece is not None and piece.piece_type == chess.PAWN:
                promotion = chess.QUEEN

    elif move_type < 64:
        # Knight move
        knight_idx = move_type - 56
        dr, df = KNIGHT_MOVES[knight_idx]
        to_rank = from_rank + dr
        to_file = from_file + df
        promotion = None
    else:
        # Underpromotion
        underpromo_idx = move_type - 64
        dir_idx = underpromo_idx // 3
        piece_idx = underpromo_idx % 3
        delta_file = [-1, 0, 1][dir_idx]
        to_rank = from_rank + 1  # Always forward one rank
        to_file = from_file + delta_file
        promotion = UNDERPROMO_PIECES[piece_idx]

    # Convert back from perspective coordinates
    from_sq = _pos_to_square(from_rank, from_file, turn)
    to_sq = _pos_to_square(to_rank, to_file, turn)

    return chess.Move(from_sq, to_sq, promotion=promotion)


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """Get a binary mask over the 4672-dim action space for legal moves."""
    mask = np.zeros(4672, dtype=np.float32)
    for move in board.legal_moves:
        try:
            idx = encode_move(move, board.turn)
            mask[idx] = 1.0
        except (ValueError, IndexError):
            continue
    return mask


def get_move_mapping(board: chess.Board) -> dict:
    """Get mapping from action indices to legal moves."""
    mapping = {}
    for move in board.legal_moves:
        try:
            idx = encode_move(move, board.turn)
            mapping[idx] = move
        except (ValueError, IndexError):
            continue
    return mapping
