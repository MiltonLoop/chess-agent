"""
Monte Carlo Tree Search (MCTS) for AlphaZero chess.

Implements PUCT (Predictor + Upper Confidence bound for Trees):
  Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

The neural network provides:
  - P(s,a): prior move probabilities (policy head)
  - V(s): position evaluation (value head)

On each simulation:
  1. SELECT: Walk down tree using PUCT until reaching a leaf
  2. EXPAND: Use neural net to evaluate leaf, create child nodes
  3. BACKUP: Propagate value back up the tree
"""

import chess
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional
import logging

from .encoding import encode_board, encode_move, decode_move, get_legal_move_mask, get_move_mapping

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """A node in the MCTS tree."""
    parent: Optional['Node'] = None
    parent_action: Optional[int] = None  # Action index that led here
    prior: float = 0.0                    # P(s,a) from parent's policy
    visit_count: int = 0                  # N(s,a)
    value_sum: float = 0.0                # W(s,a) total value
    children: dict = field(default_factory=dict)  # action_idx -> Node
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Mean action value Q(s,a) = W(s,a) / N(s,a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""

    def __init__(
        self,
        model,
        num_simulations: int = 200,
        c_puct: float = 2.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
        temp_threshold_move: int = 30,
        device: torch.device = None,
        add_noise: bool = True,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.temp_threshold_move = temp_threshold_move
        self.device = device or next(model.parameters()).device
        self.add_noise = add_noise

    def search(self, board: chess.Board, move_number: int = 0) -> tuple:
        """Run MCTS from the given position.

        Args:
            board: Current board state
            move_number: Current full-move number (for temperature scheduling)

        Returns:
            action_probs: numpy array of shape (4672,) with visit-count-based probabilities
            root_value: estimated value of the root position
        """
        root = Node()

        # Evaluate root position
        self._expand(root, board)

        # Add Dirichlet noise to root priors for exploration
        if self.add_noise and root.children:
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_board = board.copy()
            search_path = [node]

            # SELECT: walk down tree using PUCT
            while node.is_expanded and node.children:
                action_idx, node = self._select_child(node)
                move = decode_move(action_idx, sim_board)
                sim_board.push(move)
                search_path.append(node)

            # Get leaf value
            if sim_board.is_game_over():
                value = self._get_terminal_value(sim_board)
            else:
                # EXPAND: evaluate with neural net
                value = self._expand(node, sim_board)

            # BACKUP: propagate value up the tree
            self._backup(search_path, value)

        # Build action probability distribution from visit counts
        action_probs = np.zeros(4672, dtype=np.float32)
        for action_idx, child in root.children.items():
            action_probs[action_idx] = child.visit_count

        # Apply temperature
        if move_number < self.temp_threshold_move and self.temperature > 0:
            # Proportional to visit count ^ (1/temp)
            action_probs = action_probs ** (1.0 / self.temperature)
        else:
            # Deterministic: pick the most visited
            best = np.argmax(action_probs)
            action_probs = np.zeros_like(action_probs)
            action_probs[best] = 1.0

        # Normalize
        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs, root.q_value

    def get_best_move(self, board: chess.Board, move_number: int = 0) -> tuple:
        """Get the best move for the current position.

        Returns:
            move: chess.Move
            action_probs: visit count distribution
            value: position evaluation
        """
        action_probs, value = self.search(board, move_number)

        # Sample or pick best
        if move_number < self.temp_threshold_move and self.temperature > 0:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        else:
            action_idx = np.argmax(action_probs)

        move = decode_move(action_idx, board)

        # Validate move is legal
        if move not in board.legal_moves:
            # Fallback: pick highest-prob legal move
            legal_mapping = get_move_mapping(board)
            legal_probs = {idx: action_probs[idx] for idx in legal_mapping}
            if legal_probs:
                action_idx = max(legal_probs, key=legal_probs.get)
                move = legal_mapping[action_idx]
            else:
                # Last resort: random legal move
                move = list(board.legal_moves)[0]

        return move, action_probs, value

    def _expand(self, node: Node, board: chess.Board) -> float:
        """Expand a leaf node using the neural network.

        Returns the value estimate for this position.
        """
        # Encode board and get neural net prediction
        board_tensor = torch.from_numpy(encode_board(board)).to(self.device)
        policy, value = self.model.predict(board_tensor)

        # Mask illegal moves and renormalize
        legal_mask = get_legal_move_mask(board)
        policy = policy * legal_mask

        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        else:
            # If no legal moves have probability, use uniform over legal moves
            policy = legal_mask / legal_mask.sum() if legal_mask.sum() > 0 else policy

        # Create child nodes for each legal move
        for action_idx in np.nonzero(legal_mask)[0]:
            node.children[int(action_idx)] = Node(
                parent=node,
                parent_action=int(action_idx),
                prior=policy[action_idx],
            )

        node.is_expanded = True
        return value

    def _select_child(self, node: Node) -> tuple:
        """Select the child with highest PUCT score.

        PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        total_visits = sum(c.visit_count for c in node.children.values())
        sqrt_total = np.sqrt(total_visits + 1)

        best_score = -float('inf')
        best_action = None
        best_child = None

        for action_idx, child in node.children.items():
            # Exploitation term
            q = child.q_value

            # Exploration term
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)

            score = q + u

            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child

        return best_action, best_child

    def _backup(self, search_path: list, value: float):
        """Backup the value through the search path.

        Value alternates sign because players alternate.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # Flip for opponent's perspective

    def _add_dirichlet_noise(self, node: Node):
        """Add Dirichlet noise to root priors for exploration."""
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))

        for i, action_idx in enumerate(actions):
            child = node.children[action_idx]
            child.prior = (
                (1 - self.dirichlet_epsilon) * child.prior +
                self.dirichlet_epsilon * noise[i]
            )

    def _get_terminal_value(self, board: chess.Board) -> float:
        """Get the value of a terminal position."""
        result = board.result()
        if result == "1-0":
            return 1.0 if board.turn == chess.BLACK else -1.0
        elif result == "0-1":
            return 1.0 if board.turn == chess.WHITE else -1.0
        else:
            return 0.0  # Draw
