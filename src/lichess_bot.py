"""
Lichess Bot integration for autonomous online play.

Connects to Lichess via their Bot API to:
  - Accept challenges from other players
  - Play games using the MCTS engine
  - Challenge other online bots automatically
  - Log game results for analysis

Requires a Lichess Bot account and API token.
See: https://lichess.org/api#tag/Bot
"""

import chess
import berserk
import threading
import time
import json
import random
import logging
from pathlib import Path
from typing import Optional

from .model import ChessNet, get_device
from .mcts import MCTS
from .encoding import encode_board

logger = logging.getLogger(__name__)


class LichessBot:
    """Autonomous Lichess bot powered by AlphaZero MCTS."""

    def __init__(self, model: ChessNet, config: dict):
        self.model = model
        self.config = config
        self.lichess_cfg = config["lichess"]
        self.device = next(model.parameters()).device

        # Initialize Lichess client
        token = self.lichess_cfg.get("token") or ""
        if not token:
            import os
            token = os.environ.get("LICHESS_TOKEN", "")
        if not token:
            raise ValueError(
                "Lichess API token required. Set in config.yaml or LICHESS_TOKEN env var.\n"
                "Get one at: https://lichess.org/account/oauth/token/create\n"
                "Make sure to check the 'Play games with the bot API' scope."
            )

        session = berserk.TokenSession(token)
        self.client = berserk.Client(session)

        # Get bot info
        self.bot_info = self.client.account.get()
        self.bot_name = self.bot_info.get("username", "unknown")
        logger.info(f"Connected as: {self.bot_name}")

        # Active games tracking
        self.active_games = {}
        self.max_concurrent = self.lichess_cfg.get("max_concurrent_games", 1)
        self.running = False

        # Track recently challenged bots to avoid spamming
        self.recently_challenged = set()

        # Game history for logging
        self.games_dir = Path("data/games")
        self.games_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start the bot: listen for events and play games."""
        self.running = True
        logger.info(f"Bot {self.bot_name} starting... Listening for challenges.")

        # Start auto-challenge thread if enabled
        if self.lichess_cfg.get("auto_challenge", False):
            challenge_thread = threading.Thread(target=self._auto_challenge_loop, daemon=True)
            challenge_thread.start()

        # Main event loop
        try:
            for event in self.client.bots.stream_incoming_events():
                if not self.running:
                    break

                event_type = event.get("type", "")

                if event_type == "challenge":
                    self._handle_challenge(event["challenge"])
                elif event_type == "gameStart":
                    game_id = event["game"]["gameId"]
                    logger.info(f"Game started: {game_id}")
                    game_thread = threading.Thread(
                        target=self._play_game,
                        args=(game_id,),
                        daemon=True,
                    )
                    game_thread.start()
                elif event_type == "gameFinish":
                    game_id = event["game"]["gameId"]
                    logger.info(f"Game finished: {game_id}")
                    self.active_games.pop(game_id, None)

        except berserk.exceptions.ResponseError as e:
            logger.error(f"Lichess API error: {e}")
        except KeyboardInterrupt:
            logger.info("Bot stopping...")
        finally:
            self.running = False

    def stop(self):
        """Stop the bot gracefully."""
        self.running = False
        logger.info("Bot stopped.")

    def _handle_challenge(self, challenge: dict):
        """Decide whether to accept or decline a challenge."""
        challenger = challenge.get("challenger", {}).get("name", "unknown")
        if challenger.lower() == self.bot_name.lower():
            return
        game_id = challenge["id"]
        variant = challenge.get("variant", {}).get("key", "standard")
        time_control = challenge.get("speed", "unknown")
        rated = challenge.get("rated", False)

        logger.info(
            f"Challenge from {challenger}: {variant} {time_control} "
            f"{'rated' if rated else 'casual'} (id: {game_id})"
        )

        # Only accept standard chess
        if variant != "standard":
            logger.info(f"Declining: variant {variant} not supported")
            self.client.bots.decline_challenge(game_id, reason="variant")
            return

        # Check time control
        allowed_speeds = self.lichess_cfg.get("time_controls", ["bullet", "blitz", "rapid"])
        if time_control not in allowed_speeds:
            logger.info(f"Declining: {time_control} not in allowed time controls")
            self.client.bots.decline_challenge(game_id, reason="timeControl")
            return

        # Check concurrent games
        if len(self.active_games) >= self.max_concurrent:
            logger.info(f"Declining: too many active games ({len(self.active_games)})")
            self.client.bots.decline_challenge(game_id, reason="later")
            return

        # Accept!
        logger.info(f"Accepting challenge from {challenger}")
        try:
            self.client.bots.accept_challenge(game_id)
        except berserk.exceptions.ResponseError as e:
            logger.error(f"Failed to accept challenge: {e}")

    def _play_game(self, game_id: str):
        """Play a single game on Lichess."""
        self.active_games[game_id] = True
        game_moves = []
        game_info = {}

        try:
            # Create MCTS for this game
            mcts = MCTS(
                model=self.model,
                num_simulations=self.lichess_cfg.get("num_simulations", 400),
                c_puct=self.config["self_play"]["c_puct"],
                dirichlet_alpha=self.config["self_play"]["dirichlet_alpha"],
                dirichlet_epsilon=self.config["self_play"]["dirichlet_epsilon"],
                temperature=0.3,  # Slight randomness for variety
                temp_threshold_move=15,  # Be deterministic after move 15
                device=self.device,
                add_noise=True,
            )

            board = chess.Board()
            our_color = None
            move_num = 0

            for event in self.client.bots.stream_game_state(game_id):
                if not self.running:
                    break

                event_type = event.get("type", "")

                if event_type == "gameFull":
                    # Initial game state
                    game_info = event
                    white_name = event.get("white", {}).get("name", "")
                    our_color = chess.WHITE if white_name == self.bot_name else chess.BLACK
                    logger.info(
                        f"Game {game_id}: playing as "
                        f"{'white' if our_color == chess.WHITE else 'black'}"
                    )

                    # Apply existing moves (if reconnecting)
                    moves_str = event.get("state", {}).get("moves", "")
                    if moves_str:
                        for uci in moves_str.split():
                            board.push(chess.Move.from_uci(uci))
                            game_moves.append(uci)
                            move_num += 1

                    # Make our move if it's our turn
                    if board.turn == our_color and not board.is_game_over():
                        self._make_move(game_id, board, mcts, move_num, game_moves)
                        move_num += 1

                elif event_type == "gameState":
                    # Game state update
                    moves_str = event.get("moves", "")
                    if moves_str:
                        all_moves = moves_str.split()
                        # Sync board to match server state
                        board = chess.Board()
                        game_moves = []
                        for uci in all_moves:
                            board.push(chess.Move.from_uci(uci))
                            game_moves.append(uci)
                        move_num = len(all_moves)

                    status = event.get("status", "")
                    if status in ("mate", "resign", "stalemate", "timeout",
                                  "draw", "outoftime", "aborted"):
                        logger.info(f"Game {game_id} ended: {status}")
                        break

                    # Make our move if it's our turn
                    if board.turn == our_color and not board.is_game_over():
                        self._make_move(game_id, board, mcts, move_num, game_moves)
                        move_num += 1

                elif event_type == "chatLine":
                    pass  # Ignore chat

        except Exception as e:
            logger.error(f"Error in game {game_id}: {e}")
        finally:
            self.active_games.pop(game_id, None)
            self._save_game_log(game_id, game_info, game_moves)

    def _make_move(self, game_id: str, board: chess.Board, mcts: MCTS,
                   move_num: int, game_moves: list):
        """Think and make a move."""
        start = time.time()

        move, _, value = mcts.get_best_move(board, move_num)
        elapsed = time.time() - start

        logger.info(
            f"Game {game_id} move {move_num}: {move.uci()} "
            f"(eval: {value:+.3f}, time: {elapsed:.1f}s)"
        )

        try:
            self.client.bots.make_move(game_id, move.uci())
            board.push(move)
            game_moves.append(move.uci())
        except berserk.exceptions.ResponseError as e:
            logger.error(f"Failed to make move {move.uci()}: {e}")
            # Try a fallback move
            for legal_move in board.legal_moves:
                try:
                    self.client.bots.make_move(game_id, legal_move.uci())
                    board.push(legal_move)
                    game_moves.append(legal_move.uci())
                    break
                except berserk.exceptions.ResponseError:
                    continue

    def _auto_challenge_loop(self):
        """Periodically challenge other online bots."""
        interval = self.lichess_cfg.get("auto_challenge_interval", 60)

        while self.running:
            if len(self.active_games) < self.max_concurrent:
                try:
                    self._challenge_random_bot()
                except Exception as e:
                    logger.debug(f"Auto-challenge failed: {e}")

            time.sleep(interval)

    def _challenge_random_bot(self):
        """Find and challenge an online bot."""
        try:
            # Get list of online bots from the Lichess bot team
            online_bots = []
            try:
                # Fetch online bots using the player endpoint
                response = self.client.users.get_by_team("lichess-bots", count=50)
                for user in response:
                    name = user.get("username", "")
                    if (name.lower() != self.bot_name.lower()
                            and name not in self.recently_challenged
                            and user.get("online", False)):
                        online_bots.append(name)
            except Exception:
                # Fallback: try some well-known bots
                fallback_bots = [
                    "maia1", "maia5", "maia9",
                    "ChessChildren", "EasyPeasyBot",
                    "BOTdarwin", "PenguinBot",
                    "lichess-bot-pool", "RandomMover",
                    "BotKingsley", "chessbotx",
                    "turochamp-2", "ResignBot",
                    "Woezel_ansen", "AncientStar",
                    "simpleEval", "patzerbot",
                    "Nikitosik-ai", "OpeningsBot",
                ]
                online_bots = [b for b in fallback_bots if b not in self.recently_challenged]

            if not online_bots:
                # Clear recently challenged list and try again next cycle
                self.recently_challenged.clear()
                logger.debug("No bots available, cleared challenge history")
                return

            # Pick a random bot to challenge
            target = random.choice(online_bots)
            logger.info(f"Challenging bot: {target}")

            self.client.challenges.create(
                target,
                rated=self.lichess_cfg.get("rated", True),
                clock_limit=300,      # 5 minutes
                clock_increment=3,    # 3 second increment
            )

            self.recently_challenged.add(target)

            # Keep the recently challenged list from growing too large
            if len(self.recently_challenged) > 50:
                self.recently_challenged.clear()

        except berserk.exceptions.ResponseError as e:
            logger.debug(f"Challenge failed: {e}")
        except Exception as e:
            logger.debug(f"Challenge error: {e}")

    def _save_game_log(self, game_id: str, game_info: dict, moves: list):
        """Save game data for later analysis."""
        log = {
            "game_id": game_id,
            "moves": moves,
            "white": game_info.get("white", {}).get("name", ""),
            "black": game_info.get("black", {}).get("name", ""),
            "timestamp": time.time(),
        }
        path = self.games_dir / f"{game_id}.json"
        try:
            with open(path, "w") as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save game log: {e}")
