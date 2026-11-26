"""Offline game loop, the same one as docs/students/TRAINING.md.

Each player gets its own copy of the agent module, because the agent keeps
per-game memory at module level, exactly as it does in its own judge process.
"""
from __future__ import annotations

import importlib.util
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOURNAMENT = ROOT / "ai_agent_tournament"
AGENT_PATH = ROOT / "team" / "catan_agent.py"
DEMO_A_PATH = TOURNAMENT / "examples" / "team_demo_A" / "catan_agent.py"
sys.path.insert(0, str(TOURNAMENT))

from games.catan.engine import SIDE_OF, adjudicate, legal_actions, new_game, observation_for, victory_points  # noqa: E402

MAX_ROUNDS = 100
_loaded = 0


def load_agent(path: Path = AGENT_PATH):
    """A fresh, independent copy of an agent module."""
    global _loaded
    _loaded += 1
    spec = importlib.util.spec_from_file_location(f"agent_copy_{_loaded}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TimedPolicy:
    """Wraps a policy and records the time it spends, like the judge's clock."""

    def __init__(self, policy):
        self.policy = policy
        self.spent = 0.0
        self.slowest = 0.0

    def __call__(self, observation):
        started = time.perf_counter()
        choice = self.policy(observation)
        elapsed = time.perf_counter() - started
        self.spent += elapsed
        self.slowest = max(self.slowest, elapsed)
        return choice


def play_one_game(policy_a, policy_b, seed: int, on_decision=None):
    """Returns (winner or None, {"A": vp, "B": vp}, rounds played). Use a positive seed."""
    game = new_game(seed, vps_to_win=7)
    policies = {"A": policy_a, "B": policy_b}
    while game.winning_color() is None and game.state.num_turns // 2 < MAX_ROUNDS:
        actions = legal_actions(game)
        if len(actions) == 1:
            game.execute(actions[0])
            continue
        side = SIDE_OF[game.state.current_color()]
        observation = observation_for(game, side, actions, None, {"max_rounds": MAX_ROUNDS})
        action_id = policies[side](observation)
        if not isinstance(action_id, int) or not 0 <= action_id < len(actions):
            raise ValueError(f"side {side} returned illegal action {action_id!r}")
        if on_decision is not None:
            on_decision(side, observation, action_id)
        game.execute(actions[action_id])
    champion = game.winning_color()
    winner = SIDE_OF[champion] if champion is not None else adjudicate(game)[0]
    return winner, victory_points(game), game.state.num_turns // 2


def random_policy_factory(seed: int):
    rng = random.Random(seed)
    return lambda observation: rng.choice(observation["actions"])["id"]
