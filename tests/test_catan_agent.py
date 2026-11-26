"""Behaviour tests for team/catan_agent.py.  Run: python -m pytest tests"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from play import load_agent, play_one_game, random_policy_factory  # noqa: E402

SAMPLES = ROOT / "ai_agent_tournament" / "docs" / "students" / "sample_observations"


def sample(name: str) -> dict:
    return json.loads((SAMPLES / f"catan_{name}.json").read_text())


@pytest.fixture
def agent():
    return load_agent()


@pytest.fixture
def play_turn():
    return sample("play_turn")


def test_q_update_follows_the_q_learning_rule(agent):
    result = agent.q_update(1.0, 2.0, 3.0, 0.5, 0.9)

    assert result == pytest.approx(1.0 + 0.5 * (2.0 + 0.9 * 3.0 - 1.0))


def test_every_board_node_has_two_or_three_neighbours(agent, play_turn):
    board = agent.BoardModel(play_turn["board"])

    degrees = {len(board.neighbours[node]) for node in board.nodes}

    assert degrees <= {2, 3}


def test_node_production_ignores_the_robber_tile(agent, play_turn):
    board = agent.BoardModel(play_turn["board"])

    blocked = board.node_production(41, robber=(-2, 2, 0))

    assert "WHEAT" not in blocked


def test_tracker_credits_rival_second_settlement_and_rolls(agent, play_turn):
    board = agent.BoardModel(play_turn["board"])
    tracker = agent.OpponentTracker(board)
    log = [
        {"player": "opponent", "type": "BUILD_SETTLEMENT", "node": 41},
        {"player": "opponent", "type": "BUILD_ROAD", "edge": [39, 41]},
        {"player": "opponent", "type": "BUILD_SETTLEMENT", "node": 24},
        {"player": "opponent", "type": "BUILD_ROAD", "edge": [7, 24]},
        {"player": "me", "type": "ROLL", "dice": [4, 4]},  # 8: wheat tile next to node 41
    ]
    expected = {resource: 0 for resource in agent.RESOURCES}
    for resource in board.node_production(24):
        expected[resource] += 1
    expected["WHEAT"] += 1

    tracker.update(log, sum(expected.values()))

    assert tracker.hand == pytest.approx(expected)


def test_tracker_subtracts_rival_build_costs(agent, play_turn):
    board = agent.BoardModel(play_turn["board"])
    tracker = agent.OpponentTracker(board)
    tracker.rolled = True
    tracker.hand.update(WOOD=2.0, BRICK=2.0)

    tracker.update([{"player": "opponent", "type": "BUILD_ROAD", "edge": [39, 40]}], 2)

    assert tracker.hand["WOOD"] == pytest.approx(1.0) and tracker.hand["BRICK"] == pytest.approx(1.0)


def test_planner_trades_toward_a_city_it_can_then_afford(agent, play_turn):
    observation = copy.deepcopy(play_turn)
    observation["me"]["resources"] = {"WOOD": 0, "BRICK": 0, "SHEEP": 0, "WHEAT": 5, "ORE": 2}
    observation["actions"] = [
        {"id": 0, "type": "END_TURN"},
        {"id": 1, "type": "MARITIME_TRADE", "give": "WHEAT", "give_count": 3, "receive": "SHEEP"},
        {"id": 2, "type": "MARITIME_TRADE", "give": "WHEAT", "give_count": 3, "receive": "ORE"},
    ]

    choice = agent.advanced_policy(observation, q_bonus=0.0)

    assert choice == 2


def test_city_is_preferred_to_ending_the_turn(agent, play_turn):
    observation = copy.deepcopy(play_turn)
    observation["me"]["resources"] = {"WOOD": 0, "BRICK": 0, "SHEEP": 0, "WHEAT": 2, "ORE": 3}
    observation["actions"] = [{"id": 0, "type": "END_TURN"}, {"id": 1, "type": "BUILD_CITY", "node": 15}]

    choice = agent.advanced_policy(observation, q_bonus=0.0)

    assert choice == 1


def test_robber_goes_where_the_rival_loses_most(agent):
    observation = sample("move_robber")

    choice = agent.advanced_policy(observation, q_bonus=0.0)

    chosen = next(a for a in observation["actions"] if a["id"] == choice)
    assert chosen["steal_from"] == "opponent"


@pytest.mark.parametrize("name", ["initial_placement", "play_turn", "move_robber"])
def test_policies_return_legal_ids_on_samples(agent, name):
    observation = sample(name)
    legal = {a["id"] for a in observation["actions"]}

    assert agent.heuristic_policy(observation) in legal and agent.act(observation)["action"] in legal


def test_state_features_are_deterministic(agent, play_turn):
    assert agent.state_features(play_turn) == agent.state_features(copy.deepcopy(play_turn))


def test_action_value_is_finite_for_every_action(agent, play_turn):
    values = [agent.action_value(play_turn, a["id"]) for a in play_turn["actions"]]

    assert all(abs(v) < float("inf") for v in values)


def test_low_clock_falls_back_to_the_baseline(agent, play_turn):
    observation = dict(play_turn, time_remaining_s=0.05)

    assert agent.act(observation)["action"] == agent.heuristic_policy(observation)


def test_full_games_against_random_are_always_legal_and_won():
    wins = 0
    for seed in range(1, 21):
        hero = load_agent()
        hero_side = "A" if seed % 2 else "B"
        policies = (lambda o: hero.advanced_policy(o), random_policy_factory(seed))
        if hero_side == "B":
            policies = policies[::-1]

        winner, _, _ = play_one_game(*policies, seed)  # raises on an illegal id

        wins += winner == hero_side
    assert wins >= 19
