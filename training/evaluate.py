"""Win rates of one policy against another over many seeds, playing both seats.

    python training/evaluate.py advanced demo_a 200
"""
from __future__ import annotations

import random
import sys
import time

from play import DEMO_A_PATH, TimedPolicy, load_agent, play_one_game, random_policy_factory


def make_policy(name: str, seed: int):
    if name == "random":
        return random_policy_factory(seed)
    if name == "demo_a":
        demo = load_agent(DEMO_A_PATH)
        return lambda observation: demo.act(observation)["action"]
    agent = load_agent()
    if name == "baseline":
        return agent.heuristic_policy
    if name == "advanced":
        return lambda observation: agent.advanced_policy(observation, q_bonus=0.0)
    if name.startswith("q"):  # e.g. q0.3 = advanced with Q bonus weight 0.3
        weight = float(name[1:])
        return lambda observation: agent.advanced_policy(observation, q_bonus=weight)
    raise ValueError(name)


def evaluate(hero: str, rival: str, games: int, first_seed: int = 10_000) -> dict:
    random.seed(0)  # demo_A uses the global random; new_game reseeds it anyway
    tally = {"win": 0, "loss": 0, "draw": 0}
    spent, slowest, vp, rounds = [], 0.0, [], []
    for index in range(games):
        seed = first_seed + index
        hero_policy = TimedPolicy(make_policy(hero, seed))
        rival_policy = make_policy(rival, seed + 1)
        hero_side = "A" if index % 2 == 0 else "B"
        players = (hero_policy, rival_policy) if hero_side == "A" else (rival_policy, hero_policy)
        winner, points, played = play_one_game(*players, seed)
        rounds.append(played)
        outcome = "draw" if winner is None else ("win" if winner == hero_side else "loss")
        tally[outcome] += 1
        spent.append(hero_policy.spent)
        slowest = max(slowest, hero_policy.slowest)
        vp.append(points[hero_side])
    return {
        "hero": hero, "rival": rival, "games": games, **tally,
        "win_rate": round(tally["win"] / games, 3),
        "mean_vp": round(sum(vp) / games, 2),
        "mean_rounds": round(sum(rounds) / games, 1),
        "mean_time_s": round(sum(spent) / games, 3),
        "max_time_s": round(max(spent), 3),
        "slowest_decision_ms": round(slowest * 1000, 1),
    }


if __name__ == "__main__":
    hero, rival, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    started = time.time()
    print(evaluate(hero, rival, count), f"({time.time() - started:.0f} s)")
