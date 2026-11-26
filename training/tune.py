"""Hill climbing over the evaluator's weights (local search, offline).

Every candidate plays the same seeds against the same rivals, and both agents are
deterministic given the seed, so two candidates are compared on identical games
(common random numbers). That removes most of the dice noise from the comparison.

    python training/tune.py 40        # iterations; prints the best weights found
"""
from __future__ import annotations

import json
import multiprocessing
import random
import sys

from play import DEMO_A_PATH, load_agent, play_one_game

SEEDS = range(50_000, 50_240)
RIVALS = ("demo_a", "reference")
TUNABLE = ("pip", "missing_resource", "expansion_spot", "expansion_far", "road_length", "card",
           "goal_city", "goal_settlement", "goal_development", "knight_in_hand", "progress_card_in_hand",
           "knight_played", "robber_pip", "steal", "port_specific", "port_generic")


def _agent_with(weights: dict, placement: dict):
    agent = load_agent()
    agent.WEIGHTS.update(weights)
    agent.PLACEMENT_RESOURCE_WEIGHT.update(placement)
    return agent


def _play(job) -> int:
    """1 if the candidate wins this game, else 0."""
    weights, placement, rival, seed, index = job
    hero = _agent_with(weights, placement)
    hero_policy = lambda observation: hero.advanced_policy(observation, q_bonus=0.0)
    if rival == "demo_a":
        demo = load_agent(DEMO_A_PATH)
        rival_policy = lambda observation: demo.act(observation)["action"]
    else:  # the untuned agent as it is in the file
        reference = load_agent()
        rival_policy = lambda observation: reference.advanced_policy(observation, q_bonus=0.0)
    hero_side = "A" if index % 2 == 0 else "B"
    players = (hero_policy, rival_policy) if hero_side == "A" else (rival_policy, hero_policy)
    winner, _, _ = play_one_game(*players, seed)
    return int(winner == hero_side)


def fitness(pool, weights: dict, placement: dict) -> float:
    jobs = [(weights, placement, rival, seed, index) for rival in RIVALS for index, seed in enumerate(SEEDS)]
    return sum(pool.map(_play, jobs, chunksize=8)) / len(jobs)


def neighbour(rng: random.Random, weights: dict, placement: dict):
    weights, placement = dict(weights), dict(placement)
    for name in rng.sample(TUNABLE, 3):
        weights[name] *= rng.uniform(0.6, 1.5)
    if rng.random() < 0.3:
        resource = rng.choice(sorted(placement))
        placement[resource] *= rng.uniform(0.8, 1.25)
    return weights, placement


def main(iterations: int) -> None:
    rng = random.Random(7)
    base = load_agent()
    best_w = {name: base.WEIGHTS[name] for name in TUNABLE}
    best_p = dict(base.PLACEMENT_RESOURCE_WEIGHT)
    with multiprocessing.Pool() as pool:
        best = fitness(pool, best_w, best_p)
        print(f"start {best:.3f}", flush=True)
        for step in range(iterations):
            candidate_w, candidate_p = neighbour(rng, best_w, best_p)
            score = fitness(pool, candidate_w, candidate_p)
            accepted = score > best
            if accepted:
                best, best_w, best_p = score, candidate_w, candidate_p
            print(f"step {step:3d} score {score:.3f} best {best:.3f} {'ACCEPT' if accepted else ''}", flush=True)
    print(json.dumps({"weights": {k: round(v, 3) for k, v in best_w.items()},
                      "placement": {k: round(v, 3) for k, v in best_p.items()}, "fitness": best}))


if __name__ == "__main__":
    main(int(sys.argv[1]))
