"""The README comparison table: every matchup on the same held-out seeds, in parallel.

    python training/compare.py 500
"""
from __future__ import annotations

import multiprocessing
import sys

from evaluate import evaluate

MATCHUPS = [
    ("baseline", "random"), ("baseline", "demo_a"),
    ("advanced", "random"), ("advanced", "demo_a"), ("advanced", "baseline"),
    ("q0.1", "demo_a"), ("q0.3", "demo_a"), ("q1.0", "demo_a"),
    ("q0.1", "advanced"), ("q0.3", "advanced"), ("q1.0", "advanced"),
]
HELD_OUT_FIRST_SEED = 700_000  # never used by tune.py or train_q.py


def _run(args):
    hero, rival, games, chunk = args
    return hero, rival, evaluate(hero, rival, games, HELD_OUT_FIRST_SEED + chunk * games)


def main(games: int, chunks: int = 4) -> None:
    per_chunk = games // chunks
    jobs = [(hero, rival, per_chunk, chunk) for hero, rival in MATCHUPS for chunk in range(chunks)]
    with multiprocessing.Pool() as pool:
        results = pool.map(_run, jobs)
    print("| Hero | Rival | Games | Win rate | Mean VP | Mean rounds | Mean think time / game |")
    print("|---|---|---|---|---|---|---|")
    for hero, rival in MATCHUPS:
        parts = [r for h, v, r in results if (h, v) == (hero, rival)]
        n = sum(p["games"] for p in parts)
        wins = sum(p["win"] for p in parts)
        mean = lambda key: sum(p[key] * p["games"] for p in parts) / n
        print(f"| {hero} | {rival} | {n} | {wins / n:.1%} | {mean('mean_vp'):.2f} | {mean('mean_rounds'):.1f} | {mean('mean_time_s') * 1000:.0f} ms |")


if __name__ == "__main__":
    main(int(sys.argv[1]))
