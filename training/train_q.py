"""Tabular Q-learning over the agent's abstraction, offline.

MDP used for learning (see README):
  state   s = state_features(observation)     phase, VP, lead, production, affordability
  action  a = action_kind(action)             the action type
  reward  r = VP gained until my next decision; +/-1 more when the game ends
  update  agent.q_update (Q-learning, off-policy: bootstraps on max_a' Q(s', a'))

Behaviour policy: epsilon-greedy over action kinds. With probability epsilon a
random available kind is chosen; otherwise the kind maximising heuristic + Q.
Within a kind the planner's best action is played, so exploration stays sensible.

    python training/train_q.py 4000     # games; rewrites the Q line in the agent
"""
from __future__ import annotations

import random
import re
import sys
from collections import defaultdict

from play import AGENT_PATH, DEMO_A_PATH, load_agent, play_one_game

ALPHA, GAMMA = 0.1, 0.95
EPSILON_START, EPSILON_END = 0.3, 0.05
WIN_REWARD = 1.0
MIN_VISITS = 5
Q_BONUS_DURING_TRAINING = 0.3


class Learner:
    def __init__(self, agent):
        self.agent = agent
        self.q = defaultdict(float)
        self.visits = defaultdict(int)

    def best_q(self, features, kinds) -> float:
        return max((self.q[(features, kind)] for kind in kinds), default=0.0)

    def update(self, features, kind, reward, next_features, next_kinds):
        best_next = self.best_q(next_features, next_kinds) if next_features is not None else 0.0
        key = (features, kind)
        self.q[key] = self.agent.q_update(self.q[key], reward, best_next, ALPHA, GAMMA)
        self.visits[key] += 1


class LearningPlayer:
    """Plays one side, remembers its last (s, a, VP) and learns at the next decision."""

    def __init__(self, learner: Learner, epsilon: float, rng: random.Random):
        self.learner, self.epsilon, self.rng = learner, epsilon, rng
        self.agent = load_agent()  # own module copy: own per-game memory
        self.previous = None

    def __call__(self, observation) -> int:
        agent = self.agent
        if observation["phase"] == "initial_placement":
            return agent.choose_initial(observation)
        features = agent.state_features(observation)
        kinds = sorted({agent.action_kind(a) for a in observation["actions"]})
        vp = observation["me"]["victory_points"]
        if self.previous is not None:
            old_features, old_kind, old_vp = self.previous
            self.learner.update(old_features, old_kind, vp - old_vp, features, kinds)
        evaluator = agent.Evaluator(observation)
        scored = []
        for action in observation["actions"]:
            kind = agent.action_kind(action)
            value = evaluator.value(action) + agent.WEIGHTS["vp"] * Q_BONUS_DURING_TRAINING * self.learner.q[(features, kind)]
            scored.append((value, kind, action["id"]))
        if self.rng.random() < self.epsilon:
            chosen_kind = self.rng.choice(kinds)
            scored = [entry for entry in scored if entry[1] == chosen_kind]
        _, kind, action_id = max(scored)
        self.previous = (features, kind, vp)
        return action_id

    def finish(self, final_vp: int, won: bool) -> None:
        if self.previous is None:
            return
        features, kind, vp = self.previous
        reward = final_vp - vp + (WIN_REWARD if won else -WIN_REWARD)
        self.learner.update(features, kind, reward, None, [])


def train(games: int) -> Learner:
    rng = random.Random(3)
    learner = Learner(load_agent())
    wins = 0
    for index in range(games):
        epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * index / max(1, games - 1)
        hero = LearningPlayer(learner, epsilon, rng)
        if index % 2:  # self-play: both seats learn into the same table
            rival = LearningPlayer(learner, epsilon, rng)
            rival_policy = rival
        else:
            rival = None
            demo = load_agent(DEMO_A_PATH)
            rival_policy = lambda observation: demo.act(observation)["action"]
        hero_side = "A" if (index // 2) % 2 == 0 else "B"
        players = (hero, rival_policy) if hero_side == "A" else (rival_policy, hero)
        winner, points, _ = play_one_game(*players, 100_000 + index)
        hero.finish(points[hero_side], winner == hero_side)
        if rival is not None:
            rival_side = "B" if hero_side == "A" else "A"
            rival.finish(points[rival_side], winner == rival_side)
        wins += winner == hero_side
        if (index + 1) % 500 == 0:
            print(f"{index + 1} games, hero win rate so far {wins / (index + 1):.3f}, table {len(learner.q)}", flush=True)
    return learner


def write_table(learner: Learner) -> int:
    table = {f"{features}|{kind}": round(value, 4)
             for (features, kind), value in sorted(learner.q.items(), key=lambda item: str(item[0]))
             if learner.visits[(features, kind)] >= MIN_VISITS}
    source = AGENT_PATH.read_text(encoding="utf-8")
    source = re.sub(r"^Q = \{.*\}$", lambda _: f"Q = {table!r}", source, count=1, flags=re.MULTILINE)
    AGENT_PATH.write_text(source, encoding="utf-8")
    return len(table)


if __name__ == "__main__":
    trained = train(int(sys.argv[1]))
    print("entries written:", write_table(trained))
