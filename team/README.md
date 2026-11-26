# Technical explanation — Catan

Files: `team/catan_agent.py` (submission), `training/` (game loop, evaluation, Q-learning, weight tuning), `tests/` (`python -m pytest tests`).

## Catan 1v1

### Problem formulation
- **Environment.** Two players, 19-tile board, 7 VP to win, bank/port trades only, random discard above 7 cards, dice and the development deck are stochastic, the rival's hand and cards are hidden. The judge lists the legal actions; we return one id.
- **What the agent observes.** Its own full state, the rival's public state (buildings, roads, card *counts*, played cards), the board and the public log.
- **What it optimises.** Probability of reaching 7 VP first. In practice: the value of the state after each action (an *afterstate*), measured in hundredths of a VP.
- **Clock.** 5 s of decision time for the whole game, measured by the judge including the pipe and JSON parsing. So the agent never simulates the engine; every decision is a few loops over 54 nodes.

### Algorithms (all in `catan_agent.py`)
| Part | Where | What it does |
|---|---|---|
| Board model | `BoardModel` | Adjacency from tile edges, pips per node and resource, ports, resource scarcity on this board. Built once per game. |
| Logic & inference | `OpponentTracker` | Replays the public log to deduce the rival's hand (see below). |
| Classical planning | `Planner` | Builds as operators with **preconditions** (cost in the hand, legal place: distance rule, connected to my roads, rival buildings cut my network) and **effects** (VP, production, reach, longest road / largest army). Maritime trades are operators on the hand. `plan_value` searches up to 2 trades + 1 build this turn (memoised over hands). |
| Value function | `Planner.hand_value`, `TurnSituation`, `Evaluator` | V(afterstate) = structure + hand. Structure: VP (100 each, +10000 for the win), production weighted by scarcity and by what I lack, the best settlement spot my roads reach (and one road away), road length, knights. Hand: card value, progress toward city / settlement / dev card, expected discard loss when holding > 7. |
| Expectations under uncertainty | `development_card_value`, `hand_value` | A dev card is worth the expectation over the **unknown** deck (25 cards minus every card whose identity I know); holding > 7 cards costs P(a 7 before my next turn) × half the hand. |
| Robber | `TurnSituation.robber_value` | Rival pips blocked − 1.3 × my pips blocked (+ what moving it off my tile frees) + value of a steal. |
| Placement | `placement_value` | Pips × resource weight × scarcity, diversity, repeated-number penalty, complementing the first settlement, wood+brick availability, ports. Initial road points at the best free spot two steps away. |
| MDP + Q-learning | `state_features`, `action_kind`, `Q`, `q_update`, `training/train_q.py` | See below. |
| Weight tuning (local search) | `training/tune.py` | Hill climbing on the evaluator weights with common random numbers (both agents are deterministic per seed, so candidates are compared on identical games). |

#### What the agent deduces with certainty from public events
- Every production of the rival: the dice are public, its buildings are public and we track the robber position through the log, so each roll gives exact cards (bank shortages aside).
- The rival's second initial settlement pays one card per adjacent tile.
- Every build cost (roads free during placement and after Road Building), dev-card purchases, maritime trades, Year of Plenty.
- Cards stolen from us or by us (the victim and the thief see the card).
- What is uncertain: discards and steals we did not take part in, the cards it took with Monopoly. These are spread in proportion to the current estimate, and the estimate is always rescaled to the rival's public card count. The estimate drives Monopoly choice and steal value.
- Dev deck: which card types can still be drawn (unknown pool = deck − my cards − rival's played cards).

#### MDP abstraction and Q-learning
- **State** `state_features`: (phase, my VP 0–7, VP lead clamped to ±3, production bucket of 5 pips, can afford city, can afford settlement, can afford dev card).
- **Actions**: the action type (`BUILD_CITY`, `MARITIME_TRADE`, `END_TURN`, …). The planner chooses the concrete action inside a type.
- **Reward**: VP gained between my consecutive decisions; at the end, the last VP gain ± 1 for win/loss. γ = 0.95, α = 0.1.
- **Training**: `training/train_q.py`, ε-greedy over action types (ε 0.3 → 0.05), half the games in self-play and half against demo_A, playing both seats. Entries seen fewer than 5 times are dropped.
- **Use**: `action_value = planner afterstate value + 100 · q_bonus · Q(s, type)`.

### Baseline against improved version
All rows use the same 500 held-out seeds (never used for tuning or training), both seats alternated. `baseline` = `heuristic_policy` (most pips, fixed build priority, no trades). `advanced` = planner + tracker + tuned weights, no Q. `qX` = advanced with the Q bonus weighted X. Reproduce with `python training/compare.py 500`.

| Hero | Rival | Win rate | Mean VP | Mean rounds |
|---|---|---|---|---|
| baseline | random | 89.0% | 6.29 | 57.4 |
| baseline | demo_A | 36.8% | 4.86 | 42.8 |
| advanced | random | 100.0% | 7.16 | 19.9 |
| advanced | demo_A | 91.6% | 6.91 | 23.7 |
| advanced | baseline | 92.2% | 6.94 | 23.9 |
| q0.1 | demo_A | 89.6% | 6.88 | 23.8 |
| q0.3 | demo_A | 91.0% | 6.92 | 24.2 |
| q1.0 | demo_A | 88.6% | 6.84 | 25.2 |
| q0.1 | advanced | 53.6% | 5.75 | 21.1 |
| q0.3 | advanced | 49.2% | 5.61 | 21.3 |
| q1.0 | advanced | 45.2% | 5.47 | 21.7 |

- **Planner vs baseline** is the big gain: 36.8% → 91.6% against demo_A, and games finish in about 24 rounds instead of 43.
- **Weight tuning** (hill climbing, 60 steps, 480 paired games per step): on held-out seeds the tuned weights win 54.0% head-to-head against the hand-set ones and 93.3% vs 92.3% against demo_A, so a small but real gain.
- **Q-learning** (4000 training games, 1553 table entries kept): the effect is within noise (standard error about ±2.2 points per row). A small weight (0.1) is slightly better head-to-head and slightly worse against demo_A; large weights hurt. The coarse abstraction cannot tell apart situations the planner already handles. We compete with `q_bonus = 0.1`, which is no worse than the planner alone and keeps the learned table in use. We report this as an honest finding: the learned table refines, it does not replace, the model-based planner.

### Time and memory
- Pure thinking time is about 55 ms per *game* (about 1–2 ms per decision, slowest decision about 25 ms), measured in `training/evaluate.py`.
- In the real judge, each call also pays the pipe and JSON overhead (about 15–20 ms per decision on our Windows machine, the same for the demo agents). Over 6 judge games our agent kept between 4.09 and 4.53 s of its 5 s clock, with 24–43 decisions per game.
- Safety: when `time_remaining_s < 0.4` the agent switches to the O(1) baseline, and any unexpected exception also falls back to the baseline instead of forfeiting.
- Memory: the board model and tracker are rebuilt once per game; the file (Q-table included) is about 125 KB and loads instantly.

### Limits
- One-turn horizon: the planner does not plan several turns of saving explicitly; that is approximated by the "progress toward a goal" terms of the hand value.
- The rival model is passive: we do not predict where it will build, so we do not block its spots on purpose.
- The Q abstraction is coarse; its contribution is small next to the planner (see the comparison).
- With more time: expectimax over the next roll for build/hold decisions, and blocking of the rival's best expansion spot.

### Use of generative AI
This agent was developed with the help of an AI coding assistant (Claude). We reviewed and can explain every algorithm above.
