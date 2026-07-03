# Community notebook survey

Last reviewed: 2026-07-03

This is a living snapshot of public Kaggle notebooks and discussion threads for
the Pokemon TCG AI Battle Challenge. Kaggle notebook pages are dynamic and not
always fully readable outside Kaggle, so this note focuses on high-level
patterns, source links, and ideas to test locally rather than copying code.

## Immediate takeaways

- Most public competitive work is still rule-based or heuristic-first.
- Strong submissions package `main.py`, `deck.csv`, and the official `cg/`
  package together in a `.tar.gz`.
- Local matchup testing is a common workflow: iterate against random agents,
  public baselines, and known archetype counters before submitting.
- The public meta is already reacting to specific threats rather than only
  maximizing generic deck strength.
- Early RL exists, but community evidence still favors robust heuristic agents
  and search/rollout scaffolding over pure RL from scratch.

## Packaging consensus

Multiple public notebooks build a submission archive containing:

```text
main.py
deck.csv
cg/
```

This matches our current `scripts/build_submission.py` output. Including `cg/`
is still the safer choice: the sample submission includes it, public notebooks
copy it, and community posts report failures or import issues when it is absent.
The compressed size of our package is currently about 1.9 MiB, well below the
competition size limit reported on Kaggle.

Sources:

- Beginner submission guide:
  https://www.kaggle.com/code/kazutamizuta/beginner-guide-from-deck-to-first-valid-submiss
- Dragapult tempo agent:
  https://www.kaggle.com/code/zoli800/dragapult-v3-tempo-ptcg-ai-battle-agent
- Validated rule-based baseline:
  https://www.kaggle.com/code/kojimar/simple-baseline-matchup-tests
- Missing-`cg` failure note:
  https://qiita.com/Te2hi-ro/items/ecd2f882d7c3dcaa27d0

## Agent architecture patterns

The common baseline shape is:

```text
Observation -> legal options -> score/rank options -> choose legal indices
```

Common priorities:

- attack if a useful attack is available;
- take knockouts and prize-progressing lines;
- evolve into the main attacker;
- attach energy to attackers that can use it soon;
- play setup/search/draw cards before lower-impact actions;
- keep a random or simple legal fallback to avoid invalid submissions.

This aligns with our current `agents/heuristic_agent.py`. The next improvement
should be better board evaluation, not a wholesale rewrite.

Sources:

- PTCGAI Optimize Baseline:
  https://www.kaggle.com/code/suneetsaini/ptcgai-optimize-baseline
- Simple Baseline + Matchup Tests:
  https://www.kaggle.com/code/kojimar/simple-baseline-matchup-tests
- Sample Iono rule-based agent:
  https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-iono-s-deck

## Meta and deck signals

Public notebooks are not converging on one deck. They are exploring multiple
archetypes and counters:

- Dragapult ex tempo: common public reference point and good heuristic target.
- Crustle: early threat/counter target; several notebooks mention beating or
  adapting to a Crustle wall.
- Starmie/Froslass: reported as a high-performing rule-based approach with
  prize-card tracking.
- Archaludon ex / Cinderace: shared as a strong matchup-oriented sample.
- Mega Lucario ex and Lucario/Alakazam variants: popular rule-based examples.
- Iono's deck: official/sample-style rule-based baseline.

Implication for us: keep Hydrapple as a simple learning deck, but do not assume
it is a strong long-term meta deck. Our second serious target should probably be
a matchup-tested archetype such as Dragapult, Starmie/Froslass, or a metal-tempo
deck if local testing confirms the public trend.

Sources:

- Dragapult v3 Tempo:
  https://www.kaggle.com/code/zoli800/dragapult-v3-tempo-ptcg-ai-battle-agent
- Prize Card Tracking Starmie:
  https://www.kaggle.com/code/masamikobayashi/prize-card-tracking-1250-starmie
- Archaludon sample:
  https://www.kaggle.com/code/masamikobayashi/a-sample-archaludon-75-wr-vs-my-1300-starmie
- Beating Day-1 Crustle:
  https://www.kaggle.com/code/dashimaki360/beating-the-day-1-1-crustle-bot
- Lucario & Alakazam:
  https://www.kaggle.com/code/pilkwang/pokemon-tcg-lucario-alakazam
- Meta snapshot:
  https://www.kaggle.com/code/pilkwang/pok-mon-tcg-ai-battle-meta-snapshot-06-28/output

## Search and RL signals

There are public mentions of:

- MCTS or bounded lookahead for important decisions;
- heuristic search with strict time budgets;
- PPO/self-play experiments;
- behavior cloning or imitation from heuristic decisions.

The community signal so far is cautious: RL is being explored, but the practical
public agents still appear to lean heavily on heuristics, matchup testing, and
deck-specific logic. One discussion snippet reports a behavior-cloning/self-play
experiment with low direct win rate against a heuristic baseline, which supports
our plan to use heuristics first and RL later.

Sources:

- PPO agent:
  https://www.kaggle.com/code/hmnshudhmn24/pok-mon-tcg-ai-battle-challenge-ppo-agent
- Analysis mentioning MCTS budget:
  https://www.kaggle.com/code/sonika20000002/pokemon-tcg-analysis
- Mega Lucario search/lookahead:
  https://www.kaggle.com/code/nursrijan/pokemon-ai-battle-agent-mega-lucario
- Kaggle discussion on RL/self-play:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/711644

## Practical backlog for this repo

1. Keep the current heuristic agent as the stable baseline.
2. Add structured logging of `(state summary, legal options, scores, chosen option, result)`.
3. Build a local matchup harness that can run deck A/agent A vs deck B/agent B.
4. Add specific policy tests for known meta problems:
   - getting stuck with a bad Active Pokemon;
   - failing to attack;
   - bad energy attachments;
   - poor discard choices;
   - not prioritizing final-prize or KO lines.
5. Add at least one public-meta-inspired deck beyond Hydrapple/Dragapult.
6. Save self-play logs in a format usable for imitation learning.
7. Train a first value model before attempting full PPO.

## Notes for later surveys

The public meta is changing quickly. Re-run this survey regularly and track:

- notebooks claiming medal-range or rating improvements;
- decks with published matchup matrices;
- validation failures reported in discussions;
- any evidence that pure RL agents are outperforming heuristic baselines;
- changes to the official `cg` SDK or submission packaging.
