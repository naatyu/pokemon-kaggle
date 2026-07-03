# Kaggle Pokemon TCG AI Battle

This repo contains starter decks, a simple heuristic agent, local matchup
tools, and a build script for the Kaggle Pokemon TCG AI Battle Challenge.

## Build a submission

```bash
python3 scripts/build_submission.py --deck hydrapple
```

The upload-ready archive is generated at:

```text
build/kaggle_agent.tar.gz
```

You can also build another available deck profile:

```bash
python3 scripts/build_submission.py --deck dragapult
```

## Run local matchups

Quick smoke test:

```bash
python3 scripts/play_smoke_test.py --deck hydrapple --games 3
```

Matchup runner with structured JSONL logs:

```bash
python3 scripts/run_matchups.py \
  --agent heuristic \
  --deck hydrapple \
  --opponent-agent random \
  --opponent-deck abomasnow_sample \
  --games 20 \
  --log-jsonl logs/hydrapple_vs_abomasnow_random.jsonl
```

The JSONL logs include legal options, heuristic scores, chosen actions, board
summaries, final results, and per-player outcomes. They are intended for
debugging now and imitation/value-model training later.

## Run the arena

Build local baseline arena agents:

```bash
python3 scripts/build_arena_agents.py --clean
```

Run a small round-robin:

```bash
python3 scripts/run_arena.py --games 10
```

Import a downloaded public submission archive:

```bash
python3 scripts/import_arena_agent.py \
  --name public_agent_name \
  --archive path/to/submission.tar.gz
```

Then test it:

```bash
python3 scripts/run_arena.py \
  --agents public_agent_name hydrapple_heuristic dragapult_heuristic \
  --games 20
```

Pull and extract public Kaggle notebook agents after authenticating the Kaggle
CLI:

```bash
uv run kaggle kernels pull pixiux/ptcg-mega-lucario-ex-v62 -p arena/public_sources/mega_lucario_v62 -m
python3 scripts/extract_public_notebook_agent.py \
  --name public_mega_lucario_v62 \
  --notebook arena/public_sources/mega_lucario_v62/ptcg-mega-lucario-ex-v62.ipynb \
  --replace
```

See [arena/opponent_pool.md](arena/opponent_pool.md) for the current imported
public opponent set and mini-league results.

## Train the PPO baseline

The PPO environment lives in `rl/ptcg_env.py`. It uses legal action masks and
orders legal options by the current heuristic score, so action ranks are more
stable than raw simulator option indexes.

Train against the random Abomasnow deck:

```bash
uv run python scripts/train_ppo.py \
  --opponent random_abomasnow \
  --timesteps 20000 \
  --save-path models/ppo_ranked_random20k
```

Evaluate a checkpoint:

```bash
uv run python scripts/evaluate_ppo.py \
  --model models/ppo_ranked_random20k.zip \
  --opponent random_abomasnow \
  --games 100 \
  --deterministic
```

Current local results:

```text
models/ppo_ranked_random20k.zip vs random_abomasnow: 63/100 wins
models/ppo_ranked_random20k.zip vs heuristic_hydrapple: 19/50 wins
models/ppo_ranked_heuristic20k.zip vs random_abomasnow: 36/50 wins
models/ppo_ranked_heuristic20k.zip vs heuristic_hydrapple: 9/50 wins
```

The random-trained PPO is the useful first baseline. The direct heuristic
fine-tune overfit/regressed, so the next step should be mixed-opponent training
rather than one-opponent fine-tuning.
