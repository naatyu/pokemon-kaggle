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

Fast mixed-opponent training:

```bash
uv run python scripts/train_ppo.py \
  --opponent random_abomasnow,heuristic_hydrapple \
  --timesteps 100000 \
  --load-path models/ppo_ranked_random20k.zip \
  --save-path models/ppo_mixed_fixed_100k \
  --n-envs 16 \
  --n-steps 128 \
  --batch-size 512 \
  --learning-rate 0.0001 \
  --ent-coef 0.01 \
  --device cuda
```

Evaluate a checkpoint:

```bash
uv run python scripts/evaluate_ppo.py \
  --model models/ppo_mixed_fixed_100k.zip \
  --opponent random_abomasnow \
  --games 100 \
  --deterministic \
  --device cuda
```

Current local results:

```text
single-env CPU benchmark: about 212 fps
16-env CUDA/subprocess benchmark: about 903-945 fps

models/ppo_ranked_random20k.zip vs random_abomasnow: 65/100 wins
models/ppo_ranked_random20k.zip vs heuristic_hydrapple: 15/50 wins
models/ppo_mixed_fixed_100k.zip vs random_abomasnow: 66/100 wins
models/ppo_mixed_fixed_100k.zip vs heuristic_hydrapple: 21/50 wins
```

Use `--opponent a,b,c` to sample opponents per episode. The simulator is
process-global inside one Python process, so speed comes from `SubprocVecEnv`
with separate simulator processes rather than from GPU alone.
