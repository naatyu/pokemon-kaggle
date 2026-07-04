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

Fast mixed-opponent training with best-checkpoint selection:

```bash
POOL='public_metal_archaludon,public_multiply_940,public_mega_lucario_v62,public_strong_start_v10,public_baseline_1084,public_archaludon_75wr,public_alakazam_best5,public_crustle_v1,public_phantom_dragapult,public_froslass_sleep,public_kangaskhan_pressure,public_kiyota_mega_lucario,public_kiyota_dragapult,public_kiyota_iono,heuristic_hydrapple,heuristic_dragapult,random_abomasnow'
uv run python scripts/train_ppo.py \
  --deck metal_archaludon \
  --opponent "$POOL" \
  --timesteps 50000 \
  --load-path models/ppo_action_embed_effect_bc_public_metal_30k.zip \
  --save-path models/ppo_action_embed_broad_50k \
  --n-envs 16 \
  --n-steps 128 \
  --batch-size 512 \
  --learning-rate 0.00001 \
  --ent-coef 0.02 \
  --policy action_embed \
  --policy-hidden-dim 256 \
  --card-embedding-dim 32 \
  --attack-embedding-dim 16 \
  --effect-features \
  --reward-shaping-scale 0 \
  --tempo-reward-scale 0 \
  --bc-teacher public_metal_archaludon \
  --bc-samples 6000 \
  --bc-coef 0.40 \
  --bc-dataset-path data/bc_public_metal_broad_6k.npz \
  --eval-opponent "$POOL" \
  --eval-games 40 \
  --eval-freq 4096 \
  --best-save-path models/best/ppo_action_embed_broad_best \
  --device cuda
```

`--device auto` now resolves to `cuda` when PyTorch can see a CUDA GPU and
prints the resolved CUDA card at startup. Keep `--n-envs` high for speed: the
neural network updates run on GPU, but game simulation, public-agent calls, and
legal-action generation are still CPU work. In-training evaluation must use
subprocess envs; do not combine `--eval-opponent` with `--start-method dummy`
or a single env.

Behavioral-cloning pretraining has the same split: teacher trajectory
collection is CPU-bound, then cross-entropy training runs on `--device`.
Collection keeps a large observation dataset in RAM; start with 2-4 collection
workers on a 32 GB machine and increase only after watching memory use.

```bash
uv run python scripts/pretrain_ppo_bc.py \
  --deck metal_archaludon \
  --teacher public_metal_archaludon \
  --opponent "$POOL" \
  --samples 30000 \
  --epochs 12 \
  --batch-size 1024 \
  --validation-split 0.1 \
  --patience 3 \
  --dataset-path data/bc_public_metal_broad_30k.npz \
  --policy action_embed \
  --policy-hidden-dim 256 \
  --card-embedding-dim 32 \
  --attack-embedding-dim 16 \
  --effect-features \
  --collection-workers 2 \
  --device cuda \
  --save-path models/ppo_action_embed_effect_bc_public_metal_30k
```

`--policy action_embed` is the recommended policy for new checkpoints. It scores
each ranked action choice with shared action-slot weights, and also embeds card
IDs and attack IDs categorically instead of treating them only as continuous
numbers. Use `--effect-features` consistently for training, evaluation, and
analysis of checkpoints trained with those extra state signals.

`--policy action_embed_rank` adds learnable positional embeddings and logit
biases for the heuristic-sorted action slots. In a 5k public-Metal BC smoke it
improved validation accuracy from `0.378` to `0.544`, but the corresponding
20k PPO experiments did not replace `models/best/ppo_action_embed_broad_best.zip`
as the best aggregate checkpoint. Treat it as the next architecture to test
with larger BC datasets, not as the current submission default.

`train_ppo.py` can reuse BC regularization data with `--bc-dataset-path` and
`--bc-reuse-dataset`. This avoids repeatedly simulating the same teacher states
when running short PPO experiments.

`--tempo-reward-scale` adds optional dense reward for board setup, ready
attackers, useful energy placement, and prize tempo. Keep it at `0` for broad
baselines; use small values such as `0.05` only for focused experiments.

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

Check whether a checkpoint is actually copying the public teacher's decisions:

```bash
uv run python scripts/analyze_teacher_agreement.py \
  --model models/best/ppo_action_broad_best.zip \
  --deck metal_archaludon \
  --teacher public_metal_archaludon \
  --opponent public_metal_archaludon \
  --samples 3000 \
  --device cuda
```

See [docs/ppo_public_agent_findings.md](docs/ppo_public_agent_findings.md) for
public-agent experiments, behavioral cloning results, and the current diagnosis
for why the strongest public agents still beat PPO.
