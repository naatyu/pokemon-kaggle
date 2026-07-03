# PPO Public-Agent Findings

## Current result

The PPO pipeline is faster and can beat weak/local baselines, but it does not
yet beat the strongest public Kaggle agents.

Most quick sweeps below used 20 deterministic games per opponent. The current
best sweep uses 40 deterministic games per opponent.

## Current Best PPO Checkpoint

`models/best/ppo_rich_broad_best.zip` is the best current PPO-family checkpoint.
It starts from rich-observation BC, fine-tunes with:

- `--reward-shaping-scale 0`
- low learning rate: `1e-5`
- BC regularization from `public_metal_archaludon`
- mixed public/local opponent pool
- in-training masked evaluation and best-checkpoint saving

Mixed eval during training improved from `18/60` at step 0 to `22/60` at
98k timesteps. The per-opponent 40-game sweep:

```text
public_metal_archaludon: 4/40
public_multiply_940: 1/40
public_mega_lucario_v62: 3/40
public_crustle_v1: 5/40
public_phantom_dragapult: 2/40
public_froslass_sleep: 7/40
public_kangaskhan_pressure: 21/40
heuristic_hydrapple: 17/40
heuristic_dragapult: 29/40
random_abomasnow: 35/40
```

This is the first checkpoint with nonzero 40-game results against all three
strongest public agents, and it clearly beats `random_abomasnow` and
`heuristic_dragapult`. It still does not reliably beat `heuristic_hydrapple` or
the strongest public agents.

## Hydrapple PPO

`models/ppo_mixed_fixed_100k.zip`:

```text
public_metal_archaludon: 2/20
public_multiply_940: 0/20
public_mega_lucario_v62: 0/20
public_crustle_v1: 1/20
public_phantom_dragapult: 0/20
public_froslass_sleep: 1/20
public_kangaskhan_pressure: 7/20
heuristic_hydrapple: 10/20
heuristic_dragapult: 13/20
random_abomasnow: 13/20
```

`models/ppo_public_pool_200k.zip` after training against the public pool:

```text
public_metal_archaludon: 0/20
public_multiply_940: 1/20
public_mega_lucario_v62: 2/20
public_crustle_v1: 1/20
public_phantom_dragapult: 1/20
public_froslass_sleep: 1/20
public_kangaskhan_pressure: 6/20
heuristic_hydrapple: 10/20
heuristic_dragapult: 16/20
random_abomasnow: 15/20
```

Public-pool PPO improves some easier matchups but does not solve the top
public agents.

## Metal Archaludon PPO

The public Metal Archaludon deck was added as `metal_archaludon`.

Rank-0 heuristic with the metal deck is still weak:

```text
public_metal_archaludon: 2/20
public_multiply_940: 1/20
public_mega_lucario_v62: 0/20
public_crustle_v1: 2/20
public_phantom_dragapult: 3/20
public_froslass_sleep: 1/20
public_kangaskhan_pressure: 3/20
```

`models/ppo_metal_public_200k.zip`, trained from scratch with a 256x256 MLP:

```text
public_metal_archaludon: 0/20
public_multiply_940: 0/20
public_mega_lucario_v62: 0/20
public_crustle_v1: 0/20
public_phantom_dragapult: 0/20
public_froslass_sleep: 3/20
public_kangaskhan_pressure: 3/20
heuristic_hydrapple: 8/20
heuristic_dragapult: 9/20
random_abomasnow: 13/20
```

Changing deck and increasing model size did not fix the top-agent gap.

## Behavioral Cloning

`scripts/pretrain_ppo_bc.py` collects decisions from a public teacher and
pretrains a MaskablePPO policy by cross-entropy.

`models/ppo_bc_public_metal_30k.zip`, trained from the public Metal teacher:

```text
public_metal_archaludon: 0/20
public_multiply_940: 1/20
public_mega_lucario_v62: 0/20
public_crustle_v1: 8/20
public_phantom_dragapult: 1/20
public_froslass_sleep: 4/20
public_kangaskhan_pressure: 6/20
heuristic_hydrapple: 9/20
heuristic_dragapult: 10/20
random_abomasnow: 16/20
```

BC helps some mid-tier matchups, especially Crustle, but does not transfer to
the strongest public Metal/Lucario agents. PPO fine-tuning from this BC
checkpoint regressed, so the current PPO objective washes out useful imitation
before it discovers better counterplay.

## Rich Observation and Best-Checkpoint Fine-Tuning

The environment now includes a richer fixed observation:

- active and bench Pokemon slots for both players
- HP, max HP, energy, tools, evolution count, status, prize and discard counts
- per-option metadata including card id, area, target, heuristic score, card HP,
  attack damage, Pokemon/energy flags, and profile-specific card flags

`models/ppo_rich_bc_public_metal_30k.zip`, trained from the public Metal
teacher with the richer observation, improved the general baseline:

```text
public_metal_archaludon: 0/20
public_multiply_940: 0/20
public_mega_lucario_v62: 1/20
public_crustle_v1: 2/20
public_phantom_dragapult: 2/20
public_froslass_sleep: 3/20
public_kangaskhan_pressure: 12/20
heuristic_hydrapple: 10/20
heuristic_dragapult: 14/20
random_abomasnow: 16/20
```

`models/ppo_rich_bc_reg_public_100k.zip`, PPO fine-tuned from that checkpoint,
partially improved public Metal/Crustle but forgot easier matchups:

```text
public_metal_archaludon: 2/20
public_multiply_940: 0/20
public_mega_lucario_v62: 0/20
public_crustle_v1: 6/20
public_phantom_dragapult: 2/20
public_froslass_sleep: 3/20
public_kangaskhan_pressure: 0/20
heuristic_hydrapple: 4/20
heuristic_dragapult: 5/20
random_abomasnow: 15/20
```

This confirmed that BC regularization alone is not sufficient. The trainer now
supports masked in-training evaluation and `--best-save-path`, including an
initial step-0 evaluation, so fine-tuning can preserve the loaded BC policy when
PPO regresses.

## Diagnosis

The analyzer shows PPO still chooses heuristic rank 0 most of the time,
especially in losses. With `models/best/ppo_rich_broad_best.zip`:

```text
public_metal_archaludon losses:
avg_turn: 14.8
avg_prizes: (5.2, 0.7)
top_action_ranks: rank 0 dominates, then 1 and 3

public_mega_lucario_v62 losses:
avg_turn: 12.0
avg_prizes: (5.4, 1.2)
top_action_ranks: rank 0 dominates, then 1 and 3
```

That means losses are fast and decisive. The agent is not learning a distinct
strategy; it mostly follows the heuristic ordering and loses before collecting
useful corrective signal.

More raw PPO timesteps are unlikely to be enough by themselves. The current
highest-value changes are:

1. Better action representation: handle multi-select choices directly instead
   of reducing teacher choices to the first selected ranked option.
2. A policy architecture that scores each legal option with shared action
   features instead of flattening 256 ranked options into one MLP input.
3. Stronger BC before PPO: larger datasets, validation accuracy, and possibly
   sequence-level imitation for multi-select states.
4. Per-opponent curriculum with best-checkpoint selection, not final-checkpoint
   selection.
