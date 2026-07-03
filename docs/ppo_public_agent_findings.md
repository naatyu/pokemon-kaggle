# PPO Public-Agent Findings

## Current result

The PPO pipeline is faster and can beat weak/local baselines, but it does not
yet beat the strongest public Kaggle agents.

Most quick sweeps below used 20 deterministic games per opponent. The current
best sweep uses 40 deterministic games per opponent.

## Current Best PPO Checkpoint

Historical note: `models/best/ppo_rich_broad_best.zip` was the best checkpoint
before combo action choices. It should be treated as a historical result because
the action mapping changed for multi-select states.

`models/best/ppo_combo_broad_best.zip` is the best checkpoint trained with the
current combo-action environment. It starts from `models/ppo_combo_bc_public_metal_30k.zip`
and fine-tunes with:

- `--reward-shaping-scale 0`
- low learning rate: `1e-5`
- BC regularization from `public_metal_archaludon`
- mixed public/local opponent pool
- in-training masked evaluation and best-checkpoint saving

Mixed eval during training improved from `15/60` at step 0 to `21/60` at
65k timesteps. The per-opponent 40-game sweep:

```text
public_metal_archaludon: 3/40
public_multiply_940: 4/40
public_mega_lucario_v62: 1/40
public_crustle_v1: 4/40
public_phantom_dragapult: 4/40
public_froslass_sleep: 7/40
public_kangaskhan_pressure: 17/40
heuristic_hydrapple: 23/40
heuristic_dragapult: 29/40
random_abomasnow: 23/40
```

This checkpoint beats both local heuristic agents, which the previous broad
checkpoint did not. It also improves `public_multiply_940`, but it is still far
from beating the strongest public agents.

`models/best/ppo_combo_metal_focused_best.zip` was selected specifically
against `public_metal_archaludon`. Its in-training best was `4/40`, but an
external 40-game check produced `2/40`, so it is not a better general checkpoint.

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

## Combo Action Mapping

The original ranked action mapping represented a multi-select state by choosing
one option, then filling the rest greedily from the heuristic. This lost teacher
signal for decisions such as "choose 2" or "choose up to 3".

The environment now builds ranked action choices. For single-select states this
is still one option per action. For multi-select states, actions can represent
high-scoring option combinations. `scripts/analyze_action_mapping.py` measures
teacher coverage.

On 5k public-Metal teacher samples from the mixed opponent pool:

```text
old mapping exact full-list coverage: 0.904
combo mapping exact full-list coverage: 0.976
combo mapping set-equivalent coverage: 1.000
```

`models/ppo_combo_bc_public_metal_30k.zip`, trained after this change:

```text
public_metal_archaludon: 0/40
public_multiply_940: 2/40
public_mega_lucario_v62: 6/40
public_crustle_v1: 5/40
public_phantom_dragapult: 3/40
public_froslass_sleep: 5/40
public_kangaskhan_pressure: 24/40
heuristic_hydrapple: 15/40
heuristic_dragapult: 24/40
random_abomasnow: 21/40
```

The 20-game smoke for this same checkpoint showed `5/20` against
`public_metal_archaludon`, but the 40-game sweep dropped to `0/40`, so the
strong-public results remain noisy at small sample sizes.

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

1. A policy architecture that scores each legal option with shared action
   features instead of flattening 256 ranked options into one MLP input.
2. Stronger BC before PPO: larger datasets, validation accuracy, and possibly
   sequence-level imitation for multi-select states.
3. Per-opponent curriculum with best-checkpoint selection, not final-checkpoint
   selection.
4. A better reward for hard public matchups. Focused public-Metal PPO improved
   rollout reward from about `-0.29` toward `-0.04`, while held-out wins fell as
   low as `0/40`; reward and actual win probability are still misaligned.
