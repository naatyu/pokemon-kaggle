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

`models/best/ppo_action_embed_broad_best.zip` is the best aggregate checkpoint
so far. It uses the combo-action environment, opt-in extra state features, and
the embedded action-aware policy from `rl/action_policy.py`. It starts from
`models/ppo_action_embed_effect_bc_public_metal_30k.zip` and fine-tunes with:

- `--reward-shaping-scale 0`
- low learning rate: `1e-5`
- BC regularization from `public_metal_archaludon`
- `--policy action_embed`
- mixed public/local opponent pool
- in-training masked evaluation and best-checkpoint saving

The per-opponent 40-game sweep for `models/ppo_action_embed_broad_50k.zip`
copied to `models/best/ppo_action_embed_broad_best.zip`:

```text
public_metal_archaludon: 1/40
public_multiply_940: 7/40
public_mega_lucario_v62: 9/40
public_crustle_v1: 3/40
public_phantom_dragapult: 6/40
public_froslass_sleep: 23/40
public_kangaskhan_pressure: 32/40
heuristic_hydrapple: 32/40
heuristic_dragapult: 33/40
random_abomasnow: 36/40
```

This improves aggregate public-opponent wins from `64/280` for the previous
best action-aware PPO to `81/280`, and local/random wins from `99/120` to
`101/120`. It is still far from beating `public_metal_archaludon`.

Historical note: `models/best/ppo_action_broad_best.zip` was the previous best
aggregate checkpoint before card/attack embeddings:

```text
public_metal_archaludon: 2/40
public_multiply_940: 4/40
public_mega_lucario_v62: 9/40
public_crustle_v1: 4/40
public_phantom_dragapult: 7/40
public_froslass_sleep: 17/40
public_kangaskhan_pressure: 21/40
heuristic_hydrapple: 29/40
heuristic_dragapult: 36/40
random_abomasnow: 34/40
```

`models/best/ppo_combo_metal_focused_best.zip` was selected specifically
against `public_metal_archaludon`. Its in-training best was `4/40`, but an
external 40-game check produced `2/40`, so it is not a better general checkpoint.

`models/best/ppo_action_embed_metal_tempo_best.zip` tested public-Metal-only
fine-tuning from `models/ppo_action_embed_broad_50k.zip` with
`--tempo-reward-scale 0.05`, `5e-6` learning rate, and stronger BC
regularization. In-training public-Metal eval peaked at `3/40`. An external
80-game check produced:

```text
public_metal_archaludon: 6/80
```

The final checkpoint regressed to `3/80`, so best-checkpoint selection matters.
This is slightly better for public Metal, but it is not a new aggregate best:
20-game checks showed `public_multiply_940` falling to `1/20`, while easier
matchups such as `public_kangaskhan_pressure`, local heuristics, and random
remained strong.

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

Runtime note: collection is CPU-bound because it runs the simulator and public
teacher agent. The training epochs run on `--device`, and
`--collection-workers` parallelizes the collection phase. Large runs are
memory-sensitive: 60k samples require roughly 1 GB for the final arrays before
model/framework overhead, worker processes, and transfer copies. On a 32 GB
machine, prefer 2-4 workers unless memory is being watched.

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

## Action-Aware Policy

The default `MlpPolicy` receives the whole flattened observation and emits one
logit per action slot. This is a poor inductive bias because action slot 0 and
action slot 50 use unrelated output weights even though they are both legal
action choices described by the same per-action feature schema.

`ActionMaskablePolicy` scores each action choice with shared weights:

- encode global board features once
- encode each action slot's feature vector with the same option encoder
- score each action from `[global_embedding, option_embedding]`
- score the no-op action from the global embedding
- use a separate global critic

`models/ppo_action_bc_public_metal_30k.zip` trained with this policy reached
BC top-1 accuracy `0.545`, compared with `0.504` for the combo MLP BC run.

Its 40-game sweep before PPO:

```text
public_metal_archaludon: 0/40
public_multiply_940: 4/40
public_mega_lucario_v62: 6/40
public_crustle_v1: 2/40
public_phantom_dragapult: 5/40
public_froslass_sleep: 15/40
public_kangaskhan_pressure: 24/40
heuristic_hydrapple: 30/40
heuristic_dragapult: 32/40
random_abomasnow: 34/40
```

PPO fine-tuning from this checkpoint produced the current best mixed result
(`27/60`) and improved some public matchups further, but still did not solve
public Metal.

A larger BC run, `models/ppo_action_bc_public_metal_100k.zip`, used 100k
teacher samples and 10 epochs. It reached BC top-1 accuracy `0.609`, but did
not become the best general checkpoint:

```text
public_metal_archaludon: 2/40
public_multiply_940: 2/40
public_mega_lucario_v62: 5/40
public_crustle_v1: 10/40
public_phantom_dragapult: 5/40
public_froslass_sleep: 15/40
public_kangaskhan_pressure: 18/40
heuristic_hydrapple: 20/40
heuristic_dragapult: 29/40
random_abomasnow: 29/40
```

This stronger clone improved public Metal relative to 30k BC and improved
Crustle, but it lost generality on the local heuristics and Kangaskhan.
PPO fine-tuning from this 100k BC checkpoint started at `25/60` mixed eval,
then fell to `17/60` and `18/60` at the next checkpoints, so that run was
stopped early. The current best remains `models/best/ppo_action_broad_best.zip`.

`models/ppo_action512_bc_public_metal_60k.zip` tested a wider action-aware
policy with hidden size 512, 60k teacher samples, 10 epochs, CUDA training, and
8 parallel collection workers. It reached BC top-1 accuracy `0.576`. A quick
20-game sweep did not improve the current best:

```text
public_metal_archaludon: 1/20
public_multiply_940: 1/20
public_mega_lucario_v62: 1/20
public_crustle_v1: 0/20
heuristic_hydrapple: 11/20
```

Teacher agreement against public Metal over 1k states was exact `0.576` and
near-rank `0.795`, so width alone is not the limiting factor.

## Embedded Action Policy

Card IDs and attack IDs were originally presented to the policy as continuous
numbers, for example `190 / 1300`. That is a poor representation for
card-specific rules. `EmbeddedActionMaskablePolicy` keeps the same observation
shape but recovers card/attack IDs from the normalized features and embeds them
categorically.

The first 30k BC run with embeddings,
`models/ppo_action_embed_effect_bc_public_metal_30k.zip`, reached BC top-1
accuracy `0.593`, compared with `0.545` for the original 30k action-aware BC
run. Teacher agreement over 1k public-Metal states was exact `0.619`.

The raw embedded BC checkpoint was still not a strong player, but PPO
fine-tuning from it produced the current best aggregate checkpoint:
`models/ppo_action_embed_broad_50k.zip`.

PPO reduced teacher agreement slightly, from `0.619` to `0.605` on the same
1k public-Metal diagnostic, but improved aggregate wins. This suggests the
embedded policy is a better starting representation, while PPO is still
finding only shallow improvements against the strongest public agents.

## Diagnosis

`scripts/analyze_teacher_agreement.py` measures whether a checkpoint picks the
same ranked action as a public teacher while rolling forward with the teacher's
own actions.

On 3k public-Metal teacher states against `public_metal_archaludon`:

```text
models/best/ppo_action_broad_best.zip: exact 0.540, near-rank 0.835
models/ppo_action_bc_public_metal_100k.zip: exact 0.621, near-rank 0.871
models/ppo_action_bc_public_metal_mirror_60k.zip: exact 0.597, near-rank 0.874
```

So even the stronger BC models still disagree with the teacher on roughly
38-46% of tactical decisions. Most errors are close rank disagreements, but in
Pokemon that can still mean missing a setup card, attachment, evolution, or
attack timing window.

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

With `models/ppo_action_embed_broad_50k.zip`, public-Metal losses are still
fast:

```text
public_metal_archaludon losses:
avg_turn: 11.8
avg_prizes: (4.6, 2.2)

public_mega_lucario_v62 losses:
avg_turn: 10.1
avg_prizes: (5.1, 1.8)
```

That means losses are fast and decisive. The agent is not learning a distinct
strategy; it mostly follows the heuristic ordering and loses before collecting
useful corrective signal.

More raw PPO timesteps are unlikely to be enough by themselves. The current
highest-value changes are:

1. Stronger embedded BC before PPO: larger datasets, validation accuracy, and
   possibly sequence-level imitation for multi-select states.
2. Per-opponent curriculum with best-checkpoint selection, not final-checkpoint
   selection.
3. Public-Metal-specific training with a better reward. Focused public-Metal PPO improved
   rollout reward from about `-0.29` toward roughly `-0.07`, but held-out wins
   still stayed around `0-3/40`; reward and actual win probability are still
   misaligned.
4. A public-Metal-specific teacher or value target. The current public-Metal
   teacher helps general play, but imitation still does not reproduce the
   teacher's own public-Metal matchup strength.
