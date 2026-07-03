# PPO Public-Agent Findings

## Current result

The PPO pipeline is faster and can beat weak/local baselines, but it does not
yet beat the strongest public Kaggle agents.

All quick sweeps below used 20 deterministic games per opponent.

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

## Diagnosis

The analyzer shows PPO still chooses heuristic rank 0 most of the time,
especially in losses. Example against `public_metal_archaludon`:

```text
loss top_action_ranks: rank 0 dominates by a wide margin
loss average turn: about 10
loss average prizes: our prizes about 6, opponent prizes about 1
```

That means losses are fast and decisive. The agent is not learning a distinct
strategy; it mostly follows the heuristic ordering and loses before collecting
useful corrective signal.

More raw PPO timesteps are unlikely to be enough by themselves. The next
highest-value changes are:

1. Richer observations: include bench Pokemon, bench HP, attached energy,
   tools, prize counts, and target metadata in fixed slots.
2. Better action representation: handle multi-select choices directly instead
   of reducing teacher choices to the first selected ranked option.
3. BC plus PPO regularization: keep a KL/BC loss during PPO fine-tuning so
   public-agent imitation is not immediately forgotten.
4. Per-opponent curriculum/weighting: train heavily on one target at a time
   once the agent can represent the necessary choices.
