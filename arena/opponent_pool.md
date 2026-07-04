# Opponent pool

Last updated: 2026-07-04

These public agents were pulled through the authenticated Kaggle CLI, extracted
into `arena/agents/`, and smoke-tested locally with the bundled simulator.

## Imported public agents

| Arena name | Source | Notes |
| --- | --- | --- |
| `public_metal_archaludon` | `plamen06/pokemon-metall` | Strongest in the first 10-game mini league. Metal/Archaludon tempo. |
| `public_multiply_940` | `aristophanivan/multiply-agent-best-940-lb` | Lucario/MultiPly beam-search style agent. |
| `public_mega_lucario_v62` | `pixiux/ptcg-mega-lucario-ex-v62` | Mega Lucario rule agent. |
| `public_crustle_v1` | `pixiux/ptcg-crustle-v1-submit` | Crustle public submit script. |
| `public_phantom_dragapult` | `skarin/phantom-dive-or-go-home-a-dragapult-ex-deck` | Dragapult tempo agent. |
| `public_froslass_sleep` | `naoto714/en-mega-froslass-ex-is-sleep-worth-it` | Froslass sleep/control-flavored agent. |
| `public_kangaskhan_pressure` | `naoto714/en-team-rocket-s-kangaskhan-ex-220-pressure` | Kangaskhan pressure deck; useful diversity, weaker in smoke test. |
| `public_alakazam_best5` | `ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th` | Alakazam public notebook. Imported cleanly; useful archetype diversity. |
| `public_kiyota_mega_lucario` | `kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck` | Kiyota sample rule agent; requires the companion deck dataset. |
| `public_kiyota_dragapult` | `kiyotah/a-sample-rule-based-agent-dragapult-ex-deck` | Kiyota sample Dragapult rule agent; requires the companion deck dataset. |
| `public_kiyota_iono` | `kiyotah/a-sample-rule-based-agent-iono-s-deck` | Kiyota sample Iono rule agent; requires the companion deck dataset. |
| `public_strong_start_v10` | `romanrozen/strong-start-baseline-agent-v10-lb-950` | Lucario strong-start baseline; imports after downloading the Kiyota Lucario deck dataset. |
| `public_archaludon_75wr` | `masamikobayashi/a-sample-archaludon-75-wr-vs-my-1300-starmie` | Archaludon public notebook; direct `main.py` and `deck.csv` extraction. |
| `public_baseline_1084` | `makthanithin/pokemon-tcg-ai-battle-1084-5-baseline` | Public 1084.5 baseline notebook; direct extraction. |

These candidates were pulled but not imported as arena agents:

| Source | Reason |
| --- | --- |
| `masamikobayashi/prize-card-tracking-1300-starmie` | No standard `%%writefile main.py` cell; needs a custom extractor. |
| `map1e114514/starmie-cinderace-budew-skill-agent` | Notebook did not expose a runnable submission agent in the pulled source. |

## Mini league result

Command:

```bash
python3 scripts/run_arena.py \
  --agents public_metal_archaludon public_mega_lucario_v62 public_multiply_940 public_crustle_v1 public_froslass_sleep public_phantom_dragapult hydrapple_heuristic \
  --games 10 \
  --max-steps 2500 \
  --output-prefix arena/results/public_top7_10g
```

Average row win rate:

| Agent | Avg win rate |
| --- | ---: |
| `public_metal_archaludon` | 0.750 |
| `public_multiply_940` | 0.667 |
| `public_mega_lucario_v62` | 0.633 |
| `public_crustle_v1` | 0.633 |
| `public_phantom_dragapult` | 0.550 |
| `public_froslass_sleep` | 0.350 |
| `hydrapple_heuristic` | 0.000 |

Full matrix:

```text
agent,hydrapple_heuristic,public_crustle_v1,public_froslass_sleep,public_mega_lucario_v62,public_metal_archaludon,public_multiply_940,public_phantom_dragapult
hydrapple_heuristic,,0.000,0.000,0.000,0.000,0.000,0.000
public_crustle_v1,1.000,,1.000,0.100,0.100,0.600,1.000
public_froslass_sleep,1.000,0.000,,0.500,0.100,0.400,0.100
public_mega_lucario_v62,1.000,0.700,0.500,,0.600,0.500,0.500
public_metal_archaludon,1.000,0.900,1.000,0.600,,0.600,0.400
public_multiply_940,1.000,0.800,0.500,0.500,0.300,,0.900
public_phantom_dragapult,1.000,0.000,0.700,0.500,0.600,0.500,
```

## Suggested PPO opponent mix

Start with a broad curriculum:

| Pool | Weight |
| --- | ---: |
| random agents: `abomasnow_random`, `hydrapple_random` | 0.10 |
| weak/local heuristic: `hydrapple_heuristic`, `dragapult_heuristic` | 0.15 |
| strong public agents: `public_metal_archaludon`, `public_multiply_940`, `public_mega_lucario_v62`, `public_strong_start_v10`, `public_baseline_1084` | 0.40 |
| matchup specialists and deck diversity: `public_crustle_v1`, `public_froslass_sleep`, `public_phantom_dragapult`, `public_kangaskhan_pressure`, `public_alakazam_best5`, `public_archaludon_75wr`, `public_kiyota_mega_lucario`, `public_kiyota_dragapult`, `public_kiyota_iono` | 0.25 |
| previous PPO checkpoints | 0.10 initially, increasing over time |

For early PPO, do not train only against the strongest agent. Keep the weak and
random opponents in the pool so the policy learns basic game completion,
setup, attacking, and recovery before specializing.

For behavioral cloning data, use the strong public agents as teachers and the
full pool as opponents. Save the collected dataset once and reuse it for
architecture or hyperparameter experiments:

```bash
POOL='public_metal_archaludon,public_multiply_940,public_mega_lucario_v62,public_strong_start_v10,public_baseline_1084,public_archaludon_75wr,public_alakazam_best5,public_crustle_v1,public_phantom_dragapult,public_froslass_sleep,public_kangaskhan_pressure,public_kiyota_mega_lucario,public_kiyota_dragapult,public_kiyota_iono,heuristic_hydrapple,heuristic_dragapult,random_abomasnow'
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
  --collection-workers 2 \
  --policy action_embed \
  --policy-hidden-dim 256 \
  --card-embedding-dim 32 \
  --attack-embedding-dim 16 \
  --effect-features \
  --device cuda \
  --save-path models/ppo_action_embed_effect_bc_public_metal_broad_30k
```
