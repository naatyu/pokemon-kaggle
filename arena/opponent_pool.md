# Opponent pool

Last updated: 2026-07-03

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
| strong public agents: `public_metal_archaludon`, `public_multiply_940`, `public_mega_lucario_v62` | 0.40 |
| matchup specialists: `public_crustle_v1`, `public_froslass_sleep`, `public_phantom_dragapult`, `public_kangaskhan_pressure` | 0.25 |
| previous PPO checkpoints | 0.10 initially, increasing over time |

For early PPO, do not train only against the strongest agent. Keep the weak and
random opponents in the pool so the policy learns basic game completion,
setup, attacking, and recovery before specializing.
