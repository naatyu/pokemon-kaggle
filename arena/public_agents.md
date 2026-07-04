# Public agent fetch notes

Last attempted: 2026-07-04

Kaggle submissions themselves are private, but public notebooks can be pulled
through the authenticated Kaggle CLI and converted into local arena opponents
when they expose enough submission source.

The CLI is available through `uv run kaggle` in this environment. Useful
commands:

```bash
uv run kaggle kernels list \
  --competition pokemon-tcg-ai-battle \
  --page-size 50 \
  --sort-by scoreDescending

uv run kaggle kernels pull romanrozen/strong-start-baseline-agent-v10-lb-950 \
  -p arena/public_sources/strong_start_v10 \
  -m
```

For notebooks with a `%%writefile main.py` cell:

```bash
uv run python scripts/extract_public_notebook_agent.py \
  --name public_strong_start_v10 \
  --notebook arena/public_sources/strong_start_v10/strong-start-baseline-agent-v10-lb-950.ipynb \
  --replace
```

Some Kiyota notebooks reference external deck datasets. Download those beside
the notebook before extraction:

```bash
uv run kaggle datasets download \
  -d kiyotah/mega-lucario-ex-deck \
  -p arena/public_sources/strong_start_v10/deck_dataset \
  --unzip
```

For a downloaded public submission archive:

```bash
uv run python scripts/import_arena_agent.py \
  --name <agent_name> \
  --archive path/to/submission.tar.gz
```

Always smoke-test after import:

```bash
uv run python scripts/run_arena.py \
  --agents <agent_name> hydrapple_heuristic \
  --games 5
```

See [opponent_pool.md](opponent_pool.md) for the current working public arena
agents and the PPO opponent mix.
