# Public agent fetch notes

Last attempted: 2026-07-03

Kaggle submissions themselves are private. Public notebooks can describe or
generate agents, but anonymous raw notebook downloads were not accessible from
this environment. The rendered Kaggle HTML did not include notebook source, and
`/download` URLs returned 404 without Kaggle API/session access.

Useful public notebook candidates to fetch manually from Kaggle:

- Beating the Day-1 #1 Crustle Bot:
  https://www.kaggle.com/code/dashimaki360/beating-the-day-1-1-crustle-bot
- Dragapult v3 Tempo PTCG AI Battle Agent:
  https://www.kaggle.com/code/zoli800/dragapult-v3-tempo-ptcg-ai-battle-agent
- A Sample Archaludon:
  https://www.kaggle.com/code/masamikobayashi/a-sample-archaludon-75-wr-vs-my-1300-starmie
- Pokemon TCG Lucario & Alakazam:
  https://www.kaggle.com/code/pilkwang/pokemon-tcg-lucario-alakazam
- PTCGAI Optimize Baseline:
  https://www.kaggle.com/code/suneetsaini/ptcgai-optimize-baseline
- PTCG Crustle V1 Submit:
  https://www.kaggle.com/code/pixiux/ptcg-crustle-v1-submit

Manual workflow:

1. Open the notebook on Kaggle.
2. Download its generated `submission.tar.gz` / `*.tgz`, or copy the generated
   `main.py` and `deck.csv`.
3. Extract/copy into `arena/agents/<name>/`.
4. Run:

```bash
python3 scripts/run_arena.py --agents <name> hydrapple_heuristic --games 10
```

You can import a downloaded public submission archive with:

```bash
python3 scripts/import_arena_agent.py --name <agent_name> --archive path/to/submission.tar.gz
```
