# Battle arena

Drop submission-style agents into `arena/agents/<name>/`.

Each agent folder should contain at least:

```text
main.py
deck.csv
```

The arena runner loads `main.py`, calls `agent({"select": None, ...})` to get
the deck, then calls the same `agent(obs)` during games.

Generated local baseline agents are created by:

```bash
python3 scripts/build_arena_agents.py
```

Run a round-robin:

```bash
python3 scripts/run_arena.py --games 20
```

Public Kaggle agents are private unless shared through public notebooks or
downloaded manually. If you download a public notebook's built submission
archive, extract it under `arena/agents/<agent_name>/` and run the arena again.

Import a downloaded archive:

```bash
python3 scripts/import_arena_agent.py \
  --name public_crustle \
  --archive ~/Downloads/submission.tar.gz
```
