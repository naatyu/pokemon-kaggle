from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARENA_AGENTS = PROJECT_ROOT / "arena" / "agents"
SAMPLE_CG = PROJECT_ROOT / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission" / "cg"


AGENTS = {
    "hydrapple_heuristic": {
        "kind": "heuristic",
        "deck": "hydrapple",
        "deck_path": "decks/hydrapple_ex_heuristic.csv",
    },
    "dragapult_heuristic": {
        "kind": "heuristic",
        "deck": "dragapult",
        "deck_path": "decks/dragapult_ex_heuristic.csv",
    },
    "abomasnow_random": {
        "kind": "random",
        "deck": "abomasnow_sample",
        "deck_path": "decks/mega_abomasnow_sample.csv",
    },
    "hydrapple_random": {
        "kind": "random",
        "deck": "hydrapple",
        "deck_path": "decks/hydrapple_ex_heuristic.csv",
    },
}


def copytree_clean(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Remove generated local agents before rebuilding.")
    args = parser.parse_args()

    ARENA_AGENTS.mkdir(parents=True, exist_ok=True)
    for name, config in AGENTS.items():
        agent_dir = ARENA_AGENTS / name
        if args.clean and agent_dir.exists():
            shutil.rmtree(agent_dir)
        agent_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / config["deck_path"], agent_dir / "deck.csv")
        (agent_dir / "main.py").write_text(_main_py(config["kind"], config["deck"]), encoding="utf-8")
        copytree_clean(PROJECT_ROOT / "agents", agent_dir / "agents")
        copytree_clean(SAMPLE_CG, agent_dir / "cg")
        print(agent_dir)


def _main_py(kind: str, deck: str) -> str:
    if kind == "heuristic":
        return f'''from pathlib import Path
import os
import sys


AGENT_DIR = Path(globals().get("__file__", "/kaggle_simulations/agent/main.py")).resolve().parent
if not (AGENT_DIR / "deck.csv").exists():
    AGENT_DIR = Path.cwd()
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

os.environ["POKEMON_DECK"] = "{deck}"
os.environ["POKEMON_DECK_PATH"] = str(AGENT_DIR / "deck.csv")

from agents.heuristic_agent import agent  # noqa: E402,F401
'''
    if kind == "random":
        return '''from pathlib import Path
import random

from cg.api import to_observation_class


AGENT_DIR = Path(globals().get("__file__", "/kaggle_simulations/agent/main.py")).resolve().parent
if not (AGENT_DIR / "deck.csv").exists():
    AGENT_DIR = Path.cwd()


def _read_deck() -> list[int]:
    return [int(line.strip()) for line in (AGENT_DIR / "deck.csv").read_text().splitlines() if line.strip()]


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return _read_deck()
    count = obs.select.minCount if obs.select.minCount > 0 else min(1, obs.select.maxCount)
    count = min(count, len(obs.select.option))
    return random.sample(range(len(obs.select.option)), count)
'''
    raise ValueError(f"Unknown kind: {kind}")


if __name__ == "__main__":
    main()
