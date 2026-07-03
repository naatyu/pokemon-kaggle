from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.deck_profiles import PROFILES  # noqa: E402

BUILD_DIR = PROJECT_ROOT / "build" / "kaggle_agent"
ARCHIVE_PATH = PROJECT_ROOT / "build" / "kaggle_agent.tar.gz"
SAMPLE_CG = PROJECT_ROOT / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission" / "cg"


def copytree_clean(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", choices=sorted(PROFILES), default="hydrapple")
    args = parser.parse_args()

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    BUILD_DIR.mkdir(parents=True)

    (BUILD_DIR / "main.py").write_text(_main_py(), encoding="utf-8")
    shutil.copy2(PROJECT_ROOT / PROFILES[args.deck].deck_path, BUILD_DIR / "deck.csv")
    copytree_clean(PROJECT_ROOT / "agents", BUILD_DIR / "ptcg_agent")
    copytree_clean(SAMPLE_CG, BUILD_DIR / "cg")

    shutil.make_archive(str(ARCHIVE_PATH.with_suffix("").with_suffix("")), "gztar", root_dir=BUILD_DIR)

    print(ARCHIVE_PATH)


def _main_py() -> str:
    return '''from pathlib import Path
import os
import sys


AGENT_DIR = Path(globals().get("__file__", "/kaggle_simulations/agent/main.py")).resolve().parent
if not (AGENT_DIR / "deck.csv").exists():
    AGENT_DIR = Path.cwd()
if str(AGENT_DIR) in sys.path:
    sys.path.remove(str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR))

os.environ.setdefault("POKEMON_DECK_PATH", str(AGENT_DIR / "deck.csv"))

from ptcg_agent.heuristic_agent import agent  # noqa: E402,F401
'''


if __name__ == "__main__":
    main()
