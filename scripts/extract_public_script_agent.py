from __future__ import annotations

import argparse
import base64
import gzip
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARENA_AGENTS = PROJECT_ROOT / "arena" / "agents"
SAMPLE_CG = PROJECT_ROOT / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission" / "cg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    target = ARENA_AGENTS / args.name
    if target.exists():
        if not args.replace:
            raise SystemExit(f"{target} already exists. Use --replace to overwrite.")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    text = args.script.read_text(encoding="utf-8")
    main_b64 = _extract_triple_string(text, "MAIN_GZ_B64")
    deck_text = _extract_triple_string(text, "DECK_TEXT")
    (target / "main.py").write_text(gzip.decompress(base64.b64decode(main_b64)).decode("utf-8"), encoding="utf-8")
    (target / "deck.csv").write_text(deck_text.strip() + "\n", encoding="utf-8")
    shutil.copytree(SAMPLE_CG, target / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _smoke_import(target)
    print(target)


def _extract_triple_string(text: str, name: str) -> str:
    match = re.search(rf'{name}\s*=\s*"""(.*?)"""', text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find {name}")
    return match.group(1)


def _smoke_import(target: Path) -> None:
    code = """
import importlib.util
import sys
from pathlib import Path
agent_dir = Path.cwd()
sys.path.insert(0, str(agent_dir))
spec = importlib.util.spec_from_file_location('public_agent_main', agent_dir / 'main.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
deck = module.agent({'select': None, 'logs': [], 'current': None})
assert isinstance(deck, list), type(deck)
assert len(deck) == 60, len(deck)
print('deck ok', len(deck))
"""
    subprocess.run([sys.executable, "-c", code], cwd=target, check=True, timeout=20)


if __name__ == "__main__":
    main()
