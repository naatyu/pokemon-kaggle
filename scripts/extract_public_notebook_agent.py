from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARENA_AGENTS = PROJECT_ROOT / "arena" / "agents"
SAMPLE_CG = PROJECT_ROOT / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission" / "cg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    target = ARENA_AGENTS / args.name
    if target.exists():
        if not args.replace:
            raise SystemExit(f"{target} already exists. Use --replace to overwrite.")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(SAMPLE_CG, target / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    cells = json.loads(args.notebook.read_text())["cells"]
    main_source = _extract_main_py(cells)
    (target / "main.py").write_text(main_source, encoding="utf-8")

    _extract_writefile_deck(cells, target)
    _copy_nearby_deck(args.notebook, target)
    _build_deck_csv(cells, target)
    if not (target / "deck.csv").exists():
        _deck_from_main(target)
    if not (target / "deck.csv").exists():
        raise RuntimeError(f"Could not produce deck.csv for {args.notebook}")

    _smoke_import(target)
    print(target)


def _extract_main_py(cells: list[dict]) -> str:
    for cell in cells:
        source = "".join(cell.get("source", ""))
        if source.lstrip().startswith("%%writefile main.py"):
            lines = source.splitlines()
            return "\n".join(lines[1:]).strip() + "\n"
    raise RuntimeError("No %%writefile main.py cell found.")


def _extract_writefile_deck(cells: list[dict], target: Path) -> None:
    for cell in cells:
        source = "".join(cell.get("source", ""))
        if source.lstrip().startswith("%%writefile deck.csv"):
            lines = source.splitlines()
            (target / "deck.csv").write_text("\n".join(lines[1:]).strip() + "\n", encoding="utf-8")
            return


def _copy_nearby_deck(notebook: Path, target: Path) -> None:
    if (target / "deck.csv").exists():
        return
    candidates = sorted(notebook.parent.rglob("deck.csv"))
    if len(candidates) == 1:
        shutil.copy2(candidates[0], target / "deck.csv")


def _build_deck_csv(cells: list[dict], target: Path) -> None:
    snippets = []
    for cell in cells:
        source = "".join(cell.get("source", ""))
        if source.lstrip().startswith("%%writefile"):
            continue
        if "deck.csv" in source and ("DECK" in source or "deck_ids" in source or "DECK_COUNTS" in source):
            if "tarfile" in source or "shutil.copytree" in source or "subprocess" in source:
                continue
            snippets.append(source)
    if not snippets:
        return
    script = "\n\n".join(snippets)
    subprocess.run([sys.executable, "-c", script], cwd=target, check=True, timeout=20)


def _deck_from_main(target: Path) -> None:
    code = """
import importlib.util
import os
import sys
from pathlib import Path
agent_dir = Path.cwd()
sys.path.insert(0, str(agent_dir))
spec = importlib.util.spec_from_file_location('public_agent_main', agent_dir / 'main.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
deck = module.agent({'select': None, 'logs': [], 'current': None})
(agent_dir / 'deck.csv').write_text('\\n'.join(map(str, deck)) + '\\n')
"""
    subprocess.run([sys.executable, "-c", code], cwd=target, check=True, timeout=20)


def _smoke_import(target: Path) -> None:
    code = """
import importlib.util
import os
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
