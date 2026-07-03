from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARENA_AGENTS = PROJECT_ROOT / "arena" / "agents"
SAMPLE_CG = PROJECT_ROOT / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission" / "cg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--archive", type=Path, help="Path to a public submission .tar.gz/.tgz archive.")
    parser.add_argument("--folder", type=Path, help="Path to a folder containing main.py and deck.csv.")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    if bool(args.archive) == bool(args.folder):
        raise SystemExit("Provide exactly one of --archive or --folder.")

    target = ARENA_AGENTS / args.name
    if target.exists():
        if not args.replace:
            raise SystemExit(f"{target} already exists. Use --replace to overwrite.")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if args.archive:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _extract_tar(args.archive, tmp_path)
            source = _find_agent_root(tmp_path)
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        source = _find_agent_root(args.folder)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    if not (target / "cg").exists():
        shutil.copytree(SAMPLE_CG, target / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    _validate_agent(target)
    print(target)


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as tar:
        destination_resolved = destination.resolve()
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if not str(member_path).startswith(str(destination_resolved)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(destination)


def _find_agent_root(path: Path) -> Path:
    if (path / "main.py").exists() and (path / "deck.csv").exists():
        return path
    candidates = [candidate for candidate in path.rglob("main.py") if (candidate.parent / "deck.csv").exists()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one folder with main.py and deck.csv under {path}, found {len(candidates)}.")
    return candidates[0].parent


def _validate_agent(path: Path) -> None:
    missing = [name for name in ["main.py", "deck.csv"] if not (path / name).exists()]
    if missing:
        raise RuntimeError(f"{path} missing required files: {missing}")


if __name__ == "__main__":
    main()
