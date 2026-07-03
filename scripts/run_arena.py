from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SUBMISSION = PROJECT_ROOT / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SAMPLE_SUBMISSION))

from cg.api import LogType, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


ARENA_AGENTS = PROJECT_ROOT / "arena" / "agents"
ARENA_RESULTS = PROJECT_ROOT / "arena" / "results"


@dataclass
class FolderAgent:
    name: str
    path: Path
    module: object

    def deck(self) -> list[int]:
        return self._call({"select": None, "logs": [], "current": None})

    def act(self, obs: dict) -> list[int]:
        return self._call(obs)

    def _call(self, obs: dict) -> list[int]:
        old_cwd = Path.cwd()
        try:
            os.chdir(self.path)
            return self.module.agent(obs)
        finally:
            os.chdir(old_cwd)


def load_folder_agent(name: str) -> FolderAgent:
    path = ARENA_AGENTS / name
    main_py = path / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Missing agent main.py: {main_py}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    module_name = f"arena_agent_{name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, main_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {main_py}")
    module = importlib.util.module_from_spec(spec)
    old_cwd = Path.cwd()
    try:
        os.chdir(path)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return FolderAgent(name=name, path=path, module=module)


def result_from_logs(obs_dict: dict) -> int | None:
    obs = to_observation_class(obs_dict)
    if obs.current and obs.current.result != -1:
        return obs.current.result
    for log in obs.logs:
        if log.type == LogType.RESULT:
            return log.result
    return None


def run_game(player0: FolderAgent, player1: FolderAgent, game_id: int, max_steps: int, seed: int) -> dict:
    random.seed(seed)
    obs, start_data = battle_start(player0.deck(), player1.deck())
    if obs is None:
        return {
            "kind": "game_summary",
            "game_id": game_id,
            "player0": player0.name,
            "player1": player1.name,
            "result": 1,
            "steps": 0,
            "error": f"battle_start failed: player={start_data.errorPlayer} type={start_data.errorType}",
        }

    error = None
    try:
        for step in range(max_steps):
            current = to_observation_class(obs).current
            if current is None:
                error = "observation missing current state"
                return _summary(game_id, player0, player1, 2, step, error)
            if current.result != -1:
                return _summary(game_id, player0, player1, current.result, step, error)

            actor = player0 if current.yourIndex == 0 else player1
            started = perf_counter()
            try:
                action = actor.act(obs)
            except Exception as exc:
                error = f"{actor.name} action failed: {type(exc).__name__}: {exc}"
                return _summary(game_id, player0, player1, 1 - current.yourIndex, step, error)
            elapsed_ms = (perf_counter() - started) * 1000
            if elapsed_ms > 900:
                error = f"{actor.name} slow action: {elapsed_ms:.1f}ms"

            try:
                obs = battle_select(action)
            except Exception as exc:
                error = f"{actor.name} invalid action {action}: {type(exc).__name__}: {exc}"
                return _summary(game_id, player0, player1, 1 - current.yourIndex, step + 1, error)

            result = result_from_logs(obs)
            if result is not None:
                return _summary(game_id, player0, player1, result, step + 1, error)
    finally:
        battle_finish()

    return _summary(game_id, player0, player1, None, max_steps, error)


def _summary(game_id: int, player0: FolderAgent, player1: FolderAgent, result: int | None, steps: int, error: str | None) -> dict:
    return {
        "kind": "game_summary",
        "game_id": game_id,
        "player0": player0.name,
        "player1": player1.name,
        "result": result,
        "steps": steps,
        "error": error,
    }


def available_agents() -> list[str]:
    if not ARENA_AGENTS.exists():
        return []
    return sorted(path.name for path in ARENA_AGENTS.iterdir() if (path / "main.py").exists())


def write_outputs(records: list[dict], matrix: dict[tuple[str, str], float], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_prefix.with_suffix(".jsonl")
    csv_path = output_prefix.with_suffix(".csv")
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")
    names = sorted({name for pair in matrix for name in pair})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["agent"] + names)
        for row_agent in names:
            row = [row_agent]
            for col_agent in names:
                value = matrix.get((row_agent, col_agent))
                row.append("" if value is None else f"{value:.3f}")
            writer.writerow(row)
    print(jsonl_path)
    print(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="*", default=None)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-prefix", type=Path, default=ARENA_RESULTS / "arena_latest")
    args = parser.parse_args()

    names = args.agents or available_agents()
    if len(names) < 2:
        raise SystemExit("Need at least two arena agents. Run scripts/build_arena_agents.py first.")

    agents = {name: load_folder_agent(name) for name in names}
    records: list[dict] = []
    matrix: dict[tuple[str, str], float] = {}
    game_id = 0
    for player0_name in names:
        for player1_name in names:
            if player0_name == player1_name:
                continue
            wins = 0
            completed = 0
            for game_index in range(args.games):
                game_id += 1
                record = run_game(
                    agents[player0_name],
                    agents[player1_name],
                    game_id=game_id,
                    max_steps=args.max_steps,
                    seed=args.seed + game_id * 1009 + game_index,
                )
                records.append(record)
                if record["result"] is not None:
                    completed += 1
                    wins += int(record["result"] == 0)
                print(
                    f"{player0_name} vs {player1_name} "
                    f"game={game_index + 1}/{args.games} result={record['result']} "
                    f"steps={record['steps']} error={record['error']}"
                )
            matrix[(player0_name, player1_name)] = wins / completed if completed else 0.0

    write_outputs(records, matrix, args.output_prefix)


if __name__ == "__main__":
    main()
