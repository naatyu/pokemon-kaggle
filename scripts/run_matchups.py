from __future__ import annotations

import argparse
import json
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

from agents.deck_loader import load_deck  # noqa: E402
from agents.deck_profiles import PROFILES, DeckProfile  # noqa: E402
from agents.heuristic_agent import choose_action, fallback_selection, last_decision_dict  # noqa: E402


@dataclass(frozen=True)
class PlayerConfig:
    agent: str
    deck: str

    @property
    def profile(self) -> DeckProfile:
        return PROFILES[self.deck]


def random_action(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        raise ValueError("random_action received deck selection observation")
    count = obs.select.minCount if obs.select.minCount > 0 else min(1, obs.select.maxCount)
    count = min(count, len(obs.select.option))
    return random.sample(range(len(obs.select.option)), count)


def choose_for_player(obs_dict: dict, config: PlayerConfig) -> tuple[list[int], dict | None]:
    obs = to_observation_class(obs_dict)
    if config.agent == "random":
        return random_action(obs_dict), None
    if config.agent == "heuristic":
        action = choose_action(obs, config.profile)
        return action, last_decision_dict()
    raise ValueError(f"Unknown agent: {config.agent}")


def result_from_logs(obs_dict: dict) -> int | None:
    obs = to_observation_class(obs_dict)
    if obs.current and obs.current.result != -1:
        return obs.current.result
    for log in obs.logs:
        if log.type == LogType.RESULT:
            return log.result
    return None


def run_game(
    game_id: int,
    player0: PlayerConfig,
    player1: PlayerConfig,
    max_steps: int,
    seed: int,
) -> tuple[int | None, int, list[dict], str | None]:
    random.seed(seed)
    deck0 = load_deck(PROJECT_ROOT / player0.profile.deck_path)
    deck1 = load_deck(PROJECT_ROOT / player1.profile.deck_path)
    obs, start_data = battle_start(deck0, deck1)
    if obs is None:
        raise RuntimeError(f"Battle failed to start: errorPlayer={start_data.errorPlayer} errorType={start_data.errorType}")

    decisions: list[dict] = []
    error: str | None = None
    try:
        for step in range(max_steps):
            current = to_observation_class(obs).current
            if current is None:
                raise RuntimeError("battle_start returned an observation without state")
            if current.result != -1:
                return current.result, step, decisions, error

            acting_index = current.yourIndex
            config = player0 if acting_index == 0 else player1
            started = perf_counter()
            try:
                action, trace = choose_for_player(obs, config)
            except Exception as exc:
                action = fallback_selection(to_observation_class(obs))
                trace = {"fallback_error": f"{type(exc).__name__}: {exc}"}
            elapsed_ms = (perf_counter() - started) * 1000

            decisions.append(
                {
                    "kind": "decision",
                    "game_id": game_id,
                    "step": step,
                    "player_index": acting_index,
                    "agent": config.agent,
                    "deck": config.deck,
                    "action": action,
                    "elapsed_ms": elapsed_ms,
                    "trace": trace,
                }
            )

            try:
                obs = battle_select(action)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                winner = 1 - acting_index
                return winner, step + 1, decisions, error

            result = result_from_logs(obs)
            if result is not None:
                return result, step + 1, decisions, error
    finally:
        battle_finish()

    return None, max_steps, decisions, error


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--deck", choices=sorted(PROFILES), default="hydrapple")
    parser.add_argument("--opponent-agent", choices=["heuristic", "random"], default="random")
    parser.add_argument("--opponent-deck", choices=sorted(PROFILES), default="abomasnow_sample")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-jsonl", type=Path)
    args = parser.parse_args()

    player0 = PlayerConfig(args.agent, args.deck)
    player1 = PlayerConfig(args.opponent_agent, args.opponent_deck)
    results = []
    invalid_or_error = 0

    if args.log_jsonl and args.log_jsonl.exists():
        args.log_jsonl.unlink()

    for game_index in range(args.games):
        result, steps, decisions, error = run_game(
            game_id=game_index + 1,
            player0=player0,
            player1=player1,
            max_steps=args.max_steps,
            seed=args.seed + game_index,
        )
        results.append(result)
        if error:
            invalid_or_error += 1

        outcome_records = []
        for decision in decisions:
            player_index = decision["player_index"]
            decision["result"] = result
            decision["outcome_for_player"] = None if result is None or result == 2 else int(result == player_index)
            decision["terminal_error"] = error
            outcome_records.append(decision)
        outcome_records.append(
            {
                "kind": "game_summary",
                "game_id": game_index + 1,
                "result": result,
                "steps": steps,
                "player0": player0.__dict__,
                "player1": player1.__dict__,
                "terminal_error": error,
            }
        )
        if args.log_jsonl:
            write_jsonl(args.log_jsonl, outcome_records)

        print(f"game={game_index + 1} result={result} steps={steps} error={error}")

    wins = sum(1 for result in results if result == 0)
    losses = sum(1 for result in results if result == 1)
    draws = sum(1 for result in results if result == 2)
    unfinished = sum(1 for result in results if result is None)
    win_rate = wins / args.games if args.games else 0.0
    print(
        "summary "
        f"player0={args.agent}:{args.deck} "
        f"player1={args.opponent_agent}:{args.opponent_deck} "
        f"games={args.games} wins={wins} losses={losses} draws={draws} "
        f"unfinished={unfinished} errors={invalid_or_error} win_rate={win_rate:.3f}"
    )


if __name__ == "__main__":
    main()
