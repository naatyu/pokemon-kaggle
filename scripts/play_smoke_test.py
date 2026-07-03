from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SUBMISSION = PROJECT_ROOT / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SAMPLE_SUBMISSION))

from cg.api import LogType, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from agents.deck_loader import load_deck  # noqa: E402
from agents.deck_profiles import PROFILES  # noqa: E402
from agents.heuristic_agent import choose_action, fallback_selection  # noqa: E402


def random_action(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        raise ValueError("random_action received deck selection observation")
    count = obs.select.minCount if obs.select.minCount > 0 else min(1, obs.select.maxCount)
    count = min(count, len(obs.select.option))
    return random.sample(range(len(obs.select.option)), count)


def heuristic_action(obs_dict: dict, deck_name: str) -> list[int]:
    obs = to_observation_class(obs_dict)
    return choose_action(obs, PROFILES[deck_name])


def result_from_logs(obs_dict: dict) -> int | None:
    obs = to_observation_class(obs_dict)
    if obs.current and obs.current.result != -1:
        return obs.current.result
    for log in obs.logs:
        if log.type == LogType.RESULT:
            return log.result
    return None


def play_one(deck_name: str, max_steps: int, seed: int) -> tuple[int | None, int]:
    random.seed(seed)
    profile = PROFILES[deck_name]
    heuristic_deck = load_deck(PROJECT_ROOT / profile.deck_path)
    random_deck = load_deck(PROJECT_ROOT / "decks" / "hydrapple_ex_heuristic.csv")

    obs, start_data = battle_start(heuristic_deck, random_deck)
    if obs is None:
        raise RuntimeError(f"Battle failed to start: errorPlayer={start_data.errorPlayer} errorType={start_data.errorType}")

    try:
        for step in range(max_steps):
            current = to_observation_class(obs).current
            if current is None:
                raise RuntimeError("battle_start returned an observation without state")
            if current.result != -1:
                return current.result, step

            if current.yourIndex == 0:
                action = heuristic_action(obs, deck_name)
            else:
                try:
                    action = random_action(obs)
                except Exception:
                    action = fallback_selection(to_observation_class(obs))
            obs = battle_select(action)
            result = result_from_logs(obs)
            if result is not None:
                return result, step + 1
    finally:
        battle_finish()

    return None, max_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", choices=sorted(PROFILES), default="hydrapple")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    results = []
    for game_index in range(args.games):
        result, steps = play_one(args.deck, args.max_steps, args.seed + game_index)
        results.append(result)
        print(f"game={game_index + 1} result={result} steps={steps}")

    wins = sum(1 for result in results if result == 0)
    losses = sum(1 for result in results if result == 1)
    draws = sum(1 for result in results if result == 2)
    unfinished = sum(1 for result in results if result is None)
    print(f"summary wins={wins} losses={losses} draws={draws} unfinished={unfinished}")


if __name__ == "__main__":
    main()
