from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cg.api import to_observation_class  # noqa: E402
from rl.device import configure_torch_runtime, describe_torch_device, resolve_torch_device  # noqa: E402
from rl.ptcg_env import NOOP_ACTION, PTCGEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--deck", default="hydrapple")
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    args.device = resolve_torch_device(args.device)
    configure_torch_runtime(args.device)
    print(describe_torch_device(args.device), file=sys.stderr)

    model = MaskablePPO.load(args.model, device=args.device)
    totals = {"wins": 0, "losses": 0, "draws": 0, "truncated": 0}
    action_ranks = {"win": Counter(), "loss": Counter(), "draw": Counter()}
    option_types = {"win": Counter(), "loss": Counter(), "draw": Counter()}
    turn_counts = {"win": [], "loss": [], "draw": []}
    prize_counts = {"win": [], "loss": [], "draw": []}

    for game in range(args.games):
        env = PTCGEnv(deck=args.deck, opponent=args.opponent, seed=args.seed + game)
        obs, _ = env.reset(seed=args.seed + game)
        done = False
        ranks_this_game: list[int] = []
        types_this_game: list[str] = []
        truncated = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=args.deterministic)
            rank = int(action)
            ranks_this_game.append(rank)
            types_this_game.append(_chosen_option_type(env, rank))
            obs, _, terminated, truncated, _ = env.step(rank)
            done = terminated or truncated

        terminal_reward, _ = env._terminal_reward()
        final_obs = to_observation_class(env.obs_dict)
        if truncated:
            outcome = "draw"
            totals["truncated"] += 1
        elif terminal_reward > 0:
            outcome = "win"
            totals["wins"] += 1
        elif terminal_reward < 0:
            outcome = "loss"
            totals["losses"] += 1
        else:
            outcome = "draw"
            totals["draws"] += 1

        action_ranks[outcome].update(ranks_this_game)
        option_types[outcome].update(types_this_game)
        if final_obs.current is not None:
            turn_counts[outcome].append(final_obs.current.turn)
            prize_counts[outcome].append(
                (
                    len(final_obs.current.players[0].prize),
                    len(final_obs.current.players[1].prize),
                )
            )
        env.close()

    print(
        f"summary model={args.model} opponent={args.opponent} games={args.games} "
        f"wins={totals['wins']} losses={totals['losses']} draws={totals['draws']} "
        f"truncated={totals['truncated']} win_rate={totals['wins'] / args.games:.3f}"
    )
    for outcome in ["win", "loss", "draw"]:
        games = totals["wins" if outcome == "win" else "losses" if outcome == "loss" else "draws"]
        if games == 0:
            continue
        print(f"{outcome}: avg_turn={_avg(turn_counts[outcome]):.1f} avg_prizes={_avg_pair(prize_counts[outcome])}")
        print(f"{outcome}: top_action_ranks={action_ranks[outcome].most_common(8)}")
        print(f"{outcome}: top_option_types={option_types[outcome].most_common(8)}")


def _chosen_option_type(env: PTCGEnv, action: int) -> str:
    if action == NOOP_ACTION or env.obs_dict is None:
        return "NOOP"
    obs = to_observation_class(env.obs_dict)
    if obs.select is None:
        return "NO_SELECT"
    choices = env._ranked_action_choices(obs)
    if action < 0 or action >= len(choices) or not choices[action]:
        return "OUT_OF_RANGE"
    option_type = obs.select.option[choices[action][0]].type
    return getattr(option_type, "name", str(option_type))


def _avg(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _avg_pair(values: list[tuple[int, int]]) -> str:
    if not values:
        return "(0.0,0.0)"
    first = sum(value[0] for value in values) / len(values)
    second = sum(value[1] for value in values) / len(values)
    return f"({first:.1f},{second:.1f})"


if __name__ == "__main__":
    main()
