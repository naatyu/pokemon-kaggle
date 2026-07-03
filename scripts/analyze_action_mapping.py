from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cg.api import to_observation_class  # noqa: E402
from rl.opponents import make_opponent  # noqa: E402
from rl.ptcg_env import NOOP_ACTION, PTCGEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", default="metal_archaludon")
    parser.add_argument("--opponent", default="public_metal_archaludon")
    parser.add_argument("--teacher", default="public_metal_archaludon")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()

    env = PTCGEnv(deck=args.deck, opponent=args.opponent, seed=args.seed)
    teacher = make_opponent(args.teacher)
    obs, _ = env.reset(seed=args.seed)
    games = 0
    represented = exact = set_exact = 0
    lengths: Counter[int] = Counter()
    cases: Counter[tuple[int, int, int, int, bool, bool]] = Counter()

    while represented < args.samples:
        obs_class = to_observation_class(env.obs_dict)
        mask = env.action_masks()
        selected = _teacher_selection(teacher, env)
        action = _teacher_action_to_rank(env, selected)
        if action is None or not mask[action]:
            legal = np.flatnonzero(mask)
            action = int(legal[0])
        else:
            mapped = env._action_to_selection(action)
            represented += 1
            exact += int(mapped == selected)
            set_exact += int(set(mapped) == set(selected))
            lengths[len(selected)] += 1
            if obs_class.select is not None:
                cases[
                    (
                        obs_class.select.minCount,
                        obs_class.select.maxCount,
                        len(selected),
                        len(mapped),
                        mapped == selected,
                        set(mapped) == set(selected),
                    )
                ] += 1

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            games += 1
            obs, _ = env.reset(seed=args.seed + games)

    env.close()
    print(f"samples={represented} games={games}")
    print(f"exact={exact / represented:.3f} set_exact={set_exact / represented:.3f}")
    print(f"selected_lengths={lengths.most_common()}")
    print("top_cases=min,max,teacher_len,mapped_len,exact,set_exact,count")
    for case, count in cases.most_common(20):
        print(",".join(map(str, case + (count,))))


def _teacher_selection(teacher, env: PTCGEnv) -> list[int]:
    try:
        return list(teacher.act(env.obs_dict) or [])
    except Exception:
        return []


def _teacher_action_to_rank(env: PTCGEnv, selected: list[int]) -> int | None:
    mask = env.action_masks()
    if not selected:
        return NOOP_ACTION if mask[NOOP_ACTION] else None
    choices = env._ranked_action_choices(to_observation_class(env.obs_dict))
    selected_tuple = tuple(selected)
    try:
        return choices.index(selected_tuple)
    except ValueError:
        selected_set = set(selected)
        for index, choice in enumerate(choices):
            if set(choice) == selected_set:
                return index
        for index, choice in enumerate(choices):
            if choice and choice[0] == selected[0]:
                return index
    return None


if __name__ == "__main__":
    main()
