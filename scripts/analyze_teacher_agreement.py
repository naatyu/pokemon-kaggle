from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cg.api import to_observation_class  # noqa: E402
from rl.device import configure_torch_runtime, describe_torch_device, resolve_torch_device  # noqa: E402
from rl.opponents import make_opponent  # noqa: E402
from rl.ptcg_env import NOOP_ACTION, PTCGEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--deck", default="metal_archaludon")
    parser.add_argument("--teacher", default="public_metal_archaludon")
    parser.add_argument("--opponent", default="public_metal_archaludon")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--effect-features",
        action="store_true",
        help="Use optional global prompt/effect feature slots. Must match the model's training setting.",
    )
    args = parser.parse_args()
    args.device = resolve_torch_device(args.device)
    configure_torch_runtime(args.device)
    print(describe_torch_device(args.device), file=sys.stderr)

    model = MaskablePPO.load(args.model, device=args.device)
    env = PTCGEnv(
        deck=args.deck,
        opponent=args.opponent,
        reward_shaping_scale=0.0,
        effect_features=args.effect_features,
        seed=args.seed,
    )
    teacher = make_opponent(args.teacher)

    exact = 0
    near = 0
    total = 0
    rank_pairs: Counter[tuple[int, int]] = Counter()
    contexts: Counter[str] = Counter()

    obs, _ = env.reset(seed=args.seed)
    games = 0
    try:
        while total < args.samples:
            mask = env.action_masks()
            teacher_rank = teacher_action_to_rank(env, teacher)
            if teacher_rank is None or not mask[teacher_rank]:
                legal = np.flatnonzero(mask)
                action = int(legal[0])
            else:
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                predicted_rank = int(action)
                exact += int(predicted_rank == teacher_rank)
                near += int(abs(predicted_rank - teacher_rank) <= 1)
                rank_pairs[(teacher_rank, predicted_rank)] += 1
                contexts[select_context(env)] += 1
                action = teacher_rank
                total += 1

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                games += 1
                obs, _ = env.reset(seed=args.seed + games)
    finally:
        env.close()

    print(
        f"summary model={args.model} teacher={args.teacher} opponent={args.opponent} "
        f"samples={total} exact={exact / total:.3f} near_rank={near / total:.3f} games={games}"
    )
    print("top_rank_pairs teacher->model count")
    for (teacher_rank, predicted_rank), count in rank_pairs.most_common(15):
        print(f"{teacher_rank}->{predicted_rank} {count}")
    print("top_contexts count")
    for context, count in contexts.most_common(15):
        print(f"{context} {count}")


def teacher_action_to_rank(env: PTCGEnv, teacher) -> int | None:
    if env.obs_dict is None:
        return None
    mask = env.action_masks()
    try:
        selected = teacher.act(env.obs_dict)
    except Exception:
        return None
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


def select_context(env: PTCGEnv) -> str:
    if env.obs_dict is None:
        return "none"
    obs = to_observation_class(env.obs_dict)
    if obs.select is None:
        return "none"
    options = len(obs.select.option)
    return f"min={obs.select.minCount} max={obs.select.maxCount} options={options}"


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
