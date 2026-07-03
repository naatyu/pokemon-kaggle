from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rl.opponents import make_opponent  # noqa: E402
from rl.device import configure_torch_runtime, describe_torch_device, resolve_torch_device  # noqa: E402
from rl.ptcg_env import NOOP_ACTION, PTCGEnv  # noqa: E402
from rl.action_policy import ActionMaskablePolicy  # noqa: E402
from scripts.train_ppo import parse_net_arch  # noqa: E402
from cg.api import to_observation_class  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", default="metal_archaludon")
    parser.add_argument("--teacher", default="public_metal_archaludon")
    parser.add_argument("--opponent", default="public_metal_archaludon,public_multiply_940,public_mega_lucario_v62")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--collection-workers",
        type=int,
        default=1,
        help="Parallel CPU workers for teacher trajectory collection. Neural training still uses --device.",
    )
    parser.add_argument("--save-path", type=Path, default=PROJECT_ROOT / "models" / "ppo_bc_public_metal")
    parser.add_argument("--load-path", type=Path)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--policy", choices=["mlp", "action"], default="mlp")
    parser.add_argument("--policy-hidden-dim", type=int, default=256)
    parser.add_argument("--net-arch", type=parse_net_arch, default=[256, 256])
    args = parser.parse_args()
    args.device = resolve_torch_device(args.device)
    configure_torch_runtime(args.device)
    print(describe_torch_device(args.device), file=sys.stderr)

    env = PTCGEnv(deck=args.deck, opponent=args.opponent, seed=args.seed)
    teacher = make_opponent(args.teacher)
    model = _build_model(args, env)
    env.close()
    observations, actions, masks = collect_teacher_data(
        args.deck,
        args.opponent,
        args.teacher,
        args.samples,
        args.seed,
        args.collection_workers,
    )
    train_bc(model, observations, actions, masks, args.epochs, args.batch_size)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.save_path)
    env.close()
    print(args.save_path)


def _build_model(args: argparse.Namespace, env: PTCGEnv) -> MaskablePPO:
    policy = ActionMaskablePolicy if args.policy == "action" else "MlpPolicy"
    policy_kwargs = (
        {"hidden_dim": args.policy_hidden_dim}
        if args.policy == "action"
        else {"net_arch": args.net_arch}
    )
    model = MaskablePPO(
        policy,
        env,
        verbose=1,
        seed=args.seed,
        n_steps=512,
        batch_size=args.batch_size,
        gamma=0.99,
        learning_rate=args.learning_rate,
        ent_coef=0.02,
        policy_kwargs=policy_kwargs,
        device=args.device,
    )
    if args.load_path:
        loaded = MaskablePPO.load(args.load_path, device=args.device)
        model.set_parameters({"policy": loaded.get_parameters()["policy"]}, exact_match=False, device=args.device)
    return model


def collect_teacher_data(
    deck: str,
    opponent: str,
    teacher_name: str,
    samples: int,
    seed: int,
    workers: int,
):
    if workers <= 1:
        return _collect_teacher_data_worker(deck, opponent, teacher_name, samples, seed, "main")

    counts = _split_counts(samples, workers)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _collect_teacher_data_worker,
                deck,
                opponent,
                teacher_name,
                count,
                seed + worker_index * 100_000,
                f"worker={worker_index}",
            )
            for worker_index, count in enumerate(counts)
            if count > 0
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"collection_worker_done samples={len(result[1])}")

    observations, actions, masks = zip(*results)
    return (
        np.concatenate(observations).astype(np.float32, copy=False),
        np.concatenate(actions).astype(np.int64, copy=False),
        np.concatenate(masks).astype(bool, copy=False),
    )


def _split_counts(total: int, parts: int) -> list[int]:
    base = total // parts
    remainder = total % parts
    return [base + int(index < remainder) for index in range(parts)]


def _collect_teacher_data_worker(
    deck: str,
    opponent: str,
    teacher_name: str,
    samples: int,
    seed: int,
    progress_prefix: str,
):
    env = PTCGEnv(deck=deck, opponent=opponent, seed=seed)
    teacher = make_opponent(teacher_name)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    games = 0
    try:
        while len(actions) < samples:
            mask = env.action_masks()
            action = teacher_action_to_rank(env, teacher)
            if action is not None and mask[action]:
                observations.append(obs)
                actions.append(action)
                masks.append(mask)
            else:
                legal = np.flatnonzero(mask)
                action = int(legal[0])

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                games += 1
                obs, _ = env.reset(seed=seed + games)
            if len(actions) > 0 and len(actions) % 5000 == 0:
                print(f"{progress_prefix} collected={len(actions)} games={games}")
    finally:
        env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64), np.asarray(masks, dtype=bool)


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


def train_bc(model: MaskablePPO, observations: np.ndarray, actions: np.ndarray, masks: np.ndarray, epochs: int, batch_size: int):
    device = model.device
    rng = np.random.default_rng(123)
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        indices = rng.permutation(len(actions))
        losses = []
        accuracies = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            obs_tensor = torch.as_tensor(observations[batch_indices], device=device)
            action_tensor = torch.as_tensor(actions[batch_indices], device=device)
            mask_tensor = torch.as_tensor(masks[batch_indices], device=device)
            _, log_prob, entropy = model.policy.evaluate_actions(obs_tensor, action_tensor, action_masks=mask_tensor)
            loss = -log_prob.mean()
            if entropy is not None:
                loss -= 0.001 * entropy.mean()
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
            with torch.no_grad():
                predicted, _ = model.predict(observations[batch_indices], action_masks=masks[batch_indices], deterministic=True)
                accuracies.append(float((predicted == actions[batch_indices]).mean()))
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} loss={np.mean(losses):.4f} accuracy={np.mean(accuracies):.3f}")


if __name__ == "__main__":
    main()
