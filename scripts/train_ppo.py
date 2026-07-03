from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rl.ptcg_env import PTCGEnv


def parse_net_arch(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers:
        raise argparse.ArgumentTypeError("--net-arch must contain at least one layer size.")
    return layers


def make_env(deck: str, opponent: str, seed: int, rank: int):
    def _init():
        return Monitor(PTCGEnv(deck=deck, opponent=opponent, seed=seed + rank))

    return _init


def build_env(deck: str, opponent: str, seed: int, n_envs: int, start_method: str) -> PTCGEnv | VecEnv:
    if n_envs <= 1:
        return PTCGEnv(deck=deck, opponent=opponent, seed=seed)
    env_fns = [make_env(deck, opponent, seed, rank) for rank in range(n_envs)]
    if start_method == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method=start_method)


def build_model(args: argparse.Namespace, env: PTCGEnv | VecEnv) -> MaskablePPO:
    policy_kwargs = {"net_arch": args.net_arch}
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        device=args.device,
    )
    if args.load_path:
        loaded = MaskablePPO.load(args.load_path, device=args.device)
        model.set_parameters({"policy": loaded.get_parameters()["policy"]}, exact_match=False, device=args.device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", default="hydrapple")
    parser.add_argument(
        "--opponent",
        default="random_abomasnow",
        help="Opponent name, or comma-separated names sampled per episode.",
    )
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--load-path", type=Path)
    parser.add_argument("--save-path", type=Path, default=PROJECT_ROOT / "models" / "ppo_hydrapple_v0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--start-method", choices=["spawn", "forkserver", "fork", "dummy"], default="spawn")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--net-arch", type=parse_net_arch, default=[64, 64])
    args = parser.parse_args()

    env = build_env(args.deck, args.opponent, args.seed, args.n_envs, args.start_method)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(args, env)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)
    env.close()
    print(args.save_path)


if __name__ == "__main__":
    main()
