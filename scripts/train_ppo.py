from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rl.ptcg_env import PTCGEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", default="hydrapple")
    parser.add_argument("--opponent", default="random_abomasnow")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--load-path", type=Path)
    parser.add_argument("--save-path", type=Path, default=PROJECT_ROOT / "models" / "ppo_hydrapple_v0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env = PTCGEnv(deck=args.deck, opponent=args.opponent, seed=args.seed)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.load_path:
        model = MaskablePPO.load(args.load_path, env=env, device=args.device)
        model.verbose = 1
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            n_steps=512,
            batch_size=128,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=0.02,
            device=args.device,
        )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)
    env.close()
    print(args.save_path)


if __name__ == "__main__":
    main()
