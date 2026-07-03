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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--deck", default="hydrapple")
    parser.add_argument("--opponent", default="random_abomasnow")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model = MaskablePPO.load(args.model, device=args.device)
    wins = losses = draws = truncated = 0
    for game in range(args.games):
        env = PTCGEnv(deck=args.deck, opponent=args.opponent, seed=args.seed + game)
        obs, _ = env.reset(seed=args.seed + game)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=args.deterministic)
            obs, reward, terminated, was_truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or was_truncated
        terminal_reward, _ = env._terminal_reward()
        if was_truncated:
            truncated += 1
        elif terminal_reward > 0:
            wins += 1
        elif terminal_reward < 0:
            losses += 1
        else:
            draws += 1
        env.close()
        print(f"game={game + 1} reward={total_reward:.3f} terminal={terminal_reward:.1f}")
    print(
        f"summary model={args.model} opponent={args.opponent} "
        f"games={args.games} wins={wins} losses={losses} draws={draws} truncated={truncated} "
        f"win_rate={wins / args.games:.3f}"
    )


if __name__ == "__main__":
    main()
