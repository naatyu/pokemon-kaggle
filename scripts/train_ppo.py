from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rl.ptcg_env import PTCGEnv
from rl.opponents import make_opponent
from rl.ptcg_env import NOOP_ACTION
from rl.device import resolve_torch_device
from cg.api import to_observation_class


def parse_net_arch(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers:
        raise argparse.ArgumentTypeError("--net-arch must contain at least one layer size.")
    return layers


def make_env(deck: str, opponent: str, seed: int, rank: int, reward_shaping_scale: float):
    def _init():
        return Monitor(PTCGEnv(deck=deck, opponent=opponent, reward_shaping_scale=reward_shaping_scale, seed=seed + rank))

    return _init


def build_env(
    deck: str,
    opponent: str,
    seed: int,
    n_envs: int,
    start_method: str,
    reward_shaping_scale: float,
) -> PTCGEnv | VecEnv:
    if n_envs <= 1:
        return PTCGEnv(deck=deck, opponent=opponent, reward_shaping_scale=reward_shaping_scale, seed=seed)
    env_fns = [make_env(deck, opponent, seed, rank, reward_shaping_scale) for rank in range(n_envs)]
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


class BCRegularizationCallback(BaseCallback):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        masks: np.ndarray,
        coef: float,
        batch_size: int,
        epochs_per_rollout: int,
    ):
        super().__init__()
        self.observations = observations
        self.actions = actions
        self.masks = masks
        self.coef = coef
        self.batch_size = batch_size
        self.epochs_per_rollout = epochs_per_rollout
        self.rng = np.random.default_rng(123)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.coef <= 0 or len(self.actions) == 0:
            return
        policy = self.model.policy
        device = self.model.device
        policy.set_training_mode(True)
        losses = []
        accuracies = []
        for _ in range(self.epochs_per_rollout):
            indices = self.rng.choice(len(self.actions), size=min(self.batch_size, len(self.actions)), replace=False)
            obs_tensor = torch.as_tensor(self.observations[indices], device=device)
            action_tensor = torch.as_tensor(self.actions[indices], device=device)
            mask_tensor = torch.as_tensor(self.masks[indices], device=device)
            _, log_prob, entropy = policy.evaluate_actions(obs_tensor, action_tensor, action_masks=mask_tensor)
            loss = -self.coef * log_prob.mean()
            if entropy is not None:
                loss -= 0.001 * entropy.mean()
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy.optimizer.step()
            with torch.no_grad():
                predicted, _ = self.model.predict(self.observations[indices], action_masks=self.masks[indices], deterministic=True)
                accuracies.append(float((predicted == self.actions[indices]).mean()))
            losses.append(float(loss.detach().cpu()))
        self.logger.record("bc/loss", float(np.mean(losses)))
        self.logger.record("bc/accuracy", float(np.mean(accuracies)))


class MaskedEvalCallback(BaseCallback):
    def __init__(
        self,
        deck: str,
        opponent: str,
        games: int,
        eval_freq: int,
        save_path: Path | None,
        seed: int,
        deterministic: bool = True,
    ):
        super().__init__()
        self.deck = deck
        self.opponent = opponent
        self.games = games
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.seed = seed
        self.deterministic = deterministic
        self.best_win_rate = -1.0

    def _on_training_start(self) -> None:
        self._run_eval()

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        self._run_eval()
        return True

    def _run_eval(self) -> None:
        wins, losses, draws, truncated = evaluate_model(
            self.model,
            deck=self.deck,
            opponent=self.opponent,
            games=self.games,
            seed=self.seed + self.n_calls,
            deterministic=self.deterministic,
        )
        win_rate = wins / self.games if self.games else 0.0
        self.logger.record("eval/win_rate", win_rate)
        self.logger.record("eval/wins", wins)
        self.logger.record("eval/losses", losses)
        self.logger.record("eval/draws", draws)
        self.logger.record("eval/truncated", truncated)
        print(
            f"eval step={self.num_timesteps} opponent={self.opponent} "
            f"wins={wins}/{self.games} win_rate={win_rate:.3f}"
        )
        if self.save_path is not None and win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.save_path)
            print(f"new_best_eval={win_rate:.3f} saved={self.save_path}")


def collect_bc_data(deck: str, opponent: str, teacher_name: str, samples: int, seed: int):
    env = PTCGEnv(deck=deck, opponent=opponent, seed=seed)
    teacher = make_opponent(teacher_name)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    games = 0
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


def evaluate_model(model: MaskablePPO, deck: str, opponent: str, games: int, seed: int, deterministic: bool):
    wins = losses = draws = truncated = 0
    for game in range(games):
        env = PTCGEnv(deck=deck, opponent=opponent, reward_shaping_scale=0.0, seed=seed + game)
        obs, _ = env.reset(seed=seed + game)
        done = False
        was_truncated = False
        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=deterministic)
            obs, _, terminated, was_truncated, _ = env.step(int(action))
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
    return wins, losses, draws, truncated


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
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--start-method", choices=["spawn", "forkserver", "fork", "dummy"], default="spawn")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--net-arch", type=parse_net_arch, default=[64, 64])
    parser.add_argument("--reward-shaping-scale", type=float, default=1.0)
    parser.add_argument("--bc-teacher")
    parser.add_argument("--bc-samples", type=int, default=0)
    parser.add_argument("--bc-coef", type=float, default=0.0)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--bc-epochs-per-rollout", type=int, default=1)
    parser.add_argument("--eval-opponent")
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--best-save-path", type=Path)
    args = parser.parse_args()
    args.device = resolve_torch_device(args.device)
    print(f"torch_device={args.device}")
    if args.eval_opponent and args.eval_freq > 0 and (args.n_envs <= 1 or args.start_method == "dummy"):
        raise ValueError("In-training evaluation needs subprocess envs; use --n-envs > 1 and --start-method spawn/forkserver/fork.")

    env = build_env(args.deck, args.opponent, args.seed, args.n_envs, args.start_method, args.reward_shaping_scale)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(args, env)
    callbacks = []
    if args.bc_teacher and args.bc_samples > 0 and args.bc_coef > 0:
        observations, actions, masks = collect_bc_data(args.deck, args.opponent, args.bc_teacher, args.bc_samples, args.seed + 10_000)
        callbacks.append(
            BCRegularizationCallback(
                observations,
                actions,
                masks,
                coef=args.bc_coef,
                batch_size=args.bc_batch_size,
                epochs_per_rollout=args.bc_epochs_per_rollout,
            )
        )
    if args.eval_opponent and args.eval_freq > 0:
        callbacks.append(
            MaskedEvalCallback(
                deck=args.deck,
                opponent=args.eval_opponent,
                games=args.eval_games,
                eval_freq=args.eval_freq,
                save_path=args.best_save_path,
                seed=args.seed + 20_000,
            )
        )
    callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0] if callbacks else None
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.save_path)
    env.close()
    print(args.save_path)


if __name__ == "__main__":
    main()
