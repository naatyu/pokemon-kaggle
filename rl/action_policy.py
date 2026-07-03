from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from rl.ptcg_env import GLOBAL_FEATURES, MAX_OPTIONS, OBS_SIZE, OPTION_FEATURES


class ActionMaskablePolicy(MaskableActorCriticPolicy):
    """Maskable policy that scores each ranked action choice with shared weights."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        hidden_dim: int = 256,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self.global_encoder = _mlp(GLOBAL_FEATURES, self.hidden_dim, self.hidden_dim)
        self.option_encoder = _mlp(OPTION_FEATURES, self.hidden_dim, self.hidden_dim)
        self.action_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.noop_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        if self.ortho_init:
            module_gains = {
                self.global_encoder: np.sqrt(2),
                self.option_encoder: np.sqrt(2),
                self.action_scorer: 0.01,
                self.noop_scorer: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        logits, values = self._logits_and_values(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def get_distribution(self, obs, action_masks: np.ndarray | None = None):
        logits, _ = self._logits_and_values(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        logits, values = self._logits_and_values(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return values, distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        _, values = self._logits_and_values(obs)
        return values

    def _logits_and_values(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        obs = obs.float()
        if obs.shape[-1] != OBS_SIZE:
            raise ValueError(f"Expected observation width {OBS_SIZE}, got {obs.shape[-1]}.")
        global_features = obs[:, :GLOBAL_FEATURES]
        option_features = obs[:, GLOBAL_FEATURES:].reshape(-1, MAX_OPTIONS, OPTION_FEATURES)
        global_latent = self.global_encoder(global_features)
        batch_size = obs.shape[0]
        option_latent = self.option_encoder(option_features.reshape(-1, OPTION_FEATURES)).reshape(
            batch_size, MAX_OPTIONS, self.hidden_dim
        )
        expanded_global = global_latent.unsqueeze(1).expand(-1, MAX_OPTIONS, -1)
        action_logits = self.action_scorer(th.cat([expanded_global, option_latent], dim=-1)).squeeze(-1)
        noop_logit = self.noop_scorer(global_latent)
        values = self.value_net(global_latent)
        return th.cat([action_logits, noop_logit], dim=1), values


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )
