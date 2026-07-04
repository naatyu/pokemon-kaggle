from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from rl.ptcg_env import BASE_FEATURES, GLOBAL_FEATURES, MAX_OPTIONS, OBS_SIZE, OPTION_FEATURES, POKEMON_FEATURES


MAX_CARD_ID = 1400
MAX_ATTACK_ID = 2500


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


class EmbeddedActionMaskablePolicy(MaskableActorCriticPolicy):
    """Action policy with categorical embeddings for card and attack ids."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        hidden_dim: int = 256,
        card_embedding_dim: int = 32,
        attack_embedding_dim: int = 16,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.card_embedding_dim = card_embedding_dim
        self.attack_embedding_dim = attack_embedding_dim
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self.card_embedding = nn.Embedding(MAX_CARD_ID + 1, self.card_embedding_dim, padding_idx=0)
        self.attack_embedding = nn.Embedding(MAX_ATTACK_ID + 1, self.attack_embedding_dim, padding_idx=0)
        global_input_dim = GLOBAL_FEATURES + self.card_embedding_dim * 5
        option_input_dim = OPTION_FEATURES + self.card_embedding_dim + self.attack_embedding_dim
        self.global_encoder = _mlp(global_input_dim, self.hidden_dim, self.hidden_dim)
        self.option_encoder = _mlp(option_input_dim, self.hidden_dim, self.hidden_dim)
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

        global_card_ids = th.stack(
            [
                _scaled_id(global_features[:, 13], MAX_CARD_ID),
                _scaled_id(global_features[:, 16], MAX_CARD_ID),
                _scaled_id(global_features[:, 36], MAX_CARD_ID),
                _scaled_id(global_features[:, 37], MAX_CARD_ID),
            ],
            dim=1,
        )
        global_card_embeddings = self.card_embedding(global_card_ids).flatten(1)

        board_features = obs[:, BASE_FEATURES:GLOBAL_FEATURES].reshape(-1, 12, POKEMON_FEATURES)
        board_card_ids = _scaled_id(board_features[:, :, 2], MAX_CARD_ID)
        board_mask = board_features[:, :, 0:1].clamp(0.0, 1.0)
        board_embeddings = self.card_embedding(board_card_ids)
        board_embedding = (board_embeddings * board_mask).sum(dim=1) / board_mask.sum(dim=1).clamp_min(1.0)

        global_input = th.cat([global_features, global_card_embeddings, board_embedding], dim=1)
        global_latent = self.global_encoder(global_input)

        option_card_ids = _scaled_id(option_features[:, :, 1], MAX_CARD_ID)
        option_attack_ids = _scaled_id(option_features[:, :, 7], MAX_ATTACK_ID, scale=2000.0)
        option_input = th.cat(
            [
                option_features,
                self.card_embedding(option_card_ids),
                self.attack_embedding(option_attack_ids),
            ],
            dim=2,
        )
        batch_size = obs.shape[0]
        option_latent = self.option_encoder(option_input.reshape(-1, option_input.shape[-1])).reshape(
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


def _scaled_id(values: th.Tensor, max_id: int, scale: float = 1300.0) -> th.Tensor:
    return th.clamp(th.round(values * scale), 0, max_id).long()
