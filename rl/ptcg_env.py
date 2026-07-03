from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SUBMISSION = PROJECT_ROOT / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission"
if str(SAMPLE_SUBMISSION) not in sys.path:
    sys.path.append(str(SAMPLE_SUBMISSION))

from cg.api import AreaType, LogType, OptionType, SelectContext, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from agents.deck_loader import load_deck  # noqa: E402
from agents.deck_profiles import PROFILES  # noqa: E402
from agents.heuristic_agent import score_option  # noqa: E402
from rl.opponents import Opponent, make_opponent  # noqa: E402


MAX_OPTIONS = 256
NOOP_ACTION = MAX_OPTIONS
GLOBAL_FEATURES = 32
OPTION_FEATURES = 10
OBS_SIZE = GLOBAL_FEATURES + MAX_OPTIONS * OPTION_FEATURES


class PTCGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        deck: str = "hydrapple",
        opponent: str = "random_abomasnow",
        max_steps: int = 2500,
        seed: int | None = None,
    ):
        super().__init__()
        self.deck_name = deck
        self.profile = PROFILES[deck]
        self.deck = load_deck(PROJECT_ROOT / self.profile.deck_path)
        self.opponent_names = _parse_opponent_names(opponent)
        self.opponent_name = self.opponent_names[0]
        self.opponent: Opponent = make_opponent(self.opponent_name)
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.action_space = spaces.Discrete(MAX_OPTIONS + 1)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.obs_dict: dict[str, Any] | None = None
        self.steps = 0
        self.prev_my_prizes = 6
        self.prev_opp_prizes = 6
        self.finished = False
        self.battle_active = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        self._finish_if_needed()
        self.opponent_name = self.rng.choice(self.opponent_names)
        self.opponent = make_opponent(self.opponent_name)
        obs, start_data = battle_start(self.deck, self.opponent.deck())
        if obs is None:
            raise RuntimeError(f"battle_start failed: player={start_data.errorPlayer} type={start_data.errorType}")
        self.battle_active = True
        self.obs_dict = obs
        self.steps = 0
        self.finished = False
        self._sync_prize_baseline()
        reward, terminated = self._advance_opponent_until_player()
        if terminated:
            self.finished = True
        return self._features(), {"terminal_on_reset": terminated, "reward": reward, "opponent": self.opponent_name}

    def step(self, action: int):
        if self.obs_dict is None:
            raise RuntimeError("Call reset() before step().")
        if self.finished:
            return self._features(), 0.0, True, False, {}

        obs = to_observation_class(self.obs_dict)
        if obs.current is None or obs.current.yourIndex != 0:
            reward, terminated = self._advance_opponent_until_player()
            return self._features(), reward, terminated, False, {"advanced_opponent": True}

        self.steps += 1
        selected = self._action_to_selection(action)
        action_reward = self._action_shaping(action)
        try:
            self.obs_dict = battle_select(selected)
        except Exception as exc:
            self.finished = True
            self._finish_if_needed()
            return self._features(), -1.0, True, False, {"error": f"invalid action {selected}: {type(exc).__name__}: {exc}"}

        reward = action_reward + self._prize_delta_reward()
        terminal_reward, terminated = self._terminal_reward()
        if not terminated:
            opp_reward, terminated = self._advance_opponent_until_player()
            reward += opp_reward
            if terminated:
                terminal_reward, _ = self._terminal_reward()
            else:
                terminal_reward, terminated_after_opp = self._terminal_reward()
                terminated = terminated_after_opp

        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            reward -= 0.2
            self.finished = True
            self._finish_if_needed()
        elif terminated:
            reward += terminal_reward
            self.finished = True
            self._finish_if_needed()

        return self._features(), float(reward), terminated, truncated, {"selection": selected, "opponent": self.opponent_name}

    def close(self):
        self._finish_if_needed()

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(MAX_OPTIONS + 1, dtype=bool)
        if self.obs_dict is None or self.finished:
            mask[NOOP_ACTION] = True
            return mask
        obs = to_observation_class(self.obs_dict)
        if obs.select is None or obs.current is None or obs.current.yourIndex != 0:
            mask[NOOP_ACTION] = True
            return mask
        option_count = min(len(obs.select.option), MAX_OPTIONS)
        if option_count > 0:
            mask[:option_count] = True
        if obs.select.minCount == 0:
            mask[NOOP_ACTION] = True
        if not mask.any():
            mask[NOOP_ACTION] = True
        return mask

    def _advance_opponent_until_player(self) -> tuple[float, bool]:
        reward = 0.0
        for _ in range(self.max_steps):
            obs = to_observation_class(self.obs_dict)
            _, terminated = self._terminal_reward()
            if terminated:
                return reward, True
            if obs.current is None or obs.current.yourIndex == 0:
                return reward, False
            try:
                action = self.opponent.act(self.obs_dict)
                self.obs_dict = battle_select(action)
            except Exception as exc:
                return 1.0, True
            reward += self._prize_delta_reward()
        return reward - 0.2, True

    def _action_to_selection(self, action: int) -> list[int]:
        obs = to_observation_class(self.obs_dict)
        select = obs.select
        if select is None or select.maxCount == 0 or len(select.option) == 0:
            return []
        ranked_options = self._ranked_option_indices(obs)
        option_count = len(ranked_options)
        if action == NOOP_ACTION and select.minCount == 0:
            return []
        picked = int(action)
        if picked < 0 or picked >= option_count:
            picked = self.rng.randrange(option_count)
        picked_option = ranked_options[picked]
        if select.minCount <= 1 and select.maxCount == 1:
            return [picked_option]
        if select.minCount == 0:
            return [picked_option]

        selected = [picked_option]
        for index in ranked_options:
            if len(selected) >= select.minCount:
                break
            if index != picked_option:
                selected.append(index)
        return selected[: select.maxCount]

    def _action_shaping(self, action: int) -> float:
        if action == NOOP_ACTION or self.obs_dict is None:
            return 0.0
        obs = to_observation_class(self.obs_dict)
        ranked_options = self._ranked_option_indices(obs)
        if obs.select is None or action >= len(ranked_options):
            return 0.0
        option = obs.select.option[ranked_options[action]]
        reward = 0.0
        if 0 <= action < 5:
            reward += (5 - action) * 0.001
        if option.type == OptionType.ATTACK:
            reward += 0.02
        elif option.type == OptionType.EVOLVE:
            reward += 0.01
        elif option.type == OptionType.ATTACH:
            reward += 0.005
        elif option.type == OptionType.END:
            reward -= 0.005
        return reward

    def _prize_delta_reward(self) -> float:
        obs = to_observation_class(self.obs_dict)
        if obs.current is None:
            return 0.0
        my_prizes = len(obs.current.players[0].prize)
        opp_prizes = len(obs.current.players[1].prize)
        reward = (self.prev_opp_prizes - opp_prizes) * 0.2
        reward -= (self.prev_my_prizes - my_prizes) * 0.2
        self.prev_my_prizes = my_prizes
        self.prev_opp_prizes = opp_prizes
        return reward

    def _terminal_reward(self) -> tuple[float, bool]:
        if self.obs_dict is None:
            return 0.0, False
        obs = to_observation_class(self.obs_dict)
        result = None
        if obs.current and obs.current.result != -1:
            result = obs.current.result
        else:
            for log in obs.logs:
                if log.type == LogType.RESULT:
                    result = log.result
                    break
        if result is None:
            return 0.0, False
        if result == 0:
            return 1.0, True
        if result == 1:
            return -1.0, True
        return 0.0, True

    def _sync_prize_baseline(self) -> None:
        obs = to_observation_class(self.obs_dict)
        if obs.current is None:
            self.prev_my_prizes = 6
            self.prev_opp_prizes = 6
            return
        self.prev_my_prizes = len(obs.current.players[0].prize)
        self.prev_opp_prizes = len(obs.current.players[1].prize)

    def _features(self) -> np.ndarray:
        features = np.zeros(OBS_SIZE, dtype=np.float32)
        if self.obs_dict is None:
            return features
        obs = to_observation_class(self.obs_dict)
        if obs.current is None:
            return features
        state = obs.current
        me = state.players[0]
        opp = state.players[1]
        select = obs.select
        active = me.active[0] if me.active else None
        opp_active = opp.active[0] if opp.active else None
        global_values = [
            state.turn / 100.0,
            state.turnActionCount / 100.0,
            float(state.supporterPlayed),
            float(state.energyAttached),
            float(state.retreated),
            len(me.prize) / 6.0,
            len(opp.prize) / 6.0,
            me.handCount / 20.0,
            opp.handCount / 20.0,
            me.deckCount / 60.0,
            opp.deckCount / 60.0,
            len(me.bench) / max(1, me.benchMax),
            len(opp.bench) / max(1, opp.benchMax),
            _pokemon_id(active),
            _pokemon_hp(active),
            _pokemon_energy_count(active),
            _pokemon_id(opp_active),
            _pokemon_hp(opp_active),
            _pokemon_energy_count(opp_active),
            (select.type if select else 0) / 16.0,
            (select.context if select else 0) / 64.0,
            (select.minCount if select else 0) / 6.0,
            (select.maxCount if select else 0) / 6.0,
            len(select.option if select else []) / MAX_OPTIONS,
        ]
        features[: len(global_values)] = global_values

        if select is not None:
            for index, option_index in enumerate(self._ranked_option_indices(obs)):
                option = select.option[option_index]
                offset = GLOBAL_FEATURES + index * OPTION_FEATURES
                card_id = _option_card_id(obs, option)
                heuristic_score = _safe_score_option(obs, option, self.profile)
                features[offset : offset + OPTION_FEATURES] = [
                    option.type / 20.0,
                    (card_id or 0) / 1300.0,
                    (option.area or 0) / 16.0,
                    (option.index or 0) / 16.0,
                    (option.playerIndex if option.playerIndex is not None else -1) / 2.0,
                    (option.inPlayArea or 0) / 16.0,
                    (option.inPlayIndex or 0) / 8.0,
                    (option.attackId or 0) / 2000.0,
                    (option.number or 0) / 10.0,
                    np.clip(heuristic_score / 3000.0, -1.0, 1.0),
                ]
        return features

    def _ranked_option_indices(self, obs) -> list[int]:
        select = obs.select
        if select is None:
            return []
        option_count = min(len(select.option), MAX_OPTIONS)
        scored: list[tuple[int, int]] = []
        for index, option in enumerate(select.option[:option_count]):
            score = _safe_score_option(obs, option, self.profile)
            scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [index for _, index in scored]

    def _finish_if_needed(self) -> None:
        if not self.battle_active:
            return
        try:
            battle_finish()
        except Exception:
            pass
        self.battle_active = False


def _pokemon_id(pokemon) -> float:
    return 0.0 if pokemon is None else pokemon.id / 1300.0


def _pokemon_hp(pokemon) -> float:
    if pokemon is None or pokemon.maxHp <= 0:
        return 0.0
    return pokemon.hp / pokemon.maxHp


def _pokemon_energy_count(pokemon) -> float:
    return 0.0 if pokemon is None else len(pokemon.energyCards) / 8.0


def _option_card_id(obs, option) -> int | None:
    if option.cardId is not None:
        return option.cardId
    try:
        if option.type in {OptionType.PLAY, OptionType.ATTACH} and option.index is not None:
            hand = obs.current.players[obs.current.yourIndex].hand or []
            if 0 <= option.index < len(hand):
                return hand[option.index].id
        if option.area == AreaType.HAND and option.index is not None:
            hand = obs.current.players[obs.current.yourIndex].hand or []
            if 0 <= option.index < len(hand):
                return hand[option.index].id
        if option.area == AreaType.LOOKING and option.index is not None and obs.current.looking:
            card = obs.current.looking[option.index]
            return card.id if card else None
        if option.area == AreaType.ACTIVE:
            player = obs.current.players[option.playerIndex if option.playerIndex is not None else obs.current.yourIndex]
            return player.active[0].id if player.active else None
        if option.area == AreaType.BENCH and option.index is not None:
            player = obs.current.players[option.playerIndex if option.playerIndex is not None else obs.current.yourIndex]
            if 0 <= option.index < len(player.bench):
                return player.bench[option.index].id
    except Exception:
        return None
    return None


def _safe_score_option(obs, option, profile) -> int:
    try:
        return score_option(obs, option, profile)
    except Exception:
        return 0


def _parse_opponent_names(opponent: str) -> list[str]:
    names = [name.strip() for name in opponent.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one opponent is required.")
    return names
