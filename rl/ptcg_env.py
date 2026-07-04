from __future__ import annotations

import random
import sys
from itertools import combinations
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

from agents.card_db import load_card_db  # noqa: E402
from agents.deck_loader import load_deck  # noqa: E402
from agents.deck_profiles import PROFILES  # noqa: E402
from agents.heuristic_agent import score_option  # noqa: E402
from rl.opponents import Opponent, make_opponent  # noqa: E402


MAX_OPTIONS = 256
NOOP_ACTION = MAX_OPTIONS
BASE_FEATURES = 40
BOARD_SLOTS_PER_PLAYER = 6
POKEMON_FEATURES = 10
GLOBAL_FEATURES = BASE_FEATURES + 2 * BOARD_SLOTS_PER_PLAYER * POKEMON_FEATURES
OPTION_FEATURES = 16
OBS_SIZE = GLOBAL_FEATURES + MAX_OPTIONS * OPTION_FEATURES
MAX_COMBO_OPTIONS = 18
MAX_COMBO_CANDIDATES = 2048


class PTCGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        deck: str = "hydrapple",
        opponent: str = "random_abomasnow",
        max_steps: int = 2500,
        reward_shaping_scale: float = 1.0,
        effect_features: bool = False,
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
        self.reward_shaping_scale = reward_shaping_scale
        self.effect_features = effect_features
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
        opponent_deck = None
        for _ in range(max(3, len(self.opponent_names) * 2)):
            self.opponent_name = self.rng.choice(self.opponent_names)
            self.opponent = make_opponent(self.opponent_name)
            opponent_deck = self.opponent.deck()
            if len(opponent_deck) == 60:
                break
        if opponent_deck is None or len(opponent_deck) != 60:
            raise RuntimeError(f"Opponent {self.opponent_name} returned {len(opponent_deck or [])} deck cards.")
        obs, start_data = battle_start(self.deck, opponent_deck)
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
        action_reward = self.reward_shaping_scale * self._action_shaping(action)
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
        choice_count = len(self._ranked_action_choices(obs))
        if choice_count > 0:
            mask[:choice_count] = True
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
        if action == NOOP_ACTION and select.minCount == 0:
            return []
        choices = self._ranked_action_choices(obs)
        choice_count = len(choices)
        picked = int(action)
        if picked < 0 or picked >= choice_count:
            picked = self.rng.randrange(choice_count)
        return list(choices[picked])

    def _action_shaping(self, action: int) -> float:
        if action == NOOP_ACTION or self.obs_dict is None:
            return 0.0
        obs = to_observation_class(self.obs_dict)
        choices = self._ranked_action_choices(obs)
        if obs.select is None or action >= len(choices):
            return 0.0
        choice = choices[action]
        if not choice:
            return 0.0
        option = obs.select.option[choice[0]]
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
            float(me.asleep),
            float(me.burned),
            float(me.confused),
            float(me.paralyzed),
            float(me.poisoned),
            float(opp.asleep),
            float(opp.burned),
            float(opp.confused),
            float(opp.paralyzed),
            float(opp.poisoned),
            len(me.discard) / 60.0,
            len(opp.discard) / 60.0,
        ]
        if self.effect_features and select is not None:
            global_values.extend(
                [
                    _card_id(select.contextCard),
                    _card_id(select.effect),
                    _profile_hand_signal(me, self.profile),
                    _profile_discard_signal(me, self.profile),
                ]
            )
        features[: len(global_values)] = global_values
        offset = BASE_FEATURES
        for player in (me, opp):
            board = list(player.active[:1]) + list(player.bench[: BOARD_SLOTS_PER_PLAYER - 1])
            for slot in range(BOARD_SLOTS_PER_PLAYER):
                pokemon = board[slot] if slot < len(board) else None
                features[offset : offset + POKEMON_FEATURES] = _pokemon_features(pokemon, active_slot=(slot == 0))
                offset += POKEMON_FEATURES

        if select is not None:
            for index, choice in enumerate(self._ranked_action_choices(obs)):
                if not choice:
                    continue
                option = select.option[choice[0]]
                offset = GLOBAL_FEATURES + index * OPTION_FEATURES
                card_id = _option_card_id(obs, option)
                heuristic_score = _choice_score(obs, choice, self.profile)
                card = load_card_db().get(card_id) if card_id is not None else None
                features[offset : offset + OPTION_FEATURES] = [
                    option.type / 20.0,
                    (card_id or 0) / 1300.0,
                    (option.area or 0) / 16.0,
                    (option.index or 0) / 16.0,
                    (option.playerIndex if option.playerIndex is not None else -1) / 2.0,
                    (option.inPlayArea or 0) / 16.0,
                    (option.inPlayIndex or 0) / 8.0,
                    (option.attackId or 0) / 2000.0,
                    len(choice) / 6.0,
                    np.clip(heuristic_score / 3000.0, -1.0, 1.0),
                    _card_hp(card),
                    _card_attack_damage(card),
                    float(card.is_pokemon) if card else 0.0,
                    float(card.is_energy) if card else 0.0,
                    float(card_id in self.profile.main_attackers) if card_id is not None else 0.0,
                    float(card_id in self.profile.draw_search_cards) if card_id is not None else 0.0,
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

    def _ranked_action_choices(self, obs) -> list[tuple[int, ...]]:
        select = obs.select
        if select is None or select.maxCount == 0:
            return []
        ranked_options = self._ranked_option_indices(obs)
        if not ranked_options:
            return []
        min_count = max(1, select.minCount)
        max_count = max(min(select.maxCount, 6), min_count)
        if min_count == 1 and max_count == 1:
            return [(index,) for index in ranked_options[:MAX_OPTIONS]]

        candidates: set[tuple[int, ...]] = set()
        combo_options = ranked_options[: min(len(ranked_options), MAX_COMBO_OPTIONS)]
        for count in range(min_count, max_count + 1):
            if count == 1:
                candidates.update((index,) for index in ranked_options)
                continue
            for combo in combinations(combo_options, count):
                candidates.add(combo)
                if len(candidates) >= MAX_COMBO_CANDIDATES:
                    break
            if len(candidates) >= MAX_COMBO_CANDIDATES:
                break

        scored = [
            (_choice_score(obs, choice, self.profile), len(choice), tuple(ranked_options.index(index) for index in choice), choice)
            for choice in candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [choice for _, _, _, choice in scored[:MAX_OPTIONS]]

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


def _card_id(card) -> float:
    return 0.0 if card is None else card.id / 1300.0


def _pokemon_hp(pokemon) -> float:
    if pokemon is None or pokemon.maxHp <= 0:
        return 0.0
    return pokemon.hp / pokemon.maxHp


def _pokemon_energy_count(pokemon) -> float:
    return 0.0 if pokemon is None else len(pokemon.energyCards) / 8.0


def _profile_hand_signal(player, profile) -> float:
    hand = player.hand or []
    energy = sum(1 for card in hand if card and card.id in profile.energy_targets)
    setup = sum(1 for card in hand if card and card.id in profile.setup_basics)
    evolution = sum(1 for card in hand if card and card.id in profile.evolution_targets)
    draw = sum(1 for card in hand if card and card.id in profile.draw_search_cards)
    disruption = sum(1 for card in hand if card and card.id in profile.disruption_cards)
    return min((energy + 2 * setup + 2 * evolution + draw + disruption) / 20.0, 1.0)


def _profile_discard_signal(player, profile) -> float:
    discard = player.discard or []
    attackers = profile.main_attackers | profile.backup_attackers
    energy = sum(1 for card in discard if card and card.id in profile.energy_targets)
    recovery = sum(1 for card in discard if card and card.id in profile.recovery_cards)
    discarded_attackers = sum(1 for card in discard if card and card.id in attackers)
    return min((energy + recovery + 2 * discarded_attackers) / 20.0, 1.0)


def _pokemon_features(pokemon, active_slot: bool) -> np.ndarray:
    if pokemon is None:
        return np.zeros(POKEMON_FEATURES, dtype=np.float32)
    return np.asarray(
        [
            1.0,
            float(active_slot),
            pokemon.id / 1300.0,
            _pokemon_hp(pokemon),
            pokemon.maxHp / 400.0,
            len(pokemon.energyCards) / 8.0,
            len(pokemon.tools) / 4.0,
            len(pokemon.preEvolution) / 4.0,
            float(pokemon.appearThisTurn),
            pokemon.serial / 200.0,
        ],
        dtype=np.float32,
    )


def _card_hp(card) -> float:
    if card is None:
        return 0.0
    return card.hp / 400.0


def _card_attack_damage(card) -> float:
    if card is None:
        return 0.0
    return card.attack_damage / 400.0


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


def _choice_score(obs, choice: tuple[int, ...], profile) -> int:
    score = 0
    for option_index in choice:
        try:
            score += score_option(obs, obs.select.option[option_index], profile)
        except Exception:
            pass
    return score


def _parse_opponent_names(opponent: str) -> list[str]:
    names = [name.strip() for name in opponent.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one opponent is required.")
    return names
