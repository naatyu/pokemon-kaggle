from __future__ import annotations

import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SAMPLE_SUBMISSION = Path(__file__).resolve().parents[1] / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission"
if str(SAMPLE_SUBMISSION) not in sys.path:
    sys.path.append(str(SAMPLE_SUBMISSION))

from cg.api import (  # noqa: E402
    AreaType,
    Observation,
    Option,
    OptionType,
    SelectType,
    SelectContext,
    to_observation_class,
)

from .card_db import CardInfo, card_name, load_card_db
from .deck_loader import PROJECT_ROOT, load_deck
from .deck_profiles import PROFILES, DeckProfile


@dataclass(frozen=True)
class OptionTrace:
    index: int
    score: int
    selected: bool
    option_type: str
    context: str
    card_id: int | None
    card_name: str | None
    description: str


@dataclass(frozen=True)
class DecisionTrace:
    turn: int | None
    player_index: int | None
    deck_profile: str
    select_type: str
    context: str
    min_count: int
    max_count: int
    chosen: list[int]
    board: dict
    options: list[OptionTrace]


LAST_DECISION: DecisionTrace | None = None


def _profile() -> DeckProfile:
    return PROFILES[os.getenv("POKEMON_DECK", "hydrapple")]


def _deck_path(profile: DeckProfile) -> Path:
    configured = os.getenv("POKEMON_DECK_PATH")
    if configured:
        return Path(configured)
    return PROJECT_ROOT / profile.deck_path


def agent(obs_dict: dict) -> list[int]:
    """Kaggle-compatible agent entry point."""
    profile = _profile()
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return load_deck(_deck_path(profile))

    try:
        return choose_action(obs, profile)
    except Exception as exc:  # Keep the competition agent alive at all costs.
        _log(f"fallback after {type(exc).__name__}: {exc}")
        return fallback_selection(obs)


def choose_action(obs: Observation, profile: DeckProfile) -> list[int]:
    global LAST_DECISION
    select = obs.select
    option_count = len(select.option)
    if option_count == 0 or select.maxCount == 0:
        LAST_DECISION = _decision_trace(obs, profile, [], [])
        return []

    scored = [(score_option(obs, option, profile), index) for index, option in enumerate(select.option)]
    scored.sort(key=lambda item: (-item[0], item[1]))

    if select.minCount == 0 and scored and scored[0][0] < 0:
        return []

    target_count = select.maxCount
    if select.minCount == select.maxCount:
        target_count = select.minCount
    elif _is_optional_filter_context(select.context):
        target_count = min(select.maxCount, max(select.minCount, sum(1 for score, _ in scored if score > 0)))
    elif select.type == 8:  # SelectType.COUNT, kept numeric for enum forward compatibility.
        target_count = 1
    else:
        target_count = max(1, select.minCount)

    chosen = [index for score, index in scored if score >= 0][:target_count]
    if len(chosen) < select.minCount:
        chosen = [index for _, index in scored[: select.minCount]]
    elif len(chosen) == 0 and select.minCount > 0:
        chosen = [scored[0][1]]

    _log_choice(obs, scored, chosen)
    LAST_DECISION = _decision_trace(obs, profile, scored, chosen)
    return chosen


def last_decision_dict() -> dict | None:
    if LAST_DECISION is None:
        return None
    return asdict(LAST_DECISION)


def fallback_selection(obs: Observation) -> list[int]:
    select = obs.select
    if select is None or select.maxCount == 0:
        return []
    count = select.minCount if select.minCount > 0 else min(1, select.maxCount)
    count = min(count, len(select.option))
    return random.sample(range(len(select.option)), count)


def score_option(obs: Observation, option: Option, profile: DeckProfile) -> int:
    cards = load_card_db()
    score = _base_score(obs, option)

    if option.type == OptionType.END:
        return score - 200
    if option.type == OptionType.YES:
        return _score_yes_no(obs, yes=True)
    if option.type == OptionType.NO:
        return _score_yes_no(obs, yes=False)
    if option.type == OptionType.NUMBER:
        return option.number or 0
    if option.type == OptionType.ATTACK:
        return score + _score_attack(obs, option, profile)
    if option.type == OptionType.ABILITY:
        return score + _score_card_action(_option_card_id(obs, option), profile, ability=True)
    if option.type == OptionType.PLAY:
        return score + _score_play(obs, option, profile)
    if option.type == OptionType.ATTACH:
        return score + _score_attach(obs, option, profile)
    if option.type == OptionType.EVOLVE:
        return score + _score_evolve(option, profile)
    if option.type in {OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD, OptionType.ENERGY}:
        return score + _score_selection_card(obs, option, profile, cards)
    if option.type == OptionType.RETREAT:
        return score + _score_retreat(obs, profile)
    if option.type == OptionType.DISCARD:
        return score - 100
    if option.type == OptionType.SPECIAL_CONDITION:
        return score

    return score


def _base_score(obs: Observation, option: Option) -> int:
    context = obs.select.context
    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        return 800
    if context == SelectContext.SETUP_BENCH_POKEMON:
        return 500
    if context in {SelectContext.TO_ACTIVE, SelectContext.SWITCH}:
        return 600
    if context in {SelectContext.TO_HAND, SelectContext.TO_FIELD, SelectContext.TO_BENCH}:
        return 400
    if context in {SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM}:
        return -200
    if context in {SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY}:
        return 700
    if context in {SelectContext.ATTACH_FROM, SelectContext.ATTACH_TO}:
        return 500
    return 0


def _score_yes_no(obs: Observation, yes: bool) -> int:
    context = obs.select.context
    if context == SelectContext.IS_FIRST:
        return 10 if not yes else 0
    if context == SelectContext.MULLIGAN:
        return 10 if yes else 0
    if context == SelectContext.ACTIVATE:
        effect_id = obs.select.effect.id if obs.select.effect else None
        return 100 if yes and effect_id not in {1182, 1080, 1201} else 0
    return 5 if yes else 0


def _score_attack(obs: Observation, option: Option, profile: DeckProfile) -> int:
    your_active = _your_active(obs)
    active_id = your_active.id if your_active else None
    score = 1200
    if active_id in profile.main_attackers:
        score += 600
    elif active_id in profile.backup_attackers:
        score += 300
    prize_value = _probable_active_ko_prize_value(obs)
    if prize_value:
        score += 1200 + prize_value * 200
        if len(_your_player(obs).prize) <= prize_value:
            score += 3000
    if option.attackId is not None:
        score += min(option.attackId, 25)
    return score


def _score_card_action(card_id: int | None, profile: DeckProfile, ability: bool = False) -> int:
    if card_id is None:
        return 0
    score = 300 if ability else 0
    if card_id in profile.main_attackers:
        score += 500
    if card_id in profile.backup_attackers:
        score += 300
    if card_id in profile.draw_search_cards:
        score += 250
    return score


def _score_play(obs: Observation, option: Option, profile: DeckProfile) -> int:
    card_id = _option_card_id(obs, option)
    if card_id is None:
        return 0
    cards = load_card_db()
    card = cards.get(card_id)
    score = 100
    if card_id in profile.setup_basics:
        score += 650
    if card and card.is_pokemon and "Basic" in card.kind and len(_your_player(obs).bench) < _your_player(obs).benchMax:
        score += 900 if len(_your_player(obs).bench) == 0 else 250
    if card_id in profile.draw_search_cards:
        score += 500
    if card_id in profile.stadium_cards:
        score += 260
    if card_id in profile.switch_cards:
        score += 220
    if card_id in profile.recovery_cards:
        score += 180
    if card_id in profile.disruption_cards:
        score += 140
    if card and card.is_energy:
        score -= 50
    if card_id in profile.expendable_cards:
        score -= 120
    if profile.name == "hydrapple":
        score += _score_hydrapple_play(obs, card_id)
    return score


def _score_hydrapple_play(obs: Observation, card_id: int) -> int:
    board_ids = _your_board_ids(obs)
    score = 0
    if card_id == 1123 and _best_benched_attacker(obs, PROFILES["hydrapple"]) is not None:
        score += 450
    if card_id == 1094:  # Bug Catching Set
        score += 250
    if card_id == 1121:  # Ultra Ball
        score += 220
    if card_id == 1261 and 150 in board_ids:  # Forest of Vitality
        score += 180
    if card_id in {1182, 1080, 1201} and len(_opponent_player(obs).prize) > 2:
        score -= 120
    return score


def _score_attach(obs: Observation, option: Option, profile: DeckProfile) -> int:
    score = 450
    target = _attach_target_pokemon(obs, option)
    target_id = target.id if target is not None else None
    source_id = _source_card_id(obs, option)
    if target_id in profile.energy_targets:
        score += 500
    if target_id in profile.main_attackers:
        score += 450
    if target_id in profile.backup_attackers:
        score += 200
    source_card = load_card_db().get(source_id or -1)
    if source_card and source_card.is_basic_energy:
        score += 100
    if profile.name == "hydrapple":
        score += _score_hydrapple_attach_target(target)
    if target is not None:
        attached_count = len(target.energyCards)
        if target_id in profile.main_attackers and attached_count < 3:
            score += max(0, 300 - attached_count * 75)
        elif attached_count >= 4:
            score -= 250
    return score


def _score_hydrapple_attach_target(target) -> int:
    if target is None:
        return 0
    attached_count = len(target.energyCards)
    if target.id == 150:  # Hydrapple ex needs to be able to attack.
        return 500 if attached_count < 2 else 100
    if target.id == 96:
        return 450 if attached_count < 3 else -150
    if target.id == 920:
        return 250 if attached_count < 4 else -150
    if target.id == 710:
        return 200 if attached_count < 2 else -100
    if target.id in {149, 921}:
        return 100 if attached_count < 2 else -100
    return 0


def _score_evolve(option: Option, profile: DeckProfile) -> int:
    score = 700
    if option.cardId in profile.evolution_targets:
        score += 700
    if option.cardId in profile.main_attackers:
        score += 700
    return score


def _score_selection_card(
    obs: Observation,
    option: Option,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
) -> int:
    card_id = _option_card_id(obs, option)
    context = obs.select.context
    card = cards.get(card_id or -1)
    score = 0

    if context in {SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM}:
        score += _discard_score(card_id, profile, card)
    elif context in {SelectContext.TO_HAND, SelectContext.TO_FIELD, SelectContext.TO_BENCH}:
        score += _take_card_score(card_id, profile, card)
    elif context in {SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE, SelectContext.SWITCH}:
        score += _active_score(card_id, profile, card)
    elif context == SelectContext.SETUP_BENCH_POKEMON:
        score += _bench_score(card_id, profile, card)
    elif context in {SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY}:
        score += _damage_target_score(obs, option, card)
    elif context in {SelectContext.ATTACH_FROM, SelectContext.ATTACH_TO}:
        score += _attach_selection_score(card_id, profile, card)
    elif context in {SelectContext.EVOLVES_FROM, SelectContext.EVOLVES_TO, SelectContext.EVOLVE}:
        score += _take_card_score(card_id, profile, card)
    else:
        score += _take_card_score(card_id, profile, card) // 2

    return score


def _discard_score(card_id: int | None, profile: DeckProfile, card: CardInfo | None) -> int:
    score = 0
    if card_id in profile.expendable_cards:
        score += 500
    if card and card.is_basic_energy:
        score += 220
    if card_id in profile.main_attackers or card_id in profile.evolution_targets:
        score -= 700
    if card_id in profile.draw_search_cards:
        score -= 250
    return score


def _take_card_score(card_id: int | None, profile: DeckProfile, card: CardInfo | None) -> int:
    score = 0
    if card_id in profile.main_attackers:
        score += 900
    if card_id in profile.evolution_targets:
        score += 750
    if card_id in profile.setup_basics:
        score += 650
    if card_id in profile.draw_search_cards:
        score += 450
    if card_id in profile.energy_targets:
        score += 320
    if card and card.is_basic_energy:
        score += 250
    if card_id in profile.recovery_cards:
        score += 180
    if profile.name == "hydrapple":
        score += _score_hydrapple_take_card(card_id)
    return score


def _score_hydrapple_take_card(card_id: int | None) -> int:
    if card_id == 150:
        return 250
    if card_id == 921:
        return 180
    if card_id == 149:
        return 150
    if card_id == 96:
        return 250
    if card_id in {710, 709, 918, 708, 917}:
        return 120
    if card_id in {1071, 140}:
        return -250
    return 0


def _active_score(card_id: int | None, profile: DeckProfile, card: CardInfo | None) -> int:
    score = 0
    if card_id in profile.main_attackers:
        score += 1000
    if card_id in profile.backup_attackers:
        score += 700
    if card_id in profile.setup_basics:
        score += 500
    if card:
        score += min(card.hp, 350)
    if card_id in profile.expendable_cards:
        score -= 200
    return score


def _bench_score(card_id: int | None, profile: DeckProfile, card: CardInfo | None) -> int:
    score = _take_card_score(card_id, profile, card)
    if card_id in profile.expendable_cards:
        score -= 250
    return score


def _damage_target_score(obs: Observation, option: Option, card: CardInfo | None) -> int:
    target = _option_pokemon(obs, option)
    if target is None:
        return 0
    damage_taken = max(0, target.maxHp - target.hp)
    prize_bonus = 250 if card and "ex" in card.name else 0
    ko_bonus = 700 if obs.select.remainDamageCounter >= target.hp // 10 else 0
    return damage_taken + prize_bonus + ko_bonus + max(0, 350 - target.hp)


def _attach_selection_score(card_id: int | None, profile: DeckProfile, card: CardInfo | None) -> int:
    if card and card.is_energy:
        return 500
    return _take_card_score(card_id, profile, card)


def _score_retreat(obs: Observation, profile: DeckProfile) -> int:
    active = _your_active(obs)
    if active is None:
        return 0
    if active.id in profile.main_attackers:
        return -300
    benched_ids = {pokemon.id for pokemon in _your_player(obs).bench}
    if benched_ids & (profile.main_attackers | profile.backup_attackers):
        return 450
    return -50


def _is_optional_filter_context(context: SelectContext) -> bool:
    return context in {
        SelectContext.SETUP_BENCH_POKEMON,
        SelectContext.TO_HAND,
        SelectContext.TO_FIELD,
        SelectContext.TO_BENCH,
        SelectContext.DISCARD,
        SelectContext.TO_DECK,
        SelectContext.TO_DECK_BOTTOM,
        SelectContext.DAMAGE_COUNTER_ANY,
    }


def _your_player(obs: Observation):
    return obs.current.players[obs.current.yourIndex]


def _opponent_player(obs: Observation):
    return obs.current.players[1 - obs.current.yourIndex]


def _your_active(obs: Observation):
    active = _your_player(obs).active
    return active[0] if active else None


def _can_probably_ko_active(obs: Observation) -> bool:
    return _probable_active_ko_prize_value(obs) > 0


def _probable_active_ko_prize_value(obs: Observation) -> int:
    active = _your_active(obs)
    opponent_active = _opponent_player(obs).active
    if active is None or not opponent_active or opponent_active[0] is None:
        return 0
    cards = load_card_db()
    attack_damage = cards.get(active.id, CardInfo(active.id, "", "", 0, 0)).attack_damage
    if attack_damage <= 0 or attack_damage < opponent_active[0].hp:
        return 0
    opponent_card = cards.get(opponent_active[0].id)
    return _prize_value(opponent_card)


def _prize_value(card: CardInfo | None) -> int:
    if card is None:
        return 1
    if "Mega " in card.name and "ex" in card.name:
        return 3
    if "ex" in card.name:
        return 2
    return 1


def _source_card_id(obs: Observation, option: Option) -> int | None:
    if option.cardId is not None:
        return option.cardId
    if option.index is not None:
        hand = _your_player(obs).hand or []
        if 0 <= option.index < len(hand):
            return hand[option.index].id
    return None


def _attach_target_pokemon(obs: Observation, option: Option):
    if option.inPlayArea is None:
        return None
    player = _your_player(obs)
    if option.inPlayArea == AreaType.ACTIVE:
        return player.active[0] if player.active else None
    if option.inPlayArea == AreaType.BENCH and option.inPlayIndex is not None:
        if 0 <= option.inPlayIndex < len(player.bench):
            return player.bench[option.inPlayIndex]
    return None


def _option_card_id(obs: Observation, option: Option) -> int | None:
    if option.cardId is not None:
        return option.cardId
    if option.type == OptionType.PLAY:
        return _source_card_id(obs, option)
    pokemon = _option_pokemon(obs, option)
    if pokemon is not None:
        return pokemon.id
    if option.area == AreaType.HAND and option.index is not None:
        hand = _your_player(obs).hand or []
        if 0 <= option.index < len(hand):
            return hand[option.index].id
    if option.area == AreaType.LOOKING and option.index is not None and obs.current:
        looking = obs.current.looking or []
        if 0 <= option.index < len(looking):
            card = looking[option.index]
            return card.id if card else None
    if obs.select.deck is not None and option.index is not None:
        if 0 <= option.index < len(obs.select.deck):
            card = obs.select.deck[option.index]
            return card.id if card else None
    return None


def _your_board_ids(obs: Observation) -> set[int]:
    player = _your_player(obs)
    ids = {pokemon.id for pokemon in player.bench}
    if player.active:
        ids.add(player.active[0].id)
    return ids


def _best_benched_attacker(obs: Observation, profile: DeckProfile):
    best = None
    best_score = -10_000
    for pokemon in _your_player(obs).bench:
        score = _active_score(pokemon.id, profile, load_card_db().get(pokemon.id))
        score += len(pokemon.energyCards) * 180
        if pokemon.id in profile.main_attackers:
            score += 500
        if score > best_score:
            best = pokemon
            best_score = score
    return best


def _option_pokemon(obs: Observation, option: Option):
    if option.area not in {AreaType.ACTIVE, AreaType.BENCH}:
        return None
    player_index = obs.current.yourIndex if option.playerIndex is None else option.playerIndex
    player = obs.current.players[player_index]
    if option.area == AreaType.ACTIVE:
        return player.active[0] if player.active else None
    if option.index is not None and 0 <= option.index < len(player.bench):
        return player.bench[option.index]
    return None


def _log_choice(obs: Observation, scored: list[tuple[int, int]], chosen: Iterable[int]) -> None:
    if not os.getenv("POKEMON_AGENT_LOG"):
        return
    options = obs.select.option
    chosen_set = set(chosen)
    top = ", ".join(
        f"{'*' if idx in chosen_set else ''}{idx}:{score}:{_describe_option(obs, options[idx])}"
        for score, idx in scored[:5]
    )
    _log(
        f"turn={obs.current.turn if obs.current else None} "
        f"context={obs.select.context} min={obs.select.minCount} max={obs.select.maxCount} {top}"
    )


def _decision_trace(
    obs: Observation,
    profile: DeckProfile,
    scored: list[tuple[int, int]],
    chosen: Iterable[int],
) -> DecisionTrace:
    chosen_set = set(chosen)
    option_scores = {index: score for score, index in scored}
    options = []
    for index, option in enumerate(obs.select.option):
        card_id = _trace_card_id(obs, option)
        options.append(
            OptionTrace(
                index=index,
                score=option_scores.get(index, 0),
                selected=index in chosen_set,
                option_type=_enum_name(option.type, OptionType),
                context=_enum_name(obs.select.context, SelectContext),
                card_id=card_id,
                card_name=card_name(card_id) if card_id is not None else None,
                description=_describe_option(obs, option),
            )
        )
    return DecisionTrace(
        turn=obs.current.turn if obs.current else None,
        player_index=obs.current.yourIndex if obs.current else None,
        deck_profile=profile.name,
        select_type=_enum_name(obs.select.type, SelectType),
        context=_enum_name(obs.select.context, SelectContext),
        min_count=obs.select.minCount,
        max_count=obs.select.maxCount,
        chosen=list(chosen),
        board=_board_summary(obs),
        options=options,
    )


def _board_summary(obs: Observation) -> dict:
    if obs.current is None:
        return {}
    your = _your_player(obs)
    opponent = _opponent_player(obs)
    return {
        "turn": obs.current.turn,
        "turn_action_count": obs.current.turnActionCount,
        "supporter_played": obs.current.supporterPlayed,
        "energy_attached": obs.current.energyAttached,
        "your_prizes": len(your.prize),
        "opponent_prizes": len(opponent.prize),
        "your_hand_count": your.handCount,
        "opponent_hand_count": opponent.handCount,
        "your_deck_count": your.deckCount,
        "opponent_deck_count": opponent.deckCount,
        "your_active": _pokemon_summary(your.active[0] if your.active else None),
        "opponent_active": _pokemon_summary(opponent.active[0] if opponent.active else None),
        "your_bench": [_pokemon_summary(pokemon) for pokemon in your.bench],
        "opponent_bench": [_pokemon_summary(pokemon) for pokemon in opponent.bench],
    }


def _pokemon_summary(pokemon) -> dict | None:
    if pokemon is None:
        return None
    return {
        "id": pokemon.id,
        "name": card_name(pokemon.id),
        "hp": pokemon.hp,
        "max_hp": pokemon.maxHp,
        "energies": len(pokemon.energyCards),
        "tools": len(pokemon.tools),
    }


def _trace_card_id(obs: Observation, option: Option) -> int | None:
    if option.type == OptionType.ATTACH:
        target = _attach_target_pokemon(obs, option)
        source_id = _source_card_id(obs, option)
        return source_id if source_id is not None else (target.id if target is not None else None)
    return _option_card_id(obs, option)


def _describe_option(obs: Observation, option: Option) -> str:
    option_type = _enum_name(option.type, OptionType)
    if option.type == OptionType.ATTACH:
        source_id = _source_card_id(obs, option)
        target = _attach_target_pokemon(obs, option)
        source = card_name(source_id) if source_id is not None else "unknown"
        target_name = card_name(target.id) if target is not None else "unknown"
        return f"{option_type}:{source}-> {target_name}"
    card_id = _option_card_id(obs, option)
    if card_id is not None:
        return f"{option_type}:{card_name(card_id)}"
    if option.type == OptionType.NUMBER:
        return f"NUMBER:{option.number}"
    return option_type


def _enum_name(value, enum_cls) -> str:
    if hasattr(value, "name"):
        return value.name
    try:
        return enum_cls(value).name
    except ValueError:
        return str(value)


def _log(message: str) -> None:
    if os.getenv("POKEMON_AGENT_LOG"):
        print(f"[heuristic_agent] {message}", file=sys.stderr)
