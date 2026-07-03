from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARD_DATA = PROJECT_ROOT / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv"


@dataclass(frozen=True)
class CardInfo:
    id: int
    name: str
    kind: str
    hp: int
    attack_damage: int

    @property
    def is_pokemon(self) -> bool:
        return "Pokémon" in self.kind or "Pokemon" in self.kind

    @property
    def is_basic_energy(self) -> bool:
        return self.kind == "Basic Energy"

    @property
    def is_energy(self) -> bool:
        return "Energy" in self.kind


@cache
def load_card_db() -> dict[int, CardInfo]:
    if not CARD_DATA.exists():
        return _load_card_db_from_engine()

    cards: dict[int, CardInfo] = {}
    with CARD_DATA.open(newline="") as file:
        for row in csv.DictReader(file):
            card_id = int(row["Card ID"])
            hp = int(row["HP"]) if row["HP"].isdigit() else 0
            damage_text = row["Damage"].replace("×", "").replace("+", "")
            attack_damage = int(damage_text) if damage_text.isdigit() else 0
            cards[card_id] = CardInfo(
                id=card_id,
                name=row["Card Name"],
                kind=row["Stage (Pokémon)/Type (Energy and Trainer)"],
                hp=hp,
                attack_damage=attack_damage,
            )
    return cards


def _load_card_db_from_engine() -> dict[int, CardInfo]:
    from cg.api import CardType, all_attack, all_card_data

    attacks = {attack.attackId: attack for attack in all_attack()}
    cards: dict[int, CardInfo] = {}
    for card in all_card_data():
        damage = 0
        for attack_id in card.attacks:
            attack = attacks.get(attack_id)
            if attack is not None:
                damage = max(damage, attack.damage)
        if card.cardType == CardType.POKEMON:
            kind = "Pokemon"
        elif card.cardType == CardType.BASIC_ENERGY:
            kind = "Basic Energy"
        elif card.cardType == CardType.SPECIAL_ENERGY:
            kind = "Special Energy"
        else:
            kind = card.cardType.name.title()
        cards[card.cardId] = CardInfo(
            id=card.cardId,
            name=card.name,
            kind=kind,
            hp=card.hp,
            attack_damage=damage,
        )
    return cards


def card_name(card_id: int | None) -> str:
    if card_id is None:
        return "unknown"
    return load_card_db().get(card_id, CardInfo(card_id, f"Card {card_id}", "", 0, 0)).name
