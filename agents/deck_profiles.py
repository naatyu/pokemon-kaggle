from dataclasses import dataclass


@dataclass(frozen=True)
class DeckProfile:
    name: str
    deck_path: str
    main_attackers: set[int]
    backup_attackers: set[int]
    setup_basics: set[int]
    evolution_targets: set[int]
    draw_search_cards: set[int]
    disruption_cards: set[int]
    recovery_cards: set[int]
    switch_cards: set[int]
    stadium_cards: set[int]
    energy_targets: set[int]
    expendable_cards: set[int]


HYDRAPPLE = DeckProfile(
    name="hydrapple",
    deck_path="decks/hydrapple_ex_heuristic.csv",
    main_attackers={150, 96},  # Hydrapple ex, Teal Mask Ogerpon ex
    backup_attackers={920, 710, 655},  # Tapu Bulu, Meganium, Celebi
    setup_basics={96, 149, 708, 917, 172},  # Ogerpon, Applin, Chikorita, Hoothoot
    evolution_targets={150, 921, 710, 709, 918, 173},
    draw_search_cards={1227, 1094, 1152, 1121, 1231, 1184},
    disruption_cards={1182, 1080, 1201},
    recovery_cards={1097, 1184},
    switch_cards={1123},
    stadium_cards={1261},
    energy_targets={150, 96, 920, 710, 655, 921},
    expendable_cards={1071, 140},  # Late-game support Pokemon are lower priority early.
)


DRAGAPULT = DeckProfile(
    name="dragapult",
    deck_path="decks/dragapult_ex_heuristic.csv",
    main_attackers={121},
    backup_attackers={112, 235},
    setup_basics={119, 235, 65, 140},
    evolution_targets={121, 120, 66},
    draw_search_cards={1227, 1086, 1152, 1121, 1198, 1240},
    disruption_cards={1182, 1120, 1080},
    recovery_cards={1097, 1240},
    switch_cards={1123},
    stadium_cards={1260},
    energy_targets={121, 120, 112},
    expendable_cards={1071},
)


ABOMASNOW_SAMPLE = DeckProfile(
    name="abomasnow_sample",
    deck_path="decks/mega_abomasnow_sample.csv",
    main_attackers={723},
    backup_attackers={721},
    setup_basics={721, 722},
    evolution_targets={723},
    draw_search_cards={1145, 1205, 1227, 1235},
    disruption_cards=set(),
    recovery_cards=set(),
    switch_cards=set(),
    stadium_cards=set(),
    energy_targets={723, 721},
    expendable_cards=set(),
)


METAL_ARCHALUDON = DeckProfile(
    name="metal_archaludon",
    deck_path="decks/metal_archaludon_public.csv",
    main_attackers={190, 666},
    backup_attackers={57},
    setup_basics={169, 666, 57},
    evolution_targets={190},
    draw_search_cards={1121, 1122, 1147, 1152, 1185, 1227},
    disruption_cards={1182, 1213},
    recovery_cards={1097},
    switch_cards=set(),
    stadium_cards={1244},
    energy_targets={190, 666},
    expendable_cards=set(),
)


PROFILES = {
    HYDRAPPLE.name: HYDRAPPLE,
    DRAGAPULT.name: DRAGAPULT,
    ABOMASNOW_SAMPLE.name: ABOMASNOW_SAMPLE,
    METAL_ARCHALUDON.name: METAL_ARCHALUDON,
}
