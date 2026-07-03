# Starter decks

These decks are starting points for the Kaggle Pokemon TCG AI Battle Challenge.
Each `*.csv` file contains exactly 60 competition card IDs, one per line, matching
the sample submission format.

The source lists are based on current Limitless TCG Standard results checked on
2026-07-03:

- Dragapult ex: high meta share and repeated NAIC 2026 placements.
- Hydrapple ex: lower meta share than Dragapult, but more linear for a first
  heuristic agent because it mostly follows a Grass-energy acceleration plan.

`Special Red Card` appears in the Limitless lists but is not present in the
competition `EN_Card_Data.csv`, so both starter lists replace it with `Switch`.

## dragapult_ex_heuristic.csv

Based on Justin Newdorf's 3rd place NAIC 2026 Dragapult list.

Pokemon:

- 4 Dreepy
- 4 Drakloak
- 3 Dragapult ex
- 2 Munkidori
- 1 Dunsparce
- 1 Dudunsparce
- 1 Budew
- 1 Fezandipiti ex
- 1 Meowth ex

Trainer:

- 4 Lillie's Determination
- 3 Boss's Orders
- 2 Crispin
- 1 Rosa's Encouragement
- 4 Buddy-Buddy Poffin
- 4 Poke Pad
- 4 Ultra Ball
- 4 Crushing Hammer
- 3 Night Stretcher
- 1 Switch
- 1 Unfair Stamp
- 2 Risky Ruins

Energy:

- 4 Basic Psychic Energy
- 3 Basic Fire Energy
- 2 Basic Darkness Energy

## hydrapple_ex_heuristic.csv

Based on Grant Walworth's 25th place NAIC 2026 Hydrapple list.

Pokemon:

- 4 Teal Mask Ogerpon ex
- 2 Applin
- 2 Dipplin
- 2 Hydrapple ex
- 2 Chikorita
- 2 Bayleef
- 2 Meganium
- 1 Hoothoot
- 1 Noctowl
- 1 Meowth ex
- 1 Fezandipiti ex
- 1 Celebi
- 1 Tapu Bulu

Trainer:

- 4 Lillie's Determination
- 2 Boss's Orders
- 2 Dawn
- 1 Briar
- 1 Lana's Aid
- 4 Bug Catching Set
- 2 Poke Pad
- 2 Ultra Ball
- 1 Night Stretcher
- 1 Unfair Stamp
- 1 Switch
- 3 Forest of Vitality

Energy:

- 14 Basic Grass Energy

## mega_abomasnow_sample.csv

This is the official sample-submission deck copied into `decks/` so it can be
used as a public baseline opponent in local matchup runs.

Pokemon:

- 2 Kyogre
- 4 Snover
- 4 Mega Abomasnow ex

Trainer:

- 1 Maximum Belt
- 4 Mega Signal
- 2 Cyrano
- 4 Lillie's Determination
- 4 Waitress

Energy:

- 39 Basic Water Energy
