from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECK = PROJECT_ROOT / "decks" / "hydrapple_ex_heuristic.csv"


def load_deck(path: str | Path = DEFAULT_DECK) -> list[int]:
    """Load a 60-card competition deck CSV."""
    deck_path = Path(path)
    cards = [int(line.strip()) for line in deck_path.read_text().splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"{deck_path} must contain exactly 60 card IDs, found {len(cards)}.")
    return cards
