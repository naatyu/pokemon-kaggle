from __future__ import annotations

import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SUBMISSION = PROJECT_ROOT / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission"
if str(SAMPLE_SUBMISSION) not in sys.path:
    sys.path.append(str(SAMPLE_SUBMISSION))

from cg.api import to_observation_class  # noqa: E402

from agents.deck_loader import load_deck  # noqa: E402
from agents.deck_profiles import PROFILES  # noqa: E402
from agents.heuristic_agent import choose_action, fallback_selection  # noqa: E402


class Opponent:
    def deck(self) -> list[int]:
        raise NotImplementedError

    def act(self, obs_dict: dict) -> list[int]:
        raise NotImplementedError


@dataclass
class RandomOpponent(Opponent):
    deck_path: Path

    def deck(self) -> list[int]:
        return [int(line.strip()) for line in self.deck_path.read_text().splitlines() if line.strip()]

    def act(self, obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return self.deck()
        if obs.select.maxCount == 0 or len(obs.select.option) == 0:
            return []
        count = obs.select.minCount if obs.select.minCount > 0 else min(1, obs.select.maxCount)
        count = min(count, len(obs.select.option))
        return random.sample(range(len(obs.select.option)), count)


@dataclass
class HeuristicOpponent(Opponent):
    deck_name: str

    def deck(self) -> list[int]:
        return load_deck(PROJECT_ROOT / PROFILES[self.deck_name].deck_path)

    def act(self, obs_dict: dict) -> list[int]:
        profile = PROFILES[self.deck_name]
        obs = to_observation_class(obs_dict)
        try:
            return choose_action(obs, profile)
        except Exception:
            return fallback_selection(obs)


class FolderOpponent(Opponent):
    def __init__(self, path: Path):
        self.path = path
        main_py = path / "main.py"
        if not main_py.exists():
            raise FileNotFoundError(main_py)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        spec = importlib.util.spec_from_file_location(f"rl_opponent_{path.name}", main_py)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {main_py}")
        module = importlib.util.module_from_spec(spec)
        old_cwd = Path.cwd()
        try:
            os.chdir(path)
            spec.loader.exec_module(module)
        finally:
            os.chdir(old_cwd)
        self.module = module

    def deck(self) -> list[int]:
        return self._call({"select": None, "logs": [], "current": None})

    def act(self, obs_dict: dict) -> list[int]:
        return self._call(obs_dict)

    def _call(self, obs_dict: dict) -> list[int]:
        old_cwd = Path.cwd()
        try:
            os.chdir(self.path)
            return self.module.agent(obs_dict)
        finally:
            os.chdir(old_cwd)


def make_opponent(name: str) -> Opponent:
    if name == "random_abomasnow":
        return RandomOpponent(PROJECT_ROOT / "decks" / "mega_abomasnow_sample.csv")
    if name == "random_hydrapple":
        return RandomOpponent(PROJECT_ROOT / "decks" / "hydrapple_ex_heuristic.csv")
    if name == "heuristic_hydrapple":
        return HeuristicOpponent("hydrapple")
    if name == "heuristic_dragapult":
        return HeuristicOpponent("dragapult")
    arena_path = PROJECT_ROOT / "arena" / "agents" / name
    if arena_path.exists():
        return FolderOpponent(arena_path)
    raise ValueError(f"Unknown opponent: {name}")
