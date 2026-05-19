import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PLAYER_FEATURE_FILE = PROCESSED_DATA_DIR / "player_level_features.csv"

_REQUIRED_COLUMNS = [
    "player_won",
    "preflop_strength_score",
    "starting_stack",
    "hand_id",
    "source_relative_path",
]


@st.cache_data(show_spinner="Chargement des données...")
def load_player_features() -> pd.DataFrame:
    if not PLAYER_FEATURE_FILE.exists() or PLAYER_FEATURE_FILE.stat().st_size < 1024:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.player_features import save_player_level_features

        df, _ = save_player_level_features(
            output_path=str(PLAYER_FEATURE_FILE),
            data_dir=str(RAW_DATA_DIR / "pluribus"),
            max_files=1000,
        )
    else:
        df = pd.read_csv(PLAYER_FEATURE_FILE)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier de données : {missing}")

    # Composite key: source file + hand number — globally unique per hand
    df["composite_id"] = (
        df["source_relative_path"].astype(str) + "::" + df["hand_id"].astype(str)
    )

    return df


@st.cache_data(show_spinner=False)
def parse_hand_replay(source_relative_path: str) -> dict:
    """
    Parse a PHH file into structured hand data for replay.

    Returns a dict with:
      positions        — {player_idx: "SB"/"BB"/"UTG"/"HJ"/"CO"/"BTN"}
      hole_cards       — {player_idx: [(rank, suit), ...]}
      board_by_street  — {"flop": [...], "turn": [...], "river": [...]}
      actions_by_street— {"preflop": [...], "flop": [...], ...}
                          each action: {player_idx, position, action_type, amount}
      folded_at_street — {player_idx: street_name}
      folded_by_street — {street: set(player_idx)} — cumulative per street
    Returns {} if file not found or unparseable.
    """
    phh_path = RAW_DATA_DIR / source_relative_path
    if not phh_path.exists():
        return {}

    content = phh_path.read_text()
    match = re.search(r"actions\s*=\s*\[([^\]]*)\]", content, re.DOTALL)
    if not match:
        return {}

    actions_raw = re.findall(r"'([^']*)'", match.group(1))

    _POSITIONS_6 = {1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO", 6: "BTN"}

    hole_cards: dict = {}
    board_by_street: dict = {"flop": [], "turn": [], "river": []}
    actions_by_street: dict = {"preflop": [], "flop": [], "turn": [], "river": []}
    folded_at_street: dict = {}

    current_street = "preflop"
    board_deal_count = 0

    for raw in actions_raw:
        parts = raw.split()
        if not parts:
            continue

        if parts[0] == "d" and len(parts) >= 3:
            if parts[1] == "dh" and len(parts) >= 4:
                p_idx = int(parts[2][1:])
                cards_str = parts[3]
                hole_cards[p_idx] = [
                    (cards_str[i], cards_str[i + 1])
                    for i in range(0, len(cards_str) - 1, 2)
                ]
            elif parts[1] == "db" and len(parts) >= 3:
                cards_str = parts[2]
                new_cards = [
                    (cards_str[i], cards_str[i + 1])
                    for i in range(0, len(cards_str) - 1, 2)
                ]
                board_deal_count += 1
                if board_deal_count == 1:
                    board_by_street["flop"] = new_cards
                    current_street = "flop"
                elif board_deal_count == 2:
                    board_by_street["turn"] = new_cards
                    current_street = "turn"
                elif board_deal_count == 3:
                    board_by_street["river"] = new_cards
                    current_street = "river"

        elif parts[0].startswith("p") and len(parts[0]) > 1 and parts[0][1:].isdigit():
            p_idx = int(parts[0][1:])
            if len(parts) < 2:
                continue
            code = parts[1]
            amount = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None

            if code == "f":
                action_type = "fold"
                if p_idx not in folded_at_street:
                    folded_at_street[p_idx] = current_street
            elif code == "cc":
                action_type = "call"
            elif code == "cbr":
                action_type = "raise"
            else:
                action_type = code

            actions_by_street[current_street].append({
                "player_idx": p_idx,
                "position": _POSITIONS_6.get(p_idx, f"p{p_idx}"),
                "action_type": action_type,
                "amount": amount,
            })

    # Cumulative folded sets — each street includes all prior folds
    streets = ["preflop", "flop", "turn", "river"]
    folded_by_street: dict = {}
    cumulative: set = set()
    for street in streets:
        for p_idx, fold_street in folded_at_street.items():
            if fold_street == street:
                cumulative.add(p_idx)
        folded_by_street[street] = set(cumulative)

    # Parse hand metadata for pot/stack tracking
    def _parse_int_list(field: str) -> list[int]:
        m = re.search(rf"{field}\s*=\s*\[([^\]]*)\]", content)
        if not m:
            return []
        return [int(x.strip()) for x in m.group(1).split(",") if x.strip().lstrip("-").isdigit()]

    blinds_raw  = _parse_int_list("blinds_or_straddles")
    antes_raw   = _parse_int_list("antes")
    stacks_raw  = _parse_int_list("starting_stacks")
    big_blind   = max(blinds_raw) if blinds_raw else 100

    return {
        "positions":       _POSITIONS_6,
        "hole_cards":      hole_cards,
        "board_by_street": board_by_street,
        "actions_by_street": actions_by_street,
        "folded_at_street": folded_at_street,
        "folded_by_street": folded_by_street,
        "blinds":          blinds_raw,
        "antes":           antes_raw,
        "starting_stacks": stacks_raw,
        "big_blind":       big_blind,
    }


@st.cache_data(show_spinner=False)
def parse_board_cards(source_relative_path: str) -> list[tuple[str, str]]:
    """
    Extract board cards (flop, turn, river) from a raw PHH file.
    Returns a list of (rank, suit) tuples, e.g. [('7','d'),('5','h'),('9','d')].
    Returns [] if the file is not found or has no board deals.
    """
    phh_path = RAW_DATA_DIR / source_relative_path
    if not phh_path.exists():
        return []

    content = phh_path.read_text()

    match = re.search(r"actions\s*=\s*\[([^\]]*)\]", content, re.DOTALL)
    if not match:
        return []

    actions = re.findall(r"'([^']*)'", match.group(1))
    board = []
    for action in actions:
        if action.startswith("d db "):
            cards_str = action[5:]  # e.g. "7d5h9d" (flop) or "7c" (turn/river)
            for i in range(0, len(cards_str) - 1, 2):
                board.append((cards_str[i], cards_str[i + 1]))
    return board
