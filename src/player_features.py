from pathlib import Path
import re

import numpy as np
import pandas as pd


RANK_TO_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


def find_pluribus_phh_files(data_dir="data/raw/pluribus", max_files=1000):
    files = sorted(Path(data_dir).rglob("*.phh"))
    if max_files is not None:
        files = files[:max_files]
    return files


def parse_list_field(text, field_name):
    match = re.search(rf"""^\s*{re.escape(field_name)}\s*=\s*(\[[^\n]*\])""", text, re.MULTILINE)
    if not match:
        return []
    raw_value = match.group(1).strip()
    inner = raw_value[1:-1].strip()
    if not inner:
        return []
    return [item.strip().strip("'\"") for item in inner.split(",")]


def parse_scalar_field(text, field_name):
    match = re.search(rf"""^\s*{re.escape(field_name)}\s*=\s*(.+)$""", text, re.MULTILINE)
    return match.group(1).strip().strip("'\"") if match else ""


def parse_actions(text):
    match = re.search(r"""^\s*actions\s*=\s*(\[[^\]]*\])""", text, re.MULTILINE | re.DOTALL)
    if not match:
        return []
    return re.findall(r"""['"]([^'"]*)['"]""", match.group(1))


def parse_hole_card_map(actions):
    hole_card_map = {}
    for action in actions:
        match = re.match(r"""^d dh p(\d+)\s+([^\s]+)$""", action)
        if match:
            hole_card_map[int(match.group(1))] = match.group(2)
    return hole_card_map


def split_hole_cards(hole_cards):
    if not hole_cards or "?" in hole_cards or len(hole_cards) != 4:
        return (np.nan, np.nan, np.nan, np.nan)
    return (hole_cards[0], hole_cards[1], hole_cards[2], hole_cards[3])


def rank_value(rank):
    return RANK_TO_VALUE.get(rank, np.nan)


def compute_card_features(hole_cards):
    card_1_rank_raw, card_1_suit, card_2_rank_raw, card_2_suit = split_hole_cards(hole_cards)

    if pd.isna(card_1_rank_raw) or pd.isna(card_2_rank_raw):
        return {
            "card_1_rank": np.nan,
            "card_1_suit": np.nan,
            "card_2_rank": np.nan,
            "card_2_suit": np.nan,
            "is_pair": np.nan,
            "is_suited": np.nan,
            "high_card_rank": np.nan,
            "low_card_rank": np.nan,
            "rank_gap": np.nan,
            "has_ace": np.nan,
            "has_king": np.nan,
            "preflop_strength_score": np.nan,
        }

    card_1_rank = rank_value(card_1_rank_raw)
    card_2_rank = rank_value(card_2_rank_raw)
    is_pair = int(card_1_rank == card_2_rank)
    is_suited = int(card_1_suit == card_2_suit)
    high_card_rank = max(card_1_rank, card_2_rank)
    low_card_rank = min(card_1_rank, card_2_rank)
    rank_gap = abs(card_1_rank - card_2_rank)
    has_ace = int(card_1_rank == 14 or card_2_rank == 14)
    has_king = int(card_1_rank == 13 or card_2_rank == 13)
    preflop_strength_score = high_card_rank * 2 + (10 if is_pair else 0) + (3 if is_suited else 0) + (2 if rank_gap <= 1 else 0) + (5 if has_ace else 0)

    return {
        "card_1_rank": card_1_rank,
        "card_1_suit": card_1_suit,
        "card_2_rank": card_2_rank,
        "card_2_suit": card_2_suit,
        "is_pair": is_pair,
        "is_suited": is_suited,
        "high_card_rank": high_card_rank,
        "low_card_rank": low_card_rank,
        "rank_gap": rank_gap,
        "has_ace": has_ace,
        "has_king": has_king,
        "preflop_strength_score": preflop_strength_score,
    }


def count_hand_actions(actions):
    return {
        "hand_action_count": len(actions),
        "num_folds": sum(bool(re.match(r"""^p\d+\s+f\b""", action)) for action in actions),
        "num_check_calls": sum(bool(re.match(r"""^p\d+\s+cc\b""", action)) for action in actions),
        "num_bets_raises": sum(bool(re.match(r"""^p\d+\s+cbr\b""", action)) for action in actions),
        "num_hole_deals": sum(action.startswith("d dh") for action in actions),
        "num_board_deals": sum(action.startswith("d db") for action in actions),
        "num_show_or_muck": sum(bool(re.match(r"""^p\d+\s+sm\b""", action)) for action in actions),
    }


def count_player_actions(actions, player_index):
    prefix = f"p{player_index}"
    player_actions = [action for action in actions if action.startswith(prefix)]
    return {
        "player_num_folds": sum(bool(re.match(rf"""^{prefix}\s+f\b""", action)) for action in player_actions),
        "player_num_check_calls": sum(bool(re.match(rf"""^{prefix}\s+cc\b""", action)) for action in player_actions),
        "player_num_bets_raises": sum(bool(re.match(rf"""^{prefix}\s+cbr\b""", action)) for action in player_actions),
        "player_num_show_or_muck": sum(bool(re.match(rf"""^{prefix}\s+sm\b""", action)) for action in player_actions),
    }


def parse_number(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def parse_phh_hand(text, relative_path):
    players = parse_list_field(text, "players")
    starting_stacks = [parse_number(value) for value in parse_list_field(text, "starting_stacks")]
    finishing_stacks = [parse_number(value) for value in parse_list_field(text, "finishing_stacks")]
    blinds_or_straddles = [parse_number(value) for value in parse_list_field(text, "blinds_or_straddles")]
    actions = parse_actions(text)

    return {
        "source_relative_path": str(relative_path),
        "hand_id": parse_scalar_field(text, "hand"),
        "variant": parse_scalar_field(text, "variant"),
        "players": players,
        "starting_stacks": starting_stacks,
        "finishing_stacks": finishing_stacks,
        "blinds_or_straddles": blinds_or_straddles,
        "actions": actions,
        "hole_card_map": parse_hole_card_map(actions),
        "hand_action_counts": count_hand_actions(actions),
    }


def extract_player_rows(parsed_hand):
    players = parsed_hand["players"]
    starting_stacks = parsed_hand["starting_stacks"]
    finishing_stacks = parsed_hand["finishing_stacks"]
    blinds_or_straddles = parsed_hand["blinds_or_straddles"]
    actions = parsed_hand["actions"]
    hole_card_map = parsed_hand["hole_card_map"]
    hand_action_counts = parsed_hand["hand_action_counts"]

    if not players:
        return []

    positive_blinds = [value for value in blinds_or_straddles if pd.notna(value) and value > 0]
    small_blind_value = min(positive_blinds) if positive_blinds else np.nan
    big_blind_value = max(positive_blinds) if positive_blinds else np.nan
    number_of_players = len(players)
    rows = []

    for index, player_name in enumerate(players, start=1):
        starting_stack = starting_stacks[index - 1] if index - 1 < len(starting_stacks) else np.nan
        finishing_stack = finishing_stacks[index - 1] if index - 1 < len(finishing_stacks) else np.nan
        blind_or_straddle = blinds_or_straddles[index - 1] if index - 1 < len(blinds_or_straddles) else np.nan
        profit = finishing_stack - starting_stack if pd.notna(starting_stack) and pd.notna(finishing_stack) else np.nan
        player_won = int(profit > 0) if pd.notna(profit) else np.nan
        hole_cards = hole_card_map.get(index, "unknown")
        card_features = compute_card_features(hole_cards)
        player_action_counts = count_player_actions(actions, index)

        row = {
            "source_relative_path": parsed_hand["source_relative_path"],
            "hand_id": parsed_hand["hand_id"],
            "player_index": index,
            "player_name": player_name,
            "variant": parsed_hand["variant"],
            "number_of_players": number_of_players,
            "starting_stack": starting_stack,
            "finishing_stack": finishing_stack,
            "profit": profit,
            "player_won": player_won,
            "blind_or_straddle": blind_or_straddle,
            "is_small_blind": int(pd.notna(blind_or_straddle) and pd.notna(small_blind_value) and blind_or_straddle == small_blind_value and blind_or_straddle > 0),
            "is_big_blind": int(pd.notna(blind_or_straddle) and pd.notna(big_blind_value) and blind_or_straddle == big_blind_value and blind_or_straddle > 0),
            "player_position_index": index,
            "hole_cards": hole_cards if "?" not in hole_cards else "unknown",
        }
        row.update(card_features)
        row.update(hand_action_counts)
        row.update(player_action_counts)
        rows.append(row)

    return rows


def build_player_level_features(data_dir="data/raw/pluribus", max_files=1000):
    rows = []
    source_files = find_pluribus_phh_files(data_dir=data_dir, max_files=max_files)

    for path in source_files:
        text = path.read_text(encoding="utf-8")
        parsed_hand = parse_phh_hand(text, path.relative_to(Path(data_dir).parent))
        rows.extend(extract_player_rows(parsed_hand))

    return pd.DataFrame(rows)


def save_player_level_features(output_path="data/processed/player_level_features.csv", data_dir="data/raw/pluribus", max_files=1000):
    player_level_features_df = build_player_level_features(data_dir=data_dir, max_files=max_files)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    player_level_features_df.to_csv(output_file, index=False)
    return player_level_features_df, output_file


if __name__ == "__main__":
    source_files = find_pluribus_phh_files()
    player_level_features_df, output_file = save_player_level_features()
    print(f"Files processed: {len(source_files)}")
    print(f"Player rows generated: {len(player_level_features_df)}")
    print(f"Output path: {output_file}")
    print("Target distribution of player_won:")
    print(player_level_features_df["player_won"].value_counts(dropna=False))
    print(f"DataFrame shape: {player_level_features_df.shape}")
