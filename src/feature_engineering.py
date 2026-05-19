from pathlib import Path
import re

import pandas as pd

MAX_FILES = 300
SOURCE_FILTER = "pluribus"


def find_phh_files(data_dir, source_filter=SOURCE_FILTER, max_files=MAX_FILES):
    data_path = Path(data_dir)
    files = sorted([path for path in data_path.rglob("*") if path.suffix.lower() in {".phh", ".phhs"}])

    if source_filter is not None:
        source_filter_lower = str(source_filter).lower()
        files = [path for path in files if source_filter_lower in str(path).lower()]

    if max_files is not None:
        files = files[:max_files]

    return files


def read_text_file(path):
    return Path(path).read_text(encoding="utf-8")


def extract_variant(text):
    match = re.search(r"""^\s*variant\s*=\s*['"]([^'"]+)['"]""", text, re.MULTILINE)
    return match.group(1) if match else ""


def extract_actions_block(text):
    match = re.search(r"""^\s*actions\s*=\s*(\[[^\]]*\])""", text, re.MULTILINE | re.DOTALL)
    if not match:
        return []
    return re.findall(r"""['"]([^'"]*)['"]""", match.group(1))


def extract_action_counts(actions):
    counts = {
        "num_hole_deals": 0,
        "num_board_deals": 0,
        "num_folds": 0,
        "num_check_calls": 0,
        "num_bets_raises": 0,
        "num_stand_pat_or_discard": 0,
        "num_show_or_muck": 0,
    }

    for action in actions:
        if action.startswith("d dh"):
            counts["num_hole_deals"] += 1
        elif action.startswith("d db"):
            counts["num_board_deals"] += 1
        elif re.match(r"""^p\d+\s+f\b""", action):
            counts["num_folds"] += 1
        elif re.match(r"""^p\d+\s+cc\b""", action):
            counts["num_check_calls"] += 1
        elif re.match(r"""^p\d+\s+cbr\b""", action):
            counts["num_bets_raises"] += 1
        elif re.match(r"""^p\d+\s+sd\b""", action):
            counts["num_stand_pat_or_discard"] += 1
        elif re.match(r"""^p\d+\s+sm\b""", action):
            counts["num_show_or_muck"] += 1

    return counts


def estimate_player_count(actions):
    player_ids = {int(match) for action in actions for match in re.findall(r"""\bp(\d+)\b""", action)}
    return max(player_ids) if player_ids else 0


def split_text_into_hand_sections(text, suffix):
    if suffix.lower() == ".phhs":
        matches = list(re.finditer(r"""^\[(\d+)\]\s*$""", text, re.MULTILINE))
        if matches:
            sections = []
            for index, match in enumerate(matches):
                section_start = match.end()
                section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                section_id = match.group(1)
                section_text = text[section_start:section_end].strip()
                if section_text:
                    sections.append((section_id, section_text))
            return sections
    return [("", text)]


def extract_simple_field(text, field_name):
    match = re.search(rf"""^\s*{re.escape(field_name)}\s*=\s*(.+)$""", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def extract_hand_id(text):
    match = re.search(r"""^\s*hand\s*=\s*(\d+)""", text, re.MULTILINE)
    return match.group(1) if match else ""


def parse_list_count(raw_value):
    if not raw_value:
        return 0
    stripped = raw_value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return 0
    inner = stripped[1:-1].strip()
    if not inner:
        return 0
    return len([item for item in inner.split(",") if item.strip()])


def detect_likely_dataset_source(path, size_kb):
    path_text = str(path).lower()
    if "wsop" in path_text:
        return "wsop"
    if "pluribus" in path_text:
        return "pluribus"
    if "handhq" in path_text:
        return "handhq"
    if size_kb < 5 or "wikipedia" in path_text or "alice-carol" in path_text or "badugi" in path_text:
        return "example"
    return "unknown"


def build_hand_level_features(data_dir="data/raw", source_filter=SOURCE_FILTER, max_files=MAX_FILES):
    rows = []
    data_path = Path(data_dir)
    source_files = find_phh_files(data_path, source_filter=source_filter, max_files=max_files)

    for index, path in enumerate(source_files, start=1):
        text = read_text_file(path)
        source_size_bytes = path.stat().st_size
        source_size_kb = round(source_size_bytes / 1024, 2)
        likely_dataset_source = detect_likely_dataset_source(path, source_size_kb)

        for hand_section_id, section_text in split_text_into_hand_sections(text, path.suffix):
            section_lines = section_text.splitlines()
            section_non_empty_line_count = sum(1 for line in section_lines if line.strip())
            variant = extract_variant(section_text)
            hand_id = extract_hand_id(section_text)
            players_raw = extract_simple_field(section_text, "players")
            starting_stacks_raw = extract_simple_field(section_text, "starting_stacks")
            finishing_stacks_raw = extract_simple_field(section_text, "finishing_stacks")
            player_count_metadata = parse_list_count(players_raw)
            starting_stack_count = parse_list_count(starting_stacks_raw)
            finishing_stack_count = parse_list_count(finishing_stacks_raw)
            actions = extract_actions_block(section_text)
            action_counts = extract_action_counts(actions)
            player_count_estimated = estimate_player_count(actions)
            has_actions_block = "actions" in section_text and "[" in section_text
            has_starting_stacks = bool(starting_stacks_raw)
            has_finishing_stacks = bool(finishing_stacks_raw)
            has_players_field = bool(players_raw)
            has_hole_dealing = action_counts["num_hole_deals"] > 0
            has_board_dealing = action_counts["num_board_deals"] > 0
            has_showdown_or_muck = action_counts["num_show_or_muck"] > 0
            is_no_limit_texas_holdem = variant == "NT"
            is_likely_texas_holdem = is_no_limit_texas_holdem or has_board_dealing
            is_small_example = source_size_kb < 5
            usable_for_first_model = has_actions_block and len(actions) > 5 and has_hole_dealing and (has_finishing_stacks or has_showdown_or_muck)

            rows.append(
                {
                    "source_file_name": path.name,
                    "source_relative_path": str(path.relative_to(data_path)),
                    "hand_section_id": hand_section_id,
                    "parent_folder": str(path.parent.relative_to(data_path)),
                    "likely_dataset_source": likely_dataset_source,
                    "size_bytes_source_file": source_size_bytes,
                    "section_line_count": len(section_lines),
                    "section_non_empty_line_count": section_non_empty_line_count,
                    "variant": variant,
                    "hand_id": hand_id,
                    "players_raw": players_raw,
                    "player_count_metadata": player_count_metadata,
                    "player_count_estimated": player_count_estimated,
                    "starting_stacks_raw": starting_stacks_raw,
                    "finishing_stacks_raw": finishing_stacks_raw,
                    "starting_stack_count": starting_stack_count,
                    "finishing_stack_count": finishing_stack_count,
                    "has_actions_block": has_actions_block,
                    "action_count": len(actions),
                    "has_starting_stacks": has_starting_stacks,
                    "has_finishing_stacks": has_finishing_stacks,
                    "has_players_field": has_players_field,
                    "has_hole_dealing": has_hole_dealing,
                    "has_board_dealing": has_board_dealing,
                    "has_showdown_or_muck": has_showdown_or_muck,
                    "num_hole_deals": action_counts["num_hole_deals"],
                    "num_board_deals": action_counts["num_board_deals"],
                    "num_folds": action_counts["num_folds"],
                    "num_check_calls": action_counts["num_check_calls"],
                    "num_bets_raises": action_counts["num_bets_raises"],
                    "num_stand_pat_or_discard": action_counts["num_stand_pat_or_discard"],
                    "num_show_or_muck": action_counts["num_show_or_muck"],
                    "is_no_limit_texas_holdem": is_no_limit_texas_holdem,
                    "is_likely_texas_holdem": is_likely_texas_holdem,
                    "is_small_example": is_small_example,
                    "usable_for_first_model": usable_for_first_model,
                }
            )

        if index % 50 == 0:
            print(f"Processed {index} source files")

    return pd.DataFrame(rows)


def save_hand_level_features(output_path="data/processed/hand_level_features.csv", data_dir="data/raw", source_filter=SOURCE_FILTER, max_files=MAX_FILES):
    hand_level_features_df = build_hand_level_features(data_dir=data_dir, source_filter=source_filter, max_files=max_files)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hand_level_features_df.to_csv(output_file, index=False)
    return hand_level_features_df, output_file


if __name__ == "__main__":
    source_files = find_phh_files("data/raw", source_filter=SOURCE_FILTER, max_files=MAX_FILES)
    hand_level_features_df, output_file = save_hand_level_features(source_filter=SOURCE_FILTER, max_files=MAX_FILES)
    print(f"Source files scanned: {len(source_files)}")
    print(f"Hand rows generated: {len(hand_level_features_df)}")
    print(f"Output path: {output_file}")
    print(f"DataFrame shape: {hand_level_features_df.shape}")
    print("Variant counts:")
    print(hand_level_features_df["variant"].replace("", "unknown").value_counts(dropna=False))
    print("likely_dataset_source counts:")
    print(hand_level_features_df["likely_dataset_source"].value_counts(dropna=False))
    print("usable_for_first_model counts:")
    print(hand_level_features_df["usable_for_first_model"].value_counts(dropna=False))
