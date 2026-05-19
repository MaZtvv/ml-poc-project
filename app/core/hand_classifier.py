"""
Poker hand classification, preflop descriptors, and draw detection.

All functions accept cards as lists of (rank_char, suit_char) tuples,
e.g. [('A','h'), ('K','s')], consistent with the PHH parsing format.
"""

from collections import Counter

from core.equity import evaluate_hand, RANK_VAL

_RANK_NAME: dict[int, str] = {
    14: "Ace", 13: "King", 12: "Queen", 11: "Jack", 10: "Ten",
    9: "Nine", 8: "Eight", 7: "Seven", 6: "Six", 5: "Five",
    4: "Four", 3: "Three", 2: "Two",
}

_HAND_RANK_LABEL = {
    0: "High card",
    1: "Pair",
    2: "Two pair",
    3: "Trips",
    4: "Straight",
    5: "Flush",
    6: "Full house",
    7: "Quads",
    8: "Straight flush",
}


# ─── Preflop descriptors ──────────────────────────────────────────────────────

def describe_preflop(hole_cards: list) -> str:
    """
    Return a short human-readable label for a 2-card starting hand.

    Examples:
      ('A','h'), ('K','s') → "Ace-King offsuit"
      ('9','h'), ('9','c') → "Pocket Nines (medium)"
      ('J','s'), ('T','s') → "Jack-Ten suited connector"
    """
    if len(hole_cards) != 2:
        return ""

    (r1, s1), (r2, s2) = hole_cards
    v1, v2 = RANK_VAL.get(r1, 0), RANK_VAL.get(r2, 0)
    if v1 < v2:
        r1, s1, v1, r2, s2, v2 = r2, s2, v2, r1, s1, v1

    is_pair   = (r1 == r2)
    is_suited = (s1 == s2)
    gap       = v1 - v2 - 1  # 0=connector, 1=one-gap, 2=two-gap

    if is_pair:
        n = _RANK_NAME.get(v1, r1)
        if v1 >= 11: return f"Pocket {n}s (premium)"
        if v1 == 10: return "Pocket Tens (medium)"
        if v1 >= 7:  return f"Pocket {n}s (medium)"
        return               f"Pocket {n}s (small pair)"

    sc = "suited" if is_suited else "offsuit"

    conn = ""
    if gap == 0:   conn = "connector"
    elif gap == 1: conn = "one-gap"
    elif gap == 2: conn = "two-gap"

    has_ace  = v1 == 14
    has_king = v1 == 13 and v2 < 10
    both_bw  = v1 >= 10 and v2 >= 10

    if has_ace:
        base = f"Ace-{_RANK_NAME.get(v2, r2)}"
    elif both_bw:
        base = f"{_RANK_NAME.get(v1, r1)}-{_RANK_NAME.get(v2, r2)} (broadway)"
    elif has_king:
        base = f"King-{_RANK_NAME.get(v2, r2)}"
    else:
        base = f"{_RANK_NAME.get(v1, r1)}-{_RANK_NAME.get(v2, r2)}"

    parts = [base, sc]
    if conn:
        parts.append(conn)
    return " ".join(parts)


# ─── Made hand classifier ─────────────────────────────────────────────────────

def classify_made_hand(hole_cards: list, board: list) -> str:
    """
    Return a readable label for the best 5-card hand from hole_cards + board.

    Distinguishes contextually: top/middle/bottom pair, overpair, set vs trips,
    flush high-card, straight high-card, etc.
    Returns "" when board is empty (preflop — no made hand yet).
    """
    if not board or not hole_cards:
        return ""

    score      = evaluate_hand(hole_cards, board)
    cat        = score[0]
    board_vals = sorted([RANK_VAL[r] for r, _ in board], reverse=True)
    hole_vals  = [RANK_VAL[r] for r, _ in hole_cards]
    h_cnt      = Counter(hole_vals)

    n = _RANK_NAME.get

    if cat == 0:
        return f"High card ({n(score[1], '?')})"

    if cat == 1:
        pr = score[1]  # pair rank value
        # Pocket pair situation (both hole cards same rank)
        if len(hole_vals) == 2 and hole_vals[0] == hole_vals[1]:
            if pr > board_vals[0]:
                return f"Overpair ({n(pr,'?')}s)"
            return f"Pair of {n(pr,'?')}s (pocket)"
        # Paired with board
        if pr == board_vals[0]:
            return f"Top pair ({n(pr,'?')}s)"
        if board_vals and pr == board_vals[-1]:
            return f"Bottom pair ({n(pr,'?')}s)"
        return f"Middle pair ({n(pr,'?')}s)"

    if cat == 2:
        return f"Two pair ({n(score[1],'?')}s and {n(score[2],'?')}s)"

    if cat == 3:
        tr = score[1]
        # Set: pocket pair + one board card of same rank
        if h_cnt.get(tr, 0) == 2:
            return f"Set of {n(tr,'?')}s"
        return f"Trips ({n(tr,'?')}s)"

    if cat == 4:
        return f"Straight ({n(score[1],'?')}-high)"

    if cat == 5:
        return f"Flush ({n(score[1],'?')}-high)"

    if cat == 6:
        return f"Full house ({n(score[1],'?')}s full of {n(score[2],'?')}s)"

    if cat == 7:
        return f"Quads ({n(score[1],'?')}s)"

    if cat == 8:
        return "Royal flush" if score[1] == 14 else f"Straight flush ({n(score[1],'?')}-high)"

    return _HAND_RANK_LABEL.get(cat, "?")


# ─── Draw detection ───────────────────────────────────────────────────────────

def detect_draws(hole_cards: list, board: list) -> list[str]:
    """
    Return a list of draw/texture descriptors for hole_cards given board.

    Detects:
      Flush draw, Backdoor flush draw,
      Open-ended straight draw (OESD), Gutshot, Backdoor straight draw,
      Overcard(s) — only when no pair or better exists.
    Returns [] when board is empty or hand is already a made hand of rank ≥ 2
    (draws are irrelevant once you have two pair or better).
    """
    if not board or not hole_cards:
        return []

    all_cards  = list(hole_cards) + list(board)
    n_to_come  = 5 - len(board)
    draws: list[str] = []

    # ── Flush draws ───────────────────────────────────────────────────────────
    suits_all  = Counter(s for _, s in all_cards)
    suits_hole = Counter(s for _, s in hole_cards)

    for suit, total in suits_all.items():
        hc = suits_hole.get(suit, 0)
        if hc >= 1:
            if total == 4 and n_to_come >= 1:
                draws.append("Flush draw")
            elif total == 3 and n_to_come >= 2:
                draws.append("Backdoor flush draw")

    # ── Straight draws ────────────────────────────────────────────────────────
    all_vals  = [RANK_VAL[r] for r, _ in all_cards]
    hole_vals = {RANK_VAL[r] for r, _ in hole_cards}

    # Include Ace as low (rank 1) for wheel straights
    if 14 in all_vals:  all_vals  = all_vals + [1]
    if 14 in hole_vals: hole_vals = hole_vals | {1}

    unique_all  = set(all_vals)

    oesd_found = gutshot_found = bd_str_found = False

    for low in range(1, 11):   # windows A-5 through T-A
        window = set(range(low, low + 5))
        have   = window & unique_all
        missing = window - unique_all
        hole_in_window = window & hole_vals

        if not hole_in_window:   # must use at least one hole card
            continue

        n_have = len(have)
        n_miss = len(missing)

        if n_have == 4 and n_miss == 1 and n_to_come >= 1:
            miss_val = next(iter(missing))
            if miss_val == low or miss_val == low + 4:
                if not oesd_found:
                    draws.append("Open-ended straight draw")
                    oesd_found = True
            else:
                if not gutshot_found:
                    draws.append("Gutshot")
                    gutshot_found = True
        elif n_have == 3 and n_miss == 2 and n_to_come >= 2 and not bd_str_found:
            draws.append("Backdoor straight draw")
            bd_str_found = True

    # ── Overcards (only when high card — no pair yet) ─────────────────────────
    score = evaluate_hand(hole_cards, board)
    if score[0] == 0 and board:
        max_board = max(RANK_VAL[r] for r, _ in board)
        oc = [r for r, _ in hole_cards if RANK_VAL[r] > max_board]
        if len(oc) == 2:
            draws.append("Two overcards")
        elif len(oc) == 1:
            draws.append("Overcard")

    return draws
