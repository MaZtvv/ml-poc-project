"""
Texas Hold'em hand equity calculator.

Uses exact enumeration for turn/river (fast), Monte Carlo for preflop/flop.
No external poker libraries — pure Python.
"""

from collections import Counter
from itertools import combinations
from math import comb
import random

RANK_VAL: dict[str, int] = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

_ALL_RANKS = "23456789TJQKA"
_ALL_SUITS = "hdsc"
_FULL_DECK: list[tuple[str, str]] = [(r, s) for r in _ALL_RANKS for s in _ALL_SUITS]


def _score_5(cards: list[tuple[str, str]]) -> tuple:
    """Comparable hand score for exactly 5 cards. Higher tuple = better hand."""
    ranks = sorted([RANK_VAL[r] for r, _ in cards], reverse=True)
    suits = [s for _, s in cards]
    rc    = Counter(ranks)

    is_flush    = len(set(suits)) == 1
    uniq        = sorted(set(ranks))
    is_straight = len(uniq) == 5 and uniq[-1] - uniq[0] == 4
    is_wheel    = uniq == [2, 3, 4, 5, 14]          # A-2-3-4-5
    str_high    = ranks[0] if is_straight else (5 if is_wheel else 0)
    is_straight = is_straight or is_wheel

    if is_straight and is_flush:
        return (8, str_high)

    # Sort groups: descending by count, then by rank within same count
    groups  = sorted(rc.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ordered = tuple(r for r, _ in groups)
    counts  = [c for _, c in groups]

    if counts[0] == 4:          return (7,) + ordered           # quads
    if counts[:2] == [3, 2]:    return (6,) + ordered           # full house
    if is_flush:                return (5,) + tuple(ranks)      # flush
    if is_straight:             return (4, str_high)            # straight
    if counts[0] == 3:          return (3,) + ordered           # trips
    if counts[:2] == [2, 2]:    return (2,) + ordered           # two pair
    if counts[0] == 2:          return (1,) + ordered           # one pair
    return                             (0,) + tuple(ranks)      # high card


def _best_hand(cards: list[tuple[str, str]]) -> tuple:
    """Best 5-card hand score from 5, 6, or 7 cards."""
    if len(cards) <= 5:
        return _score_5(cards)
    return max(_score_5(list(c)) for c in combinations(cards, 5))


def compute_equity(
    hole_cards: dict,
    board_cards: list,
    n_samples: int = 1200,
    exact_threshold: int = 80_000,
) -> dict:
    """
    Return {player_idx: equity_float} summing to 1.0.

    hole_cards    : {player_idx: [(rank, suit), (rank, suit)]} — active players only
    board_cards   : community cards already revealed (0, 3, 4, or 5 cards)
    n_samples     : Monte Carlo iterations when exact enumeration is too expensive
    exact_threshold: use exact if C(remaining, n_to_deal) ≤ this value

    River  → always exact (n_to_deal == 0)
    Turn   → exact  (C(~44, 1) ≈ 44)
    Flop   → exact  (C(~45, 2) ≈ 990)
    Preflop→ Monte Carlo (C(~40, 5) ≈ 658 000)
    """
    if not hole_cards:
        return {}

    players = list(hole_cards.keys())

    # Uncontested pot — no calculation needed
    if len(players) == 1:
        return {players[0]: 1.0}

    n_to_deal = 5 - len(board_cards)

    known: set = {tuple(c) for c in board_cards}
    for cards in hole_cards.values():
        known.update(tuple(c) for c in cards)
    remaining = [c for c in _FULL_DECK if tuple(c) not in known]

    wins: dict = {p: 0.0 for p in players}

    def _tally(runout: list) -> None:
        full    = list(board_cards) + runout
        scores  = {p: _best_hand(list(hole_cards[p]) + full) for p in players}
        best    = max(scores.values())
        winners = [p for p, s in scores.items() if s == best]
        share   = 1.0 / len(winners)
        for w in winners:
            wins[w] += share

    if n_to_deal == 0:
        _tally([])
    elif comb(len(remaining), n_to_deal) <= exact_threshold:
        for runout in combinations(remaining, n_to_deal):
            _tally(list(runout))
    else:
        rng = random.Random(42)
        for _ in range(n_samples):
            _tally(rng.sample(remaining, n_to_deal))

    total = sum(wins.values())
    if total <= 0:
        n = len(players)
        return {p: 1.0 / n for p in players}
    return {p: wins[p] / total for p in players}


def evaluate_hand(hole_cards: list, board: list) -> tuple:
    """
    Score the best 5-card hand from hole_cards + board.
    Public entry point for hand classification use.
    Returns a comparable tuple (category, tiebreakers…):
      0=high card, 1=pair, 2=two pair, 3=trips, 4=straight,
      5=flush, 6=full house, 7=quads, 8=straight flush
    """
    return _best_hand(list(hole_cards) + list(board))
