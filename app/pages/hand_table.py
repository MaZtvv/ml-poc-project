import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd

from core.data import load_player_features, parse_hand_replay
from core.hand_classifier import classify_made_hand, describe_preflop, detect_draws
from core.model import (
    INHAND_FEATURES,
    build_hand_summary,
    compute_all_probabilities,
    get_inhand_lr_pipeline,
)
from core.table_renderer import render_poker_table
from core.ui import callout, page_header, section_label

# ─── Constants ────────────────────────────────────────────────────────────────

_POS_LABEL  = {1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO", 6: "BTN"}
_POS_COLOR  = {
    "SB": "#4A6FA5", "BB": "#4A6FA5",
    "UTG": "#4A5568", "HJ": "#4A5568", "CO": "#4A5568", "BTN": "#6B5B95",
}
_ACTION_COL = {"fold": "#C94040", "call": "#4CAF82", "raise": "#D4A017"}
_SUIT_SYM   = {"h": "♥", "d": "♦", "s": "♠", "c": "♣"}
_SUIT_COL   = {"h": "#E03535", "d": "#E03535", "s": "#D1D9E6", "c": "#D1D9E6"}
_RANK_DISP  = {"T": "10"}
_STREET_LABEL = {
    "preflop": "PREFLOP", "flop": "FLOP",
    "turn": "TURN", "river": "RIVER", "showdown": "SHOWDOWN",
}
_PLR_COLORS = ["#5B8DD9", "#4CAF82", "#C94040", "#7B68EE", "#F97316", "#94AECF"]


# ─── Equity cache (module-level for st.cache_data compatibility) ───────────────

@st.cache_data(show_spinner="Calcul des équités poker...")
def _compute_equity_cached(hc_key: tuple, board_key: tuple) -> dict:
    """
    Compute and cache hand equity.
    hc_key   : tuple of (player_idx, tuple-of-card-tuples)
    board_key: tuple of card tuples
    """
    from core.equity import compute_equity
    hc = {p: [list(c) for c in cards] for p, cards in hc_key}
    bc = [list(c) for c in board_key]
    return compute_equity(hc, bc)


# ─── Pure helpers ─────────────────────────────────────────────────────────────

def _normalize(raw: dict, folded) -> dict:
    active = {k: v for k, v in raw.items() if k not in folded}
    total  = sum(active.values())
    if total <= 0:
        n = max(len(raw), 1)
        return {k: (1 / n if k not in folded else 0.0) for k in raw}
    return {k: (active.get(k, 0.0) / total) for k in raw}


def _build_inhand_step_probs(
    hand_df: pd.DataFrame,
    replay_steps: list[dict],
) -> list[dict]:
    """
    Compute in-hand model probabilities at every replay step.

    Uses INHAND_FEATURES (no player_num_folds) with running counters:
    - player_num_check_calls / player_num_bets_raises accumulate as actions happen
    - num_board_deals increments at each new street (0→1→2→3)
    - Folded players are explicitly zeroed, remaining players renormalised

    Falls back to uniform distribution if the pipeline is unavailable.
    """
    _STREET_BOARD = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 3}

    try:
        pipe = get_inhand_lr_pipeline()
    except Exception:
        n = len(hand_df)
        return [{idx: 1.0 / n for idx in hand_df["player_position_index"]}
                for _ in replay_steps]

    player_indices = hand_df["player_position_index"].tolist()

    # Preflop feature values — constant across all steps
    _preflop_cols = [f for f in INHAND_FEATURES
                     if f not in ("player_num_check_calls", "player_num_bets_raises",
                                  "num_board_deals", "hand_action_count",
                                  "num_check_calls", "num_bets_raises")]
    preflop_base = [
        {c: hand_df.iloc[i].get(c, 0) for c in _preflop_cols}
        for i in range(len(player_indices))
    ]

    # Running action counters — updated before scoring each step
    player_cc  = {idx: 0 for idx in player_indices}
    player_cbr = {idx: 0 for idx in player_indices}
    total_cc = total_cbr = total_actions = 0

    result = []
    for step in replay_steps:
        # Update counters from this step's action
        if step["type"] == "action":
            p  = step.get("active")
            at = step.get("action_type", "")
            if p is not None:
                total_actions += 1
                if at == "call":
                    player_cc[p]  = player_cc.get(p, 0) + 1
                    total_cc += 1
                elif at == "raise":
                    player_cbr[p] = player_cbr.get(p, 0) + 1
                    total_cbr += 1

        board_count = _STREET_BOARD.get(step.get("street", "preflop"), 0)
        folded      = step.get("folded", frozenset())

        feat_rows = []
        for i, idx in enumerate(player_indices):
            feat = dict(preflop_base[i])
            feat["player_num_check_calls"] = player_cc.get(idx, 0)
            feat["player_num_bets_raises"] = player_cbr.get(idx, 0)
            feat["num_board_deals"]        = board_count
            feat["hand_action_count"]      = total_actions
            feat["num_check_calls"]        = total_cc
            feat["num_bets_raises"]        = total_cbr
            feat_rows.append(feat)

        X_step = pd.DataFrame(feat_rows, columns=INHAND_FEATURES)
        raw    = pipe.predict_proba(X_step)[:, 1]

        prob_dict = {idx: float(p) for idx, p in zip(player_indices, raw)}
        for f_idx in folded:
            prob_dict[f_idx] = 0.0
        total = sum(prob_dict.values())
        if total > 0:
            prob_dict = {k: v / total for k, v in prob_dict.items()}

        result.append(prob_dict)

    return result


def _card_text(rank: str, suit: str) -> str:
    col  = _SUIT_COL.get(suit, "#D1D9E6")
    sym  = _SUIT_SYM.get(suit, suit)
    disp = _RANK_DISP.get(rank, rank)
    return f'<span style="color:{col};font-weight:700">{disp}{sym}</span>'


def _fmtc(v: float) -> str:
    return f"{int(v):,}".replace(",", " ")


def _build_replay_steps(replay: dict, idx_to_name: dict) -> list[dict]:
    """Build the ordered list of discrete events for action-by-action replay."""
    actions_by_street = replay.get("actions_by_street", {})
    board_by_street   = replay.get("board_by_street", {})

    blinds_raw = replay.get("blinds", [])
    antes_raw  = replay.get("antes", [])
    stacks_raw = replay.get("starting_stacks", [])
    big_blind  = replay.get("big_blind") or 100

    def _g(lst, i):
        return lst[i] if i < len(lst) else 0

    n = max(len(stacks_raw), len(blinds_raw), len(antes_raw), 1)
    player_stacks  = {i + 1: _g(stacks_raw, i) - _g(blinds_raw, i) - _g(antes_raw, i)
                      for i in range(n)}
    invested_round = {i + 1: _g(blinds_raw, i) for i in range(n)}
    facing_bet     = max(blinds_raw) if blinds_raw else 0
    pot            = sum(blinds_raw) + sum(antes_raw)
    last_raise_to  = max(blinds_raw) if blinds_raw else 0

    state = {"board": [], "folded": set()}
    steps = []

    def _snap_state() -> dict:
        return {
            "pot":           pot,
            "facing_bet":    facing_bet,
            "last_raise_to": last_raise_to,
            "player_stacks": dict(player_stacks),
            "big_blind":     big_blind,
        }

    def snap(type_, street, label, *, active=None, atype=None, amount=None,
             pos=None, new_cards=None, context=None):
        return {
            "type":        type_,
            "street":      street,
            "label":       label,
            "board_cards": list(state["board"]),
            "folded":      frozenset(state["folded"]),
            "active":      active,
            "action_type": atype,
            "amount":      amount,
            "position":    pos,
            "new_cards":   new_cards or [],
            "context":     context,
            "state_after": _snap_state(),
        }

    def _get_context(p_idx, atype, amount):
        inv = invested_round.get(p_idx, 0)
        if atype == "raise" and amount is not None:
            chips_added = max(0, amount - inv)
        elif atype == "call":
            chips_added = max(0, facing_bet - inv)
        else:
            chips_added = 0
        return {
            "pot":          pot,
            "chips_added":  chips_added,
            "stack_before": player_stacks.get(p_idx, 0),
            "big_blind":    big_blind,
            "facing_bet":   facing_bet,
        }

    def _fmt_action_rich(act, ctx) -> str:
        pos    = act.get("position", "")
        name   = idx_to_name.get(act["player_idx"], f"p{act['player_idx']}")
        atype  = act["action_type"]
        chips  = ctx.get("chips_added", 0)
        amount = act.get("amount")
        facing = ctx.get("facing_bet", 0)
        if atype == "raise" and amount is not None:
            verb = f"{'Raise' if facing > 0 else 'Bet'} to {_fmtc(amount)}"
        elif atype == "call" and chips > 0:
            verb = f"Call {_fmtc(chips)}"
        elif atype == "call":
            verb = "Check"
        else:
            verb = "Fold"
        return f"[{pos}] {name} — {verb}"

    def _apply_action(p_idx, atype, amount):
        nonlocal pot, facing_bet, last_raise_to
        inv = invested_round.get(p_idx, 0)
        if atype == "raise" and amount is not None:
            chips = max(0, amount - inv)
            pot += chips
            player_stacks[p_idx] = player_stacks.get(p_idx, 0) - chips
            invested_round[p_idx] = amount
            facing_bet    = amount
            last_raise_to = amount
        elif atype == "call":
            chips = max(0, facing_bet - inv)
            pot += chips
            player_stacks[p_idx] = player_stacks.get(p_idx, 0) - chips
            invested_round[p_idx] = facing_bet

    def _reset_round():
        nonlocal facing_bet, last_raise_to
        facing_bet    = 0
        last_raise_to = 0
        for k in list(invested_round.keys()):
            invested_round[k] = 0

    def _process_actions(street_name):
        for act in actions_by_street.get(street_name, []):
            p_idx  = act["player_idx"]
            atype  = act["action_type"]
            amount = act.get("amount")
            ctx   = _get_context(p_idx, atype, amount)
            label = _fmt_action_rich(act, ctx)
            if atype == "fold":
                state["folded"] = state["folded"] | {p_idx}
            _apply_action(p_idx, atype, amount)
            steps.append(snap("action", street_name, label,
                               active=p_idx, atype=atype,
                               amount=amount, pos=act.get("position"),
                               context=ctx))

    def process_street(street_name):
        cards = board_by_street.get(street_name, [])
        if not cards:
            return
        _reset_round()
        state["board"] = state["board"] + cards
        card_str = " ".join(f"{_RANK_DISP.get(r, r)}{_SUIT_SYM.get(s, s)}" for r, s in cards)
        steps.append(snap("board_reveal", street_name,
                          f"▶ {street_name.upper()}: {card_str}", new_cards=cards))
        _process_actions(street_name)

    steps.append(snap("start", "preflop", "Deal — hole cards"))
    _process_actions("preflop")

    process_street("flop")
    process_street("turn")
    process_street("river")

    steps.append(snap("showdown", "showdown", "Showdown — result"))
    return steps


# ─── Cached data ──────────────────────────────────────────────────────────────

df      = load_player_features()
probs   = compute_all_probabilities(df)
summary = build_hand_summary(df)

all_composite_ids = summary["composite_id"].tolist()


def _short_label(cid: str) -> str:
    path, hid = cid.rsplit("::", 1)
    return f"{'/'.join(path.split('/')[-2:])} · main {hid}"


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("")
    section_label("Changer de main")

    current_id  = st.session_state.get("selected_composite_id")
    default_idx = all_composite_ids.index(current_id) if current_id in all_composite_ids else 0

    chosen_id = st.selectbox("Main", options=all_composite_ids, index=default_idx,
                              format_func=_short_label, label_visibility="collapsed")
    st.session_state["selected_composite_id"] = chosen_id

    st.divider()
    if st.button("← Retour au navigateur", use_container_width=True):
        st.switch_page("pages/hand_browser.py")

# ─── Load one hand ────────────────────────────────────────────────────────────

composite_id = st.session_state.get("selected_composite_id")
if not composite_id:
    st.info("Aucune main sélectionnée. Utilisez le navigateur pour en choisir une.")
    st.stop()

hand_df = df[df["composite_id"] == composite_id].copy()
hand_df["win_probability"] = probs[hand_df.index].values
hand_df = hand_df.sort_values("player_position_index").reset_index(drop=True)

if len(hand_df) == 0:
    st.error(f"Main introuvable : {composite_id}")
    st.stop()

# ─── Derived values ───────────────────────────────────────────────────────────

source_path   = hand_df["source_relative_path"].iloc[0]
n_players     = len(hand_df)
winner_mask   = hand_df["player_won"] == 1
winner_name   = hand_df.loc[winner_mask, "player_name"].iloc[0] if winner_mask.any() else "—"
fav_df_idx    = hand_df["win_probability"].idxmax()
fav_name      = hand_df.loc[fav_df_idx, "player_name"]
fav_prob      = hand_df.loc[fav_df_idx, "win_probability"]
correct       = fav_name == winner_name
correct_label = "✓ Correcte" if correct else "✗ Incorrecte"

idx_to_name   = dict(zip(hand_df["player_position_index"], hand_df["player_name"]))
initial_probs = dict(zip(hand_df["player_position_index"], hand_df["win_probability"]))

replay        = parse_hand_replay(source_path)
replay_steps  = _build_replay_steps(replay, idx_to_name) if replay else []
big_blind     = replay.get("big_blind") or 100 if replay else 100

# ML probabilities: in-hand model (no player_num_folds), updates at each action
# and at each street transition (num_board_deals), zeroes folds by explicit logic.
step_probs = (
    _build_inhand_step_probs(hand_df, replay_steps)
    if replay_steps else []
)

# Street jump points
street_jumps: dict[str, int] = {}
for _i, _s in enumerate(replay_steps):
    if _s["type"] == "board_reveal" and _s["street"] not in street_jumps:
        street_jumps[_s["street"]] = _i

# ─── Poker equity (actual hand strength, computed per street) ──────────────────
#
# This is different from the ML model output.
# Equity = probability that a hand wins given the visible cards + remaining deck.
# Changes only when new community cards are revealed.

_raw_hc = replay.get("hole_cards", {}) if replay else {}
_boards = replay.get("board_by_street", {}) if replay else {}
_fbs    = replay.get("folded_by_street", {}) if replay else {}

_b_flop  = _boards.get("flop", [])
_b_turn  = _b_flop + _boards.get("turn", [])
_b_river = _b_turn + _boards.get("river", [])


def _hc_key(folded: set) -> tuple:
    return tuple(sorted(
        (p, tuple(tuple(c) for c in cards))
        for p, cards in _raw_hc.items()
        if p not in folded
    ))


def _bk(board: list) -> tuple:
    return tuple(tuple(c) for c in board)


if _raw_hc:
    eq_preflop = _compute_equity_cached(_hc_key(set()),                      _bk([]))
    eq_flop    = _compute_equity_cached(_hc_key(_fbs.get("preflop", set())), _bk(_b_flop))
    eq_turn    = _compute_equity_cached(_hc_key(_fbs.get("flop",    set())), _bk(_b_turn))
    eq_river   = _compute_equity_cached(_hc_key(_fbs.get("turn",    set())), _bk(_b_river))
else:
    eq_preflop = eq_flop = eq_turn = eq_river = {}

_eq_map = {
    "preflop":  eq_preflop,
    "flop":     eq_flop,
    "turn":     eq_turn,
    "river":    eq_river,
    "showdown": eq_river,
}
equity_at_step = [_eq_map.get(s["street"], eq_preflop) for s in replay_steps]

# ─── Session state ────────────────────────────────────────────────────────────

step_key = f"replay_step__{composite_id}"
if step_key not in st.session_state:
    st.session_state[step_key] = 0

total     = len(replay_steps)
step_idx  = min(max(st.session_state[step_key], 0), max(total - 1, 0))
cur_step  = replay_steps[step_idx] if total else {}
cur_probs = step_probs[step_idx]   if total else _normalize(initial_probs, frozenset())
cur_equity = equity_at_step[step_idx] if total else {}

# ─── Current hand state (post-action snapshot) ────────────────────────────────

_sa           = cur_step.get("state_after", {})
panel_pot     = _sa.get("pot", 0)
panel_facing  = _sa.get("facing_bet", 0)
panel_raise   = _sa.get("last_raise_to", 0)
panel_stacks  = _sa.get("player_stacks", {})
panel_bb      = _sa.get("big_blind") or big_blind
panel_active  = sum(1 for p in hand_df["player_position_index"]
                    if p not in cur_step.get("folded", frozenset()))

# ─── Page header ──────────────────────────────────────────────────────────────

page_header("Table de jeu", f"Replay · {_short_label(composite_id)}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Joueurs",          str(n_players))
c2.metric("Favori ML",        fav_name)
c3.metric("Score ML",         f"{fav_prob:.1%}")
c4.metric("Prédiction ML",    correct_label)

st.divider()

# ─── Replay controls ──────────────────────────────────────────────────────────

section_label("Replay pas à pas")

c_prev, c_banner, c_next = st.columns([1, 8, 1])

with c_prev:
    st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
    if st.button("◀", use_container_width=True, disabled=step_idx == 0, key="btn_prev"):
        st.session_state[step_key] = step_idx - 1
        st.rerun()

with c_next:
    st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
    if st.button("▶", use_container_width=True, disabled=step_idx >= total - 1, key="btn_next"):
        st.session_state[step_key] = step_idx + 1
        st.rerun()

with c_banner:
    _atype  = cur_step.get("action_type") or ""
    _acolor = _ACTION_COL.get(_atype, "#60A5FA")
    _st_fr  = _STREET_LABEL.get(cur_step.get("street", ""), "")
    _lbl    = cur_step.get("label", "")
    _num    = f"{step_idx + 1} / {total}" if total else "—"
    st.markdown(
        f'<div style="background:#0C1826;border:1px solid #1A2840;border-radius:10px;'
        f'padding:11px 20px;text-align:center">'
        f'<div style="font-size:7.5px;color:#2D3F52;font-weight:800;letter-spacing:0.14em;'
        f'text-transform:uppercase;margin-bottom:4px">{_st_fr} &nbsp;·&nbsp; Step {_num}</div>'
        f'<div style="font-size:13px;font-weight:600;color:{_acolor};letter-spacing:0.01em">{_lbl}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Jump buttons
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
jump_def: dict[str, int] = {"⏮ Start": 0}
if "flop"  in street_jumps: jump_def["Flop"]  = street_jumps["flop"]
if "turn"  in street_jumps: jump_def["Turn"]  = street_jumps["turn"]
if "river" in street_jumps: jump_def["River"] = street_jumps["river"]
jump_def["Fin ⏭"] = total - 1 if total else 0

jcols = st.columns(len(jump_def))
for _ji, (_jlbl, _jtgt) in enumerate(jump_def.items()):
    with jcols[_ji]:
        if st.button(_jlbl, use_container_width=True, key=f"jump_{_ji}"):
            st.session_state[step_key] = _jtgt
            st.rerun()

# ─── Persistent hand-state panel ──────────────────────────────────────────────

_pot_disp   = f"{_fmtc(panel_pot)} chips ({panel_pot / panel_bb:.1f} BB)"
_call_disp  = f"{_fmtc(panel_facing)} ({panel_facing / panel_bb:.1f} BB)" if panel_facing > 0 else "—"
_raise_disp = f"{_fmtc(panel_raise)} ({panel_raise / panel_bb:.1f} BB)"  if panel_raise  > 0 else "—"
_call_col   = "#D4A017" if panel_facing > 0 else "#3A4A5E"
_raise_col  = "#E08030" if panel_raise  > 0 else "#3A4A5E"
_street_lbl = _STREET_LABEL.get(cur_step.get("street", ""), "—")

_sep = 'border-right:1px solid rgba(255,255,255,0.05);padding-right:20px;margin-right:20px'
_lbl_css = 'font-size:7px;color:#2D3F52;font-weight:800;letter-spacing:0.13em;text-transform:uppercase;margin-bottom:5px'
_val_css = 'font-size:13.5px;font-weight:700;white-space:nowrap;letter-spacing:-0.01em'

st.markdown(
    f'<div style="background:linear-gradient(135deg,#0C1826 0%,#09121E 100%);'
    f'border:1px solid #1A2840;border-radius:12px;padding:13px 24px;margin:8px 0 12px;'
    f'display:grid;grid-template-columns:repeat(5,1fr);gap:0;position:relative">'

    f'<div style="position:absolute;top:0;left:0;right:0;height:2px;'
    f'background:linear-gradient(90deg,#0C3050,rgba(56,189,248,0.15),transparent);'
    f'border-radius:12px 12px 0 0"></div>'

    f'<div style="{_sep}">'
    f'<div style="{_lbl_css}">Pot</div>'
    f'<div style="{_val_css};color:#E2E8F0">{_pot_disp}</div>'
    f'</div>'

    f'<div style="{_sep}">'
    f'<div style="{_lbl_css}">Mise actuelle</div>'
    f'<div style="{_val_css};color:{_call_col}">{_call_disp}</div>'
    f'</div>'

    f'<div style="{_sep}">'
    f'<div style="{_lbl_css}">Dernière relance</div>'
    f'<div style="{_val_css};color:{_raise_col}">{_raise_disp}</div>'
    f'</div>'

    f'<div style="{_sep}">'
    f'<div style="{_lbl_css}">Joueurs actifs</div>'
    f'<div style="{_val_css};color:#60A5FA">{panel_active}</div>'
    f'</div>'

    f'<div>'
    f'<div style="{_lbl_css}">Street</div>'
    f'<div style="{_val_css};color:#4B5E72">{_street_lbl}</div>'
    f'</div>'

    f'</div>',
    unsafe_allow_html=True,
)

# ─── Poker table (seat bars show equity, not ML score) ───────────────────────

board_now   = cur_step.get("board_cards", [])
folded_now  = cur_step.get("folded", frozenset())
active_now  = cur_step.get("active")
is_showdown = cur_step.get("type") == "showdown"
act_type    = cur_step.get("action_type", "")

# Dynamic favorite and winner based on equity at current street
_active_eq  = {k: v for k, v in cur_equity.items() if k not in folded_now and v > 0}
dyn_fav_idx = max(_active_eq, key=_active_eq.get) if _active_eq else None

players_render = []
for _, _row in hand_df.iterrows():
    p_idx  = int(_row["player_position_index"])
    name   = str(_row["player_name"])
    is_win = bool(_row["player_won"] == 1)

    live_stack = panel_stacks.get(p_idx, int(_row["starting_stack"]))

    last_act, last_act_type = "", ""
    if p_idx == active_now and act_type:
        last_act      = cur_step.get("label", "").split(" — ", 1)[-1]
        last_act_type = act_type

    players_render.append({
        "name":             name,
        "stack":            live_stack,
        "hole_cards":       str(_row.get("hole_cards") or ""),
        "probability":      cur_equity.get(p_idx, 0.0),
        "position":         _POS_LABEL.get(p_idx, f"p{p_idx}"),
        "is_winner":        is_win and is_showdown,
        "is_favorite":      (p_idx == dyn_fav_idx) and not is_showdown,
        "folded":           p_idx in folded_now,
        "is_active":        p_idx == active_now,
        "last_action":      last_act,
        "last_action_type": last_act_type,
    })

hand_label = composite_id.rsplit("::", 1)[-1]
table_html = render_poker_table(
    players_render,
    hand_id=hand_label,
    board_cards=board_now if board_now else None,
)
components.html(table_html, height=590, scrolling=False)
st.markdown(
    '<p style="font-size:0.75rem;color:#2D3F52;margin:-4px 0 0;padding-left:2px">'
    'Barres de sièges (%) = équité vraie (Monte Carlo préflop / exact turn-river). '
    'Badge <span style="color:#22C55E">●</span> vert = leader en équité à ce stade. '
    'Badge <span style="color:#F59E0B">●</span> or = vainqueur au showdown.'
    '</p>',
    unsafe_allow_html=True,
)

# ─── Hand analysis expander ───────────────────────────────────────────────────

with st.expander("Hand analysis", expanded=False):
    _board_list = list(board_now)
    _th_style = (
        "font-size:9px;font-weight:800;letter-spacing:0.1em;text-transform:uppercase;"
        "color:#4A5568;padding:4px 10px 4px 0;border-bottom:1px solid rgba(255,255,255,0.07);"
        "white-space:nowrap"
    )
    _td_style = "font-size:11px;color:#D1D9E6;padding:4px 10px 4px 0;vertical-align:top"
    _an_html = (
        '<table style="width:100%;border-collapse:collapse">'
        f'<tr><th style="{_th_style}">Player</th>'
        f'<th style="{_th_style}">Pos</th>'
        f'<th style="{_th_style}">Hole cards</th>'
        f'<th style="{_th_style}">Preflop type</th>'
        f'<th style="{_th_style}">Best hand</th>'
        f'<th style="{_th_style}">Draws / texture</th></tr>'
    )

    for _, _arow in hand_df.iterrows():
        _ap = int(_arow["player_position_index"])
        _is_fold = _ap in folded_now
        _op = "0.35" if _is_fold else "1.0"
        _apos = _POS_LABEL.get(_ap, f"p{_ap}")
        _aname = str(_arow["player_name"])
        _ahc = _raw_hc.get(_ap, [])

        if _ahc:
            _hc_html = " ".join(_card_text(r, s) for r, s in _ahc)
            _pre = describe_preflop(_ahc)
            _made = classify_made_hand(_ahc, _board_list) if _board_list else "—"
            _draws = detect_draws(_ahc, _board_list) if _board_list else []
            _draws_str = ", ".join(_draws) if _draws else "—"
        else:
            _hc_html = '<span style="color:#3A4A5E">unknown</span>'
            _pre = _made = _draws_str = "—"

        _made_col = "#D1D9E6"
        if _made and _made not in ("—", ""):
            if any(k in _made for k in ("Straight flush", "Quads", "Full house", "Flush", "Straight")):
                _made_col = "#F4B942"
            elif any(k in _made for k in ("Set", "Trips", "Two pair")):
                _made_col = "#4CAF82"

        _draws_col = "#5B8DD9" if _draws_str != "—" else "#3A4A5E"

        _an_html += (
            f'<tr style="opacity:{_op}">'
            f'<td style="{_td_style};font-weight:600;color:#94AECF">{_aname}</td>'
            f'<td style="{_td_style}">{_apos}</td>'
            f'<td style="{_td_style}">{_hc_html}</td>'
            f'<td style="{_td_style};color:#94AECF">{_pre}</td>'
            f'<td style="{_td_style};color:{_made_col}">{_made}</td>'
            f'<td style="{_td_style};color:{_draws_col}">{_draws_str}</td>'
            f'</tr>'
        )

    _an_html += "</table>"
    if not _board_list:
        st.caption("Board cards will appear here once the flop is dealt.")
    st.markdown(_an_html, unsafe_allow_html=True)

st.divider()

# ─── Chart (left) + Action log (right) ───────────────────────────────────────

col_chart, col_log = st.columns([6, 4])

with col_chart:
    section_label("Modèle In-Hand vs Équité poker")

    fig = go.Figure()

    for _, _row in hand_df.iterrows():
        p_idx  = int(_row["player_position_index"])
        name   = str(_row["player_name"])
        is_win = bool(_row["player_won"] == 1)
        color  = "#F4B942" if is_win else _PLR_COLORS[p_idx % len(_PLR_COLORS)]

        # In-hand ML prediction (solid) — updates at every action + street change
        y_ml = [sp.get(p_idx, 0.0) * 100 for sp in step_probs]
        fig.add_trace(go.Scatter(
            x=list(range(len(y_ml))),
            y=y_ml,
            mode="lines",
            name=name,
            line=dict(color=color, width=2.5 if is_win else 1.5, shape="hv"),
            hovertemplate=f"{name} in-hand: %{{y:.1f}}%<extra></extra>",
        ))

        # Equity (dashed) — true hand equity, updates only at street changes
        y_eq = [eq.get(p_idx, 0.0) * 100 for eq in equity_at_step]
        fig.add_trace(go.Scatter(
            x=list(range(len(y_eq))),
            y=y_eq,
            mode="lines",
            showlegend=False,
            line=dict(color=color, width=1.5, shape="hv", dash="dot"),
            opacity=0.50,
            hovertemplate=f"{name} equity: %{{y:.1f}}%<extra></extra>",
        ))

    if total > 0:
        fig.add_vline(x=step_idx, line_dash="dot",
                      line_color="rgba(210,220,255,0.6)", line_width=2)

    for _street, _sidx in street_jumps.items():
        _fr = {"flop": "Flop", "turn": "Turn", "river": "River"}.get(_street, _street)
        fig.add_annotation(x=_sidx, y=105, text=_fr, showarrow=False,
                           font=dict(size=8.5, color="#5B6880"), xanchor="left")

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94AECF", size=10), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showticklabels=False,
                   gridcolor="rgba(255,255,255,0.04)",
                   linecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(ticksuffix="%", range=[0, 108],
                   tickfont=dict(color="#6B7280", size=9),
                   gridcolor="rgba(255,255,255,0.04)",
                   linecolor="rgba(255,255,255,0.08)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Trait plein = modèle in-hand (LR sans player_num_folds, mis à jour à chaque action et à chaque street). "
        "Pointillés = équité vraie (Monte Carlo / exact). "
        "Les deux courbes mesurent des choses différentes."
    )

with col_log:
    section_label("Journal de la main")

    log_html = (
        '<div style="font-size:11px;line-height:1.65;max-height:350px;overflow-y:auto;'
        'padding-right:4px">'
    )

    if replay_steps:
        log_html += (
            '<div style="margin-bottom:6px">'
            '<span style="font-size:7.5px;font-weight:800;letter-spacing:0.14em;'
            'color:#1E3047;text-transform:uppercase;border-left:2px solid #1A3047;'
            'padding-left:6px">PREFLOP</span>'
            '</div>'
        )

    for _li, _step in enumerate(replay_steps):
        _stype  = _step["type"]
        _street = _step["street"]

        if _stype == "start":
            continue

        if _stype == "board_reveal":
            _cards_html = " ".join(_card_text(r, s) for r, s in _step.get("new_cards", []))
            _st_fr_lbl  = _street.upper()
            log_html += (
                f'<div style="margin-top:14px;margin-bottom:6px;display:flex;'
                f'align-items:center;gap:7px;flex-wrap:wrap">'
                f'<span style="font-size:7.5px;font-weight:800;letter-spacing:0.14em;'
                f'color:#1E3047;text-transform:uppercase;border-left:2px solid #1A3047;'
                f'padding-left:6px">{_st_fr_lbl}</span>'
                f'<span style="font-size:14px">{_cards_html}</span>'
                f'</div>'
            )
            _sa_br  = _step.get("state_after", {})
            _pot_br = _sa_br.get("pot", 0)
            _bb_br  = _sa_br.get("big_blind") or big_blind
            if _pot_br > 0:
                log_html += (
                    f'<div style="font-size:8.5px;color:#1E2D3E;margin-bottom:5px;'
                    f'padding-left:8px">Pot: '
                    f'<strong style="color:#2D4056">{_fmtc(_pot_br)}'
                    f'&nbsp;({_pot_br / _bb_br:.1f} BB)</strong></div>'
                )
            continue

        if _stype == "showdown":
            log_html += (
                f'<div style="margin-top:12px;padding:7px 10px;'
                f'background:rgba(245,158,11,0.08);border-radius:7px;'
                f'border-left:3px solid rgba(245,158,11,0.45)">'
                f'<span style="font-size:7.5px;color:#6B4500;font-weight:800;'
                f'letter-spacing:0.12em;text-transform:uppercase;display:block;margin-bottom:2px">Showdown</span>'
                f'<span style="color:#F59E0B;font-weight:700;font-size:12px">★ {winner_name}</span>'
                f'</div>'
            )
            continue

        if _stype != "action":
            continue

        _p_idx  = _step.get("active")
        _pos    = _step.get("position", "")
        _atype  = _step.get("action_type", "")
        _ctx    = _step.get("context") or {}
        _chips  = _ctx.get("chips_added", 0)
        _amount = _step.get("amount")
        _name   = idx_to_name.get(_p_idx, f"p{_p_idx}") if _p_idx is not None else ""

        _facing = _ctx.get("facing_bet", 0)
        if _atype == "raise" and _amount is not None:
            _verb = f"{'Raise' if _facing > 0 else 'Bet'} to {_fmtc(_amount)}"
        elif _atype == "call" and _chips > 0:
            _verb = f"Call {_fmtc(_chips)}"
        elif _atype == "call":
            _verb = "Check"
        else:
            _verb = "Fold"

        _acolor = _ACTION_COL.get(_atype, "#6B7280")
        _is_cur = (_li == step_idx)
        _bg     = "background:rgba(56,189,248,0.07);border-radius:5px;" if _is_cur else ""
        _fw     = "font-weight:700;" if _is_cur else "font-weight:500;"

        if _pos == "BTN":
            _pbg, _pcol, _pbr = "#1C1000", "#F59E0B", "10px"
        elif _pos in ("SB", "BB"):
            _pbg, _pcol, _pbr = "#091C35", "#60A5FA", "4px"
        else:
            _pbg, _pcol, _pbr = "#111B28", "#3D5166", "4px"

        log_html += (
            f'<div style="display:flex;align-items:center;gap:6px;'
            f'margin-bottom:2px;padding:3px 5px;{_bg}">'
            f'<span style="background:{_pbg};color:{_pcol};font-size:7px;font-weight:800;'
            f'padding:1.5px 5px;border-radius:{_pbr};min-width:26px;text-align:center;'
            f'flex-shrink:0;letter-spacing:0.07em">{_pos}</span>'
            f'<span style="color:#94AECF;{_fw}flex:1;white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis;font-size:11px">{_name}</span>'
            f'<span style="color:{_acolor};{_fw}white-space:nowrap;font-size:11px">{_verb}</span>'
            f'</div>'
        )

    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)

# ─── Academic note ────────────────────────────────────────────────────────────

st.divider()
section_label("Note méthodologique")
callout(
    "<strong>Deux couches de probabilité distinctes :</strong><br>"
    "<strong>Modèle in-hand (trait plein)</strong> — Régression Logistique entraînée sur les variables "
    "préflop + style d'action (<code>player_num_check_calls</code>, <code>player_num_bets_raises</code>, "
    "<code>num_board_deals</code>…). "
    "<strong>Sans <code>player_num_folds</code></strong> — la tautologie de fuite principale est supprimée. "
    "La probabilité est recalculée à chaque action (les compteurs d'actions s'accumulent) "
    "et à chaque nouvelle street (<code>num_board_deals</code> passe de 0 à 3). "
    "Les joueurs couchés sont exclus par logique de jeu explicite, pas par variable de fuite.<br>"
    "<em>Limitation résiduelle :</em> <code>player_num_check_calls</code> et <code>player_num_bets_raises</code> "
    "restent des résumés de la main entière — le modèle est une approximation, pas une prédiction strictement chronologique.<br><br>"
    "<strong>Équité vraie (pointillés / barres de sièges)</strong> — probabilité réelle de gagner "
    "contre les autres mains visibles. Énumération exacte au turn/river ; Monte Carlo "
    "(1 200 tirages, graine fixe) au préflop/flop. Ne change qu'à chaque nouvelle street.<br><br>"
    "<strong>Périmètre du dataset :</strong> Pluribus (10 000 mains) est un benchmark de recherche IA, "
    "<em>pas</em> des données de tournoi. Chaque main repart avec 10 000 jetons et les blindes 50/100 — "
    "pas d'élimination, pas d'escalade des blindes. "
    "Le modèle prédit <em>qui gagne une main donnée</em>, pas un vainqueur de tournoi."
)
