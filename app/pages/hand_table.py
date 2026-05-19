import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from core.data import load_player_features, parse_hand_replay
from core.model import build_hand_summary, compute_all_probabilities
from core.table_renderer import render_poker_table
from core.ui import callout, page_header, section_label

# ─── Constants ────────────────────────────────────────────────────────────────

_POS_LABEL  = {1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO", 6: "BTN"}
_POS_COLOR  = {
    "SB": "#4A6FA5", "BB": "#4A6FA5",
    "UTG": "#4A5568", "HJ": "#4A5568", "CO": "#4A5568", "BTN": "#6B5B95",
}
_ACTION_FR  = {"fold": "Se couche", "call": "Appel / Check", "raise": "Relance"}
_ACTION_COL = {"fold": "#C94040", "call": "#4CAF82", "raise": "#D4A017"}
_SUIT_SYM    = {"h": "♥", "d": "♦", "s": "♠", "c": "♣"}
_SUIT_COL    = {"h": "#E03535", "d": "#E03535", "s": "#D1D9E6", "c": "#D1D9E6"}
_RANK_DISP   = {"T": "10"}   # T = Ten in PHH notation
_STREET_FR  = {
    "preflop": "PRÉFLOP", "flop": "FLOP",
    "turn": "TURN",       "river": "RIVER", "showdown": "FIN",
}
_PLR_COLORS = ["#5B8DD9", "#4CAF82", "#C94040", "#7B68EE", "#F97316", "#94AECF"]


# ─── Pure helpers ─────────────────────────────────────────────────────────────

def _normalize(raw: dict, folded) -> dict:
    active = {k: v for k, v in raw.items() if k not in folded}
    total  = sum(active.values())
    if total <= 0:
        n = max(len(raw), 1)
        return {k: (1 / n if k not in folded else 0.0) for k in raw}
    return {k: (active.get(k, 0.0) / total) for k in raw}


def _card_text(rank: str, suit: str) -> str:
    col  = _SUIT_COL.get(suit, "#D1D9E6")
    sym  = _SUIT_SYM.get(suit, suit)
    disp = _RANK_DISP.get(rank, rank)
    return f'<span style="color:{col};font-weight:700">{disp}{sym}</span>'


def _build_replay_steps(replay: dict, idx_to_name: dict) -> list[dict]:
    """Build the ordered list of discrete events for action-by-action replay."""
    actions_by_street = replay.get("actions_by_street", {})
    board_by_street   = replay.get("board_by_street", {})

    # Pot / stack tracking — initialised from PHH header fields
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

    state = {"board": [], "folded": set()}
    steps = []

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
        }

    def fmt_action(act) -> str:
        pos  = act.get("position", "")
        name = idx_to_name.get(act["player_idx"], f"p{act['player_idx']}")
        verb = _ACTION_FR.get(act["action_type"], act["action_type"])
        if act["action_type"] == "raise" and act.get("amount"):
            verb = f"Relance {act['amount']:,}".replace(",", " ")
        return f"[{pos}] {name} — {verb}"

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
        }

    def _apply_action(p_idx, atype, amount):
        nonlocal pot, facing_bet
        inv = invested_round.get(p_idx, 0)
        if atype == "raise" and amount is not None:
            chips = max(0, amount - inv)
            pot += chips
            player_stacks[p_idx] = player_stacks.get(p_idx, 0) - chips
            invested_round[p_idx] = amount
            facing_bet = amount
        elif atype == "call":
            chips = max(0, facing_bet - inv)
            pot += chips
            player_stacks[p_idx] = player_stacks.get(p_idx, 0) - chips
            invested_round[p_idx] = facing_bet

    def _reset_round():
        nonlocal facing_bet
        facing_bet = 0
        for k in list(invested_round.keys()):
            invested_round[k] = 0

    def _process_actions(street_name):
        for act in actions_by_street.get(street_name, []):
            p_idx  = act["player_idx"]
            atype  = act["action_type"]
            amount = act.get("amount")
            ctx = _get_context(p_idx, atype, amount)
            if atype == "fold":
                state["folded"] = state["folded"] | {p_idx}
            _apply_action(p_idx, atype, amount)
            steps.append(snap("action", street_name, fmt_action(act),
                               active=p_idx, atype=atype,
                               amount=amount, pos=act.get("position"),
                               context=ctx))

    def process_street(street_name, fr_prefix):
        cards = board_by_street.get(street_name, [])
        if not cards:
            return
        _reset_round()
        state["board"] = state["board"] + cards
        card_str = " ".join(f"{_RANK_DISP.get(r, r)}{_SUIT_SYM.get(s, s)}" for r, s in cards)
        steps.append(snap("board_reveal", street_name,
                          f"▶ {fr_prefix} : {card_str}", new_cards=cards))
        _process_actions(street_name)

    steps.append(snap("start", "preflop", "Début de main — distribution des cartes"))
    _process_actions("preflop")

    process_street("flop",  "FLOP")
    process_street("turn",  "TURN")
    process_street("river", "RIVER")

    steps.append(snap("showdown", "showdown", "Fin de main — résultat"))
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

replay       = parse_hand_replay(source_path)
replay_steps = _build_replay_steps(replay, idx_to_name) if replay else []

# Precompute normalized probs at every step
step_probs = [_normalize(initial_probs, step["folded"]) for step in replay_steps]

# Street jump points: step index of first board_reveal per street
street_jumps: dict[str, int] = {}
for _i, _s in enumerate(replay_steps):
    if _s["type"] == "board_reveal" and _s["street"] not in street_jumps:
        street_jumps[_s["street"]] = _i

# ─── Session state ────────────────────────────────────────────────────────────

step_key = f"replay_step__{composite_id}"
if step_key not in st.session_state:
    st.session_state[step_key] = 0

total     = len(replay_steps)
step_idx  = min(max(st.session_state[step_key], 0), max(total - 1, 0))
cur_step  = replay_steps[step_idx] if total else {}
cur_probs = step_probs[step_idx]   if total else _normalize(initial_probs, frozenset())

# ─── Page header ──────────────────────────────────────────────────────────────

page_header("Table de jeu", f"Replay · {_short_label(composite_id)}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Joueurs",       str(n_players))
c2.metric("Favori prédit", fav_name)
c3.metric("Probabilité",   f"{fav_prob:.1%}")
c4.metric("Prédiction",    correct_label)

st.divider()

# ─── Replay controls ──────────────────────────────────────────────────────────

section_label("Replay pas à pas")

# Prev / Banner / Next
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
    _acolor = _ACTION_COL.get(_atype, "#94AECF")
    _st_fr  = _STREET_FR.get(cur_step.get("street", ""), "")
    _lbl    = cur_step.get("label", "")
    _num    = f"{step_idx + 1} / {total}" if total else "—"
    st.markdown(
        f'<div style="background:#1A2236;border-radius:8px;padding:10px 18px;text-align:center">'
        f'<div style="font-size:8.5px;color:#4A5568;font-weight:800;letter-spacing:0.12em;'
        f'text-transform:uppercase;margin-bottom:3px">{_st_fr} · Étape {_num}</div>'
        f'<div style="font-size:13px;font-weight:600;color:{_acolor}">{_lbl}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Jump buttons
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
jump_def: dict[str, int] = {"⏮ Début": 0}
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

# ─── Betting context bar ──────────────────────────────────────────────────────

_ctx = cur_step.get("context")
if _ctx and _ctx.get("chips_added", 0) > 0:
    _pot     = _ctx["pot"]
    _chip    = _ctx["chips_added"]
    _stk_rem = _ctx["stack_before"] - _chip
    _bb      = _ctx["big_blind"] or 100

    def _fmt(v):
        return f"{int(v):,}".replace(",", " ")

    _pot_bb  = _pot / _bb
    _chip_bb = _chip / _bb
    _pct_pot = _chip / _pot * 100 if _pot > 0 else 0

    st.markdown(
        f'<div style="background:#0C1520;border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:8px;padding:8px 18px;margin:6px 0 10px;'
        f'font-size:11.5px;color:#6B7E96;display:flex;gap:24px;flex-wrap:wrap">'
        f'<span>Pot&nbsp;<strong style="color:#D1D9E6">{_fmt(_pot)}&nbsp;chips'
        f'&nbsp;({_pot_bb:.1f}&nbsp;BB)</strong></span>'
        f'<span>Mise/Appel&nbsp;<strong style="color:#D4A017">{_fmt(_chip)}&nbsp;chips'
        f'&nbsp;({_chip_bb:.1f}&nbsp;BB&nbsp;·&nbsp;{_pct_pot:.0f}%&nbsp;du&nbsp;pot)'
        f'</strong></span>'
        f'<span>Stack&nbsp;restant&nbsp;<strong style="color:#4CAF82">'
        f'{_fmt(_stk_rem)}</strong></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─── Poker table ──────────────────────────────────────────────────────────────

board_now   = cur_step.get("board_cards", [])
folded_now  = cur_step.get("folded", frozenset())
active_now  = cur_step.get("active")
is_showdown = cur_step.get("type") == "showdown"
act_type    = cur_step.get("action_type", "")

# Dynamic favorite: highest-prob active player at this step
_active_probs = {k: v for k, v in cur_probs.items() if k not in folded_now and v > 0}
dyn_fav_idx   = max(_active_probs, key=_active_probs.get) if _active_probs else None

players_render = []
for _, _row in hand_df.iterrows():
    p_idx  = int(_row["player_position_index"])
    name   = str(_row["player_name"])
    is_win = bool(_row["player_won"] == 1)

    last_act, last_act_type = "", ""
    if p_idx == active_now and act_type:
        last_act = _ACTION_FR.get(act_type, act_type)
        if act_type == "raise" and cur_step.get("amount"):
            last_act = f"Relance {cur_step['amount']:,}".replace(",", " ")
        last_act_type = act_type

    players_render.append({
        "name":             name,
        "stack":            _row["starting_stack"],
        "hole_cards":       str(_row.get("hole_cards") or ""),
        "probability":      cur_probs.get(p_idx, 0.0),
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

st.divider()

# ─── Chart (left) + Action log (right) ───────────────────────────────────────

col_chart, col_log = st.columns([6, 4])

# ── Probability evolution chart ────────────────────────────────────────────────

with col_chart:
    section_label("Évolution des probabilités")

    fig = go.Figure()

    for _, _row in hand_df.iterrows():
        p_idx  = int(_row["player_position_index"])
        name   = str(_row["player_name"])
        is_win = bool(_row["player_won"] == 1)
        color  = "#F4B942" if is_win else _PLR_COLORS[p_idx % len(_PLR_COLORS)]
        y_vals = [sp.get(p_idx, 0.0) * 100 for sp in step_probs]

        fig.add_trace(go.Scatter(
            x=list(range(len(y_vals))),
            y=y_vals,
            mode="lines",
            name=name,
            line=dict(color=color, width=3 if is_win else 1.5, shape="hv"),
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
        "Probabilités normalisées (somme = 100 %). Évoluent uniquement "
        "par effet des retraits — pas de réévaluation sur les cartes communes."
    )

# ── Action log ─────────────────────────────────────────────────────────────────

with col_log:
    section_label("Journal de la main")

    log_html = '<div style="font-size:11px;line-height:1.65;max-height:320px;overflow-y:auto">'

    # Always show PRÉFLOP header first
    if replay_steps:
        log_html += (
            '<div style="margin-bottom:5px">'
            '<span style="font-size:8.5px;font-weight:800;letter-spacing:0.12em;'
            'color:#4A5568;text-transform:uppercase">PRÉFLOP</span>'
            '</div>'
        )

    for _li, _step in enumerate(replay_steps):
        _stype  = _step["type"]
        _street = _step["street"]

        if _stype == "start":
            continue

        if _stype == "board_reveal":
            _cards_html = " ".join(_card_text(r, s) for r, s in _step.get("new_cards", []))
            _st_fr_lbl  = _STREET_FR.get(_street, _street.upper())
            log_html += (
                f'<div style="margin-top:12px;margin-bottom:5px;display:flex;'
                f'align-items:baseline;gap:6px;flex-wrap:wrap">'
                f'<span style="font-size:8.5px;font-weight:800;letter-spacing:0.12em;'
                f'color:#4A5568;text-transform:uppercase">{_st_fr_lbl}</span>'
                f'<span style="font-size:13px">{_cards_html}</span>'
                f'</div>'
            )
            continue

        if _stype == "showdown":
            log_html += (
                f'<div style="margin-top:10px;padding:6px 8px;'
                f'background:rgba(244,185,66,0.1);border-radius:5px;'
                f'border-left:3px solid #F4B942">'
                f'<span style="color:#F4B942;font-weight:700">★ {winner_name}</span>'
                f'</div>'
            )
            continue

        if _stype != "action":
            continue

        _p_idx  = _step.get("active")
        _pos    = _step.get("position", "")
        _name   = idx_to_name.get(_p_idx, f"p{_p_idx}") if _p_idx is not None else ""
        _atype  = _step.get("action_type", "")
        _amount = _step.get("amount")
        _verb   = _ACTION_FR.get(_atype, _atype)
        if _atype == "raise" and _amount:
            _verb = f"Relance {_amount:,}".replace(",", " ")

        _acolor  = _ACTION_COL.get(_atype, "#6B7280")
        _pc      = _POS_COLOR.get(_pos, "#4A5568")
        _is_cur  = (_li == step_idx)
        _bg      = "background:rgba(220,225,255,0.09);border-radius:3px;" if _is_cur else ""
        _fw      = "font-weight:700;" if _is_cur else ""

        log_html += (
            f'<div style="display:flex;align-items:center;gap:6px;'
            f'margin-bottom:3px;padding:2px 4px;{_bg}">'
            f'<span style="background:{_pc};color:#FFF;font-size:7.5px;font-weight:700;'
            f'padding:1px 4px;border-radius:3px;min-width:26px;text-align:center;'
            f'flex-shrink:0">{_pos}</span>'
            f'<span style="color:#94AECF;{_fw}flex:1;white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis">{_name}</span>'
            f'<span style="color:{_acolor};{_fw}white-space:nowrap">{_verb}</span>'
            f'</div>'
        )

    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)

# ─── Academic note ────────────────────────────────────────────────────────────

st.divider()
section_label("Note méthodologique")
callout(
    "<strong>Modèle :</strong> Régression Logistique préflop avec "
    "<code>class_weight='balanced'</code>. "
    "Les probabilités évoluent au fil du replay uniquement par l'effet des retraits : "
    "quand un joueur se couche, sa probabilité passe à 0 et les autres sont renormalisées. "
    "Il ne s'agit pas d'une réévaluation basée sur les cartes communes — "
    "le modèle n'a pas de composante post-préflop."
)
