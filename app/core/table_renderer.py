"""
Generates a self-contained HTML poker table visualization.
No external JS dependencies — pure HTML/CSS, rendered via st.components.v1.html().
"""

SUIT_SYMBOL = {"h": "♥", "d": "♦", "s": "♠", "c": "♣"}
SUIT_COLOR  = {"h": "#E03535", "d": "#E03535", "s": "#1A1A1A", "c": "#1A1A1A"}

RANK_DISPLAY = {"T": "10"}

_SEAT_POSITIONS = [
    {"left": "365px", "top": "415px"},  # 0 — south
    {"left": "80px",  "top": "340px"},  # 1 — SW
    {"left": "4px",   "top": "178px"},  # 2 — W
    {"left": "90px",  "top": "14px"},   # 3 — NW
    {"left": "640px", "top": "14px"},   # 4 — NE
    {"left": "725px", "top": "178px"},  # 5 — E
]

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: transparent; overflow: hidden; }
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.tc {
    position: relative;
    width: 900px;
    height: 575px;
    background: #080D14;
    border-radius: 18px;
    font-family: 'Inter', -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ── Table felt ── */
.felt {
    position: absolute;
    width: 490px;
    height: 205px;
    left: 205px;
    top: 183px;
    border-radius: 50%;
    background: radial-gradient(ellipse at 38% 38%,
        #0F5C33 0%, #0A4526 50%, #063018 100%);
    border: 18px solid #050A0E;
    box-shadow:
        inset 0 0 70px rgba(0,0,0,0.6),
        0 0 0 2px #0E1A12,
        0 16px 70px rgba(0,0,0,0.8);
}

.board-label {
    position: absolute;
    font-size: 7.5px;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: rgba(255,255,255,0.13);
    top: 13px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}

.board {
    position: absolute;
    display: flex;
    gap: 8px;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    align-items: center;
}

/* ── Playing cards ── */
.card {
    width: 36px;
    height: 50px;
    background: #F8F9FA;
    border-radius: 6px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.9);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    padding: 3px 3px 4px;
    font-weight: 800;
    flex-shrink: 0;
}
.cr { font-size: 11px; align-self: flex-start; line-height: 1; }
.cs { font-size: 14px; line-height: 1; }

.card-back {
    width: 36px; height: 50px;
    background: repeating-linear-gradient(
        -45deg,
        #0C2D56, #0C2D56 4px, #0A2347 4px, #0A2347 8px
    );
    border-radius: 6px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.6);
    flex-shrink: 0;
}

.card-ph {
    width: 36px; height: 50px;
    background: rgba(255,255,255,0.04);
    border: 1px dashed rgba(255,255,255,0.18);
    border-radius: 6px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: rgba(255,255,255,0.15);
    font-size: 15px;
}

/* ── Seat blocks ── */
.seat {
    position: absolute;
    width: 172px;
    background: linear-gradient(160deg, #0E1929 0%, #0B1520 100%);
    border: 1px solid #1A2840;
    border-radius: 13px;
    padding: 10px 12px 9px;
    box-shadow: 0 6px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.03);
}
.seat-winner {
    border-color: rgba(245,158,11,0.60);
    box-shadow: 0 0 0 2px rgba(245,158,11,0.15), 0 6px 32px rgba(0,0,0,0.6);
    background: linear-gradient(160deg, #150F00 0%, #0E0B00 100%);
}
.seat-favorite {
    border-color: rgba(34,197,94,0.50);
    box-shadow: 0 0 0 2px rgba(34,197,94,0.10), 0 6px 32px rgba(0,0,0,0.6);
    background: linear-gradient(160deg, #051A0E 0%, #041510 100%);
}
.seat-active {
    border-color: rgba(56,189,248,0.65);
    box-shadow: 0 0 0 2px rgba(56,189,248,0.18), 0 6px 32px rgba(0,0,0,0.6);
}
.seat-folded { opacity: 0.28; }

.paction {
    font-size: 9.5px;
    font-weight: 600;
    text-align: center;
    margin-top: 3px;
    letter-spacing: 0.02em;
}

/* ── Seat header row ── */
.seat-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    gap: 4px;
    min-height: 18px;
}
.pname {
    font-size: 11px;
    font-weight: 600;
    color: #B8C8DA;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
    min-width: 0;
}
.seat-badges {
    display: flex;
    align-items: center;
    gap: 3px;
    flex-shrink: 0;
}

/* ── Winner / favorite badges ── */
.badge-w {
    background: rgba(245,158,11,0.20);
    color: #F59E0B;
    border: 1px solid rgba(245,158,11,0.40);
    font-size: 7.5px;
    font-weight: 800;
    padding: 1.5px 5px;
    border-radius: 8px;
    white-space: nowrap;
    letter-spacing: 0.04em;
    flex-shrink: 0;
}
.badge-f {
    background: rgba(34,197,94,0.12);
    color: #22C55E;
    border: 1px solid rgba(34,197,94,0.30);
    font-size: 7.5px;
    font-weight: 700;
    padding: 1.5px 5px;
    border-radius: 8px;
    white-space: nowrap;
    flex-shrink: 0;
}

/* ── Position badges ── */
.pos-badge {
    font-size: 7px;
    font-weight: 800;
    padding: 1.5px 5px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    white-space: nowrap;
    flex-shrink: 0;
}
.pos-sb, .pos-bb {
    background: #091C35;
    color: #60A5FA;
    border: 1px solid #1A3A5E;
    border-radius: 4px;
}
.pos-utg, .pos-hj, .pos-co {
    background: #111B28;
    color: #3D5166;
    border: 1px solid #1A2840;
    border-radius: 4px;
}
.pos-btn {
    background: #1C1000;
    color: #F59E0B;
    border: 1.5px solid #6B4500;
    border-radius: 20px;
    padding: 1.5px 7px;
    font-size: 7px;
    font-weight: 800;
    letter-spacing: 0.08em;
}

/* ── Cards area ── */
.pcards {
    display: flex;
    gap: 5px;
    margin-bottom: 7px;
}

/* ── Stack & bar ── */
.pstack {
    font-size: 9px;
    color: #2D3F52;
    margin-bottom: 5px;
    letter-spacing: 0.01em;
}
.pbar {
    display: flex;
    align-items: center;
    gap: 7px;
}
.ptrack {
    flex: 1;
    height: 3px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
}
.pfill {
    height: 100%;
    border-radius: 2px;
}
.plabel {
    font-size: 11px;
    font-weight: 700;
    min-width: 32px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Footer label ── */
.hmeta {
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 8.5px;
    color: rgba(255,255,255,0.10);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    white-space: nowrap;
}
"""

_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{css}</style>
</head>
<body>
<div class="tc">
  <div class="felt">
    <div class="board-label">Community Cards</div>
    <div class="board">{board_html}</div>
  </div>
  {seats_html}
  <div class="hmeta">{meta}</div>
</div>
</body>
</html>"""


def _card_html(rank: str, suit: str) -> str:
    sym  = SUIT_SYMBOL.get(suit, "?")
    clr  = SUIT_COLOR.get(suit, "#CCC")
    disp = RANK_DISPLAY.get(rank, rank)
    return (
        f'<div class="card">'
        f'<span class="cr" style="color:{clr}">{disp}</span>'
        f'<span class="cs" style="color:{clr}">{sym}</span>'
        f'</div>'
    )


def _parse_hole_cards(hole_cards_str: str) -> tuple | None:
    s = str(hole_cards_str).strip()
    if len(s) >= 4:
        return (s[0], s[1]), (s[2], s[3])
    return None


def _pos_badge_html(pos: str) -> str:
    if not pos:
        return ""
    pos_lower = pos.lower()
    if pos == "BTN":
        return f'<span class="pos-badge pos-btn">BTN</span>'
    elif pos in ("SB", "BB"):
        return f'<span class="pos-badge pos-{pos_lower}">{pos}</span>'
    else:
        return f'<span class="pos-badge pos-{pos_lower}">{pos}</span>'


def _seat_html(player: dict, position_idx: int) -> str:
    pos = _SEAT_POSITIONS[position_idx % 6]

    name             = str(player.get("name") or "?")[:15]
    stack            = player.get("stack", 0)
    prob             = float(player.get("probability", 0.0))
    winner           = bool(player.get("is_winner", False))
    fav              = bool(player.get("is_favorite", False))
    folded           = bool(player.get("folded", False))
    is_active        = bool(player.get("is_active", False))
    last_action      = str(player.get("last_action") or "")
    last_action_type = str(player.get("last_action_type") or "")
    cards_s          = str(player.get("hole_cards") or "")
    position_label   = str(player.get("position") or "")

    parsed = _parse_hole_cards(cards_s)
    if parsed:
        (r1, s1), (r2, s2) = parsed
        cards_html = f'<div class="pcards">{_card_html(r1,s1)}{_card_html(r2,s2)}</div>'
    else:
        cards_html = (
            '<div class="pcards">'
            '<div class="card-back"></div>'
            '<div class="card-back"></div>'
            '</div>'
        )

    if winner:
        bar_col, label_col = "#F59E0B", "#F59E0B"
    elif fav:
        bar_col, label_col = "#22C55E", "#22C55E"
    else:
        bar_col, label_col = "#2A4A72", "#3A5A8A"

    pos_badge = _pos_badge_html(position_label)

    badge_html = ""
    if winner:
        badge_html = '<span class="badge-w">★ Winner</span>'
    elif fav:
        badge_html = '<span class="badge-f">Favori</span>'

    classes = "seat"
    if winner:
        classes += " seat-winner"
    elif fav:
        classes += " seat-favorite"
    if is_active:
        classes += " seat-active"
    if folded and not is_active:
        classes += " seat-folded"

    action_html = ""
    if last_action:
        act_col = {"fold": "#EF4444", "call": "#22C55E", "raise": "#F59E0B"}.get(
            last_action_type, "#94AECF"
        )
        action_html = f'<div class="paction" style="color:{act_col}">{last_action}</div>'

    stack_display = f"{int(stack):,}".replace(",", " ")

    return f"""
  <div class="{classes}" style="left:{pos['left']};top:{pos['top']}">
    <div class="seat-head">
      <span class="pname">{name}</span>
      <div class="seat-badges">
        {pos_badge}
        {badge_html}
      </div>
    </div>
    {cards_html}
    {action_html}
    <div class="pstack">{stack_display} chips</div>
    <div class="pbar">
      <div class="ptrack">
        <div class="pfill" style="width:{prob*100:.1f}%;background:{bar_col}"></div>
      </div>
      <span class="plabel" style="color:{label_col}">{prob:.0%}</span>
    </div>
  </div>"""


def render_poker_table(
    players: list[dict],
    hand_id: str = "",
    board_cards: list[tuple] | None = None,
) -> str:
    seats_html = "\n".join(
        _seat_html(p, i) for i, p in enumerate(players)
    )

    if board_cards:
        board_html = "".join(_card_html(r, s) for r, s in board_cards)
    else:
        board_html = "".join('<div class="card-ph">?</div>' for _ in range(5))

    meta = f"Hand #{hand_id}" if hand_id else "PokerMind AI"

    return _TEMPLATE.format(
        css=_CSS,
        seats_html=seats_html,
        board_html=board_html,
        meta=meta,
    )
