"""
Generates a self-contained HTML poker table visualization.
No external JS dependencies — pure HTML/CSS, rendered via st.components.v1.html().
"""

SUIT_SYMBOL = {"h": "♥", "d": "♦", "s": "♠", "c": "♣"}
SUIT_COLOR  = {"h": "#E03535", "d": "#E03535", "s": "#1A1A1A", "c": "#1A1A1A"}

# T is Ten in standard poker notation — display as 10 for non-expert readability
RANK_DISPLAY = {"T": "10"}

# Six fixed seat positions (top-left corner of a 170×148 seat block)
# arranged clockwise starting from south, within a 900×580 component.
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.tc {
    position: relative;
    width: 900px;
    height: 575px;
    background: #0B1120;
    border-radius: 18px;
    font-family: 'Inter', -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ── Table felt ── */
.felt {
    position: absolute;
    width: 480px;
    height: 200px;
    left: 210px;
    top: 185px;
    border-radius: 50%;
    background: radial-gradient(ellipse at 40% 40%,
        #1D6B3A 0%, #115C2C 45%, #0A3D1C 100%);
    border: 20px solid #0C0C0C;
    box-shadow:
        inset 0 0 60px rgba(0,0,0,0.55),
        0 0 0 3px #222,
        0 12px 60px rgba(0,0,0,0.75);
}

.board-label {
    position: absolute;
    font-size: 8px;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: rgba(255,255,255,0.16);
    top: 14px;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
}

.board {
    position: absolute;
    display: flex;
    gap: 7px;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    align-items: center;
}

/* ── Cards ── */
.card {
    width: 34px;
    height: 48px;
    background: #FAFAFA;
    border-radius: 5px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.55);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    padding: 3px 3px 4px;
    font-weight: 800;
    flex-shrink: 0;
}
.cr { font-size: 10.5px; align-self: flex-start; line-height: 1; }
.cs { font-size: 13px; line-height: 1; }

.card-back {
    width: 34px; height: 48px;
    background: repeating-linear-gradient(
        -45deg,
        #1B3A6B, #1B3A6B 4px, #152E56 4px, #152E56 8px
    );
    border-radius: 5px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    flex-shrink: 0;
}

.card-ph {
    width: 34px; height: 48px;
    background: rgba(255,255,255,0.07);
    border: 1px dashed rgba(255,255,255,0.25);
    border-radius: 5px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: rgba(255,255,255,0.2);
    font-size: 14px;
}

/* ── Seat blocks ── */
.seat {
    position: absolute;
    width: 170px;
    background: #141C2E;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 9px 11px 8px;
    box-shadow: 0 4px 28px rgba(0,0,0,0.55);
}
.seat-winner {
    border-color: rgba(244,185,66,0.55);
    box-shadow: 0 0 0 2px rgba(244,185,66,0.18), 0 4px 28px rgba(0,0,0,0.55);
}
.seat-favorite {
    border-color: rgba(76,175,130,0.45);
    box-shadow: 0 0 0 2px rgba(76,175,130,0.12), 0 4px 28px rgba(0,0,0,0.55);
}
.seat-active {
    border-color: rgba(200,215,255,0.7);
    box-shadow: 0 0 0 2px rgba(200,215,255,0.22), 0 4px 28px rgba(0,0,0,0.55);
}
.seat-folded { opacity: 0.32; }
.paction {
    font-size: 9.5px;
    font-weight: 600;
    text-align: center;
    margin-top: 3px;
    letter-spacing: 0.02em;
}

.seat-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 7px;
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
}
.badge-w {
    background: #F4B942;
    color: #1A0E00;
    font-size: 8px;
    font-weight: 800;
    padding: 2px 5px;
    border-radius: 8px;
    white-space: nowrap;
    letter-spacing: 0.03em;
    flex-shrink: 0;
}
.badge-f {
    background: rgba(76,175,130,0.18);
    color: #4CAF82;
    font-size: 8px;
    font-weight: 700;
    padding: 2px 5px;
    border-radius: 8px;
    white-space: nowrap;
    flex-shrink: 0;
}

.pcards {
    display: flex;
    gap: 5px;
    margin-bottom: 7px;
}
.pstack {
    font-size: 9.5px;
    color: #3A4A5E;
    margin-bottom: 5px;
}
.pbar {
    display: flex;
    align-items: center;
    gap: 6px;
}
.ptrack {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.07);
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
    min-width: 30px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Footer label ── */
.hmeta {
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 9px;
    color: rgba(255,255,255,0.15);
    text-transform: uppercase;
    letter-spacing: 0.12em;
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
    <div class="board-label">Cartes communes</div>
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


def _parse_hole_cards(hole_cards_str: str) -> tuple[str, str] | None:
    s = str(hole_cards_str).strip()
    if len(s) >= 4:
        return (s[0], s[1]), (s[2], s[3])
    return None


def _seat_html(player: dict, position_idx: int) -> str:
    pos = _SEAT_POSITIONS[position_idx % 6]

    name            = str(player.get("name") or "?")[:15]
    stack           = player.get("stack", 0)
    prob            = float(player.get("probability", 0.0))
    winner          = bool(player.get("is_winner", False))
    fav             = bool(player.get("is_favorite", False))
    folded          = bool(player.get("folded", False))
    is_active       = bool(player.get("is_active", False))
    last_action     = str(player.get("last_action") or "")
    last_action_type= str(player.get("last_action_type") or "")
    cards_s         = str(player.get("hole_cards") or "")

    # Cards
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

    # Probability bar color
    if winner:
        bar_col, label_col = "#F4B942", "#F4B942"
    elif fav:
        bar_col, label_col = "#4CAF82", "#4CAF82"
    else:
        bar_col, label_col = "#3A5A8A", "#4A6A9A"

    badge_html = ""
    if winner:
        badge_html = '<span class="badge-w">★ Gagnant</span>'
    elif fav:
        badge_html = '<span class="badge-f">Favori</span>'

    # active border overrides winner/fav so it's always additive
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
        act_col = {"fold": "#C94040", "call": "#4CAF82", "raise": "#D4A017"}.get(
            last_action_type, "#94AECF"
        )
        action_html = f'<div class="paction" style="color:{act_col}">{last_action}</div>'

    stack_display = f"{int(stack):,}".replace(",", " ")

    return f"""
  <div class="{classes}" style="left:{pos['left']};top:{pos['top']}">
    <div class="seat-head">
      <span class="pname">{name}</span>
      {badge_html}
    </div>
    {cards_html}
    {action_html}
    <div class="pstack">{stack_display}</div>
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
    board_cards: list[tuple[str, str]] | None = None,
    height: int = 580,
) -> str:
    """
    Build a self-contained HTML string for the poker table component.

    players: list of dicts with keys:
        name, stack, hole_cards (e.g. "AhKs"), probability,
        is_winner, is_favorite, folded
    board_cards: list of (rank, suit) tuples, or None for placeholder
    """
    seats_html = "\n".join(
        _seat_html(p, i) for i, p in enumerate(players)
    )

    if board_cards:
        board_html = "".join(_card_html(r, s) for r, s in board_cards)
    else:
        board_html = "".join('<div class="card-ph">?</div>' for _ in range(5))

    meta = f"Main #{hand_id}" if hand_id else ""

    return _TEMPLATE.format(
        css=_CSS,
        seats_html=seats_html,
        board_html=board_html,
        meta=meta,
    )
