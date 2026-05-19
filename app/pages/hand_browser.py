import streamlit as st

from core.data import load_player_features
from core.model import build_hand_summary
from core.ui import callout, page_header, section_label

# ─── Data ─────────────────────────────────────────────────────────────────────

df      = load_player_features()
summary = build_hand_summary(df)

total_hands  = len(summary)
correct_pct  = summary["correct"].mean()
avg_fav_prob = summary["prob_max"].mean()


def _short_label(composite_id: str) -> str:
    """'pluribus/100/0.phh::0' → '100/0.phh · main 0'"""
    path, hid = composite_id.rsplit("::", 1)
    parts = path.split("/")
    return f"{'/'.join(parts[-2:])!s} · main {hid}"


# ─── Page header ──────────────────────────────────────────────────────────────

page_header(
    "Navigateur de mains",
    "Parcourez les mains disponibles et sélectionnez-en une pour l'analyser.",
)

# ─── KPI row ──────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mains disponibles",       f"{total_hands:,}".replace(",", " "))
c2.metric("Prédictions correctes",   f"{correct_pct:.1%}")
c3.metric("Prob. moyenne du favori", f"{avg_fav_prob:.1%}")
c4.metric("Joueurs par main",        "6")

st.divider()

# ─── Filters ──────────────────────────────────────────────────────────────────

section_label("Filtrer les mains")

col_f1, col_f2, _ = st.columns([1, 1, 2])

with col_f1:
    filter_correct = st.selectbox(
        "Prédiction",
        options=["Toutes", "Correctes ✓", "Incorrectes ✗"],
    )
with col_f2:
    filter_player = st.selectbox(
        "Joueur favori",
        options=["Tous"] + sorted(summary["favorite"].unique().tolist()),
    )

display_df = summary.copy()
if filter_correct == "Correctes ✓":
    display_df = display_df[display_df["correct"]]
elif filter_correct == "Incorrectes ✗":
    display_df = display_df[~display_df["correct"]]
if filter_player != "Tous":
    display_df = display_df[display_df["favorite"] == filter_player]

st.divider()

# ─── Hand list ────────────────────────────────────────────────────────────────

section_label(f"{len(display_df)} mains — sélectionnez une ligne puis cliquez sur Analyser")

display_df_ui = display_df.copy()
display_df_ui["main"]    = display_df_ui["composite_id"].map(_short_label)
display_df_ui["correct"] = display_df_ui["correct"].map({True: "✓", False: "✗"})

COLUMNS_TO_SHOW = ["main", "source", "n_players", "favorite", "prob_max", "winner", "correct"]

selected = st.dataframe(
    display_df_ui[COLUMNS_TO_SHOW].rename(columns={
        "main":      "Main",
        "source":    "Source",
        "n_players": "Joueurs",
        "favorite":  "Favori prédit",
        "prob_max":  "Probabilité",
        "winner":    "Gagnant réel",
        "correct":   "Correct",
    }),
    column_config={
        "Main":          st.column_config.TextColumn(width="medium"),
        "Source":        st.column_config.TextColumn(width="medium"),
        "Joueurs":       st.column_config.NumberColumn(width="small"),
        "Favori prédit": st.column_config.TextColumn(width="medium"),
        "Probabilité":   st.column_config.ProgressColumn(
            min_value=0.0, max_value=1.0, format="%.1%%", width="medium"
        ),
        "Gagnant réel":  st.column_config.TextColumn(width="medium"),
        "Correct":       st.column_config.TextColumn(width="small"),
    },
    use_container_width=True,
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun",
    key="hand_browser_selection",
)

# ─── Selection + navigation ───────────────────────────────────────────────────

selected_rows = selected.selection.rows if hasattr(selected, "selection") else []

if selected_rows:
    row_idx      = selected_rows[0]
    composite_id = display_df.iloc[row_idx]["composite_id"]
    winner       = display_df.iloc[row_idx]["winner"]
    favorite     = display_df.iloc[row_idx]["favorite"]
    prob         = display_df.iloc[row_idx]["prob_max"]
    correct      = display_df.iloc[row_idx]["correct"]

    st.divider()
    info_col, btn_col = st.columns([3, 1])

    with info_col:
        correct_label = "✓ Prédiction correcte" if correct else "✗ Prédiction incorrecte"
        callout(
            f"<strong>{_short_label(composite_id)}</strong><br>"
            f"Favori prédit : <strong>{favorite}</strong> ({prob:.1%})"
            f" &nbsp;·&nbsp; Gagnant réel : <strong>{winner}</strong>"
            f" &nbsp;·&nbsp; {correct_label}"
        )

    with btn_col:
        st.markdown("")
        if st.button("Analyser →", type="primary", use_container_width=True):
            st.session_state["selected_composite_id"] = composite_id
            st.switch_page("pages/hand_table.py")
else:
    st.caption("Sélectionnez une ligne dans le tableau pour analyser la main.")
