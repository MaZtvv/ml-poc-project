import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="PokerMind AI",
    page_icon="♠",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css_path = Path(__file__).parent / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        '<div style="padding:0.4rem 0 0.6rem">'
        '<div style="font-size:1.05rem;font-weight:700;color:#E2E8F0;'
        'letter-spacing:-0.01em;line-height:1.3">♠ PokerMind AI</div>'
        '<div style="font-size:0.72rem;color:#2D3F52;margin-top:0.25rem;'
        'letter-spacing:0.02em;font-weight:500">Intelligence Artificielle · Poker</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

_pages = [
    st.Page("pages/hand_browser.py",     title="Navigateur de mains",     icon="🃏"),
    st.Page("pages/hand_table.py",       title="Table de jeu",            icon="♠"),
    st.Page("pages/analysis.py",         title="Analyse ML",              icon="📈"),
    st.Page("pages/model_comparison.py", title="Comparaison des modèles", icon="🎯"),
]

pg = st.navigation(_pages)

with st.sidebar:
    st.divider()
    st.markdown(
        '<div style="font-size:0.71rem;color:#1E2D3E;line-height:1.9;'
        'letter-spacing:0.01em">'
        'Dataset · Pluribus NL Texas Hold\'em<br>'
        'Modèle · Logistic Regression préflop<br>'
        'Projet universitaire · 2025–2026'
        '</div>',
        unsafe_allow_html=True,
    )

pg.run()
