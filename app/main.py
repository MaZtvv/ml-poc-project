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
    st.markdown("### ♠ PokerMind AI")
    st.caption("Analyse de poker par apprentissage automatique")
    st.divider()

_pages = [
    st.Page("pages/hand_browser.py",     title="Navigateur de mains",     icon="🃏"),
    st.Page("pages/hand_table.py",       title="Table de jeu",            icon="♠"),
    st.Page("pages/model_comparison.py", title="Comparaison des modèles", icon="🎯"),
]

pg = st.navigation(_pages)

with st.sidebar:
    st.divider()
    st.caption("Dataset : Pluribus NL Texas Hold'em")
    st.caption("Projet universitaire · 2025–2026")

pg.run()
