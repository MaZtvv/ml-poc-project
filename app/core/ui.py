import plotly.graph_objects as go
import streamlit as st

# ─── Color palette ────────────────────────────────────────────────────────────

COLOR_PRIMARY  = "#1B3A6B"
COLOR_WIN      = "#2E8B57"
COLOR_LOSS     = "#C0392B"
COLOR_NEUTRAL  = "#4A6FA5"
COLOR_ACCENT   = "#D4A017"
COLOR_GRAY     = "#6B7280"
COLOR_SEQUENCE = [COLOR_PRIMARY, COLOR_NEUTRAL, COLOR_ACCENT, COLOR_WIN, COLOR_LOSS]

PLOTLY_TEMPLATE = "plotly_white"


# ─── Chart defaults ───────────────────────────────────────────────────────────

def apply_chart_defaults(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="Inter, -apple-system, sans-serif",
        font_color="#1A1D23",
        title_font_color=COLOR_PRIMARY,
        title_font_size=14,
        margin=dict(l=0, r=0, t=44, b=0),
        hoverlabel=dict(bgcolor="white", font_size=13, bordercolor="#DDE1E7"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F0F0F0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0", zeroline=False)
    return fig


# ─── UI helpers ───────────────────────────────────────────────────────────────

def callout(text: str, kind: str = "info") -> None:
    css_class = "callout-info" if kind == "info" else "callout-warning"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def section_label(title: str) -> None:
    st.markdown(f'<p class="section-label">{title.upper()}</p>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"## {title}")
    st.caption(subtitle)
    st.markdown("")
