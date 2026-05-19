import plotly.graph_objects as go
import streamlit as st

# ─── Color palette (dark-theme optimised) ────────────────────────────────────

COLOR_PRIMARY  = "#38BDF8"   # sky blue  — main accent on dark background
COLOR_WIN      = "#22C55E"   # green     — positive outcomes
COLOR_LOSS     = "#EF4444"   # red       — negative outcomes / leakage
COLOR_NEUTRAL  = "#60A5FA"   # lighter blue — supporting accent
COLOR_ACCENT   = "#F59E0B"   # amber     — warnings / dealer button
COLOR_GRAY     = "#6B7280"   # muted gray
COLOR_SEQUENCE = [COLOR_PRIMARY, COLOR_NEUTRAL, COLOR_ACCENT, COLOR_WIN, COLOR_LOSS]

PLOTLY_TEMPLATE = "plotly_dark"


# ─── Chart defaults (transparent dark, replaces plotly_white) ─────────────────

def apply_chart_defaults(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, -apple-system, sans-serif",
        font_color="#D1D9E6",
        title_font=dict(size=13, color="#9CA3AF"),
        margin=dict(l=0, r=0, t=44, b=0),
        hoverlabel=dict(
            bgcolor="#111827",
            font_size=13,
            bordercolor="#263244",
            font_color="#F9FAFB",
        ),
        legend=dict(
            bgcolor="rgba(13,20,32,0.90)",
            bordercolor="#1A2840",
            borderwidth=1,
            font=dict(color="#D1D9E6", size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#6B7280", size=10),
        title_font=dict(color="#9CA3AF", size=11),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#6B7280", size=10),
        title_font=dict(color="#9CA3AF", size=11),
    )
    return fig


# ─── UI helpers ───────────────────────────────────────────────────────────────

def callout(text: str, kind: str = "info") -> None:
    css_class = "callout-info" if kind == "info" else "callout-warning"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def section_label(title: str) -> None:
    st.markdown(f'<p class="section-label">{title.upper()}</p>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    sub_html = (
        f'<p style="font-size:0.82rem;color:#3D4F62;margin:0.25rem 0 0;'
        f'font-weight:400;letter-spacing:0.01em">{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="padding:0.1rem 0 1.35rem;'
        f'border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:0.2rem">'
        f'<h1 style="font-size:1.55rem;font-weight:700;color:#F1F5F9;'
        f'letter-spacing:-0.02em;margin:0;line-height:1.25">{title}</h1>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )
