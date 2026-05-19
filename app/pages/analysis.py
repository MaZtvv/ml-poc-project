import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.data import load_player_features
from core.model import (
    compute_preflop_analysis,
    compute_winrate_stats,
)
from core.ui import (
    COLOR_ACCENT,
    COLOR_GRAY,
    COLOR_LOSS,
    COLOR_NEUTRAL,
    COLOR_PRIMARY,
    COLOR_WIN,
    PLOTLY_TEMPLATE,
    apply_chart_defaults,
    callout,
    page_header,
    section_label,
)

# ─── Data ─────────────────────────────────────────────────────────────────────

df      = load_player_features()
ana     = compute_preflop_analysis(df)
wr      = compute_winrate_stats(df)
y_test  = ana["y_test"]
y_proba = ana["y_proba"]
metrics = ana["metrics"]
coef_df = ana["coef_df"]

baseline_wr = float(df["player_won"].mean())


# ─── Shared chart builder ──────────────────────────────────────────────────────

def _fig(title: str, height: int = 340, **kw) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        height=height,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=0, r=0, t=44, b=0),
        **kw,
    )
    return apply_chart_defaults(fig)


# ─── Page header ──────────────────────────────────────────────────────────────

page_header(
    "Analyse ML",
    "Baseline préflop · Régression Logistique · Pluribus NL Texas Hold'em",
)

# ═══════════════════════════════════════════════════════════════════════════════
# A. DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

section_label("A — Vue d'ensemble du dataset")

n_hands   = int(df["composite_id"].nunique())
n_rows    = len(df)
n_winners = int(df["player_won"].sum())
n_losers  = n_rows - n_winners
avg_profit = float(df["profit"].mean()) if "profit" in df.columns else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mains uniques",   f"{n_hands:,}".replace(",", " "))
c2.metric("Lignes joueurs",  f"{n_rows:,}".replace(",", " "))
c3.metric("Gagnants",        f"{n_winners:,}".replace(",", " "))
c4.metric("Taux de victoire", f"{baseline_wr:.1%}")
c5.metric("Ratio 0 / 1",     f"{(1 - baseline_wr) / baseline_wr:.1f}×")

callout(
    f"Le dataset Pluribus contient <strong>{n_hands:,} mains distinctes</strong> "
    f"({n_rows:,} lignes joueurs, 6 joueurs par main). "
    "La cible <code>player_won</code> est fortement déséquilibrée : "
    f"<strong>{baseline_wr:.1%} de gagnants</strong>, cohérent avec une partie à 6 joueurs (1/6 ≈ 16.7 %). "
    "Ce déséquilibre de classes impose d'utiliser <code>class_weight='balanced'</code> "
    "et de choisir l'<strong>AUC-ROC</strong> comme métrique principale plutôt que l'accuracy brute, "
    "qui serait artificiellement élevée (un modèle qui prédit toujours « perdant » obtiendrait ~83 %)."
)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# B. ML PERFORMANCE — METRICS + ROC + PR
# ═══════════════════════════════════════════════════════════════════════════════

section_label("B — Performances du baseline préflop  (Logistic Regression · ensemble de test 20 %)")

m = metrics
ca, cb, cc, cd, ce = st.columns(5)
ca.metric("AUC-ROC",   f"{m['roc_auc']:.3f}")
cb.metric("F1",        f"{m['f1']:.3f}")
cc.metric("Recall",    f"{m['recall']:.3f}")
cd.metric("Précision", f"{m['precision']:.3f}")
ce.metric("Accuracy",  f"{m['accuracy']:.3f}")

st.markdown("")
col_roc, col_pr = st.columns(2)

# ── ROC curve ──────────────────────────────────────────────────────────────────
with col_roc:
    fig_roc = _fig("Courbe ROC — Logistic Regression préflop")
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name=f"Aléatoire (AUC = 0.500)",
        line=dict(dash="dash", color=COLOR_GRAY, width=1.5),
    ))
    fig_roc.add_trace(go.Scatter(
        x=ana["fpr"].tolist(), y=ana["tpr"].tolist(),
        mode="lines",
        name=f"LR préflop  (AUC = {m['roc_auc']:.3f})",
        line=dict(color=COLOR_PRIMARY, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.08)",
    ))
    fig_roc.update_layout(
        xaxis_title="Taux de faux positifs (FPR)",
        yaxis_title="Taux de vrais positifs (TPR)",
        xaxis_range=[-0.02, 1.02],
        yaxis_range=[-0.02, 1.05],
        legend=dict(x=0.38, y=0.08, bgcolor="rgba(10,18,30,0.92)",
                    bordercolor="#1A2840", borderwidth=1,
                    font=dict(color="#D1D9E6", size=11)),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# ── PR curve ───────────────────────────────────────────────────────────────────
with col_pr:
    fig_pr = _fig("Courbe Précision–Rappel")
    fig_pr.add_shape(
        type="line", x0=0, x1=1, y0=baseline_wr, y1=baseline_wr,
        line=dict(dash="dash", color=COLOR_GRAY, width=1.5),
    )
    fig_pr.add_trace(go.Scatter(
        x=ana["rec"].tolist(), y=ana["prec"].tolist(),
        mode="lines",
        name="LR préflop",
        line=dict(color=COLOR_NEUTRAL, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(96,165,250,0.08)",
    ))
    fig_pr.add_annotation(
        x=0.88, y=baseline_wr + 0.018,
        text=f"Aléatoire ({baseline_wr:.1%})",
        showarrow=False, font=dict(size=10, color=COLOR_GRAY),
    )
    fig_pr.update_layout(
        xaxis_title="Rappel (Recall)",
        yaxis_title="Précision (Precision)",
        xaxis_range=[-0.02, 1.02],
        yaxis_range=[-0.02, 1.05],
        showlegend=False,
    )
    st.plotly_chart(fig_pr, use_container_width=True)

callout(
    f"L'<strong>AUC-ROC de {m['roc_auc']:.3f}</strong> confirme une capacité discriminante réelle mais modeste, "
    "bien au-dessus du hasard (0.50). "
    "Le recall élevé traduit la priorité donnée à la détection des gagnants "
    "via <code>class_weight='balanced'</code>, au prix d'une précision plus faible — "
    "arbitrage visible sur la courbe Précision–Rappel. "
    "La baseline aléatoire sur cette courbe est la fréquence de la classe positive "
    f"({baseline_wr:.1%}), rappelant que la tâche est intrinsèquement difficile : "
    "prédire la victoire à partir des seules cartes préflop sans voir les actions ni le board."
)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# C. PROBABILITY DISTRIBUTION & CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

section_label("C — Distribution des probabilités prédites & calibration")

col_hist, col_cal = st.columns(2)

# ── Histogram by class ─────────────────────────────────────────────────────────
with col_hist:
    mask0 = y_test == 0
    mask1 = y_test == 1
    fig_hist = _fig("Distribution des probabilités prédites (par classe réelle)")
    fig_hist.add_trace(go.Histogram(
        x=y_proba[mask0].tolist(),
        nbinsx=28,
        name=f"Perdant (0)  n={mask0.sum()}",
        opacity=0.58,
        marker_color=COLOR_LOSS,
        histnorm="probability density",
    ))
    fig_hist.add_trace(go.Histogram(
        x=y_proba[mask1].tolist(),
        nbinsx=28,
        name=f"Gagnant (1)  n={mask1.sum()}",
        opacity=0.72,
        marker_color=COLOR_WIN,
        histnorm="probability density",
    ))
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title="Probabilité prédite de victoire",
        yaxis_title="Densité",
        legend=dict(x=0.50, y=0.96, bgcolor="rgba(10,18,30,0.92)",
                    bordercolor="#1A2840", borderwidth=1,
                    font=dict(color="#D1D9E6", size=11)),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Calibration curve ──────────────────────────────────────────────────────────
with col_cal:
    fig_cal = _fig("Courbe de calibration (reliability diagram)")
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Calibration parfaite",
        line=dict(dash="dash", color=COLOR_GRAY, width=1.5),
    ))
    fig_cal.add_trace(go.Scatter(
        x=ana["cal_mean"].tolist(),
        y=ana["cal_frac"].tolist(),
        mode="lines+markers",
        name="LR préflop",
        line=dict(color=COLOR_PRIMARY, width=2.5),
        marker=dict(size=8, color=COLOR_PRIMARY),
        hovertemplate="Prédit: %{x:.2f}<br>Observé: %{y:.2f}<extra></extra>",
    ))
    fig_cal.update_layout(
        xaxis_title="Probabilité prédite moyenne (par bin)",
        yaxis_title="Fréquence observée de victoire",
        xaxis_range=[-0.02, 1.02],
        yaxis_range=[-0.02, 1.02],
        legend=dict(x=0.04, y=0.94, bgcolor="rgba(10,18,30,0.92)",
                    bordercolor="#1A2840", borderwidth=1,
                    font=dict(color="#D1D9E6", size=11)),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

callout(
    "L'histogramme révèle un <strong>chevauchement important</strong> entre les distributions des gagnants et "
    "des perdants — les deux classes occupent une plage similaire de probabilités. "
    "C'est un indicateur direct de la difficulté de la tâche : les cartes préflop seules "
    "ne permettent pas une séparation franche. "
    "La courbe de calibration compare probabilité prédite et fréquence observée. "
    "Un modèle parfaitement calibré suivrait la diagonale. "
    "<strong>Une déviation systématique vers le haut ou le bas signifie que les probabilités "
    "affichées dans le replay ne sont pas des estimations absolues fiables — "
    "elles reflètent un classement relatif, pas une probabilité de gain exacte.</strong> "
    "Si la calibration est mauvaise, il faudrait appliquer une post-calibration "
    "(ex. Platt scaling ou isotonic regression) avant d'afficher des probabilités aux utilisateurs."
)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# D. MODEL INTERPRETATION — LR COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════════

section_label("D — Interprétation du modèle  (coefficients Logistic Regression)")

top_coef    = coef_df.head(12).copy()
coef_colors = [COLOR_WIN if c > 0 else COLOR_LOSS for c in top_coef["coef"]]

fig_coef = go.Figure(go.Bar(
    x=top_coef["coef"].tolist(),
    y=top_coef["feature"].tolist(),
    orientation="h",
    marker_color=coef_colors,
    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    text=[f"{c:+.3f}" for c in top_coef["coef"]],
    textposition="outside",
    textfont=dict(size=11, color="#9CA3AF"),
))
fig_coef.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1.2)
fig_coef.update_layout(
    title="Coefficients LR — top 12 par valeur absolue  (positif ↑ victoire · négatif ↓ victoire)",
    xaxis_title="Coefficient (espace log-odds)",
    yaxis_title="",
    yaxis=dict(autorange="reversed"),
    height=400,
    template=PLOTLY_TEMPLATE,
    margin=dict(l=0, r=60, t=44, b=0),
    showlegend=False,
)
fig_coef = apply_chart_defaults(fig_coef)
st.plotly_chart(fig_coef, use_container_width=True)

callout(
    "Les coefficients mesurent l'effet de chaque variable en <strong>log-odds</strong> : "
    "un coefficient positif augmente la probabilité prédite de victoire, "
    "un coefficient négatif la diminue (après normalisation StandardScaler). "
    "<strong>preflop_strength_score</strong> est le signal le plus fort — "
    "cohérent avec l'intuition : une main plus forte gagne plus souvent. "
    "Les variables de rang (<strong>high_card_rank</strong>, <strong>low_card_rank</strong>) "
    "contribuent aussi positivement. "
    "À l'inverse, <strong>is_small_blind</strong> et <strong>is_big_blind</strong> ont un effet négatif : "
    "les blindes sont forcées d'investir en aveugle, ce qui dégrade leur résultat moyen. "
    "<strong>rank_gap</strong> négatif signifie que les mains avec un grand écart entre les cartes "
    "(ex. As-2) gagnent moins que les connecteurs. "
    "Note : les variables constantes dans Pluribus (<em>starting_stack</em>, <em>number_of_players</em>) "
    "ont été exclues — elles n'apportent aucun signal dans ce dataset homogène.",
    kind="info",
)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# E. STATISTICAL PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

section_label("E — Patterns statistiques observés dans le dataset")

col_buck, col_cat = st.columns(2)

# ── Win rate by score quintile ─────────────────────────────────────────────────
with col_buck:
    by_bucket = wr["by_bucket"]
    bucket_colors = [
        COLOR_LOSS if i < 2 else (COLOR_ACCENT if i == 2 else COLOR_WIN)
        for i in range(len(by_bucket))
    ]
    fig_buck = go.Figure(go.Bar(
        x=by_bucket["bucket"].tolist(),
        y=by_bucket["win_rate"].tolist(),
        marker_color=bucket_colors,
        hovertemplate="%{x}<br>Taux de victoire : %{y:.1%}<extra></extra>",
        text=[f"{v:.1%}" for v in by_bucket["win_rate"]],
        textposition="outside",
        textfont=dict(size=12, color="#9CA3AF"),
    ))
    fig_buck.add_hline(
        y=baseline_wr, line_dash="dash",
        line_color=COLOR_NEUTRAL, line_width=1.5,
        annotation_text=f"Moyenne {baseline_wr:.1%}",
        annotation_position="top left",
        annotation_font=dict(size=10, color=COLOR_NEUTRAL),
    )
    fig_buck.update_layout(
        title="Taux de victoire observé par quintile de force préflop",
        xaxis_title="Quintile (Q1 = mains faibles, Q5 = mains fortes)",
        yaxis_title="Taux de victoire",
        yaxis_tickformat=".0%",
        yaxis_range=[0, max(by_bucket["win_rate"]) * 1.28],
        showlegend=False,
        height=350,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=0, r=0, t=44, b=0),
    )
    fig_buck = apply_chart_defaults(fig_buck)
    st.plotly_chart(fig_buck, use_container_width=True)

# ── Win rate by hand category ──────────────────────────────────────────────────
with col_cat:
    by_cat   = wr["by_category"]
    cat_cols  = ["is_pair", "is_suited", "has_ace", "has_king"]
    cat_names = {"is_pair": "Paire", "is_suited": "Couleur", "has_ace": "As", "has_king": "Roi"}

    yes_rows = by_cat[by_cat["yn"] == "Oui"].set_index("col").reindex(cat_cols)
    no_rows  = by_cat[by_cat["yn"] == "Non"].set_index("col").reindex(cat_cols)
    x_labels = [cat_names[c] for c in cat_cols]

    fig_cat = go.Figure()
    fig_cat.add_trace(go.Bar(
        name="Oui",
        x=x_labels,
        y=yes_rows["win_rate"].tolist(),
        marker_color=COLOR_WIN,
        opacity=0.88,
        text=[f"{v:.1%}" for v in yes_rows["win_rate"]],
        textposition="outside",
        textfont=dict(size=12, color="#9CA3AF"),
        hovertemplate="%{x} — Oui: %{y:.1%}<extra></extra>",
    ))
    fig_cat.add_trace(go.Bar(
        name="Non",
        x=x_labels,
        y=no_rows["win_rate"].tolist(),
        marker_color=COLOR_NEUTRAL,
        opacity=0.70,
        text=[f"{v:.1%}" for v in no_rows["win_rate"]],
        textposition="outside",
        textfont=dict(size=12, color="#9CA3AF"),
        hovertemplate="%{x} — Non: %{y:.1%}<extra></extra>",
    ))
    fig_cat.add_hline(
        y=baseline_wr, line_dash="dash",
        line_color=COLOR_ACCENT, line_width=1.5,
        annotation_text=f"Moyenne {baseline_wr:.1%}",
        annotation_position="top left",
        annotation_font=dict(size=10, color=COLOR_ACCENT),
    )
    fig_cat.update_layout(
        barmode="group",
        title="Taux de victoire observé par caractéristique de main",
        xaxis_title="Caractéristique",
        yaxis_title="Taux de victoire",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 0.38],
        legend=dict(x=0.72, y=0.96, bgcolor="rgba(10,18,30,0.92)",
                    bordercolor="#1A2840", borderwidth=1,
                    font=dict(color="#D1D9E6", size=11)),
        height=350,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=0, r=0, t=44, b=0),
    )
    fig_cat = apply_chart_defaults(fig_cat)
    st.plotly_chart(fig_cat, use_container_width=True)

callout(
    "Le quintile Q5 (mains les plus fortes) gagne environ <strong>2× plus souvent</strong> que Q1 "
    "— la progressivité valide la cohérence du <em>preflop_strength_score</em> comme signal. "
    "Pour les caractéristiques de main : les <strong>paires</strong> et les mains <strong>avec un As</strong> "
    "montrent l'avantage statistique le plus net. "
    "L'effet des mains colorées (<em>suited</em>) est plus modeste : "
    "l'avantage suited se réalise principalement en jouant un flush ou un draw, "
    "situations moins fréquentes que celles exploitées par une paire de poche ou un As dominant. "
    "Ces patterns sont observés sur le dataset Pluribus — ils peuvent différer sur un dataset de poker en ligne "
    "où la sélection de mains varie selon les joueurs.",
    kind="info",
)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# F. LEAKAGE REMINDER (thin)
# ═══════════════════════════════════════════════════════════════════════════════

section_label("F — Repère méthodologique  (effet de la fuite d'information)")

callout(
    "Ce modèle préflop est le seul baseline académiquement valide. "
    "Un second modèle « main complète » existe à titre diagnostique — il inclut "
    "<code>player_num_folds</code>, une variable de <strong>fuite directe</strong> : "
    "un joueur couché ne peut pas gagner, donc cette variable encode l'issue plutôt que la prédire. "
    "Pour la comparaison AUC et l'anatomie complète de la fuite, "
    "voir l'onglet <strong>Comparaison des modèles</strong>.",
    kind="warning",
)
