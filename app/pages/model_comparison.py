import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.data import load_player_features
from core.model import (
    INHAND_FEATURES,
    LEAKAGE_FEATURES,
    compute_inhand_analysis,
    compute_preflop_analysis,
    evaluate_models,
)
from core.ui import (
    COLOR_ACCENT,
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

df         = load_player_features()
pre_ana    = compute_preflop_analysis(df)
inhand_ana = compute_inhand_analysis(df)
results    = evaluate_models(df)

pre_lr_auc    = float(pre_ana["metrics"]["roc_auc"])
inhand_lr_auc = float(inhand_ana["lr_auc"])

# ─── Page header ──────────────────────────────────────────────────────────────

page_header(
    "Comparaison des modèles",
    "Préflop baseline propre · Approximation in-hand · Référence diagnostique (fuite)",
)

# ─── Model hierarchy badges ───────────────────────────────────────────────────

st.markdown(
    '<div style="display:flex;gap:0.75rem;align-items:center;margin-bottom:0.5rem;flex-wrap:wrap">'
    '<span style="background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.25);'
    'color:#38BDF8;font-size:0.72rem;font-weight:700;padding:3px 10px;border-radius:20px;'
    'letter-spacing:0.06em;text-transform:uppercase">① Baseline principal (préflop)</span>'
    '<span style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.22);'
    'color:#22C55E;font-size:0.72rem;font-weight:700;padding:3px 10px;border-radius:20px;'
    'letter-spacing:0.06em;text-transform:uppercase">② Approximation in-hand (propre)</span>'
    '<span style="background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.22);'
    'color:#F59E0B;font-size:0.72rem;font-weight:700;padding:3px 10px;border-radius:20px;'
    'letter-spacing:0.06em;text-transform:uppercase">③ Référence diagnostique (fuite)</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ─── Column config shared ─────────────────────────────────────────────────────

METRICS_COLUMN_CONFIG = {
    "modèle": st.column_config.TextColumn("Modèle", width="medium"),
    "roc_auc": st.column_config.ProgressColumn(
        "AUC-ROC ↑", min_value=0.0, max_value=1.0, format="%.3f", width="medium",
    ),
    "rappel":    st.column_config.NumberColumn("Rappel",    format="%.3f"),
    "f1":        st.column_config.NumberColumn("F1",        format="%.3f"),
    "précision": st.column_config.NumberColumn("Précision", format="%.3f"),
    "accuracy":  st.column_config.NumberColumn("Accuracy",  format="%.3f"),
    "acc_train": st.column_config.NumberColumn("Acc. train", format="%.3f"),
}
DISPLAY_ORDER = ["modèle", "roc_auc", "rappel", "f1", "précision", "accuracy", "acc_train"]

# ─── Three tabs ───────────────────────────────────────────────────────────────

tab_pre, tab_inhand, tab_diag = st.tabs([
    "✅  Préflop — baseline propre",
    "🔵  In-Hand — approximation propre",
    "⚠️  Référence diagnostique (fuite)",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Preflop baseline (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

with tab_pre:
    callout(
        "<strong>Ce modèle est le seul baseline académiquement valide de ce projet.</strong><br>"
        "Il utilise uniquement les informations disponibles <em>avant</em> les décisions : "
        "cartes initiales, position, blindes et taille de pile. "
        "Aucune variable de ce jeu ne dépend de l'issue de la main — "
        "la prédiction est chronologiquement antérieure aux décisions.<br>"
        "Un AUC-ROC de ~0.60–0.65 à partir des seules cartes préflop est un signal réel, "
        "pas un artefact. Entraîné avec <code>class_weight='balanced'</code>, "
        "évalué sur un ensemble de test stratifié (80/20)."
    )

    section_label("Résultats sur l'ensemble de test")
    pre_df = results["preflop"][DISPLAY_ORDER]
    st.dataframe(pre_df, column_config=METRICS_COLUMN_CONFIG, use_container_width=True, hide_index=True)

    st.markdown("")
    section_label("Importance des variables — Forêt Aléatoire préflop")

    imp_df  = results["importance"]
    top_imp = imp_df.head(12).copy()

    fig_imp = px.bar(
        top_imp[::-1],
        x="importance", y="variable", orientation="h",
        template=PLOTLY_TEMPLATE,
        color="importance",
        color_continuous_scale=[[0, "#0F2035"], [0.5, "#1D4A7A"], [1, COLOR_PRIMARY]],
        title="Top 12 variables — modèle préflop",
        labels={"importance": "Importance relative", "variable": ""},
    )
    fig_imp.update_traces(textfont=dict(color="#9CA3AF"))
    fig_imp.update_layout(coloraxis_showscale=False)
    fig_imp = apply_chart_defaults(fig_imp)
    st.plotly_chart(fig_imp, use_container_width=True)

    callout(
        "Les variables liées aux cartes (<strong>preflop_strength_score</strong>, "
        "<strong>high_card_rank</strong>, <strong>low_card_rank</strong>) dominent. "
        "À l'inverse, <strong>starting_stack</strong>, <strong>number_of_players</strong> "
        "et <strong>variant</strong> ont une importance nulle : dans Pluribus, "
        "tous les joueurs démarrent avec 10 000 jetons, toutes les parties ont 6 joueurs "
        "et une seule variante est présente — ces variables sont <em>constantes</em> et n'apportent aucun signal."
    )

    st.divider()
    section_label("Validation croisée stratifiée — 5 plis")

    callout(
        "La validation croisée stratifiée à 5 plis produit une estimation robuste des performances, "
        "indépendante du partage train/test choisi. "
        "Chaque pli respecte la proportion de classes (83/17). "
        "Les résultats sont présentés sous la forme <strong>moyenne ± écart-type</strong>.",
        kind="info",
    )

    cv_df = results["cv"]
    st.dataframe(
        cv_df,
        column_config={
            "modèle":   st.column_config.TextColumn("Modèle",   width="medium"),
            "accuracy": st.column_config.TextColumn("Accuracy"),
            "rappel":   st.column_config.TextColumn("Rappel"),
            "f1":       st.column_config.TextColumn("F1"),
            "roc_auc":  st.column_config.TextColumn("AUC-ROC"),
        },
        use_container_width=True,
        hide_index=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Clean in-hand approximation (new, the main upgrade)
# ══════════════════════════════════════════════════════════════════════════════

with tab_inhand:

    callout(
        "<strong>Approximation in-hand — sans <code>player_num_folds</code>.</strong><br>"
        "Ce modèle conserve les variables préflop et ajoute des informations sur le style d'action "
        "(<code>player_num_check_calls</code>, <code>player_num_bets_raises</code>, "
        "<code>num_board_deals</code>…), sans inclure la tautologie principale. "
        "Dans le replay, les probabilités sont recalculées à chaque action et à chaque nouvelle street — "
        "les joueurs couchés sont exclus par logique de jeu explicite, pas par variable de fuite.<br>"
        f"AUC-ROC préflop seul : <strong>{pre_lr_auc:.3f}</strong> · "
        f"AUC-ROC in-hand propre : <strong>{inhand_lr_auc:.3f}</strong> · "
        "Gain réel — informations d'action supplémentaires sans tautologie.",
        kind="info",
    )

    section_label("Résultats sur l'ensemble de test (mêmes splits que le préflop)")
    iha_df = inhand_ana["metrics_df"][DISPLAY_ORDER]
    st.dataframe(iha_df, column_config=METRICS_COLUMN_CONFIG, use_container_width=True, hide_index=True)

    st.markdown("")

    # AUC visual comparison
    section_label("Comparaison AUC-ROC — préflop vs in-hand (propre)")

    _auc_fig = go.Figure()
    _models  = ["Préflop (baseline)", "In-Hand (propre)"]
    _aucs    = [pre_lr_auc, inhand_lr_auc]
    _cols    = [COLOR_PRIMARY, COLOR_WIN]
    for _m, _a, _c in zip(_models, _aucs, _cols):
        _auc_fig.add_trace(go.Bar(
            x=[_a], y=[_m], orientation="h",
            marker_color=_c, opacity=0.85,
            text=[f"AUC = {_a:.3f}"],
            textposition="outside",
            textfont=dict(color="#9CA3AF", size=12),
            hovertemplate=f"{_m}: %{{x:.3f}}<extra></extra>",
        ))
    _auc_fig.update_layout(
        height=140,
        xaxis=dict(range=[0.50, 1.0], title="AUC-ROC"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        title="AUC-ROC par modèle (LR, même split 80/20)",
        barmode="overlay",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=0, r=80, t=44, b=0),
    )
    _auc_fig = apply_chart_defaults(_auc_fig)
    _auc_fig.add_vline(x=pre_lr_auc, line_dash="dot",
                       line_color="rgba(56,189,248,0.3)", line_width=1.5)
    st.plotly_chart(_auc_fig, use_container_width=True)

    st.divider()
    section_label("Importance des variables — modèle in-hand (Forêt Aléatoire)")

    _inhand_imp = inhand_ana["importance"].head(14).copy()

    _inhand_colors = []
    for v in _inhand_imp["variable"]:
        if v in ("player_num_check_calls", "player_num_bets_raises"):
            _inhand_colors.append(COLOR_WIN)
        elif v in ("num_board_deals", "hand_action_count", "num_check_calls", "num_bets_raises"):
            _inhand_colors.append(COLOR_ACCENT)
        else:
            _inhand_colors.append(COLOR_PRIMARY)

    fig_inh = px.bar(
        _inhand_imp[::-1],
        x="importance", y="variable", orientation="h",
        template=PLOTLY_TEMPLATE,
        title="Variables in-hand — vert = style d'action · orange = état de la main · bleu = préflop",
        labels={"importance": "Importance relative", "variable": ""},
    )
    fig_inh.update_traces(
        marker_color=_inhand_colors[::-1],
        textfont=dict(color="#9CA3AF"),
    )
    fig_inh = apply_chart_defaults(fig_inh)
    st.plotly_chart(fig_inh, use_container_width=True)

    callout(
        "Les variables d'action (<code>player_num_check_calls</code>, <code>player_num_bets_raises</code>) "
        "contribuent significativement au-delà des cartes préflop — elles capturent le style de jeu du joueur "
        "(agressif / passif) sans encoder directement l'issue. "
        "<code>num_board_deals</code> encode le nombre de streets atteintes (0 = fin préflop, "
        "1 = flop, 2 = turn, 3 = river) — les mains longues ont statistiquement un profil différent. "
        "Aucune de ces variables n'est une tautologie directe : un joueur peut avoir beaucoup relancé "
        "et quand même perdre au river."
    )

    st.divider()
    section_label("Limites résiduelles de ce modèle")

    callout(
        "<strong>Ce modèle est une approximation, pas une prédiction strictement chronologique.</strong><br>"
        "<code>player_num_check_calls</code> et <code>player_num_bets_raises</code> résument "
        "la main <em>entière</em> du joueur, pas seulement les actions passées. "
        "Un joueur qui fold tôt aura mécaniquement moins d'actions — la corrélation avec la victoire "
        "est réelle mais <em>pas tautologique</em> (aucune règle de jeu ne l'impose à 100 %). "
        "La variable <code>player_num_folds</code>, elle, est tautologique : "
        "fold → <code>player_won = 0</code> dans 100 % des cas.<br>"
        "Le gain d'AUC par rapport au préflop reflète des informations d'action réelles — "
        "mais une partie de ce gain reste due à la corrélation indirecte avec les folds.",
        kind="warning",
    )

    # ── Leakage anatomy (collapsed) ────────────────────────────────────────────
    with st.expander("Anatomie de la fuite directe — pourquoi le modèle main complète est invalide", expanded=False):
        st.markdown(
            """
**Mécanisme exact de la fuite principale :**

`player_num_folds` est une variable binaire (0 ou 1). Elle vaut 1 si le joueur s'est couché.

**Un joueur qui se couche ne peut jamais gagner le pot.** Sa valeur de `player_won` est mécaniquement contrainte à 0 dès que `player_num_folds = 1`.

> *A player who has folded is almost certain not to win the hand — fold-related features encode the result rather than predict it.*

**Ce n'est pas une corrélation apprise : c'est une tautologie encodée dans les règles du jeu.**

| `player_num_folds` | `player_won` |
|---|---|
| 1 (le joueur s'est couché) | Toujours 0 |
| 0 (le joueur n'a pas plié) | 0 ou 1 |

**Problème chronologique :** `player_num_folds` est calculée *après* que le fold a eu lieu — elle est chronologiquement en aval de l'issue. Inclure cette variable constitue une fuite directe.

Le modèle main complète (avec fuite) atteignait AUC ~0.98–0.99 grâce à cette tautologie. Le modèle in-hand propre, sans cette variable, est honnête.
            """
        )

        section_label("Importance des variables — modèle main complète (fuite visible)")

        leak_df  = results["leakage"]
        top_leak = leak_df.head(12).copy()
        bar_colors = [
            COLOR_LOSS if v in LEAKAGE_FEATURES else COLOR_NEUTRAL
            for v in top_leak["variable"]
        ]

        fig_leak = px.bar(
            top_leak[::-1],
            x="importance", y="variable", orientation="h",
            template=PLOTLY_TEMPLATE,
            title="Top 12 variables — modèle main complète (rouge = fuite directe)",
            labels={"importance": "Importance relative", "variable": ""},
        )
        fig_leak.update_traces(
            marker_color=bar_colors[::-1],
            textfont=dict(color="#9CA3AF"),
        )
        fig_leak = apply_chart_defaults(fig_leak)
        st.plotly_chart(fig_leak, use_container_width=True)

        callout(
            "<strong><code>player_num_folds</code> représente ~84 % de l'importance totale "
            "du modèle main complète.</strong><br>"
            "Le modèle n'a pas appris à jouer au poker — il a appris à détecter les joueurs "
            "qui se sont couchés, information disponible uniquement après la fin de la main. "
            "Le modèle in-hand propre (onglet précédent) corrige ce problème.",
            kind="warning",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Diagnostic reference (full leaky model, shown for comparison only)
# ══════════════════════════════════════════════════════════════════════════════

with tab_diag:
    callout(
        "<strong>⚠ Ce modèle n'est pas un baseline valide — il est présenté ici uniquement comme référence diagnostique.</strong><br>"
        "Il inclut <code>player_num_folds</code> et d'autres variables d'action qui "
        "ne sont disponibles qu'après les décisions. "
        "Ses performances apparentes (AUC quasi parfait) ne reflètent <em>pas</em> une capacité prédictive réelle — "
        "elles mesurent la capacité à détecter un fait déjà accompli.<br>"
        "Le modèle in-hand propre (onglet précédent) est la version utile.",
        kind="warning",
    )

    section_label("Résultats sur l'ensemble de test")
    full_df = results["full"][DISPLAY_ORDER]
    st.dataframe(full_df, column_config=METRICS_COLUMN_CONFIG, use_container_width=True, hide_index=True)

st.divider()

# ─── Global methodology note ──────────────────────────────────────────────────

section_label("Note méthodologique commune")
callout(
    "<strong>Protocole commun :</strong> "
    "partition stratifiée 80/20 (random_state=42), "
    "<code>class_weight='balanced'</code> sur tous les modèles, "
    "prétraitement en pipeline (imputation médiane → normalisation pour le numérique, "
    "imputation mode → OneHotEncoding pour <code>variant</code>). "
    "L'AUC-ROC est la métrique principale — robuste au déséquilibre de classes et indépendante du seuil."
)
callout(
    "<strong>Hiérarchie des modèles dans ce projet :</strong><br>"
    "① <strong>Modèle préflop</strong> — baseline principal, propre, utilisé dans le navigateur de mains.<br>"
    "② <strong>Modèle in-hand propre</strong> — approximation dynamique, utilisée dans le replay hand-by-hand. "
    "Sans <code>player_num_folds</code>. Mis à jour à chaque action et à chaque street.<br>"
    "③ <strong>Modèle main complète (fuite)</strong> — référence diagnostique uniquement. "
    "Ne doit pas être interprété comme une capacité prédictive réelle.",
    kind="info",
)
