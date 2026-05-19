import plotly.express as px
import streamlit as st

from core.data import load_player_features
from core.model import evaluate_models
from core.ui import (
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

# ─── Data & models ────────────────────────────────────────────────────────────

df      = load_player_features()
results = evaluate_models(df)

# ─── Page header ──────────────────────────────────────────────────────────────

page_header(
    "Comparaison des modèles de base",
    "Trois modèles × deux jeux de variables — baseline préflop honnête vs. référence main complète avec fuite d'information.",
)

# ─── Column config for metrics table ─────────────────────────────────────────

METRICS_COLUMN_CONFIG = {
    "modèle": st.column_config.TextColumn("Modèle", width="medium"),
    "roc_auc": st.column_config.ProgressColumn(
        "AUC-ROC ↑",
        min_value=0.0,
        max_value=1.0,
        format="%.3f",
        width="medium",
    ),
    "rappel": st.column_config.NumberColumn("Rappel", format="%.3f"),
    "f1": st.column_config.NumberColumn("F1", format="%.3f"),
    "précision": st.column_config.NumberColumn("Précision", format="%.3f"),
    "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
    "acc_train": st.column_config.NumberColumn("Acc. entraînement", format="%.3f"),
}

DISPLAY_ORDER = ["modèle", "roc_auc", "rappel", "f1", "précision", "accuracy", "acc_train"]

# ─── Two tabs: preflop / full-hand ────────────────────────────────────────────

tab_pre, tab_full = st.tabs([
    "📊  Préflop — baseline honnête",
    "⚠️  Main complète — référence avec fuite",
])

# ── Tab 1 : Preflop ───────────────────────────────────────────────────────────

with tab_pre:
    callout(
        "<strong>Ce modèle est le baseline académiquement valide.</strong><br>"
        "Il utilise uniquement les informations disponibles <em>avant</em> les décisions : "
        "cartes initiales, position, blindes et taille de pile. "
        "Ses performances modestes reflètent la difficulté réelle de la tâche — "
        "prédire la victoire à partir de la seule main préflop. "
        "Entraîné avec <code>class_weight='balanced'</code> et évalué sur un ensemble de test stratifié (80/20)."
    )

    section_label("Résultats sur l'ensemble de test")
    pre_df = results["preflop"][DISPLAY_ORDER]
    st.dataframe(pre_df, column_config=METRICS_COLUMN_CONFIG, use_container_width=True, hide_index=True)

    st.markdown("")
    section_label("Importance des variables — Forêt Aléatoire préflop")

    imp_df = results["importance"]
    top_imp = imp_df.head(12).copy()

    fig_imp = px.bar(
        top_imp[::-1],
        x="importance",
        y="variable",
        orientation="h",
        template=PLOTLY_TEMPLATE,
        color="importance",
        color_continuous_scale=[[0, "#ECEEF2"], [1, COLOR_PRIMARY]],
        title="Top 12 variables — modèle préflop",
        labels={"importance": "Importance relative", "variable": ""},
    )
    fig_imp.update_layout(coloraxis_showscale=False)
    fig_imp = apply_chart_defaults(fig_imp)
    st.plotly_chart(fig_imp, use_container_width=True)

    callout(
        "Les variables liées aux cartes (<strong>preflop_strength_score</strong>, "
        "<strong>high_card_rank</strong>, <strong>low_card_rank</strong>) dominent. "
        "À l'inverse, <strong>starting_stack</strong>, <strong>number_of_players</strong> "
        "et <strong>variant</strong> ont une importance nulle : dans Pluribus, "
        "tous les joueurs démarrent avec 10 000 jetons, toutes les parties ont exactement 6 joueurs "
        "et une seule variante est présente. Ces variables sont <em>constantes</em> — "
        "elles n'apportent aucun signal discriminant dans ce dataset."
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
            "modèle": st.column_config.TextColumn("Modèle", width="medium"),
            "accuracy": st.column_config.TextColumn("Accuracy"),
            "rappel": st.column_config.TextColumn("Rappel"),
            "f1": st.column_config.TextColumn("F1"),
            "roc_auc": st.column_config.TextColumn("AUC-ROC"),
        },
        use_container_width=True,
        hide_index=True,
    )

# ── Tab 2 : Full-hand ─────────────────────────────────────────────────────────

with tab_full:
    callout(
        "<strong>Attention — fuite d'information directe.</strong><br>"
        "Ce modèle inclut des variables d'action (nombre de plis, mises, relances…) "
        "qui ne sont disponibles qu'<em>après</em> les décisions. "
        "Ses performances très élevées ne reflètent <em>pas</em> une capacité prédictive réelle — "
        "elles mesurent principalement la capacité du modèle à détecter si un joueur s'est couché, "
        "information triviale une fois la main terminée. "
        "Ce modèle est présenté ici comme <strong>référence de comparaison uniquement</strong>.",
        kind="warning",
    )

    section_label("Résultats sur l'ensemble de test")
    full_df = results["full"][DISPLAY_ORDER]
    st.dataframe(full_df, column_config=METRICS_COLUMN_CONFIG, use_container_width=True, hide_index=True)

    st.markdown("")
    section_label("Analyse de la fuite — importance des variables (main complète)")

    leak_df = results["leakage"]
    top_leak = leak_df.head(12).copy()

    bar_colors = [
        COLOR_LOSS if "player_num_folds" in v else COLOR_NEUTRAL
        for v in top_leak["variable"]
    ]

    fig_leak = px.bar(
        top_leak[::-1],
        x="importance",
        y="variable",
        orientation="h",
        template=PLOTLY_TEMPLATE,
        title="Top 12 variables — modèle main complète",
        labels={"importance": "Importance relative", "variable": ""},
    )
    fig_leak.update_traces(marker_color=bar_colors[::-1])
    fig_leak = apply_chart_defaults(fig_leak)
    st.plotly_chart(fig_leak, use_container_width=True)

    callout(
        "<strong>player_num_folds représente à lui seul ~84 % de l'importance totale.</strong><br>"
        "Se coucher (<em>fold</em>) détermine directement l'issue : un joueur qui se couche "
        "ne peut pas gagner le pot. Le modèle a donc appris à <em>détecter les plis</em>, "
        "pas à <em>prédire la victoire</em>. "
        "Les 97 %+ d'accuracy ne reflètent aucune capacité prédictive réelle — "
        "ils mesurent une tautologie encodée dans les données.",
        kind="warning",
    )

st.divider()

# ─── Methodology note ─────────────────────────────────────────────────────────

section_label("Note méthodologique")
callout(
    "<strong>Protocole commun aux deux modèles :</strong> "
    "partition stratifiée 80/20 (random_state=42), "
    "<code>class_weight='balanced'</code> sur tous les modèles, "
    "prétraitement en pipeline (imputation médiane → normalisation pour les variables numériques, "
    "imputation mode → OneHotEncoding pour <code>variant</code>). "
    "L'AUC-ROC est la métrique principale car elle est indépendante du seuil de décision "
    "et robuste au déséquilibre de classes."
)
