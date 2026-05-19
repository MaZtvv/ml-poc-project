import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ─── Feature sets ─────────────────────────────────────────────────────────────

TARGET = "player_won"

PREFLOP_FEATURES = [
    "number_of_players",
    "starting_stack",
    "blind_or_straddle",
    "is_small_blind",
    "is_big_blind",
    "player_position_index",
    "is_pair",
    "is_suited",
    "high_card_rank",
    "low_card_rank",
    "rank_gap",
    "has_ace",
    "has_king",
    "preflop_strength_score",
    "variant",
]

# ── Leakage features (direct tautology or showdown-only) ─────────────────────
# player_num_folds is the primary leakage vector: fold → player_won = 0, always.
# Removing it from any predictive model is mandatory.
LEAKAGE_FEATURES = [
    "player_num_folds",        # fold → cannot win (near-deterministic tautology)
    "player_num_show_or_muck", # only known at showdown
    "num_folds",               # table-level fold count also encodes the outcome
    "num_show_or_muck",        # showdown-only
]

# ── Clean in-hand feature set — no direct leakage ─────────────────────────────
# Adds action-style features that are accumulated step-by-step during replay.
# Residual limitation: player_num_check_calls / player_num_bets_raises still
# summarise the whole hand. Their correlation with winning is real but not
# tautological — they encode action style, not the fold decision itself.
INHAND_FEATURES = PREFLOP_FEATURES + [
    "player_num_check_calls",  # player action style (not fold-driven)
    "player_num_bets_raises",  # player aggression
    "num_board_deals",         # streets reached (0-3); changes during replay
    "hand_action_count",       # hand length proxy
    "num_check_calls",         # table-level passivity
    "num_bets_raises",         # table-level aggression
]

# ── Diagnostic full-hand feature set (kept for methodological comparison only) ─
FULL_HAND_FEATURES = PREFLOP_FEATURES + [
    "hand_action_count", "num_check_calls", "num_bets_raises",
    "num_hole_deals", "num_board_deals",
] + LEAKAGE_FEATURES


# ─── Pipeline builder ─────────────────────────────────────────────────────────

def _split_features(feature_list: list[str]) -> tuple[list[str], list[str]]:
    numeric = [f for f in feature_list if f != "variant"]
    categorical = [f for f in feature_list if f == "variant"]
    return numeric, categorical


def _build_pipeline(feature_list: list[str], model) -> Pipeline:
    numeric, categorical = _split_features(feature_list)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _model_definitions() -> dict:
    return {
        "Régression Logistique": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Arbre de Décision": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Forêt Aléatoire": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
    }


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def _evaluate_pipeline(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "modèle": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "précision": precision_score(y_test, y_pred, zero_division=0),
        "rappel": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "acc_train": pipeline.score(X_train, y_train),
    }


def _run_cv(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, model in _model_definitions().items():
        pipe = _build_pipeline(PREFLOP_FEATURES, model)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        rows.append({
            "modèle": name,
            "accuracy": f"{scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}",
            "rappel": f"{scores['test_recall'].mean():.3f} ± {scores['test_recall'].std():.3f}",
            "f1": f"{scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}",
            "roc_auc": f"{scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}",
        })
    return rows


def _compute_importance(
    X: pd.DataFrame, y: pd.Series, feature_list: list[str]
) -> pd.DataFrame:
    pipe = _build_pipeline(
        feature_list,
        RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    )
    pipe.fit(X, y)
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    df = pd.DataFrame({"variable": names, "importance": importances})
    df["variable"] = (
        df["variable"]
        .str.replace(r"^num__", "", regex=True)
        .str.replace(r"^cat__variant_", "variante=", regex=True)
    )
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ─── Main cached entry point ──────────────────────────────────────────────────

# ─── Live inference (preflop LR only) ───────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du modèle préflop...")
def get_preflop_lr_pipeline():
    from core.data import load_player_features

    df = load_player_features()
    pipe = _build_pipeline(
        PREFLOP_FEATURES,
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    )
    pipe.fit(df[PREFLOP_FEATURES], df[TARGET])
    return pipe


@st.cache_resource(show_spinner="Chargement du modèle in-hand...")
def get_inhand_lr_pipeline():
    """Logistic Regression trained on INHAND_FEATURES (no player_num_folds)."""
    from core.data import load_player_features

    df = load_player_features()
    df_clean = df.dropna(subset=INHAND_FEATURES + [TARGET])
    pipe = _build_pipeline(
        INHAND_FEATURES,
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    )
    pipe.fit(df_clean[INHAND_FEATURES], df_clean[TARGET])
    return pipe


@st.cache_data(show_spinner="Calcul des probabilités...")
def compute_all_probabilities(df: pd.DataFrame) -> pd.Series:
    pipe = get_preflop_lr_pipeline()
    probs = pipe.predict_proba(df[PREFLOP_FEATURES])[:, 1]
    return pd.Series(probs, index=df.index, name="win_probability")


@st.cache_data(show_spinner="Génération du résumé des mains...")
def build_hand_summary(df: pd.DataFrame) -> pd.DataFrame:
    probs = compute_all_probabilities(df)
    df_p = df.assign(win_probability=probs)
    rows = []
    for composite_id, grp in df_p.groupby("composite_id"):
        winner_mask = grp["player_won"] == 1
        winner = (
            grp.loc[winner_mask, "player_name"].iloc[0]
            if winner_mask.any()
            else "—"
        )
        fav_idx  = grp["win_probability"].idxmax()
        favorite = grp.loc[fav_idx, "player_name"]
        max_prob = grp.loc[fav_idx, "win_probability"]
        source   = str(grp["source_relative_path"].iloc[0])
        rows.append({
            "composite_id": composite_id,
            "source":       source,
            "n_players":    len(grp),
            "favorite":     favorite,
            "prob_max":     max_prob,
            "winner":       winner,
            "correct":      favorite == winner,
        })
    return pd.DataFrame(rows).sort_values("composite_id").reset_index(drop=True)


# ─── ML analysis artifacts (analysis page) ──────────────────────────────────

@st.cache_data(show_spinner="Calcul des analyses ML préflop...")
def compute_preflop_analysis(df: pd.DataFrame) -> dict:
    """All sklearn curve artifacts for the analysis page (preflop LR, 80/20 split)."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

    y = df[TARGET]
    X = df[PREFLOP_FEATURES]
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    pipe = _build_pipeline(
        PREFLOP_FEATURES,
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    )
    pipe.fit(X_tr, y_tr)

    y_proba = pipe.predict_proba(X_te)[:, 1]
    y_pred  = pipe.predict(X_te)

    fpr, tpr, _        = roc_curve(y_te, y_proba)
    prec, rec, _       = precision_recall_curve(y_te, y_proba)
    cal_frac, cal_mean = calibration_curve(y_te, y_proba, n_bins=10, strategy="quantile")
    cm                 = confusion_matrix(y_te, y_pred)

    feat_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    coefs      = pipe.named_steps["model"].coef_[0]
    coef_df    = pd.DataFrame({"feature": feat_names, "coef": coefs})
    coef_df["feature"] = (
        coef_df["feature"]
        .str.replace(r"^num__", "", regex=True)
        .str.replace(r"^cat__variant_", "variant=", regex=True)
    )
    # Drop features that are constant in Pluribus (zero-variance after scaling → meaningless coef)
    _constant = {"number_of_players", "starting_stack", "variant=NT"}
    coef_df = coef_df[~coef_df["feature"].isin(_constant)].copy()
    coef_df = coef_df.reindex(
        coef_df["coef"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    return {
        "y_test":   y_te.values,
        "y_proba":  y_proba,
        "y_pred":   y_pred,
        "fpr":      fpr,
        "tpr":      tpr,
        "prec":     prec,
        "rec":      rec,
        "cal_frac": cal_frac,
        "cal_mean": cal_mean,
        "cm":       cm,
        "coef_df":  coef_df,
        "metrics": {
            "accuracy":  accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall":    recall_score(y_te, y_pred, zero_division=0),
            "f1":        f1_score(y_te, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_te, y_proba),
        },
        "pos_rate": float(y_te.mean()),
        "n_test":   len(y_te),
        "n_train":  len(y_tr),
    }


@st.cache_data(show_spinner="Entraînement du modèle in-hand (propre)...")
def compute_inhand_analysis(df: pd.DataFrame) -> dict:
    """
    Train and evaluate the clean in-hand model (INHAND_FEATURES — no player_num_folds).

    Returns metrics table, RF feature importance, and LR coefficients for the
    model comparison page.  Uses the same 80/20 stratified split as the preflop
    analysis so AUC numbers are directly comparable.
    """
    df_clean = df.dropna(subset=INHAND_FEATURES + [TARGET]).copy()
    y = df_clean[TARGET]
    X = df_clean[INHAND_FEATURES]
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    rows = []
    for name, model_proto in _model_definitions().items():
        params = model_proto.get_params()
        pipe = _build_pipeline(INHAND_FEATURES, model_proto.__class__(**params))
        pipe.fit(X_tr, y_tr)
        rows.append(_evaluate_pipeline(name, pipe, X_tr, X_te, y_tr, y_te))

    metrics_df = (
        pd.DataFrame(rows)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    importance_df = _compute_importance(X, y, INHAND_FEATURES)

    # LR coefficients for interpretability
    lr_pipe = _build_pipeline(
        INHAND_FEATURES,
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    )
    lr_pipe.fit(X_tr, y_tr)
    feat_names = lr_pipe.named_steps["preprocessor"].get_feature_names_out()
    coefs      = lr_pipe.named_steps["model"].coef_[0]
    coef_df    = pd.DataFrame({"feature": feat_names, "coef": coefs})
    coef_df["feature"] = (
        coef_df["feature"]
        .str.replace(r"^num__", "", regex=True)
        .str.replace(r"^cat__variant_", "variant=", regex=True)
    )
    _constant = {"number_of_players", "starting_stack", "variant=NT"}
    coef_df = coef_df[~coef_df["feature"].isin(_constant)].copy()
    coef_df = coef_df.reindex(
        coef_df["coef"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    lr_auc = float(
        metrics_df.loc[
            metrics_df["modèle"].str.contains("Logistique"), "roc_auc"
        ].iloc[0]
    ) if len(metrics_df) else 0.0

    return {
        "metrics_df":  metrics_df,
        "importance":  importance_df,
        "coef_df":     coef_df,
        "lr_auc":      lr_auc,
        "n_train":     len(y_tr),
        "n_test":      len(y_te),
    }


@st.cache_data(show_spinner=False)
def compute_winrate_stats(df: pd.DataFrame) -> dict:
    """Win rate by preflop score quintile and by hand category."""
    d = df.copy()
    d["_bucket"] = pd.qcut(
        d["preflop_strength_score"], q=5,
        labels=["Q1 (faibles)", "Q2", "Q3", "Q4", "Q5 (fortes)"],
    )
    by_bucket = (
        d.groupby("_bucket", observed=False)["player_won"]
        .agg(win_rate="mean", count="count")
        .reset_index()
        .rename(columns={"_bucket": "bucket"})
    )
    by_bucket["bucket"] = by_bucket["bucket"].astype(str)

    cats = [
        ("is_pair",   "Paire",    "Non-paire"),
        ("is_suited", "Couleur",  "Offsuit"),
        ("has_ace",   "Avec As",  "Sans As"),
        ("has_king",  "Avec Roi", "Sans Roi"),
    ]
    rows = []
    for col, yes_lbl, no_lbl in cats:
        if col not in d.columns:
            continue
        for val, lbl in [(1, yes_lbl), (0, no_lbl)]:
            sub = d[d[col] == val]
            rows.append({
                "col":      col,
                "label":    lbl,
                "yn":       "Oui" if val else "Non",
                "win_rate": float(sub["player_won"].mean()),
                "count":    len(sub),
            })
    return {
        "by_bucket":   by_bucket,
        "by_category": pd.DataFrame(rows),
    }


# ─── Street probability evolution ───────────────────────────────────────────

def compute_street_probabilities(
    initial_probs: dict,
    folded_by_street: dict,
) -> dict:
    """
    Normalize raw LR probabilities to sum=1, then re-normalize at each street
    after applying cumulative folds (folded players → 0).

    initial_probs:   {player_idx: float} — raw model outputs
    folded_by_street:{street: set(player_idx)} — cumulative folds per street

    Returns: {"Début": {idx: p}, "Après préflop": {...}, "Après flop": {...},
              "Après turn": {...}, "Après river": {...}}
    """
    def normalize(probs: dict, folded: set) -> dict:
        active = {k: v for k, v in probs.items() if k not in folded}
        total = sum(active.values())
        if total <= 0:
            n = max(len(probs), 1)
            return {k: (1 / n if k not in folded else 0.0) for k in probs}
        result = {k: 0.0 for k in probs}
        for k, v in active.items():
            result[k] = v / total
        return result

    return {
        "Début":         normalize(initial_probs, set()),
        "Après préflop": normalize(initial_probs, folded_by_street.get("preflop", set())),
        "Après flop":    normalize(initial_probs, folded_by_street.get("flop", set())),
        "Après turn":    normalize(initial_probs, folded_by_street.get("turn", set())),
        "Après river":   normalize(initial_probs, folded_by_street.get("river", set())),
    }


# ─── Batch evaluation ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Entraînement des modèles en cours...")
def evaluate_models(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    y = df[TARGET]
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )

    X_pre = df[PREFLOP_FEATURES]
    X_full = df[FULL_HAND_FEATURES]

    X_pre_train, X_pre_test = X_pre.iloc[train_idx], X_pre.iloc[test_idx]
    X_full_train, X_full_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    preflop_rows, full_rows = [], []

    for name, model_proto in _model_definitions().items():
        params = model_proto.get_params()

        pre_pipe = _build_pipeline(PREFLOP_FEATURES, model_proto.__class__(**params))
        pre_pipe.fit(X_pre_train, y_train)
        preflop_rows.append(_evaluate_pipeline(name, pre_pipe, X_pre_train, X_pre_test, y_train, y_test))

        full_pipe = _build_pipeline(FULL_HAND_FEATURES, model_proto.__class__(**params))
        full_pipe.fit(X_full_train, y_train)
        full_rows.append(_evaluate_pipeline(name, full_pipe, X_full_train, X_full_test, y_train, y_test))

    return {
        "preflop": pd.DataFrame(preflop_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True),
        "full": pd.DataFrame(full_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True),
        "cv": pd.DataFrame(_run_cv(X_pre, y)),
        "importance": _compute_importance(X_pre, y, PREFLOP_FEATURES),
        "leakage": _compute_importance(X_full, y, FULL_HAND_FEATURES),
    }
