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

FULL_HAND_FEATURES = PREFLOP_FEATURES + [
    "hand_action_count",
    "num_folds",
    "num_check_calls",
    "num_bets_raises",
    "num_hole_deals",
    "num_board_deals",
    "num_show_or_muck",
    "player_num_folds",
    "player_num_check_calls",
    "player_num_bets_raises",
    "player_num_show_or_muck",
]


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
