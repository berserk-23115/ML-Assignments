"""
Kaggle-style tabular classification pipeline.

What this script does:
1) Performs brief train/test feature analysis.
2) Uses 10-fold Stratified CV.
3) Trains XGBoost, LightGBM, and CatBoost with stable hyperparameters.
4) Builds a weighted soft-voting ensemble from CV performance.
5) Writes submission.csv with exactly: id, Target.

Notes on leaderboard strategy:
- Public leaderboard is only a subset and can be noisy.
- Trust cross-validation signal more than small public-LB gains.
- Favor stable CV and conservative hyperparameters to reduce overfitting risk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 10
TARGET_COL = "Target"
ID_COL = "id"


@dataclass
class FoldArtifacts:
    xgb_test_proba: np.ndarray
    lgbm_test_proba: np.ndarray
    cat_test_proba: np.ndarray


def resolve_input_files() -> tuple[str, str]:
    """Support both common Kaggle names and workspace-specific names."""
    candidates = [
        ("train.csv", "test.csv"),
        ("train_assignment.csv", "test_assignment.csv"),
    ]
    for train_path, test_path in candidates:
        if os.path.exists(train_path) and os.path.exists(test_path):
            return train_path, test_path
    raise FileNotFoundError(
        "Could not find train/test files. Expected either train.csv/test.csv "
        "or train_assignment.csv/test_assignment.csv."
    )


def detect_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Detect categorical-like features for encoding decisions."""
    categorical_cols: list[str] = []
    for col in df.columns:
        s = df[col]
        if s.dtype == "object" or str(s.dtype).startswith("category"):
            categorical_cols.append(col)
            continue
        if col.endswith("_code") or col.endswith("_flag"):
            categorical_cols.append(col)
            continue
        # Low-cardinality integer features are usually categorical in tabular competitions.
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) <= 25:
            categorical_cols.append(col)
    return sorted(set(categorical_cols))


def brief_data_analysis(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Print a concise feature analysis and encoding context."""
    feature_cols = [c for c in train_df.columns if c not in [ID_COL, TARGET_COL]]
    combined = pd.concat([train_df[feature_cols], test_df[feature_cols]], axis=0, ignore_index=True)

    categorical_cols = [c for c in detect_categorical_columns(combined) if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    print("=" * 90)
    print("1) Brief Data Analysis")
    print("=" * 90)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Features: {len(feature_cols)} | Numeric: {len(numeric_cols)} | Categorical-like: {len(categorical_cols)}")

    train_missing = train_df[feature_cols].isna().mean().sort_values(ascending=False)
    test_missing = test_df[feature_cols].isna().mean().sort_values(ascending=False)
    print("Top train missing rates:")
    print(train_missing.head(5).to_string())
    print("Top test missing rates:")
    print(test_missing.head(5).to_string())

    if TARGET_COL in train_df.columns:
        target_dist = train_df[TARGET_COL].value_counts(normalize=True).sort_index()
        print("Target distribution:")
        print(target_dist.to_string())

    show_num = [c for c in [
        "age_at_enrollment",
        "admission_grade",
        "previous_qualification_grade",
        "cu1_grade",
        "cu2_grade",
    ] if c in numeric_cols]
    if show_num:
        print("Numeric feature quantiles (train):")
        print(train_df[show_num].describe(percentiles=[0.05, 0.5, 0.95]).T.to_string())

    if categorical_cols:
        cardinality = combined[categorical_cols].nunique(dropna=False).sort_values(ascending=False)
        print("Categorical-like cardinalities (top 15):")
        print(cardinality.head(15).to_string())

    print("\nEncoding requirements:")
    print("- XGBoost/LightGBM: ordinal-encode categorical-like features + median-impute numerics.")
    print("- CatBoost: pass categorical column indices directly (native categorical handling).")

    print("\nPrivate vs Public Leaderboard guidance:")
    print("- Public LB can mislead due to subset variance; optimize against 10-fold CV first.")
    print("- Keep hyperparameters conservative (lower learning rate, controlled depth).")
    print("- Avoid frequent tweaking based only on tiny public-LB deltas.")

    return categorical_cols, numeric_cols


def fit_transform_for_xgb_lgbm(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model-specific preprocessing for XGBoost/LightGBM."""
    X_tr_num = X_train[numeric_cols].copy()
    X_va_num = X_valid[numeric_cols].copy()
    X_te_num = X_test[numeric_cols].copy()

    # Median imputation for numeric features.
    medians = X_tr_num.median()
    X_tr_num = X_tr_num.fillna(medians)
    X_va_num = X_va_num.fillna(medians)
    X_te_num = X_te_num.fillna(medians)

    X_tr_cat = X_train[categorical_cols].copy().fillna("MISSING").astype(str)
    X_va_cat = X_valid[categorical_cols].copy().fillna("MISSING").astype(str)
    X_te_cat = X_test[categorical_cols].copy().fillna("MISSING").astype(str)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    X_tr_cat_enc = encoder.fit_transform(X_tr_cat)
    X_va_cat_enc = encoder.transform(X_va_cat)
    X_te_cat_enc = encoder.transform(X_te_cat)

    X_tr_all = np.hstack([X_tr_num.to_numpy(dtype=np.float32), X_tr_cat_enc.astype(np.float32)])
    X_va_all = np.hstack([X_va_num.to_numpy(dtype=np.float32), X_va_cat_enc.astype(np.float32)])
    X_te_all = np.hstack([X_te_num.to_numpy(dtype=np.float32), X_te_cat_enc.astype(np.float32)])
    return X_tr_all, X_va_all, X_te_all


def prepare_for_catboost(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int]]:
    """CatBoost input with native categorical features."""
    X_tr = X_train.copy()
    X_va = X_valid.copy()
    X_te = X_test.copy()

    medians = X_tr[numeric_cols].median()
    X_tr[numeric_cols] = X_tr[numeric_cols].fillna(medians)
    X_va[numeric_cols] = X_va[numeric_cols].fillna(medians)
    X_te[numeric_cols] = X_te[numeric_cols].fillna(medians)

    for col in categorical_cols:
        X_tr[col] = X_tr[col].fillna("MISSING").astype(str)
        X_va[col] = X_va[col].fillna("MISSING").astype(str)
        X_te[col] = X_te[col].fillna("MISSING").astype(str)

    cat_feature_indices = [X_tr.columns.get_loc(c) for c in categorical_cols]
    return X_tr, X_va, X_te, cat_feature_indices


def get_models(num_classes: int) -> tuple[XGBClassifier, LGBMClassifier, CatBoostClassifier]:
    """Stable baseline hyperparameters for robust generalization."""
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=1600,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss",
        tree_method="hist",
    )

    lgbm = LGBMClassifier(
        objective="multiclass",
        n_estimators=1800,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=48,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    cat = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=2200,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=RANDOM_STATE,
        verbose=0,
    )

    return xgb, lgbm, cat


def main() -> None:
    train_path, test_path = resolve_input_files()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    if ID_COL not in train_df.columns or ID_COL not in test_df.columns:
        raise ValueError(f"Missing required id column: {ID_COL}")

    categorical_cols, numeric_cols = brief_data_analysis(train_df, test_df)

    feature_cols = [c for c in train_df.columns if c not in [ID_COL, TARGET_COL]]
    X = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_df[TARGET_COL])
    num_classes = len(label_encoder.classes_)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Out-of-fold and test probability containers.
    oof_xgb = np.zeros((len(train_df), num_classes), dtype=np.float64)
    oof_lgbm = np.zeros((len(train_df), num_classes), dtype=np.float64)
    oof_cat = np.zeros((len(train_df), num_classes), dtype=np.float64)

    fold_artifacts: list[FoldArtifacts] = []
    xgb_fold_acc: list[float] = []
    lgbm_fold_acc: list[float] = []
    cat_fold_acc: list[float] = []

    print("\n" + "=" * 90)
    print("2) 10-Fold Stratified Training")
    print("=" * 90)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        xgb_tr, xgb_va, xgb_te = fit_transform_for_xgb_lgbm(
            X_tr, X_va, X_test, categorical_cols, numeric_cols
        )
        cat_tr, cat_va, cat_te, cat_idx = prepare_for_catboost(
            X_tr, X_va, X_test, categorical_cols, numeric_cols
        )

        xgb_model, lgbm_model, cat_model = get_models(num_classes=num_classes)

        xgb_model.fit(xgb_tr, y_tr, eval_set=[(xgb_va, y_va)], verbose=False)
        lgbm_model.fit(
            xgb_tr,
            y_tr,
            eval_set=[(xgb_va, y_va)],
            callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
        )
        cat_model.fit(cat_tr, y_tr, cat_features=cat_idx, eval_set=(cat_va, y_va), verbose=False)

        xgb_va_proba = xgb_model.predict_proba(xgb_va)
        lgbm_va_proba = lgbm_model.predict_proba(xgb_va)
        cat_va_proba = cat_model.predict_proba(cat_va)

        oof_xgb[va_idx] = xgb_va_proba
        oof_lgbm[va_idx] = lgbm_va_proba
        oof_cat[va_idx] = cat_va_proba

        xgb_acc = accuracy_score(y_va, np.argmax(xgb_va_proba, axis=1))
        lgbm_acc = accuracy_score(y_va, np.argmax(lgbm_va_proba, axis=1))
        cat_acc = accuracy_score(y_va, np.argmax(cat_va_proba, axis=1))
        xgb_fold_acc.append(xgb_acc)
        lgbm_fold_acc.append(lgbm_acc)
        cat_fold_acc.append(cat_acc)

        fold_artifacts.append(
            FoldArtifacts(
                xgb_test_proba=xgb_model.predict_proba(xgb_te),
                lgbm_test_proba=lgbm_model.predict_proba(xgb_te),
                cat_test_proba=cat_model.predict_proba(cat_te),
            )
        )

        print(
            f"Fold {fold:02d}/{N_SPLITS} | "
            f"XGB={xgb_acc:.5f} LGBM={lgbm_acc:.5f} CAT={cat_acc:.5f}"
        )

    # Derive model weights from mean CV accuracy to create a robust voting ensemble.
    xgb_mean_acc = float(np.mean(xgb_fold_acc))
    lgbm_mean_acc = float(np.mean(lgbm_fold_acc))
    cat_mean_acc = float(np.mean(cat_fold_acc))

    raw_weights = np.array([xgb_mean_acc, lgbm_mean_acc, cat_mean_acc], dtype=np.float64)
    weights = raw_weights / raw_weights.sum()

    oof_ensemble = (
        weights[0] * oof_xgb
        + weights[1] * oof_lgbm
        + weights[2] * oof_cat
    )
    ensemble_oof_pred = np.argmax(oof_ensemble, axis=1)
    ensemble_oof_acc = accuracy_score(y, ensemble_oof_pred)

    print("\n" + "=" * 90)
    print("3) CV Summary")
    print("=" * 90)
    print(f"XGBoost mean CV acc : {xgb_mean_acc:.5f}")
    print(f"LightGBM mean CV acc: {lgbm_mean_acc:.5f}")
    print(f"CatBoost mean CV acc: {cat_mean_acc:.5f}")
    print(f"Voting weights       : XGB={weights[0]:.3f}, LGBM={weights[1]:.3f}, CAT={weights[2]:.3f}")
    print(f"Ensemble OOF acc     : {ensemble_oof_acc:.5f}")

    test_xgb = np.mean([f.xgb_test_proba for f in fold_artifacts], axis=0)
    test_lgbm = np.mean([f.lgbm_test_proba for f in fold_artifacts], axis=0)
    test_cat = np.mean([f.cat_test_proba for f in fold_artifacts], axis=0)

    test_ensemble = (
        weights[0] * test_xgb
        + weights[1] * test_lgbm
        + weights[2] * test_cat
    )

    test_pred_idx = np.argmax(test_ensemble, axis=1)
    test_pred_labels = label_encoder.inverse_transform(test_pred_idx)

    submission = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL].copy(),
            TARGET_COL: test_pred_labels,
        }
    )

    # Strict format and ordering checks required by competition submission rules.
    assert list(submission.columns) == [ID_COL, TARGET_COL]
    assert len(submission) == len(test_df)
    assert submission[ID_COL].reset_index(drop=True).equals(test_df[ID_COL].reset_index(drop=True))

    submission.to_csv("submission.csv", index=False)
    print("\n" + "=" * 90)
    print("4) Submission")
    print("=" * 90)
    print(f"Saved submission.csv with shape: {submission.shape}")
    print("Columns:", submission.columns.tolist())
    print("Preview:")
    print(submission.head(5).to_string(index=False))


if __name__ == "__main__":
    main()