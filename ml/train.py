"""Model training.

Two models, deliberately: a logistic regression baseline and a gradient boosting
challenger. Reporting only the strong model would hide whether the extra
complexity earned anything.

Threshold selection happens on the validation split and never on test. This is
the discipline that makes the held-out number meaningful: if we picked the
threshold that flattered the test set, the test set would no longer be held out.

Run:
    python -m ml.train --seed 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import FEATURE_COLUMNS, build_features, feature_matrix, load_events
from .splits import assign_transaction_splits

ARTIFACTS = C.ROOT / "artifacts"

# If the held-out score ever exceeds this, something is leaking. A near-perfect
# result on synthetic data is a bug report, not an achievement.
LEAKAGE_ALARM_PR_AUC = 0.99


@dataclass
class TrainedModel:
    name: str
    threshold: float
    validation_pr_auc: float


def load_risk_scores() -> pd.Series:
    """Per-transaction risk from the primary model, indexed by transaction id.

    Shared by the spike detector and the ring investigator. Features are
    point-in-time correct, so scoring the full history here is safe.
    """
    from .features import build_features, feature_matrix, load_events

    meta = json.loads((ARTIFACTS / "training_meta.json").read_text())
    name = meta["primary_model"]
    pipe = joblib.load(ARTIFACTS / f"model_{name}.joblib")
    feats = build_features(load_events())
    scores = pipe.predict_proba(feature_matrix(feats).to_numpy(dtype=float))[:, 1]
    series = pd.Series(scores, index=feats["transaction_id"], name="risk")
    series.attrs["threshold"] = meta["models"][name]["threshold"]
    return series


def load_labelled() -> pd.DataFrame:
    """Features joined to labels. Only train/evaluate may call this."""
    features = build_features(load_events())
    labels = pd.read_parquet(C.LABELS_DIR / "transaction_labels.parquet")
    df = features.merge(labels, on="transaction_id", how="left", validate="1:1")
    if df["is_fraud"].isna().any():
        raise ValueError("unlabelled transactions found")
    df["split"] = assign_transaction_splits(df).values
    return df


def _pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 on the validation split.

    F1 rather than accuracy because the classes are imbalanced, and rather than
    a fixed 0.5 because a fixed cut on an imbalanced problem is arbitrary.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(
            (precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0
        )
    # precision_recall_curve returns one more point than thresholds.
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best])


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=1.0,
                        random_state=0,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.08,
                        max_leaf_nodes=31,
                        min_samples_leaf=40,
                        l2_regularization=1.0,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=0,
                    ),
                ),
            ]
        ),
    }


def train(seed: int = C.SEED) -> dict:
    df = load_labelled()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]

    X_train = feature_matrix(train_df).to_numpy(dtype=float)
    y_train = train_df["is_fraud"].to_numpy()
    X_val = feature_matrix(val_df).to_numpy(dtype=float)
    y_val = val_df["is_fraud"].to_numpy()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for name, pipe in build_models().items():
        pipe.fit(X_train, y_train)
        val_scores = pipe.predict_proba(X_val)[:, 1]
        trained = TrainedModel(
            name=name,
            threshold=_pick_threshold(y_val, val_scores),
            validation_pr_auc=float(average_precision_score(y_val, val_scores)),
        )
        joblib.dump(pipe, ARTIFACTS / f"model_{name}.joblib")
        summary[name] = asdict(trained)
        print(
            f"{name:24s} val PR-AUC {trained.validation_pr_auc:.4f}  "
            f"threshold {trained.threshold:.4f}"
        )

    meta = {
        "seed": seed,
        "feature_columns": FEATURE_COLUMNS,
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "train_positive_rate": round(float(y_train.mean()), 5),
        "validation_positive_rate": round(float(y_val.mean()), 5),
        "primary_model": "hist_gradient_boosting",
        "baseline_model": "logistic_regression",
        "models": summary,
        "threshold_selection": "max F1 on the validation split; test never used",
    }
    (ARTIFACTS / "training_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RazorShield risk models.")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    meta = train(seed=args.seed)
    print(f"\nwritten: {ARTIFACTS / 'training_meta.json'}")
    print(f"train rows {meta['n_train']:,}  validation rows {meta['n_validation']:,}")


if __name__ == "__main__":
    main()
