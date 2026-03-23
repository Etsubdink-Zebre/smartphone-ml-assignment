import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.config import CLASSIFICATION_MODEL_DIR, OUTPUTS_DIR
from src.data_loader import load_raw_data
from src.evaluate_classification import classification_metrics
from src.feature_engineering import add_engineered_features
from src.preprocessing import build_preprocessor
from src.utils import ensure_dir, save_json


def _safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        df.to_csv(fallback, index=False)
        print(f"Warning: {path.name} was locked. Saved fallback file: {fallback.name}")
        return fallback


def run_classification_training(random_state: int = 42) -> dict[str, float]:
    df = add_engineered_features(load_raw_data())
    target_col = "addiction_level"
    df = df[df[target_col].notna()].copy()

    drop_cols = {"transaction_id", "user_id", "addicted_label", target_col}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
        ]
    )
    param_grid = {
        "model__n_estimators": [150, 250],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    metrics = classification_metrics(y_test, y_pred)
    metrics["cv_best_score_f1_weighted"] = float(grid.best_score_)

    ensure_dir(CLASSIFICATION_MODEL_DIR)
    ensure_dir(OUTPUTS_DIR / "predictions")
    ensure_dir(OUTPUTS_DIR / "tuning_results")

    joblib.dump(grid.best_estimator_, CLASSIFICATION_MODEL_DIR / "best_classification_model.pkl")
    save_json(metrics, OUTPUTS_DIR / "classification_test_metrics.json")
    _safe_to_csv(
        pd.DataFrame({"actual": y_test, "predicted": y_pred}),
        OUTPUTS_DIR / "predictions" / "classification_predictions.csv",
    )
    _safe_to_csv(
        pd.DataFrame(grid.cv_results_),
        OUTPUTS_DIR / "tuning_results" / "classification_gridsearch_results.csv",
    )
    return metrics

