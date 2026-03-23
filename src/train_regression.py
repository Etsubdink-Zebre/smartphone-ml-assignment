import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.config import OUTPUTS_DIR, REGRESSION_MODEL_DIR
from src.data_loader import load_raw_data
from src.evaluate_regression import regression_metrics
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


def run_regression_training(random_state: int = 42) -> dict[str, float]:
    df = add_engineered_features(load_raw_data())
    target_col = "daily_screen_time_hours"
    df = df[df[target_col].notna()].copy()

    drop_cols = {"transaction_id", "user_id", "addicted_label", "addiction_level", target_col}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=random_state, n_jobs=-1)),
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
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    metrics["cv_best_score_rmse"] = float(-grid.best_score_)

    ensure_dir(REGRESSION_MODEL_DIR)
    ensure_dir(OUTPUTS_DIR / "predictions")
    ensure_dir(OUTPUTS_DIR / "tuning_results")

    joblib.dump(grid.best_estimator_, REGRESSION_MODEL_DIR / "best_regression_model.pkl")
    save_json(metrics, OUTPUTS_DIR / "regression_test_metrics.json")
    _safe_to_csv(
        pd.DataFrame({"actual": y_test, "predicted": y_pred}),
        OUTPUTS_DIR / "predictions" / "regression_predictions.csv",
    )
    _safe_to_csv(
        pd.DataFrame(grid.cv_results_),
        OUTPUTS_DIR / "tuning_results" / "regression_gridsearch_results.csv",
    )
    return metrics

