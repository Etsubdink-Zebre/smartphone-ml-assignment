import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.feature_engineering import add_engineered_features
from src.preprocessing import build_preprocessor


REQUIRED_BASE_COLUMNS = [
    "age",
    "gender",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "stress_level",
    "academic_work_impact",
]


def validate_base_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def prepare_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_base_columns(df)
    work = add_engineered_features(df)
    if "daily_screen_time_hours" not in work.columns:
        work["daily_screen_time_hours"] = (work["social_media_hours"] + work["gaming_hours"]).clip(lower=1.0)
    return work


def build_regression_predict_features(df: pd.DataFrame) -> pd.DataFrame:
    work = add_engineered_features(df)
    drop_cols = [c for c in ["transaction_id", "user_id", "addiction_level", "addicted_label", "daily_screen_time_hours"] if c in work.columns]
    return work.drop(columns=drop_cols)


def train_models_from_uploaded_csv(train_df: pd.DataFrame):
    validate_base_columns(train_df)
    df = add_engineered_features(train_df)

    clf_model = None
    if "addiction_level" in df.columns and df["addiction_level"].notna().any():
        clf_df = df[df["addiction_level"].notna()].copy()
        drop_cols = [c for c in ["transaction_id", "user_id", "addicted_label", "addiction_level"] if c in clf_df.columns]
        X_clf = clf_df.drop(columns=drop_cols)
        y_clf = clf_df["addiction_level"]
        clf_model = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_clf)),
                ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
            ]
        )
        clf_model.fit(X_clf, y_clf)

    reg_model = None
    if "daily_screen_time_hours" in df.columns and df["daily_screen_time_hours"].notna().any():
        reg_df = df[df["daily_screen_time_hours"].notna()].copy()
        drop_cols = [
            c
            for c in ["transaction_id", "user_id", "addiction_level", "addicted_label", "daily_screen_time_hours"]
            if c in reg_df.columns
        ]
        X_reg = reg_df.drop(columns=drop_cols)
        y_reg = reg_df["daily_screen_time_hours"]
        reg_model = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_reg)),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        )
        reg_model.fit(X_reg, y_reg)

    return clf_model, reg_model
