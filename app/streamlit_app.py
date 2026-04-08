import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_modeling import train_models_from_uploaded_csv

CLASSIFICATION_MODEL_PATH = PROJECT_ROOT / "models" / "classification" / "best_classification_model.pkl"
REGRESSION_MODEL_PATH = PROJECT_ROOT / "models" / "regression" / "best_regression_model.pkl"


INPUT_COLUMNS = [
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
    "social_gaming_total_hours",
    "notifications_per_open",
]


def get_expected_feature_columns(model) -> list[str]:
    preprocessor = model.named_steps.get("preprocessor") if hasattr(model, "named_steps") else None
    if preprocessor is None:
        return []
    feature_names = getattr(preprocessor, "feature_names_in_", None)
    if feature_names is None:
        return []
    return list(feature_names)


def align_input_to_model_features(input_df: pd.DataFrame, model, *, default_text: str = "Unknown") -> pd.DataFrame:
    expected_cols = get_expected_feature_columns(model)
    if not expected_cols:
        return input_df.copy()

    aligned = input_df.copy()
    missing = [c for c in expected_cols if c not in aligned.columns]
    extra = [c for c in aligned.columns if c not in expected_cols]

    for col in missing:
        # Old leaked models may expect `predicted_*` columns.
        # Fill safely so pipeline transform can run.
        aligned[col] = default_text if col.endswith("level") else pd.NA

    if extra:
        aligned = aligned.drop(columns=extra)

    return aligned.reindex(columns=expected_cols)


def make_input_df(
    age: int,
    gender: str,
    social_media_hours: float,
    gaming_hours: float,
    work_study_hours: float,
    sleep_hours: float,
    notifications_per_day: int,
    app_opens_per_day: int,
    weekend_screen_time: float,
    stress_level: str,
    academic_work_impact: str,
):
    data = {
        "age": age,
        "gender": gender,
        "social_media_hours": social_media_hours,
        "gaming_hours": gaming_hours,
        "work_study_hours": work_study_hours,
        "sleep_hours": sleep_hours,
        "notifications_per_day": notifications_per_day,
        "app_opens_per_day": app_opens_per_day,
        "weekend_screen_time": weekend_screen_time,
        "stress_level": stress_level,
        "academic_work_impact": academic_work_impact,
    }
    row = pd.DataFrame([data])
    # Used by the classification model as a practical proxy input at demo time.
    row["daily_screen_time_hours"] = max(
        float((row["social_media_hours"] + row["gaming_hours"]).iloc[0]), 1.0
    )
    row["social_gaming_total_hours"] = row["social_media_hours"] + row["gaming_hours"]
    row["notifications_per_open"] = row["notifications_per_day"] / row["app_opens_per_day"].replace(0, 1)
    return row


def build_recommendation(predicted_addiction_level: str, predicted_screen_time: float) -> str:
    level = str(predicted_addiction_level).strip().lower()

    severe_levels = {"severe", "high", "very high"}
    moderate_levels = {"moderate", "medium"}
    mild_levels = {"mild", "low"}

    # Prefer class label from model. Use screen-time thresholds only as fallback.
    if level in severe_levels:
        return (
            "High-risk usage pattern detected. Reduce non-essential app use, enable focus/bedtime mode, "
            "and schedule phone-free blocks (for example: study time and before sleep)."
        )
    if level in moderate_levels:
        return (
            "Moderate usage pattern detected. Set daily app limits, reduce notifications, and keep at least "
            "one short phone-free period each day."
        )
    if level in mild_levels:
        return (
            "Mild usage pattern detected. Maintain current habits and continue healthy phone routines."
        )

    # Fallback path if model label is unexpected or missing.
    if predicted_screen_time > 6.0:
        return (
            "High-risk usage pattern detected from screen-time estimate. Reduce non-essential app use, "
            "enable focus/bedtime mode, and schedule phone-free blocks."
        )
    if 4.0 <= predicted_screen_time <= 6.0:
        return (
            "Moderate usage pattern detected from screen-time estimate. Set daily app limits and "
            "reduce notification interruptions."
        )
    return (
        "Mild usage pattern detected from screen-time estimate. Maintain current habits and healthy sleep/work balance."
    )


def main():
    st.set_page_config(page_title="Smartphone ML Demo", layout="centered")
    st.title("Smartphone ML Assignment Demo")
    st.write("Upload a CSV, train models from it, then use those trained models for prediction.")

    if "active_clf_model" not in st.session_state:
        st.session_state.active_clf_model = None
    if "active_reg_model" not in st.session_state:
        st.session_state.active_reg_model = None

    model_source = st.radio(
        "Choose model source",
        ["Train from uploaded CSV", "Use existing saved models"],
        horizontal=True,
    )

    if model_source == "Train from uploaded CSV":
        st.subheader("Upload Training CSV")
        st.caption(
            "Required feature columns: age, gender, social_media_hours, gaming_hours, work_study_hours, sleep_hours, "
            "notifications_per_day, app_opens_per_day, weekend_screen_time, stress_level, academic_work_impact. "
            "Training target columns: addiction_level (classification), daily_screen_time_hours (regression)."
        )
        train_file = st.file_uploader("Training CSV", type=["csv"], key="train_csv")
        save_trained_models = st.checkbox("Save trained models", value=False)
        if train_file is not None and st.button("Train Models From Uploaded CSV"):
            try:
                train_df = pd.read_csv(train_file)
                with st.spinner("Training models..."):
                    clf_model, reg_model = train_models_from_uploaded_csv(train_df)
                st.session_state.active_clf_model = clf_model
                st.session_state.active_reg_model = reg_model

                if save_trained_models:
                    CLASSIFICATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    REGRESSION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    if clf_model is not None:
                        joblib.dump(clf_model, CLASSIFICATION_MODEL_PATH)
                    if reg_model is not None:
                        joblib.dump(reg_model, REGRESSION_MODEL_PATH)

                if clf_model is None and reg_model is None:
                    st.error(
                        "No model was trained. Add non-null `addiction_level` and/or `daily_screen_time_hours` in training CSV."
                    )
                else:
                    st.success("Training complete. Uploaded-file models are now active in this UI session.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
    else:
        st.subheader("Use Existing Saved Models")
        if st.button("Load Saved Models"):
            clf_model = joblib.load(CLASSIFICATION_MODEL_PATH) if CLASSIFICATION_MODEL_PATH.exists() else None
            reg_model = joblib.load(REGRESSION_MODEL_PATH) if REGRESSION_MODEL_PATH.exists() else None
            st.session_state.active_clf_model = clf_model
            st.session_state.active_reg_model = reg_model
            if clf_model is None and reg_model is None:
                st.error("No saved models found in models/ folder.")
            else:
                st.success("Saved models loaded and active in this session.")

    clf_model = st.session_state.active_clf_model
    reg_model = st.session_state.active_reg_model

    st.subheader("Quick Single-Row Test (Optional)")
    age = st.slider("Age", min_value=10, max_value=70, value=22)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    social_media_hours = st.slider("Social Media Hours", 0.0, 12.0, 3.0, 0.1)
    gaming_hours = st.slider("Gaming Hours", 0.0, 12.0, 1.0, 0.1)
    work_study_hours = st.slider("Work/Study Hours", 0.0, 14.0, 4.0, 0.1)
    sleep_hours = st.slider("Sleep Hours", 2.0, 12.0, 7.0, 0.1)
    notifications_per_day = st.slider("Notifications per Day", 0, 500, 120)
    app_opens_per_day = st.slider("App Opens per Day", 0, 300, 80)
    weekend_screen_time = st.slider("Weekend Screen Time (hours)", 0.0, 16.0, 5.0, 0.1)
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    academic_work_impact = st.selectbox("Academic/Work Impact", ["No", "Yes"])

    input_df = make_input_df(
        age=age,
        gender=gender,
        social_media_hours=social_media_hours,
        gaming_hours=gaming_hours,
        work_study_hours=work_study_hours,
        sleep_hours=sleep_hours,
        notifications_per_day=notifications_per_day,
        app_opens_per_day=app_opens_per_day,
        weekend_screen_time=weekend_screen_time,
        stress_level=stress_level,
        academic_work_impact=academic_work_impact,
    )
    st.write("Single-row input preview:")
    st.dataframe(input_df[INPUT_COLUMNS + ["daily_screen_time_hours"]])

    if st.button("Predict Single Row With Trained Models"):
        if clf_model is None or reg_model is None:
            if model_source == "Use existing saved models":
                st.error("Load saved models first.")
            else:
                st.error("Train models from uploaded CSV first.")
            return
        clf_input_df = align_input_to_model_features(input_df, clf_model)
        reg_input_df = align_input_to_model_features(input_df, reg_model)
        if any(c.startswith("predicted_") for c in clf_input_df.columns):
            st.warning(
                "Classification model was trained with `predicted_*` columns. "
                "Using compatibility mode for this run. Retrain models to remove leakage columns."
            )
        predicted_addiction_level = clf_model.predict(clf_input_df)[0]
        predicted_screen_time = reg_model.predict(reg_input_df)[0]
        recommendation = build_recommendation(predicted_addiction_level, float(predicted_screen_time))
        st.success(f"Predicted Addiction Level: {predicted_addiction_level}")
        st.info(f"Predicted Daily Screen Time: {predicted_screen_time:.2f} hours")
        st.markdown("### Suggested Action")
        st.write(recommendation)


if __name__ == "__main__":
    main()

