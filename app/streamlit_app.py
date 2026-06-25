import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_PATH
from src.ui_modeling import train_models_from_uploaded_csv


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


@st.cache_resource(show_spinner="Training models from bundled dataset...")
def train_bundled_models():
    """Train fresh models from the repo CSV (once per server process)."""
    train_df = pd.read_csv(RAW_DATA_PATH)
    return train_models_from_uploaded_csv(train_df)


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
    st.write(
        "Models are trained automatically from the bundled dataset on startup, "
        "so deployment does not depend on pickled files from another machine."
    )

    if "custom_clf_model" not in st.session_state:
        st.session_state.custom_clf_model = None
    if "custom_reg_model" not in st.session_state:
        st.session_state.custom_reg_model = None

    model_source = st.radio(
        "Choose model source",
        ["Bundled dataset (default)", "Custom uploaded CSV"],
        horizontal=True,
    )

    clf_model = None
    reg_model = None

    if model_source == "Bundled dataset (default)":
        st.subheader("Bundled Dataset")
        st.caption(f"Training data: `{RAW_DATA_PATH.name}` (auto-trained once per app restart).")
        if not RAW_DATA_PATH.exists():
            st.error(f"Bundled dataset not found at `{RAW_DATA_PATH}`.")
            return
        clf_model, reg_model = train_bundled_models()
        if clf_model is None or reg_model is None:
            st.error("Bundled dataset training failed. Check that target columns are present in the CSV.")
            return
        st.success("Models ready (trained from bundled dataset).")
    else:
        st.subheader("Upload Training CSV")
        st.caption(
            "Required feature columns: age, gender, social_media_hours, gaming_hours, work_study_hours, sleep_hours, "
            "notifications_per_day, app_opens_per_day, weekend_screen_time, stress_level, academic_work_impact. "
            "Training target columns: addiction_level (classification), daily_screen_time_hours (regression)."
        )
        train_file = st.file_uploader("Training CSV", type=["csv"], key="train_csv")
        if train_file is not None and st.button("Train Models From Uploaded CSV"):
            try:
                train_df = pd.read_csv(train_file)
                with st.spinner("Training models..."):
                    clf_model, reg_model = train_models_from_uploaded_csv(train_df)
                st.session_state.custom_clf_model = clf_model
                st.session_state.custom_reg_model = reg_model
                if clf_model is None and reg_model is None:
                    st.error(
                        "No model was trained. Add non-null `addiction_level` and/or "
                        "`daily_screen_time_hours` in training CSV."
                    )
                else:
                    st.success("Training complete. Custom models are active for this session.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

        clf_model = st.session_state.custom_clf_model
        reg_model = st.session_state.custom_reg_model
        if clf_model is None or reg_model is None:
            st.info("Upload a CSV and click **Train Models From Uploaded CSV** to enable predictions.")

    st.subheader("Quick Single-Row Test")
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
            st.error("Train or load models first.")
            return
        clf_input_df = align_input_to_model_features(input_df, clf_model)
        reg_input_df = align_input_to_model_features(input_df, reg_model)
        predicted_addiction_level = clf_model.predict(clf_input_df)[0]
        predicted_screen_time = reg_model.predict(reg_input_df)[0]
        recommendation = build_recommendation(predicted_addiction_level, float(predicted_screen_time))
        st.success(f"Predicted Addiction Level: {predicted_addiction_level}")
        st.info(f"Predicted Daily Screen Time: {predicted_screen_time:.2f} hours")
        st.markdown("### Suggested Action")
        st.write(recommendation)


if __name__ == "__main__":
    main()
