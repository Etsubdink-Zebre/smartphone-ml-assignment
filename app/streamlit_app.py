from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


@st.cache_resource
def load_models():
    clf_model = joblib.load(CLASSIFICATION_MODEL_PATH) if CLASSIFICATION_MODEL_PATH.exists() else None
    reg_model = joblib.load(REGRESSION_MODEL_PATH) if REGRESSION_MODEL_PATH.exists() else None
    return clf_model, reg_model


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

    high_risk_levels = {"high", "severe", "very high"}
    medium_risk_levels = {"medium", "moderate"}

    if level in high_risk_levels or predicted_screen_time > 6.0:
        return (
            "High-risk usage pattern detected. Reduce non-essential app use, enable focus/bedtime mode, "
            "and schedule phone-free blocks (for example: study time and before sleep)."
        )
    if level in medium_risk_levels or 4.0 <= predicted_screen_time <= 6.0:
        return (
            "Moderate usage pattern detected. Set daily app limits, reduce notifications, and keep at least "
            "one short phone-free period each day."
        )
    return (
        "Lower-risk usage pattern detected. Continue current habits and maintain good sleep and study/work balance."
    )


def main():
    st.set_page_config(page_title="Smartphone ML Demo", layout="centered")
    st.title("Smartphone ML Assignment Demo")
    st.write("Predict smartphone addiction level and daily screen time.")

    clf_model, reg_model = load_models()
    if clf_model is None or reg_model is None:
        st.error(
            "Trained models not found. Run training first:\n"
            "`python run_pipeline.py --task all`"
        )
        return

    st.subheader("Input User Behavior")
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

    st.write("Model input preview:")
    st.dataframe(input_df[INPUT_COLUMNS + ["daily_screen_time_hours"]])

    if st.button("Predict"):
        predicted_addiction_level = clf_model.predict(input_df)[0]
        predicted_screen_time = reg_model.predict(input_df)[0]
        recommendation = build_recommendation(predicted_addiction_level, float(predicted_screen_time))

        st.success(f"Predicted Addiction Level: {predicted_addiction_level}")
        st.info(f"Predicted Daily Screen Time: {predicted_screen_time:.2f} hours")
        st.markdown("### Suggested Action")
        st.write(recommendation)


if __name__ == "__main__":
    main()

