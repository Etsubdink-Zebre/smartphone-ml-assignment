---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 21px;
    padding: 40px;
    color: #0f172a;
    background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
  }
  h1 {
    color: #1d4ed8;
    font-size: 50px;
    margin-bottom: 10px;
  }
  h2 {
    color: #1e40af;
    font-size: 34px;
    border-bottom: 3px solid #bfdbfe;
    padding-bottom: 8px;
    margin-bottom: 18px;
  }
  ul, ol {
    font-size: 20px;
    line-height: 1.32;
  }
  img {
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
    display: block;
    margin: 12px auto 0 auto;
    object-fit: contain;
    max-height: 340px;
  }
  .two-col {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 18px;
    margin-top: 12px;
  }
  .two-col img {
    width: 48%;
    max-height: 300px;
    margin: 0;
  }
  section.lead {
    text-align: center;
    background: radial-gradient(circle at top left, #e0ecff 0%, #dbeafe 35%, #bfdbfe 100%);
  }
  section.lead h1 {
    color: #1e3a8a;
    border-bottom: none;
    font-size: 52px;
    line-height: 1.15;
    margin-bottom: 12px;
  }
  section.lead .subtitle {
    font-size: 24px;
    color: #1f2937;
    margin-bottom: 22px;
  }
  section.lead .meta {
    display: inline-block;
    text-align: left;
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid #bfdbfe;
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 10px 24px rgba(30, 58, 138, 0.12);
    font-size: 20px;
    line-height: 1.4;
  }
  section.lead .meta b {
    color: #1e40af;
  }
  section.end {
    text-align: center;
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
  }
  section.end h2 {
    border-bottom: none;
    font-size: 56px;
    margin-bottom: 10px;
    color: #1e3a8a;
  }
  section.end p {
    font-size: 26px;
    color: #1f2937;
  }
---

<!-- _class: lead -->
<!-- _paginate: false -->
# Smartphone Addiction Classification  
# and Screen Time Prediction

<div class="subtitle">Machine Learning Assignment Presentation</div>

<div class="meta">
<b>Student:</b> Etsubdink Zebre<br/>
<b>ID:</b> GSE/0523/18<br/>
<b>Date:</b> 23/3/2026
</div>

---

## Problem Statement
- This project solves two supervised machine learning problems from smartphone usage data:
  - **Task 1 (Classification):** Predict smartphone addiction level (`addiction_level`)
  - **Task 2 (Regression):** Predict daily screen time in hours (`daily_screen_time_hours`)
- Why this matters:
  - Helps identify high-risk phone usage patterns
  - Supports digital wellbeing planning for students/users
  - Demonstrates real-world use of ML in behavior analytics

---

## Dataset Overview
- Dataset source: Public smartphone usage dataset (CSV)
- File used: `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`
- Total records: **7,500**
- Data type: tabular, mixed numerical + categorical
- Example input features:
  - age, gender, social_media_hours, gaming_hours
  - notifications_per_day, app_opens_per_day
  - work_study_hours, sleep_hours, weekend_screen_time, stress_level
- Target variables:
  - Classification target: `addiction_level`
  - Regression target: `daily_screen_time_hours`

---

## ML Process Workflow
1. Problem Definition
2. Data Collection
3. EDA and Data Preparation
4. Algorithm Selection
5. Model Development and Training
6. Evaluation and Hyperparameter Tuning
7. Final Testing
8. Deployment Demo
9. Monitoring Plan and Documentation

**Approach:** baseline models first, then tuning the best model.

---

## EDA: Class and Target Distribution
- Key insights:
  - Addiction classes are not perfectly separated, so classification is harder
  - Daily screen time shows a stable pattern suitable for regression
  - Dataset size is enough for model comparison and tuning

![w:520](../reports/figures/eda/class_distribution_addiction_level.png)

---

## EDA: Target Distribution (Screen Time)
- The target for regression is continuous and suitable for regression modeling
- The distribution supports learning stable patterns for prediction

![w:760](../reports/figures/eda/daily_screen_time_distribution.png)

---

## EDA: Correlation and Feature Behavior
- Key interpretation:
  - Usage-related features show relationship with both targets
  - Higher social/gaming activity is generally linked with higher usage outcomes
  - EDA insights guided feature engineering and model selection
  - Correlation is useful, but not equal to causation

<div class="two-col">
  <img src="../reports/figures/eda/correlation_heatmap.png" />
  <img src="../reports/figures/eda/box_social_media_hours_by_addiction_level.png" />
</div>

---

## Classification Modeling
- Objective: Predict `addiction_level`
- Models tested:
  - Logistic Regression
  - Random Forest Classifier
  - SVM (RBF)
- Metrics used:
  - Accuracy
  - Weighted F1 (main metric)
- Why Weighted F1:
  - It balances precision and recall across multiple classes

![w:650](../reports/figures/classification/validation_f1_model_comparison.png)

---

## Classification Final Result
- Best model (after tuning): **Random Forest Classifier**
- Final classification performance:
  - Test Accuracy: **0.5617**
  - Test Weighted F1: **0.5583**
  - Best CV Weighted F1: **0.5730**
- Interpretation:
  - Performance is moderate; class behavior overlap reduces separability
  - Useful as a baseline, can improve with richer features and class balancing

![w:520](../reports/figures/classification/confusion_matrix_test.png)

---

## Regression Modeling
- Objective: Predict `daily_screen_time_hours`
- Models tested:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Main metrics: RMSE (main), MAE, R2
- Tuning method: GridSearchCV
- RMSE chosen because it penalizes larger errors more strongly

![w:520](../reports/figures/regression/validation_rmse_model_comparison.png)

---

## Regression Final Result
- Best model (after tuning): **Random Forest Regressor**
- Final regression performance:
  - MAE: **0.5937**
  - RMSE: **0.7008**
  - R2: **0.9300**
  - Best CV RMSE: **0.6793**
- Interpretation:
  - Regression model performs strongly and predicts screen time well
  - High R2 means the model explains most variance in the target

<div class="two-col">
  <img src="../reports/figures/regression/actual_vs_predicted_test.png" />
  <img src="../reports/figures/regression/residual_distribution_test.png" />
</div>

---

## Deployment Demo
- Local deployment demo completed using Streamlit
- App file: `app/streamlit_app.py`
- Demonstration:
  - User enters behavior values
  - App returns predicted addiction level and predicted daily screen time
- Deployment screenshots (input + prediction output):
- Run command:
  - `streamlit run app/streamlit_app.py`
- Status: deployment workflow tested successfully on local machine

<div class="two-col">
  <img src="../reports/figures/deployment/streamlit_demo.png" />
  <img src="../reports/figures/deployment/streamlit_demo1.png" />
</div>

---

## Conclusion and Future Work
- Key outcomes:
  - Completed full ML pipeline for both classification and regression tasks
  - Classification result is moderate (Accuracy 0.5617, Weighted F1 0.5583)
  - Regression result is strong (RMSE 0.7008, R2 0.9300)
- Limitations:
  - Behavior overlap reduces classification performance
  - Dataset is not time-series/longitudinal
- Future improvements:
  - Add richer features and class balancing methods
  - Try advanced models and explainability tools
  - Deploy to free cloud platform for live demo
  - Monitor drift and retrain periodically with new data

---

<!-- _class: end -->
## Thank You  
## Questions and Answers

