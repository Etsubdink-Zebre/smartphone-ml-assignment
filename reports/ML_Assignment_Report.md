# Machine Learning Assignment Report

## Project Titles
- Classification of Smartphone Addiction Levels Based on User Behavior Patterns
- Modeling and Predicting Daily Screen Time Using Regression Techniques

## Student Information
- Student Name: Etsubdink Zebre
- ID Number: GSE/0523/18
- Course: Machine Learning
- Date: 23/3/2026

## Abstract
This report presents a complete machine learning workflow for two supervised tasks using smartphone usage data: addiction level classification and daily screen time regression. The dataset includes 7,500 records with behavioral and demographic features. Data exploration, cleaning checks, preprocessing, feature engineering, and model comparison were performed before final model selection. For classification, the best tuned model achieved moderate performance (Accuracy = 0.5617, Weighted F1 = 0.5583). For regression, the best tuned model achieved strong performance (MAE = 0.5937, RMSE = 0.7008, R2 = 0.9300). A deployment demonstration was also completed through a Streamlit application.

## 1. Problem Definition
This assignment addresses two supervised machine learning problems in the smartphone behavior domain:
- **Classification:** predict `addiction_level`
- **Regression:** predict `daily_screen_time_hours`

### Success Criteria
- **Classification:** weighted F1 (primary), accuracy/recall (secondary)
- **Regression:** RMSE (primary), MAE and R2 (secondary)

## 2. Data Collection
- **Source:** `data/raw/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`
- **Total samples:** 7500
- **Domain relevance:** features represent realistic usage behavior, including app usage intensity, notifications, social media and gaming time, and lifestyle variables.

## 3. Data Exploration and Preparation

### 3.1 Exploratory Data Analysis
Performed in `notebooks/01_data_understanding_and_eda.ipynb`:
- class distribution (`addiction_level`)
- target distribution (`daily_screen_time_hours`)
- correlation heatmap for numeric features
- boxplots of important features across addiction classes

### 3.2 Data Cleaning
- missing values addressed with median/mode imputation inside model pipelines
- duplicate checks performed during EDA
- outlier checks performed using IQR analysis

### 3.3 Feature Engineering
Engineered features:
- `social_gaming_total_hours = social_media_hours + gaming_hours`
- `notifications_per_open = notifications_per_day / app_opens_per_day`

### 3.4 Data Splitting
- classification: train/validation/test split with stratification
- regression: train/validation/test split

## 4. Algorithm Selection

### 4.1 Classification Candidates
- Logistic Regression
- Random Forest Classifier
- SVM (RBF)

### 4.2 Regression Candidates
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Selection rationale: a balanced set of interpretable baseline models and nonlinear models was used to compare performance and generalization.

## 5. Model Development and Training
Implemented in:
- `notebooks/02_classification_modeling.ipynb`
- `notebooks/03_regression_modeling.ipynb`

All models are trained with preprocessing pipelines:
- numeric: median imputation + standard scaling
- categorical: mode imputation + one-hot encoding

## 6. Model Evaluation and Hyperparameter Tuning

### 6.1 Classification Results
Use `reports/tables/classification_baseline_metrics.csv` and `reports/tables/classification_metrics.csv`.

Final metrics:
- Test Accuracy: `0.5617`
- Test Weighted F1: `0.5583`
- Best CV Weighted F1: `0.5730`

### 6.2 Regression Results
Use `reports/tables/regression_baseline_metrics.csv` and `reports/tables/regression_metrics.csv`.

Final metrics:
- Test MAE: `0.5937`
- Test RMSE: `0.7008`
- Test R2: `0.9300`
- Best CV RMSE: `0.6793`

### 6.3 Overfitting / Underfitting Discussion
Compare validation and test metrics:
- if test performance is much worse than validation => likely overfitting
- if both are weak => likely underfitting

Observed behavior:
- Classification shows moderate performance, suggesting current features have limited separation power for addiction classes.
- Regression generalizes well, with high R2 and low prediction error, indicating the selected features are strongly informative for screen time.

## 7. Final Testing and Comparison
Consolidated in `notebooks/04_model_comparison_and_final_testing.ipynb`.

Final comparison artifacts:
- `reports/tables/final_task_summary.csv`
- `reports/figures/comparison/classification_baseline_comparison.png`
- `reports/figures/comparison/regression_baseline_comparison.png`

## 8. Deployment
Deployment demo provided through:
- `app/streamlit_app.py`
- `notebooks/05_deployment_demo.ipynb`
- Live app URL: [https://smartphoneml.streamlit.app/](https://smartphoneml.streamlit.app/)
- Status: deployed and publicly accessible for demonstration

Run locally:
```bash
streamlit run app/streamlit_app.py
```

## 9. Monitoring and Maintenance
Proposed monitoring plan:
- track prediction quality over time
- monitor latency and input drift
- retrain periodically or when metric thresholds degrade

### 9.1 User-Level Suggestions from Predictions
To make the model output actionable, the deployment layer provides simple recommendations based on predicted addiction level and predicted daily screen time:
- **Low risk / lower screen time:** maintain current digital habits and healthy sleep routine.
- **Medium risk / moderate screen time:** reduce non-essential notifications, set app limits, and schedule short phone-free periods.
- **High risk / high screen time:** apply stricter usage controls (focus mode, bedtime mode), reduce late-night use, and follow a structured daily screen-time plan.

This recommendation layer is rule-based and transparent, so users can easily understand why a specific suggestion is shown.

## 10. Conclusion
Key findings:
- Best classification model (current pipeline): Random Forest classifier with tuned hyperparameters, achieving weighted F1 of `0.5583`.
- Best regression model (current pipeline): Random Forest regressor with tuned hyperparameters, achieving RMSE of `0.7008` and R2 of `0.9300`.
- Important predictive behavior signals include social media usage, gaming usage, app open frequency, notifications, and derived usage-intensity features.

Interpretation:
- The classification task is more challenging due to overlapping behavior patterns across addiction categories.
- The regression task performs strongly, indicating daily screen time is highly predictable from behavioral usage attributes in this dataset.

Limitations:
- cross-sectional data and limited behavioral context
- no temporal sequence modeling

Future work:
- collect longitudinal data
- incorporate richer behavioral signals
- evaluate advanced models and explainability tools

## References
- Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
- Streamlit documentation: [https://streamlit.io](https://streamlit.io)

## 11. Reproducibility Checklist
- [ ] `pip install -r requirements.txt`
- [ ] run notebooks 01 -> 05 in order
- [ ] run `python run_pipeline.py --task all`
- [ ] run `streamlit run app/streamlit_app.py`

