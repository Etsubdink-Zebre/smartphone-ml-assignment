## Smartphone ML Assignment

This project has 2 machine learning tasks:
- Classification: predict smartphone addiction level
- Regression: predict daily screen time

## Project Structure

- `data/raw/` original dataset files
- `notebooks/` EDA and modeling notebooks
- `src/` reusable Python scripts and pipeline code
- `models/` saved trained models
- `reports/` report, figures, and result tables
- `presentation/` presentation markdown and exported PPT
- `outputs/` predictions and tuning results
- `app/` deployment demo (Streamlit)

## Dataset

- `data/raw/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`

## Setup (Beginner Friendly)

1. Open terminal in the project folder:

```powershell
cd C:\Users\HP\Documents\AI\Y1S2\ML\smartphone_ml_assignment
```

2. Install packages:

```bash
pip install -r requirements.txt
```

## Step-by-Step Commands (Windows / PowerShell)

### 1) Train models quickly from script

```powershell
python run_pipeline.py --task all
```

### 2) Execute all notebooks (recommended for report figures/tables)

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_understanding_and_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_classification_modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_regression_modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_model_comparison_and_final_testing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_deployment_demo.ipynb
```

### 3) Run deployment demo (optional)

```powershell
streamlit run app/streamlit_app.py
```

### 4) Export presentation to PPTX from Marp markdown

```powershell
marp presentation/ML_Assignment_Presentation_Outline.md --pptx --allow-local-files --output presentation/ML_Assignment_Presentation.pptx
```

## Notebook Order

1. `notebooks/01_data_understanding_and_eda.ipynb`
2. `notebooks/02_classification_modeling.ipynb`
3. `notebooks/03_regression_modeling.ipynb`
4. `notebooks/04_model_comparison_and_final_testing.ipynb`
5. `notebooks/05_deployment_demo.ipynb`

## Quick Run

```bash
python run_pipeline.py --task all
```

Available tasks:
- `classification`
- `regression`
- `all`

## Deployment Demo

```bash
streamlit run app/streamlit_app.py
```
