from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CLASSIFICATION_MODEL_DIR = MODELS_DIR / "classification"
REGRESSION_MODEL_DIR = MODELS_DIR / "regression"

