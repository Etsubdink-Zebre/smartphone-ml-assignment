import pandas as pd

from src.config import RAW_DATA_PATH


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)

