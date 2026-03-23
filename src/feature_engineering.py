import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add assignment-safe engineered features.
    Keep it simple for now so you can expand in notebooks.
    """
    out = df.copy()
    if {"social_media_hours", "gaming_hours"}.issubset(out.columns):
        out["social_gaming_total_hours"] = out["social_media_hours"] + out["gaming_hours"]
    if {"notifications_per_day", "app_opens_per_day"}.issubset(out.columns):
        out["notifications_per_open"] = out["notifications_per_day"] / out["app_opens_per_day"].replace(0, 1)
    return out

