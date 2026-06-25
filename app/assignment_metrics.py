"""Fixed evaluation metrics from the assignment report (tuned pipeline, hold-out test set)."""

REPORT_METRICS = {
    "classification": {
        "model": "Random Forest (tuned)",
        "target": "addiction_level",
        "accuracy": 0.5617,
        "f1_weighted": 0.5583,
        "precision_weighted": 0.5558,
        "recall_weighted": 0.5617,
    },
    "regression": {
        "model": "Random Forest (tuned)",
        "target": "daily_screen_time_hours",
        "mae": 0.5937,
        "rmse": 0.7008,
        "r2": 0.9300,
    },
}
