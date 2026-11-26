"""Run the complete reproducible CDS4003 analysis pipeline."""
from __future__ import annotations

import json
import time

import pandas as pd

from src.config import DATA_PATH, FIGURE_DIR, TABLE_DIR
from src.data_processing import audit_data, clean_dataset, prepare_model_data
from src.modeling import train_final_model
from src.reporting import generate_eda, generate_evaluation_figures


def main():
    start = time.time()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    raw_audit = audit_data(raw, "raw")
    clean, cleaning_summary = clean_dataset(raw)
    clean_audit = audit_data(clean, "after_deterministic_cleaning")
    combined_audit = pd.concat([raw_audit, clean_audit], ignore_index=True)
    combined_audit.to_csv(TABLE_DIR / "data_quality_audit.csv", index=False)
    pd.DataFrame([cleaning_summary]).drop(columns=["invalid_by_column"]).to_csv(
        TABLE_DIR / "cleaning_summary.csv", index=False
    )
    pd.DataFrame(cleaning_summary["invalid_by_column"]).to_csv(
        TABLE_DIR / "domain_invalid_values.csv", index=False
    )
    clean.head(2000).to_csv(DATA_PATH.parent / "cleaned_sample_humanoid_robot_tasks.csv", index=False)
    print(json.dumps({k: v for k, v in cleaning_summary.items() if k != "invalid_by_column"}, indent=2))

    print("Generating EDA")
    eda_summary = generate_eda(raw, clean)
    X, y = prepare_model_data(clean)
    print(f"Model matrix: {X.shape}; failure rate={y.mean():.4f}")
    print("Tuning and evaluating models")
    results = train_final_model(X, y)
    print(results["comparison"].to_string(index=False))
    print(json.dumps(results["metadata"]["test_metrics_selected_threshold"], indent=2))
    print("Generating evaluation figures")
    generate_evaluation_figures(results)

    summary = {
        "cleaning": {k: v for k, v in cleaning_summary.items() if k != "invalid_by_column"},
        "eda": eda_summary,
        "model_rows": int(len(X)),
        "model_features": int(X.shape[1]),
        "selected_model": results["metadata"]["selected_model"],
        "decision_threshold": results["metadata"]["decision_threshold"],
        "test_metrics": results["metadata"]["test_metrics_selected_threshold"],
        "runtime_seconds": round(time.time() - start, 2),
    }
    (TABLE_DIR.parent / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Completed in {summary['runtime_seconds']} seconds")


if __name__ == "__main__":
    main()
