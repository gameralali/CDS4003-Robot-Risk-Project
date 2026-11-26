#!/usr/bin/env python3
"""Add missing values, inconsistent formatting, outliers, and fully empty rows.

This is for data-cleaning practice. It intentionally corrupts a clean synthetic
humanoid-robot dataset; it does not represent real Agility Robotics data.

Usage:
    pip install pandas numpy
    python make_dataset_messy.py humanoid_robot_tasks_200k.csv \
        --output humanoid_robot_tasks_200k_messy.csv --seed 20261027
"""
import argparse
import numpy as np
import pandas as pd


def make_messy(df: pd.DataFrame, seed: int = 20261027) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()

    protected = {"task_id", "task_success", "task_failure"}
    for col in df.columns:
        if col not in protected:
            df.loc[rng.random(len(df)) < 0.02, col] = np.nan

    mappings = {
        "shift": {"morning": "Morning", "afternoon": "AFTERNOON", "night": "night "},
        "warehouse_zone": {
            "receiving": "Receiving", "storage": "storage ",
            "packing": "PACKING", "shipping": "Shippping"
        },
        "task_type": {
            "pick_and_place": "Pick and Place", "transport": "TRANSPORT ",
            "shelf_retrieval": "shelf retrieval", "tote_sorting": "Tote_Sorting"
        },
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            mask = (rng.random(len(df)) < 0.025) & df[col].notna()
            df.loc[mask, col] = df.loc[mask, col].map(mapping).fillna(df.loc[mask, col])

    outlier_rates = {
        "object_weight_kg": (lambda x: -abs(x), lambda x: 999),
        "battery_level_pct": (lambda x: -10, lambda x: 150),
        "vision_confidence": (lambda x: -0.2, lambda x: 1.8),
        "completion_time_sec": (lambda x: -5, lambda x: 9999),
        "temperature_c": (lambda x: -100, lambda x: 200),
    }
    for col, funcs in outlier_rates.items():
        if col not in df.columns:
            continue
        candidates = df.index[df[col].notna()].to_numpy()
        if len(candidates) == 0:
            continue
        count = max(1, int(len(candidates) * 0.0015))
        chosen = rng.choice(candidates, size=min(count, len(candidates)), replace=False)
        half = len(chosen) // 2
        df.loc[chosen[:half], col] = funcs[0](pd.to_numeric(df.loc[chosen[:half], col]))
        df.loc[chosen[half:], col] = funcs[1](pd.to_numeric(df.loc[chosen[half:], col]))

    for col in ["human_nearby", "operator_intervention", "task_success", "task_failure"]:
        if col not in df.columns:
            continue
        candidates = df.index[df[col].notna()].to_numpy()
        if len(candidates) == 0:
            continue
        chosen = rng.choice(candidates, size=max(1, int(len(candidates) * 0.001)), replace=False)
        values = df[col].astype(object)
        for i, idx in enumerate(chosen):
            values.loc[idx] = ["yes", "No", "TRUE", "unknown"][i % 4]
        df[col] = values

    empty_count = max(1, int(round(len(df) * 0.001)))
    positions = set(rng.choice(np.arange(len(df) + empty_count), size=empty_count, replace=False).tolist())
    rows = []
    iterator = iter(df.itertuples(index=False, name=None))
    for pos in range(len(df) + empty_count):
        if pos in positions:
            rows.append([np.nan] * len(df.columns))
        else:
            rows.append(list(next(iterator)))
    return pd.DataFrame(rows, columns=df.columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="path to clean CSV")
    parser.add_argument("--output", default="humanoid_robot_tasks_200k_messy.csv")
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()

    clean = pd.read_csv(args.input)
    messy = make_messy(clean, args.seed)
    messy.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Rows including blank rows: {len(messy):,}")
    print(f"Columns: {len(messy.columns)}")
    print(f"Fully empty rows: {messy.isna().all(axis=1).sum():,}")
    print(f"Missing cells: {int(messy.isna().sum().sum()):,}")


if __name__ == "__main__":
    main()
