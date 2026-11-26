"""Project configuration and feature contracts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "humanoid_robot_tasks_200k_messy.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
FIGURE_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"

RANDOM_STATE = 4003
TARGET = "task_failure"
ID_COLUMN = "task_id"
LEAKAGE_COLUMNS = ["task_success", "failure_probability_simulated"]
POST_TASK_COLUMNS = ["completion_time_sec", "operator_intervention"]
EXCLUDED_COLUMNS = [ID_COLUMN, *LEAKAGE_COLUMNS, *POST_TASK_COLUMNS]

CATEGORICAL_FEATURES = ["shift", "warehouse_zone", "task_type"]
BINARY_FEATURES = ["human_nearby"]
BASE_NUMERIC_FEATURES = [
    "object_weight_kg", "object_size_liters", "travel_distance_m",
    "obstacles_count", "human_nearby", "aisle_width_m", "floor_friction",
    "lighting_lux", "temperature_c", "humidity_pct", "battery_level_pct",
    "motor_temperature_c", "joint_load_pct", "grasp_difficulty_1_to_5",
    "vision_confidence", "localization_error_cm", "network_latency_ms",
    "previous_task_failures", "maintenance_hours_since_service",
    "payload_balance_offset_cm", "path_planning_complexity_1_to_10",
]
ENGINEERED_FEATURES = [
    "load_distance_index", "thermal_load_index", "obstacles_per_10m",
    "battery_deficit", "maintenance_failure_interaction",
    "low_battery_flag", "low_vision_flag",
]
NUMERIC_FEATURES = [*BASE_NUMERIC_FEATURES, *ENGINEERED_FEATURES]
MODEL_FEATURES = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]

# Inclusive physical/domain ranges. Values outside these limits are set to missing.
VALID_RANGES = {
    "object_weight_kg": (0.0, 50.0),
    "object_size_liters": (0.0, 100.0),
    "travel_distance_m": (0.0, 100.0),
    "obstacles_count": (0.0, 20.0),
    "aisle_width_m": (0.3, 10.0),
    "floor_friction": (0.0, 1.0),
    "lighting_lux": (0.0, 2000.0),
    "temperature_c": (-20.0, 60.0),
    "humidity_pct": (0.0, 100.0),
    "battery_level_pct": (0.0, 100.0),
    "motor_temperature_c": (10.0, 120.0),
    "joint_load_pct": (0.0, 100.0),
    "grasp_difficulty_1_to_5": (1.0, 5.0),
    "vision_confidence": (0.0, 1.0),
    "localization_error_cm": (0.0, 100.0),
    "network_latency_ms": (0.0, 2000.0),
    "previous_task_failures": (0.0, 20.0),
    "maintenance_hours_since_service": (0.0, 2000.0),
    "payload_balance_offset_cm": (0.0, 50.0),
    "path_planning_complexity_1_to_10": (1.0, 10.0),
    "completion_time_sec": (0.0, 3600.0),
    "failure_probability_simulated": (0.0, 1.0),
}

# Illustrative business assumptions, not observed financial data.
FALSE_NEGATIVE_COST_AED = 500
FALSE_POSITIVE_COST_AED = 40
TUNING_SAMPLE_SIZE = 30_000
PERMUTATION_SAMPLE_SIZE = 2_500
