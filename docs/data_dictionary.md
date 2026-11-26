# Data dictionary

## Scope

The raw file `data/humanoid_robot_tasks_200k_messy.csv` contains 30 columns. Values below describe the intended schema; the messy file deliberately contains inconsistent types and labels. Cleaning rules are implemented in `src/data_processing.py`, and valid numeric ranges are centralised in `src/config.py`.

All data are synthetic course-project records. No column should be interpreted as a measurement from a real robot fleet.

## Raw columns

| Column | Intended type / units | Valid values or range | Meaning and model role |
|---|---|---|---|
| `task_id` | Identifier | Unique integer-like ID | Record identifier; excluded to prevent memorisation. |
| `shift` | Category | `morning`, `afternoon`, `night` | Work shift; pre-task categorical input. |
| `warehouse_zone` | Category | `receiving`, `storage`, `packing`, `shipping` | Warehouse area; pre-task categorical input. The typo `Shippping` is corrected. |
| `task_type` | Category | `pick_and_place`, `transport`, `shelf_retrieval`, `tote_sorting` | Requested operation; pre-task categorical input. |
| `object_weight_kg` | Numeric, kg | 0–50 | Object mass; numeric input. |
| `object_size_liters` | Numeric, litres | 0–100 | Object volume; numeric input. |
| `travel_distance_m` | Numeric, metres | 0–100 | Planned travel distance; numeric input. |
| `obstacles_count` | Integer count | 0–20 | Known obstacles on the planned path; numeric input. |
| `human_nearby` | Binary | 0 = no, 1 = yes | Whether a person is nearby; numeric input. Mixed labels such as `yes`, `No`, and `TRUE` are normalised. |
| `aisle_width_m` | Numeric, metres | 0.3–10 | Width of the operating aisle; numeric input. |
| `floor_friction` | Numeric coefficient | 0–1 | Floor-friction estimate; numeric input. |
| `lighting_lux` | Numeric, lux | 0–2,000 | Ambient illumination; numeric input. |
| `temperature_c` | Numeric, °C | −20–60 | Ambient temperature; numeric input. |
| `humidity_pct` | Numeric, percent | 0–100 | Relative humidity; numeric input. |
| `battery_level_pct` | Numeric, percent | 0–100 | Battery level before execution; numeric input. |
| `motor_temperature_c` | Numeric, °C | 10–120 | Motor temperature before execution; numeric input. |
| `joint_load_pct` | Numeric, percent | 0–100 | Joint-load estimate; numeric input. |
| `grasp_difficulty_1_to_5` | Ordinal integer | 1–5 | Expected grasp difficulty; numeric input. |
| `vision_confidence` | Numeric proportion | 0–1 | Perception confidence before execution; numeric input. |
| `localization_error_cm` | Numeric, centimetres | 0–100 | Estimated localisation error; numeric input. |
| `network_latency_ms` | Numeric, milliseconds | 0–2,000 | Communication latency; numeric input. |
| `previous_task_failures` | Integer count | 0–20 | Number of previous failures known before the new task; numeric input. |
| `maintenance_hours_since_service` | Numeric, hours | 0–2,000 | Operating hours since the last service; numeric input. |
| `payload_balance_offset_cm` | Numeric, centimetres | 0–50 | Payload offset from its balanced position; numeric input. |
| `path_planning_complexity_1_to_10` | Ordinal integer | 1–10 | Planned-path complexity; numeric input. |
| `operator_intervention` | Binary | 0 = no, 1 = yes | Intervention observed during/after the task; **excluded as post-task leakage**. |
| `completion_time_sec` | Numeric, seconds | 0–3,600 | Observed completion time; **excluded as post-task leakage**. |
| `failure_probability_simulated` | Numeric proportion | 0–1 | Simulation-generated probability related to the target; **excluded as direct leakage**. |
| `task_failure` | Binary target | 0 = success, 1 = failure | Prediction target. Records with unknown labels are excluded from supervised fitting and evaluation. |
| `task_success` | Binary | 0 = no, 1 = yes | Target complement; **excluded as direct leakage**. |

## Engineered model features

These seven features are recreated deterministically by `add_engineered_features()` for training, single prediction, and batch scoring. If a source value is missing, the dependent engineered value is missing and is handled by the fitted pipeline imputer.

| Feature | Formula / definition | Rationale |
|---|---|---|
| `load_distance_index` | `object_weight_kg × travel_distance_m` | Combines payload and distance into a workload proxy. |
| `thermal_load_index` | `motor_temperature_c × joint_load_pct / 100` | Combines thermal condition and mechanical loading. |
| `obstacles_per_10m` | `10 × obstacles_count / (travel_distance_m + 1)` | Obstacle-density proxy; the +1 stabilises zero-distance tasks. |
| `battery_deficit` | `100 − battery_level_pct` | Expresses remaining battery as distance from full charge. |
| `maintenance_failure_interaction` | `maintenance_hours_since_service × (previous_task_failures + 1)` | Represents the joint burden of service age and failure history. |
| `low_battery_flag` | 1 if battery < 20%, otherwise 0; missing remains missing | Makes an operational low-battery regime explicit. |
| `low_vision_flag` | 1 if vision confidence < 0.70, otherwise 0; missing remains missing | Makes a low-confidence perception regime explicit. |

## Final preprocessing contract

The model receives 31 original pipeline inputs: 3 categorical fields, 21 base numeric fields, and 7 engineered numeric fields.

- Categorical missing values are replaced with the **training-fold mode**, then one-hot encoded with unknown categories ignored at inference.
- Numeric missing values are replaced with the **training-fold median**.
- Logistic regression also standardises numeric features; random forest does not require scaling.
- Preprocessing is fitted inside cross-validation and model pipelines, preventing validation/test imputation or encoding leakage.
- Batch scoring never removes blank or duplicate records: each uploaded row receives one output row in its original order.

## Scored CSV fields

The app preserves every uploaded column and appends:

| Output field | Meaning |
|---|---|
| `failure_probability` | Estimated probability of `task_failure = 1`, between 0 and 1. |
| `review_required` | 1 when the probability meets the selected operating threshold; otherwise 0. |

`review_required` is a workflow recommendation for this synthetic demonstration, not a guarantee or an automated control action.
