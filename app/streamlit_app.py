"""Streamlit deployment for pre-task humanoid robot failure-risk scoring."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_FEATURES
from src.scoring import score_frame

st.set_page_config(page_title="Robot Task Risk Console", page_icon="◉", layout="wide")
st.markdown("""
<style>
:root { --bg:#050810; --panel:#0c1220; --panel2:#111a2b; --line:#1a2540;
        --cyan:#22d3ee; --mint:#34d399; --amber:#fbbf24; --red:#f87171; }
.stApp { background: var(--bg); color: #e5edf7; }
.stApp::before { content:""; position:fixed; inset:0; pointer-events:none; opacity:.10;
  background:repeating-linear-gradient(0deg,transparent 0px,transparent 3px,#22d3ee 4px); z-index:0; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stMainViewContainer"] { max-width:1500px; }
h1,h2,h3 { font-family:"JetBrains Mono","Fira Code",monospace; letter-spacing:.03em; }
p,label,li { font-family:Inter,Arial,sans-serif; }
.console-title { font-family:"JetBrains Mono","Fira Code",monospace; font-size:1.8rem;
  font-weight:700; letter-spacing:.08em; margin:0; }
.console-sub { color:#9ca3af; font-size:.92rem; margin-top:.3rem; }
.live { display:inline-flex; align-items:center; gap:.5rem; border:1px solid var(--line);
  background:var(--panel); padding:.45rem .7rem; font-family:monospace; font-size:.75rem;
  letter-spacing:.12em; color:var(--mint); }
.dot { width:8px; height:8px; border-radius:50%; background:var(--mint);
  box-shadow:0 0 10px var(--mint); animation:pulse 1.6s ease-in-out infinite; }
.ticker { border-top:1px solid var(--line); border-bottom:1px solid var(--line);
  color:#9ca3af; font-family:monospace; font-size:.74rem; letter-spacing:.08em;
  padding:.55rem 0; margin:1rem 0 1.2rem; }
.panel { background:var(--panel); border:1px solid var(--line); padding:1rem; margin-bottom:.8rem; }
.risk-high { border-left:3px solid var(--red); }
.risk-low { border-left:3px solid var(--mint); }
.risk-label { font-family:monospace; font-size:.72rem; letter-spacing:.18em; color:#9ca3af; }
.risk-number { font-family:monospace; font-size:3rem; line-height:1.1; font-weight:700; }
.high-text { color:var(--red); } .low-text { color:var(--mint); }
.action { color:#e5edf7; line-height:1.55; margin-top:.6rem; }
.small { color:#9ca3af; font-size:.82rem; line-height:1.5; }
.stButton>button, .stDownloadButton>button { background:var(--cyan); color:#050810;
  border:1px solid var(--cyan); border-radius:4px; font-weight:700; min-height:44px; }
.stButton>button:hover, .stDownloadButton>button:hover { background:#67e8f9; border-color:#67e8f9; color:#050810; }
.stButton>button:focus-visible, .stDownloadButton>button:focus-visible { outline:3px solid var(--amber); outline-offset:2px; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:.7rem; }
[data-testid="stDataFrame"] { border:1px solid var(--line); }
@keyframes pulse { 0%,100%{opacity:.55} 50%{opacity:1} }
@media (prefers-reduced-motion:reduce) { .dot{animation:none} }
@media (max-width:768px) { .console-title{font-size:1.25rem} .risk-number{font-size:2.2rem} }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "models" / "best_model.joblib")
    metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text())
    importance = pd.read_csv(ROOT / "outputs" / "tables" / "permutation_importance.csv")
    return model, metadata, importance


def input_checks(record: dict) -> list[str]:
    checks = []
    if record["battery_level_pct"] < 20:
        checks.append("Battery is below 20%.")
    if record["vision_confidence"] < 0.70:
        checks.append("Vision confidence is below 0.70.")
    if record["joint_load_pct"] > 80:
        checks.append("Joint load exceeds 80%.")
    if record["maintenance_hours_since_service"] > 200:
        checks.append("More than 200 hours have elapsed since service.")
    if record["previous_task_failures"] >= 2:
        checks.append("The robot has at least two previous task failures.")
    if record["object_weight_kg"] * record["travel_distance_m"] > 200:
        checks.append("The load–distance index exceeds 200 kg·m.")
    return checks


model, meta, importance = load_artifacts()
left, right = st.columns([4, 1.2])
with left:
    st.markdown('<p class="console-title">ROBOT TASK / RISK CONSOLE</p>', unsafe_allow_html=True)
    st.markdown('<p class="console-sub">Pre-task decision support for warehouse operations · synthetic-data demonstration</p>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="live"><span class="dot"></span>MODEL READY</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="ticker">MODEL {meta["selected_model"].upper()} &nbsp;·&nbsp; '
    f'TEST ROC AUC {meta["test_metrics_selected_threshold"]["roc_auc"]:.3f} &nbsp;·&nbsp; '
    f'INPUTS {len(MODEL_FEATURES)} ENGINEERED PIPELINE FEATURES &nbsp;·&nbsp; NO POST-TASK FIELDS</div>',
    unsafe_allow_html=True,
)

single_tab, batch_tab, evidence_tab = st.tabs(["Single task", "Batch scoring", "Model evidence"])
with single_tab:
    controls, result_col, action_col = st.columns([1.25, 1.0, 1.0], gap="large")
    with controls:
        st.subheader("Task parameters")
        with st.form("risk_form"):
            st.markdown("**Operation**")
            c1, c2 = st.columns(2)
            shift = c1.selectbox("Shift", ["morning", "afternoon", "night"])
            zone = c2.selectbox("Warehouse zone", ["receiving", "storage", "packing", "shipping"])
            task_type = st.selectbox("Task type", ["pick_and_place", "transport", "shelf_retrieval", "tote_sorting"])
            with st.expander("Load and path", expanded=True):
                a, b = st.columns(2)
                weight = a.number_input("Object weight (kg)", 0.01, 50.0, 4.0, 0.1)
                size = b.number_input("Object size (L)", 0.01, 100.0, 10.0, 0.1)
                distance = a.number_input("Travel distance (m)", 0.0, 100.0, 8.0, 0.1)
                obstacles = b.number_input("Obstacles", 0, 20, 2, 1)
                human = a.selectbox("Human nearby", [0, 1], format_func=lambda x: "Yes" if x else "No")
                complexity = b.slider("Path complexity", 1, 10, 3)
                aisle = a.number_input("Aisle width (m)", 0.3, 10.0, 1.8, 0.1)
                offset = b.number_input("Payload offset (cm)", 0.0, 50.0, 2.0, 0.1)
                grasp = a.slider("Grasp difficulty", 1, 5, 2)
                friction = b.slider("Floor friction", 0.0, 1.0, 0.70, 0.01)
            with st.expander("Robot condition", expanded=True):
                a, b = st.columns(2)
                battery = a.slider("Battery (%)", 0.0, 100.0, 70.0, 0.1)
                motor = b.number_input("Motor temperature (°C)", 10.0, 120.0, 42.0, 0.1)
                joint = a.slider("Joint load (%)", 0.0, 100.0, 35.0, 0.1)
                vision = b.slider("Vision confidence", 0.0, 1.0, 0.90, 0.01)
                localisation = a.number_input("Localisation error (cm)", 0.0, 100.0, 4.0, 0.1)
                latency = b.number_input("Network latency (ms)", 0.0, 2000.0, 45.0, 1.0)
                previous = a.number_input("Previous failures", 0, 20, 0, 1)
                maintenance = b.number_input("Hours since service", 0.0, 2000.0, 40.0, 1.0)
            with st.expander("Environment"):
                a, b = st.columns(2)
                lighting = a.number_input("Lighting (lux)", 0.0, 2000.0, 300.0, 1.0)
                temperature = b.number_input("Temperature (°C)", -20.0, 60.0, 23.0, 0.1)
                humidity = a.slider("Humidity (%)", 0.0, 100.0, 45.0, 0.1)
            submitted = st.form_submit_button("Assess task risk", width="stretch")

    record = {
        "shift": shift, "warehouse_zone": zone, "task_type": task_type,
        "object_weight_kg": weight, "object_size_liters": size,
        "travel_distance_m": distance, "obstacles_count": obstacles,
        "human_nearby": human, "aisle_width_m": aisle, "floor_friction": friction,
        "lighting_lux": lighting, "temperature_c": temperature, "humidity_pct": humidity,
        "battery_level_pct": battery, "motor_temperature_c": motor,
        "joint_load_pct": joint, "grasp_difficulty_1_to_5": grasp,
        "vision_confidence": vision, "localization_error_cm": localisation,
        "network_latency_ms": latency, "previous_task_failures": previous,
        "maintenance_hours_since_service": maintenance,
        "payload_balance_offset_cm": offset, "path_planning_complexity_1_to_10": complexity,
    }
    if submitted:
        probability = float(score_frame(model, pd.DataFrame([record]))[0])
        st.session_state["single_probability"] = probability
        st.session_state["single_checks"] = input_checks(record)

    with result_col:
        st.subheader("Risk assessment")
        mode = st.radio("Operating point", ["Balanced (F1)", "Standard (0.50)", "High recall (cost-based)"])
        threshold = {
            "Balanced (F1)": meta["balanced_f1_threshold"],
            "Standard (0.50)": 0.5,
            "High recall (cost-based)": meta["decision_threshold"],
        }[mode]
        if "single_probability" not in st.session_state:
            st.markdown('<div class="panel"><span class="risk-label">AWAITING INPUT</span><p class="action">Set task parameters and select <b>Assess task risk</b>. The model uses only pre-task information.</p></div>', unsafe_allow_html=True)
        else:
            probability = st.session_state["single_probability"]
            predicted = probability >= threshold
            status = "REVIEW TASK" if predicted else "ROUTINE MONITORING"
            css = "risk-high" if predicted else "risk-low"
            text_css = "high-text" if predicted else "low-text"
            st.markdown(f'''<div class="panel {css}"><span class="risk-label">FAILURE PROBABILITY</span>
              <div class="risk-number {text_css}">{probability:.1%}</div>
              <b>{status}</b><br><span class="small">Threshold {threshold:.2f} · {mode}</span></div>''', unsafe_allow_html=True)
            st.progress(min(max(probability, 0.0), 1.0))
    with action_col:
        st.subheader("Operational guidance")
        if "single_probability" in st.session_state:
            predicted = st.session_state["single_probability"] >= threshold
            guidance = ("Pause automatic release and have a supervisor review the task. Check the flagged conditions before deciding whether to proceed."
                        if predicted else "No threshold-based review is indicated. Continue normal monitoring; this is not a guarantee of success.")
            st.markdown(f'<div class="panel"><span class="risk-label">RECOMMENDED WORKFLOW</span><p class="action">{guidance}</p></div>', unsafe_allow_html=True)
            checks = st.session_state["single_checks"]
            st.markdown("**Input checks — not causal explanations**")
            if checks:
                for item in checks:
                    st.write("• " + item)
            else:
                st.write("No rule-based input checks were triggered.")
        st.markdown("**Decision limits**")
        st.caption("Synthetic data only. Predictions support—not replace—operational judgment. Validate thresholds with real costs and a pilot before operational use.")

with batch_tab:
    st.subheader("Batch CSV scoring")
    st.write("Upload a CSV containing the 24 raw pre-task input columns. Engineered features are recreated automatically; target and post-task columns are not required.")
    uploaded = st.file_uploader("Task CSV", type=["csv"])
    batch_mode = st.selectbox("Batch operating point", ["Balanced (F1)", "Standard (0.50)", "High recall (cost-based)"])
    batch_threshold = {
        "Balanced (F1)": meta["balanced_f1_threshold"], "Standard (0.50)": 0.5,
        "High recall (cost-based)": meta["decision_threshold"],
    }[batch_mode]
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded, skip_blank_lines=False)
            if len(batch) > 50_000:
                st.error("For this demonstration, upload at most 50,000 rows per file.")
            else:
                probabilities = score_frame(model, batch)
                output = batch.copy()
                output["failure_probability"] = probabilities
                output["review_required"] = (probabilities >= batch_threshold).astype(int)
                m1, m2, m3 = st.columns(3)
                m1.metric("Tasks", f"{len(output):,}")
                m2.metric("Review required", f"{int(output.review_required.sum()):,}")
                m3.metric("Mean failure probability", f"{probabilities.mean():.1%}")
                st.dataframe(output.head(100), width="stretch")
                st.download_button("Download scored CSV", output.to_csv(index=False).encode("utf-8"),
                                   "scored_robot_tasks.csv", "text/csv")
        except Exception as exc:
            st.error(str(exc))
    else:
        st.info("Choose a CSV to enable scoring and download.")

with evidence_tab:
    st.subheader("Validation and model evidence")
    metrics = {
        "Standard 0.50": meta["test_metrics_default_threshold"],
        "Balanced F1": meta["test_metrics_balanced_threshold"],
        "High recall / cost": meta["test_metrics_selected_threshold"],
    }
    metric_table = pd.DataFrame([
        {"Operating point": name, "Threshold": values["threshold"],
         "Accuracy": values["accuracy"], "Precision": values["precision"],
         "Recall": values["recall"], "F1": values["f1"],
         "ROC AUC": values["roc_auc"], "PR AUC": values["pr_auc"],
         "Error cost (AED)": values["total_error_cost_aed"]}
        for name, values in metrics.items()
    ])
    metric_formats = {c: "{:.3f}" for c in ["Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "PR AUC"]}
    metric_formats["Error cost (AED)"] = "{:,.0f}"
    st.dataframe(metric_table.style.format(metric_formats), width="stretch", hide_index=True)
    st.caption(f'Untouched test set: {meta["split"]["test_n"]:,} records. Thresholds were selected using validation data only; AED costs are illustrative assumptions.')
    c1, c2 = st.columns([1.2, 1])
    with c1:
        top = importance.head(15)[["feature", "importance_mean"]]
        st.markdown("**Permutation importance — decrease in average precision**")
        chart = (alt.Chart(top).mark_bar(color="#22d3ee").encode(
            x=alt.X("importance_mean:Q", title="Decrease in average precision"),
            y=alt.Y("feature:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=260, labelFontSize=11)),
            tooltip=[alt.Tooltip("feature:N", title="Feature"),
                     alt.Tooltip("importance_mean:Q", title="Importance", format=".5f")],
        ).properties(height=420))
        st.altair_chart(chart, width="stretch")
    with c2:
        st.markdown("**Model configuration**")
        st.json({"selected_model": meta["selected_model"], "parameters": meta["best_parameters"],
                 "split": meta["split"], "excluded_columns": meta["excluded_columns"]})
    st.warning("The high-recall threshold is aggressive and offers only a small illustrative cost improvement over a flag-all baseline. The balanced threshold may be more practical for a pilot.")
    st.caption("Data source: user-provided synthetic humanoid-robot task data. No external licence was supplied. This app is not validated for physical robot control.")
