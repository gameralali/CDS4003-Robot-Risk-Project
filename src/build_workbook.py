"""Build a formula-driven Excel results workbook from saved project outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.formatting.rule import DataBarRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from src.config import ROOT

OUTPUT = ROOT / "outputs" / "Project_Results.xlsx"
TABLES = ROOT / "outputs" / "tables"
NAVY = "0B1F33"
TEAL = "008C95"
PALE = "E8F3F4"
WHITE = "FFFFFF"
GREY = "5B6573"
BLUE = "0000FF"
GREEN = "008000"
BLACK = "000000"
THIN_GREY = Side(style="thin", color="D6DCE4")


def title(ws, text: str, subtitle: str, end_col: int = 10) -> None:
    ws.sheet_view.showGridLines = False
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=end_col)
    ws["A1"] = text
    ws["A1"].font = Font(name="Calibri", size=18, bold=True, color=WHITE)
    ws["A1"].fill = PatternFill("solid", fgColor=NAVY)
    ws["A1"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 32
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=end_col)
    ws["A2"] = subtitle
    ws["A2"].font = Font(name="Calibri", size=10, color=GREY)
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[2].height = 32


def header(ws, row: int, start: int, end: int) -> None:
    for col in range(start, end + 1):
        cell = ws.cell(row, col)
        cell.font = Font(name="Calibri", size=10, bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=TEAL)
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.border = Border(bottom=THIN_GREY)
    ws.row_dimensions[row].height = 30


def style_data(ws, start_row: int, end_row: int, start_col: int, end_col: int) -> None:
    for row in range(start_row, end_row + 1):
        for col in range(start_col, end_col + 1):
            cell = ws.cell(row, col)
            cell.font = Font(name="Calibri", size=10, color=BLUE)
            cell.border = Border(bottom=THIN_GREY)
            cell.alignment = Alignment(vertical="center")


def formula_style(ws, columns: list[str], start_row: int, end_row: int, cross_sheet: bool = False) -> None:
    for column in columns:
        for row in range(start_row, end_row + 1):
            ws[f"{column}{row}"].font = Font(
                name="Calibri", size=10, color=GREEN if cross_sheet else BLACK
            )


def widths(ws, mapping: dict[str, float]) -> None:
    for column, width in mapping.items():
        ws.column_dimensions[column].width = width


def percent(ws, columns: list[str], rows: range) -> None:
    for column in columns:
        for row in rows:
            ws[f"{column}{row}"].number_format = "0.0%"


def money(ws, columns: list[str], rows: range) -> None:
    for column in columns:
        for row in rows:
            ws[f"{column}{row}"].number_format = '"AED" #,##0'


def build() -> Path:
    meta = json.loads((ROOT / "models" / "model_metadata.json").read_text())
    run = json.loads((ROOT / "outputs" / "run_summary.json").read_text())
    validation = pd.read_csv(TABLES / "model_comparison_validation.csv")
    test = pd.read_csv(TABLES / "final_test_metrics.csv")
    thresholds = pd.read_csv(TABLES / "selected_model_validation_threshold_curve.csv")
    importance = pd.read_csv(TABLES / "permutation_importance.csv")
    task_type = pd.read_csv(TABLES / "failure_rate_by_task_type.csv")
    battery = pd.read_csv(TABLES / "failure_rate_by_battery_band.csv")

    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.title = "Humanoid Robot Task Failure-Risk Project Results"
    wb.properties.subject = "CDS4003 CRISP-DM technical results"
    wb.properties.creator = "CDS4003 project team"
    wb.calculation.fullCalcOnLoad = True
    wb.calculation.forceFullCalc = True
    wb.calculation.calcMode = "auto"

    # Assumptions: hardcoded scenario inputs and run configuration.
    ws = wb.create_sheet("Assumptions")
    title(ws, "Assumptions and experiment contract", "Blue cells are hardcoded inputs. AED values are illustrative scenario assumptions, not observed financial data.", 5)
    rows = [
        ("Selected model", meta["selected_model"], "model_metadata.json", "Chosen by minimum validation cost."),
        ("False-negative cost (AED)", meta["false_negative_cost_aed"], "model_metadata.json", "Illustrative missed-failure assumption."),
        ("False-positive cost (AED)", meta["false_positive_cost_aed"], "model_metadata.json", "Illustrative unnecessary-review assumption."),
        ("Random seed", 4003, "src/config.py", "Reproducibility setting."),
        ("Tuning sample rows", 30000, "src/config.py", "Stratified training-only sample."),
        ("Permutation sample rows", 2500, "src/config.py", "Test-sample size for pipeline-level importance."),
        ("Overall failure rate", run["eda"]["failure_rate"], "run_summary.json", "Valid labeled rows only."),
        ("Untouched test rows", meta["split"]["test_n"], "model_metadata.json", "Not used for model or threshold selection."),
        ("Balanced threshold", meta["balanced_f1_threshold"], "model_metadata.json", "Maximum validation F1."),
        ("High-recall threshold", meta["decision_threshold"], "model_metadata.json", "Minimum validation illustrative cost."),
    ]
    ws.append([])
    ws.append(["Parameter", "Value", "Source", "Interpretation"])
    header(ws, 4, 1, 4)
    for item in rows:
        ws.append(item)
    style_data(ws, 5, 14, 1, 4)
    ws["B6"].number_format = '"AED" #,##0'
    ws["B7"].number_format = '"AED" #,##0'
    ws["B11"].number_format = "0.00%"
    for row in (13, 14):
        ws[f"B{row}"].number_format = "0.00"
    widths(ws, {"A": 30, "B": 24, "C": 28, "D": 58})
    ws.freeze_panes = "A5"

    # Data quality.
    ws = wb.create_sheet("Data Quality")
    title(ws, "Data-quality and cleaning summary", "Structural cleaning is deterministic; imputation is fitted inside training pipelines.", 5)
    ws.append([])
    ws.append(["Measure", "Value", "Unit", "Source", "Decision / note"])
    header(ws, 4, 1, 5)
    quality_rows = [
        ("Raw rows", run["cleaning"]["rows_raw"], "rows", "run_summary.json", "Includes fully blank records."),
        ("Blank rows removed", run["cleaning"]["blank_rows_removed"], "rows", "run_summary.json", "Removed for training; preserved for batch scoring."),
        ("Raw columns", run["cleaning"]["columns"], "columns", "run_summary.json", "Meets the 20-column requirement."),
        ("Raw missing cells", run["cleaning"]["missing_cells_raw"], "cells", "run_summary.json", "Before structural cleaning."),
        ("Domain-invalid values set missing", run["cleaning"]["domain_invalid_values_set_missing"], "values", "run_summary.json", "Inclusive physical/domain ranges."),
        ("Rows after structural cleaning", run["cleaning"]["rows_after_structural_cleaning"], "rows", "run_summary.json", "Blank records removed."),
        ("Missing cells after domain validation", run["cleaning"]["missing_cells_after_domain_validation"], "cells", "run_summary.json", "Before pipeline imputation."),
        ("Valid labeled modelling rows", run["eda"]["labeled_rows"], "rows", "run_summary.json", "Unknown targets excluded."),
        ("Overall failure rate", run["eda"]["failure_rate"], "proportion", "run_summary.json", "Among valid labeled rows."),
    ]
    for item in quality_rows:
        ws.append(item)
    ws.append(["Blank-row share of raw data", "=B6/B5", "proportion", "Formula", "Derived in workbook."])
    ws.append(["Valid-label share of structurally cleaned data", "=B12/B10", "proportion", "Formula", "Derived in workbook."])
    style_data(ws, 5, 15, 1, 5)
    formula_style(ws, ["B"], 14, 15)
    for row in (13, 14, 15):
        ws[f"B{row}"].number_format = "0.00%"
    widths(ws, {"A": 44, "B": 20, "C": 16, "D": 24, "E": 52})
    ws.freeze_panes = "A5"

    # Validation model comparison with formula-derived classification metrics and costs.
    ws = wb.create_sheet("Validation Models")
    title(ws, "Validation model comparison", "Three-fold randomized search on training-only samples; thresholds minimise illustrative validation cost. Metrics below are calculated from confusion counts.", 17)
    headers = ["Model", "Tuning seconds", "Threshold", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "PR AUC", "FP cost", "FN cost", "Total cost", "Saving vs dummy"]
    ws.append([])
    ws.append(headers)
    header(ws, 4, 1, len(headers))
    model_labels = {"random_forest": "Random forest", "logistic_regression": "Logistic regression", "dummy_prior": "Dummy prior"}
    for _, record in validation.iterrows():
        row = ws.max_row + 1
        ws.append([
            model_labels[record["model"]], float(record["tuning_seconds"]), float(record["validation_threshold"]),
            int(record["validation_tn"]), int(record["validation_fp"]), int(record["validation_fn"]), int(record["validation_tp"]),
            f"=IFERROR((D{row}+G{row})/SUM(D{row}:G{row}),0)",
            f"=IFERROR(G{row}/(E{row}+G{row}),0)",
            f"=IFERROR(G{row}/(F{row}+G{row}),0)",
            f"=IFERROR(2*I{row}*J{row}/(I{row}+J{row}),0)",
            float(record["validation_roc_auc"]), float(record["validation_pr_auc"]),
            f"=E{row}*Assumptions!$B$7", f"=F{row}*Assumptions!$B$6",
            f"=N{row}+O{row}", f"=$P$7-P{row}",
        ])
    style_data(ws, 5, 7, 1, 17)
    formula_style(ws, list("HIJK"), 5, 7)
    formula_style(ws, ["N", "O"], 5, 7, cross_sheet=True)
    formula_style(ws, ["P", "Q"], 5, 7)
    percent(ws, list("HIJKLM"), range(5, 8))
    money(ws, ["N", "O", "P", "Q"], range(5, 8))
    for row in range(5, 8):
        ws[f"C{row}"].number_format = "0.00"
        ws[f"B{row}"].number_format = "0.00"
    ws.auto_filter.ref = "A4:Q7"
    ws.freeze_panes = "D5"
    widths(ws, {"A": 24, "B": 16, "C": 12, **{c: 12 for c in "DEFGHIJKLM"}, "N": 17, "O": 17, "P": 17, "Q": 18})
    chart = BarChart()
    chart.type = "col"
    chart.title = "Validation ROC AUC and PR AUC"
    chart.y_axis.title = "AUC"
    chart.x_axis.title = "Model"
    chart.y_axis.scaling.min = 0
    chart.y_axis.scaling.max = 1
    chart.add_data(Reference(ws, min_col=12, max_col=13, min_row=4, max_row=7), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=5, max_row=7))
    chart.height = 7
    chart.width = 14
    ws.add_chart(chart, "A10")

    # Untouched test metrics, including a formula-built flag-all baseline.
    ws = wb.create_sheet("Test Metrics")
    title(ws, "Untouched test-set operating points", "Thresholds were selected using validation data only. The flag-all comparator exposes the small cost advantage of the aggressive policy.", 20)
    headers = ["Operating point", "Model", "Threshold", "TN", "FP", "FN", "TP", "Total N", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "PR AUC", "FP cost", "FN cost", "Total cost", "Saving vs flag-all", "FP rate", "FN rate"]
    ws.append([])
    ws.append(headers)
    header(ws, 4, 1, len(headers))
    labels = [("Standard 0.50", test.iloc[0]), ("Balanced F1", test.iloc[1]), ("High recall / cost", test.iloc[2])]
    for label, record in labels:
        row = ws.max_row + 1
        ws.append([
            label, "Random forest", float(record["threshold"]), int(record["tn"]), int(record["fp"]), int(record["fn"]), int(record["tp"]),
            f"=SUM(D{row}:G{row})", f"=IFERROR((D{row}+G{row})/H{row},0)",
            f"=IFERROR(G{row}/(E{row}+G{row}),0)", f"=IFERROR(G{row}/(F{row}+G{row}),0)",
            f"=IFERROR(2*J{row}*K{row}/(J{row}+K{row}),0)", float(record["roc_auc"]), float(record["pr_auc"]),
            f"=E{row}*Assumptions!$B$7", f"=F{row}*Assumptions!$B$6", f"=O{row}+P{row}",
            "=$Q$8-Q8" if row == 8 else f"=$Q$8-Q{row}",
            f"=IFERROR(E{row}/(D{row}+E{row}),0)", f"=IFERROR(F{row}/(F{row}+G{row}),0)",
        ])
    # Flag every task for review: TN=FN=0; all actual negatives are FP and all actual positives are TP.
    ws.append([
        "Flag all tasks", "Baseline", 0.0, 0, "=D7+E7", 0, "=F7+G7", "=SUM(D8:G8)",
        "=IFERROR((D8+G8)/H8,0)", "=IFERROR(G8/(E8+G8),0)", "=IFERROR(G8/(F8+G8),0)",
        "=IFERROR(2*J8*K8/(J8+K8),0)", 0.5, "=G8/H8", "=E8*Assumptions!$B$7",
        "=F8*Assumptions!$B$6", "=O8+P8", "=Q8-Q8", "=IFERROR(E8/(D8+E8),0)", "=IFERROR(F8/(F8+G8),0)",
    ])
    style_data(ws, 5, 8, 1, 20)
    formula_style(ws, ["E", "G"], 8, 8)
    formula_style(ws, list("HIJKL"), 5, 8)
    formula_style(ws, ["N"], 8, 8)
    formula_style(ws, ["O", "P"], 5, 8, cross_sheet=True)
    formula_style(ws, ["Q", "R", "S", "T"], 5, 8)
    percent(ws, list("IJKLMNST"), range(5, 9))
    money(ws, ["O", "P", "Q", "R"], range(5, 9))
    for row in range(5, 9):
        ws[f"C{row}"].number_format = "0.00"
    ws.auto_filter.ref = "A4:T8"
    ws.freeze_panes = "D5"
    widths(ws, {"A": 24, "B": 18, "C": 12, **{c: 11 for c in "DEFGHIJKLMN"}, "O": 17, "P": 17, "Q": 17, "R": 19, "S": 12, "T": 12})
    ws.conditional_formatting.add("R5:R7", DataBarRule(start_type="num", start_value=0, end_type="max", color=TEAL, showValue=True))
    chart = BarChart()
    chart.type = "col"
    chart.title = "Test recall and false-positive rate"
    chart.y_axis.title = "Rate"
    chart.y_axis.scaling.min = 0
    chart.y_axis.scaling.max = 1.1
    chart.add_data(Reference(ws, min_col=11, max_col=11, min_row=4, max_row=8), titles_from_data=True)
    chart.add_data(Reference(ws, min_col=19, max_col=19, min_row=4, max_row=8), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=5, max_row=8))
    chart.height = 7
    chart.width = 15
    ws.add_chart(chart, "A11")
    cost_chart = BarChart()
    cost_chart.type = "col"
    cost_chart.title = "Illustrative total error cost"
    cost_chart.y_axis.title = "AED"
    cost_chart.add_data(Reference(ws, min_col=17, min_row=4, max_row=8), titles_from_data=True)
    cost_chart.set_categories(Reference(ws, min_col=1, min_row=5, max_row=8))
    cost_chart.height = 7
    cost_chart.width = 15
    ws.add_chart(cost_chart, "J11")

    # Full validation threshold curve, formula-derived from confusion counts.
    ws = wb.create_sheet("Threshold Curve")
    title(ws, "Selected random forest — validation threshold curve", "Each row recalculates metrics and illustrative costs from validation confusion counts. Test data plays no role in threshold selection.", 12)
    headers = ["Threshold", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "F1", "FP cost", "FN cost", "Total cost"]
    ws.append([])
    ws.append(headers)
    header(ws, 4, 1, 12)
    start = 5
    for _, record in thresholds.iterrows():
        row = ws.max_row + 1
        ws.append([
            float(record["threshold"]), int(record["tn"]), int(record["fp"]), int(record["fn"]), int(record["tp"]),
            f"=IFERROR((B{row}+E{row})/SUM(B{row}:E{row}),0)", f"=IFERROR(E{row}/(C{row}+E{row}),0)",
            f"=IFERROR(E{row}/(D{row}+E{row}),0)", f"=IFERROR(2*G{row}*H{row}/(G{row}+H{row}),0)",
            f"=C{row}*Assumptions!$B$7", f"=D{row}*Assumptions!$B$6", f"=J{row}+K{row}",
        ])
    end = ws.max_row
    style_data(ws, start, end, 1, 12)
    formula_style(ws, list("FGHI"), start, end)
    formula_style(ws, ["J", "K"], start, end, cross_sheet=True)
    formula_style(ws, ["L"], start, end)
    percent(ws, list("FGHI"), range(start, end + 1))
    money(ws, ["J", "K", "L"], range(start, end + 1))
    for row in range(start, end + 1):
        ws[f"A{row}"].number_format = "0.00"
    ws.auto_filter.ref = f"A4:L{end}"
    ws.freeze_panes = "B5"
    widths(ws, {"A": 12, **{c: 11 for c in "BCDEFGHI"}, "J": 16, "K": 16, "L": 17})
    line = LineChart()
    line.title = "Validation recall and F1 by threshold"
    line.y_axis.title = "Score"
    line.x_axis.title = "Threshold"
    line.y_axis.scaling.min = 0
    line.y_axis.scaling.max = 1
    line.add_data(Reference(ws, min_col=8, max_col=9, min_row=4, max_row=end), titles_from_data=True)
    line.set_categories(Reference(ws, min_col=1, min_row=start, max_row=end))
    line.height = 8
    line.width = 16
    ws.add_chart(line, "N4")
    cost_line = LineChart()
    cost_line.title = "Validation illustrative cost by threshold"
    cost_line.y_axis.title = "AED"
    cost_line.x_axis.title = "Threshold"
    cost_line.add_data(Reference(ws, min_col=12, min_row=4, max_row=end), titles_from_data=True)
    cost_line.set_categories(Reference(ws, min_col=1, min_row=start, max_row=end))
    cost_line.height = 8
    cost_line.width = 16
    ws.add_chart(cost_line, "N21")

    # Feature importance.
    ws = wb.create_sheet("Feature Importance")
    title(ws, "Pipeline-level permutation importance", "Mean decrease in test average precision on a 2,500-row sample; three repeats. Association is not causation.", 5)
    ws.append([])
    ws.append(["Rank", "Feature", "Importance mean", "Importance std", "Source"])
    header(ws, 4, 1, 5)
    for rank, (_, record) in enumerate(importance.iterrows(), start=1):
        ws.append([rank, record["feature"], float(record["importance_mean"]), float(record["importance_std"]), "permutation_importance.csv"])
    end = ws.max_row
    style_data(ws, 5, end, 1, 5)
    for row in range(5, end + 1):
        ws[f"C{row}"].number_format = "0.00000"
        ws[f"D{row}"].number_format = "0.00000"
    ws.auto_filter.ref = f"A4:E{end}"
    ws.freeze_panes = "A5"
    widths(ws, {"A": 10, "B": 38, "C": 20, "D": 20, "E": 32})
    ws.conditional_formatting.add("C5:C14", DataBarRule(start_type="num", start_value=0, end_type="max", color=TEAL, showValue=True))
    chart = BarChart()
    chart.type = "bar"
    chart.title = "Top 10 features by mean importance"
    chart.x_axis.title = "Decrease in average precision"
    chart.add_data(Reference(ws, min_col=3, min_row=4, max_row=14), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=2, min_row=5, max_row=14))
    chart.height = 9
    chart.width = 17
    ws.add_chart(chart, "G4")

    # EDA group summaries with workbook-derived failure rates where counts allow.
    ws = wb.create_sheet("EDA Groups")
    title(ws, "Exploratory group failure rates", "Observed synthetic associations only. Missing task type is shown explicitly; group differences do not establish causation.", 8)
    ws["A4"] = "Task type"
    ws["A4"].font = Font(size=12, bold=True, color=NAVY)
    ws.append([])
    ws.append(["Task type", "Records", "Failures", "Failure rate", "Source"])
    header(ws, 6, 1, 5)
    for _, record in task_type.iterrows():
        row = ws.max_row + 1
        label = record["task_type"] if pd.notna(record["task_type"]) else "Missing"
        ws.append([label, int(record["records"]), int(record["failures"]), f"=IFERROR(C{row}/B{row},0)", "failure_rate_by_task_type.csv"])
    task_end = ws.max_row
    style_data(ws, 7, task_end, 1, 5)
    formula_style(ws, ["D"], 7, task_end)
    percent(ws, ["D"], range(7, task_end + 1))
    chart = BarChart()
    chart.type = "col"
    chart.title = "Failure rate by task type"
    chart.y_axis.title = "Failure rate"
    chart.y_axis.scaling.min = 0
    chart.y_axis.scaling.max = 1
    chart.add_data(Reference(ws, min_col=4, min_row=6, max_row=task_end), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=7, max_row=task_end))
    chart.height = 7
    chart.width = 14
    ws.add_chart(chart, "G4")

    battery_title = task_end + 3
    ws[f"A{battery_title}"] = "Battery band"
    ws[f"A{battery_title}"].font = Font(size=12, bold=True, color=NAVY)
    battery_header = battery_title + 2
    ws.append([])
    ws.append(["Battery band", "Records", "Failure rate", "Source"])
    header(ws, battery_header, 1, 4)
    for _, record in battery.iterrows():
        ws.append([record["battery_band"], int(record["count"]), float(record["mean"]), "failure_rate_by_battery_band.csv"])
    battery_end = ws.max_row
    style_data(ws, battery_header + 1, battery_end, 1, 4)
    percent(ws, ["C"], range(battery_header + 1, battery_end + 1))
    widths(ws, {"A": 28, "B": 16, "C": 16, "D": 36, "E": 36})
    chart2 = BarChart()
    chart2.type = "col"
    chart2.title = "Failure rate by battery band"
    chart2.y_axis.title = "Failure rate"
    chart2.y_axis.scaling.min = 0
    chart2.y_axis.scaling.max = 1
    chart2.add_data(Reference(ws, min_col=3, min_row=battery_header, max_row=battery_end), titles_from_data=True)
    chart2.set_categories(Reference(ws, min_col=1, min_row=battery_header + 1, max_row=battery_end))
    chart2.height = 7
    chart2.width = 14
    ws.add_chart(chart2, f"G{battery_title}")

    # Overview first, with cross-sheet formula pull-throughs.
    ws = wb.create_sheet("Overview", 0)
    title(ws, "Humanoid Robot Task Failure-Risk Project", "CDS4003 · CRISP-DM · synthetic-data demonstration · formula-driven technical results", 8)
    ws["A4"] = "Key result"
    ws["B4"] = "Value"
    ws["C4"] = "Interpretation / caveat"
    header(ws, 4, 1, 3)
    overview_rows = [
        ("Selected model", "=Assumptions!B5", "Minimum validation cost; RF advantage over logistic regression is negligible."),
        ("Test ROC AUC", "='Test Metrics'!M7", "Threshold-independent ranking measure."),
        ("Test PR AUC", "='Test Metrics'!N7", "Precision-recall ranking measure."),
        ("Balanced F1", "='Test Metrics'!L6", "Validation-selected F1 operating point."),
        ("Balanced recall", "='Test Metrics'!K6", "Higher recall with substantial reviewer workload."),
        ("High-recall recall", "='Test Metrics'!K7", "Aggressive policy: almost every task reviewed."),
        ("High-recall total cost", "='Test Metrics'!Q7", "Illustrative AED assumptions only."),
        ("Saving vs flag-all", "='Test Metrics'!R7", "Small advantage; do not claim a large business saving."),
        ("Valid labeled rows", "='Data Quality'!B12", "199,950 rows used for supervised modelling."),
    ]
    for item in overview_rows:
        ws.append(item)
    style_data(ws, 5, 13, 1, 3)
    formula_style(ws, ["B"], 5, 13, cross_sheet=True)
    for row in (6, 7, 8, 9, 10):
        ws[f"B{row}"].number_format = "0.0%" if row not in (6, 7) else "0.000"
    for row in (11, 12):
        ws[f"B{row}"].number_format = '"AED" #,##0'
    ws["B13"].number_format = "#,##0"
    widths(ws, {"A": 30, "B": 24, "C": 68, "D": 4, "E": 22, "F": 22, "G": 22, "H": 22})
    ws["A16"] = "Workbook index"
    ws["A16"].font = Font(size=12, bold=True, color=NAVY)
    index = [
        ("Assumptions", "Scenario costs, thresholds, and run configuration"),
        ("Data Quality", "Measured quality issues and cleaning decisions"),
        ("Validation Models", "Model comparison and validation costs"),
        ("Test Metrics", "Untouched test metrics and flag-all comparator"),
        ("Threshold Curve", "Validation threshold trade-offs"),
        ("Feature Importance", "Pipeline-level permutation importance"),
        ("EDA Groups", "Task-type and battery-band outcomes"),
        ("Sources & Notes", "File lineage, metric definitions, and limitations"),
    ]
    for row, (sheet, description) in enumerate(index, start=17):
        ws[f"A{row}"] = sheet
        ws[f"A{row}"].hyperlink = f"#'{sheet}'!A1"
        ws[f"A{row}"].font = Font(color=TEAL, underline="single", bold=True)
        ws[f"B{row}"] = description
    ws["A27"] = "Decision warning"
    ws["A27"].font = Font(size=12, bold=True, color=NAVY)
    ws.merge_cells("A28:H30")
    ws["A28"] = ("Synthetic results are not evidence of safe or effective physical robot control. "
                  "Validate with real representative data, real costs, reviewer capacity, human oversight, and a pilot before operational use.")
    ws["A28"].alignment = Alignment(wrap_text=True, vertical="top")
    ws["A28"].fill = PatternFill("solid", fgColor=PALE)

    # Sources and notes.
    ws = wb.create_sheet("Sources & Notes")
    title(ws, "Sources, metric definitions, and limitations", "All values come from this project's saved outputs; no external financial or operational data are claimed.", 5)
    ws.append([])
    ws.append(["Source name", "Source path", "Used in", "Notes"])
    header(ws, 4, 1, 4)
    sources = [
        ("Run summary", "outputs/run_summary.json", "Overview; Data Quality", "Full reproducible pipeline run."),
        ("Model metadata", "models/model_metadata.json", "Assumptions; Test Metrics", "Selected configuration, splits, thresholds, and metrics."),
        ("Validation comparison", "outputs/tables/model_comparison_validation.csv", "Validation Models", "Confusion counts are inputs; displayed metrics/costs are workbook formulas."),
        ("Final test metrics", "outputs/tables/final_test_metrics.csv", "Test Metrics", "Untouched test evaluation; flag-all baseline added by formula."),
        ("Threshold curve", "outputs/tables/selected_model_validation_threshold_curve.csv", "Threshold Curve", "Validation only; not test-based selection."),
        ("Permutation importance", "outputs/tables/permutation_importance.csv", "Feature Importance", "Average-precision scoring on original pipeline inputs."),
        ("Task-type summary", "outputs/tables/failure_rate_by_task_type.csv", "EDA Groups", "Failure rate recalculated as failures / records."),
        ("Battery-band summary", "outputs/tables/failure_rate_by_battery_band.csv", "EDA Groups", "Grouped cleaned labeled data."),
        ("Raw synthetic dataset", "data/humanoid_robot_tasks_200k_messy.csv", "Upstream analysis", "No external licence supplied; public redistribution requires confirmation."),
    ]
    for item in sources:
        ws.append(item)
    style_data(ws, 5, 13, 1, 4)
    widths(ws, {"A": 28, "B": 62, "C": 32, "D": 70})
    notes = [
        "Accuracy = (TN + TP) / total records.",
        "Precision = TP / (FP + TP); recall = TP / (FN + TP).",
        "F1 = harmonic mean of precision and recall.",
        "Total illustrative cost = FP × AED 40 + FN × AED 500.",
        "Saving vs flag-all = flag-all total cost − operating-point total cost.",
        "ROC AUC and PR AUC are threshold-independent ranking metrics imported from the saved evaluation.",
        "High recall is aggressive: 14,198 false positives and 3 false negatives on the test set.",
        "The high-recall saving versus flag-all is AED 1,780 (approximately 0.31%), not a large demonstrated business benefit.",
        "Synthetic-data approval, licence/permission, public GitHub, and public Streamlit deployment remain student responsibilities.",
    ]
    ws["A16"] = "Definitions and limitations"
    ws["A16"].font = Font(size=12, bold=True, color=NAVY)
    for row, note in enumerate(notes, start=17):
        ws[f"A{row}"] = "•"
        ws[f"B{row}"] = note
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=4)
        ws[f"B{row}"].alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row].height = 28
    ws.freeze_panes = "A5"

    # Common print/layout settings.
    for sheet in wb.worksheets:
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.page_margins.left = 0.3
        sheet.page_margins.right = 0.3
        sheet.page_margins.top = 0.5
        sheet.page_margins.bottom = 0.5
        sheet.sheet_properties.tabColor = TEAL

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT)

    # Reopen to ensure the package and chart series are readable before the recalc gate.
    checked = load_workbook(OUTPUT, data_only=False)
    for sheet in checked.worksheets:
        for chart in sheet._charts:
            assert chart.series, f"Chart without series on {sheet.title}"
    return OUTPUT


if __name__ == "__main__":
    print(build())
