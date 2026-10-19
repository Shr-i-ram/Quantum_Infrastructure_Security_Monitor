"""Generate Comprehensive Publication Figures for QISM Benchmark.

Reads the saved benchmark tables:
- results/qism_model_benchmark/tables/aggregate_metrics.csv
- results/qism_model_benchmark/tables/compute_cost_table.csv

Generates high-resolution publication-quality PNG and PDF figures:
1. 01_overall_auroc_comparison (PNG & PDF)
2. 02_recall_at_1pct_fpr_comparison (PNG & PDF)
3. 03_performance_vs_compute_frontier (PNG & PDF)
4. 04_qubit_count_sweep (PNG & PDF)
5. 05_encoding_comparison (PNG & PDF)
6. 06_temporal_ablation_comparison (PNG & PDF)
7. 07_generalization_robustness (PNG & PDF)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "results" / "qism_model_benchmark" / "tables"
FIGS_DIR = REPO_ROOT / "results" / "qism_model_benchmark" / "figures"

(FIGS_DIR / "comparisons").mkdir(parents=True, exist_ok=True)
(FIGS_DIR / "pitqsvdd").mkdir(parents=True, exist_ok=True)
(FIGS_DIR / "generalization").mkdir(parents=True, exist_ok=True)

if "seaborn-v0_8-whitegrid" in plt.style.available:
    plt.style.use("seaborn-v0_8-whitegrid")
else:
    plt.style.use("default")

df_agg = pd.read_csv(TABLES_DIR / "aggregate_metrics.csv")
df_cost = pd.read_csv(TABLES_DIR / "compute_cost_table.csv")

print(f"Loaded {len(df_agg)} models from aggregate_metrics.csv")
print(f"Loaded {len(df_cost)} cost records from compute_cost_table.csv")

# -------------------------------------------------------------
# 1. Overall AUROC Comparison Bar Chart
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
top_models = df_agg.sort_values(by="auroc_in_dist", ascending=True)
colors = [
    "#2b5c8f" if "classical" in fam else ("#8e44ad" if "quantum" in fam else "#d35400")
    for fam in top_models["family"]
]
bars = ax.barh(top_models["model_id"], top_models["auroc_in_dist"], color=colors, edgecolor="black", alpha=0.85)
ax.set_xlim(0.4, 1.02)
ax.axvline(0.70, color="gray", linestyle="--", alpha=0.6, label="70% Baseline")
ax.axvline(0.75, color="green", linestyle=":", alpha=0.8, label="75% Excellence Line")
ax.set_title("QISM Model Benchmark: Overall In-Distribution Anomaly Detection AUROC", fontsize=14, weight="bold", pad=12)
ax.set_xlabel("Area Under ROC Curve (AUROC)", fontsize=12)
for bar in bars:
    val = bar.get_width()
    if pd.notna(val):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8.5, weight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(FIGS_DIR / "comparisons" / "01_overall_auroc_comparison.png")
fig.savefig(FIGS_DIR / "comparisons" / "01_overall_auroc_comparison.pdf")
plt.close(fig)
print("Saved 01_overall_auroc_comparison (PNG & PDF)")

# -------------------------------------------------------------
# 2. Operational Recall at 1% False Positive Rate
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
top_rec = df_agg.sort_values(by="recall_at_1pct_fpr_in_dist", ascending=True)
colors = [
    "#2b5c8f" if "classical" in fam else ("#8e44ad" if "quantum" in fam else "#d35400")
    for fam in top_rec["family"]
]
bars = ax.barh(top_rec["model_id"], top_rec["recall_at_1pct_fpr_in_dist"], color=colors, edgecolor="black", alpha=0.85)
ax.set_xlim(0.0, 1.05)
ax.set_title("Operational Detection Performance: Recall at Fixed 1% False Positive Rate", fontsize=14, weight="bold", pad=12)
ax.set_xlabel("Recall @ 1% FPR (Normal Calibration)", fontsize=12)
for bar in bars:
    val = bar.get_width()
    if pd.notna(val):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val*100:.1f}%", va="center", fontsize=8.5, weight="bold")
plt.tight_layout()
fig.savefig(FIGS_DIR / "comparisons" / "02_recall_at_1pct_fpr_comparison.png")
fig.savefig(FIGS_DIR / "comparisons" / "02_recall_at_1pct_fpr_comparison.pdf")
plt.close(fig)
print("Saved 02_recall_at_1pct_fpr_comparison (PNG & PDF)")

# -------------------------------------------------------------
# 3. Performance vs Compute Cost Frontier (AUROC vs Inference Latency)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
merged_cost = pd.merge(df_agg, df_cost, on="model_id", how="inner").drop_duplicates(subset=["model_id"])

family_colors = {
    "classical_unsupervised": "#1f77b4",
    "classical_statistical": "#aec7e8",
    "classical_neural": "#2ca02c",
    "classical_temporal": "#17becf",
    "classical_supervised": "#7f7f7f",
    "quantum_supervised": "#9467bd",
    "quantum_kernel": "#8c564b",
    "quantum_unsupervised": "#e377c2",
    "quantum_one_class": "#bcbd22",
    "flagship_pitqsvdd": "#d35400",
}

for fam, group in merged_cost.groupby("family_x"):
    color = family_colors.get(fam, "#333333")
    ax.scatter(group["inference_time_per_window_s"] * 1e3, group["auroc_in_dist"], label=fam, s=120, color=color, alpha=0.85, edgecolors="black")
    for _, r in group.iterrows():
        short_name = r["model_id"].replace("PITQSVDD_", "").replace("Classical", "")
        ax.annotate(short_name, (r["inference_time_per_window_s"] * 1e3, r["auroc_in_dist"]), fontsize=7.5, xytext=(4, 2), textcoords="offset points")

ax.set_xscale("log")
ax.set_xlabel("Inference Latency per Window (ms) [Log Scale]", fontsize=12)
ax.set_ylabel("In-Distribution AUROC", fontsize=12)
ax.set_title("Pareto Frontier: Detection Performance vs. Inference Latency", fontsize=14, weight="bold", pad=12)
ax.legend(title="Model Family", loc="lower right", fontsize=8.5)
plt.tight_layout()
fig.savefig(FIGS_DIR / "comparisons" / "03_performance_vs_compute_frontier.png")
fig.savefig(FIGS_DIR / "comparisons" / "03_performance_vs_compute_frontier.pdf")
plt.close(fig)
print("Saved 03_performance_vs_compute_frontier (PNG & PDF)")

# -------------------------------------------------------------
# 4. PI-TQSVDD Qubit Count Sweep (2, 4, 6, 8 qubits)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
qubit_rows = df_agg[df_agg["model_id"].isin(["PITQSVDD_2Q_Angle", "PITQSVDD_4Q_Angle_Ref", "PITQSVDD_6Q_Angle", "PITQSVDD_8Q_Angle"])].copy()
qubit_map = {"PITQSVDD_2Q_Angle": 2, "PITQSVDD_4Q_Angle_Ref": 4, "PITQSVDD_6Q_Angle": 6, "PITQSVDD_8Q_Angle": 8}
qubit_rows["qubits"] = qubit_rows["model_id"].map(qubit_map)
qubit_rows = qubit_rows.sort_values(by="qubits")

ax1.plot(qubit_rows["qubits"], qubit_rows["auroc_in_dist"], marker="o", linewidth=2.5, color="#d35400", markersize=8, label="In-Distribution")
ax1.plot(qubit_rows["qubits"], qubit_rows["auroc_generalization"], marker="s", linewidth=2.0, color="#2980b9", linestyle="--", markersize=7, label="Generalization (OOD)")
ax1.plot(qubit_rows["qubits"], qubit_rows["auroc_stress"], marker="^", linewidth=2.0, color="#c0392b", linestyle=":", markersize=7, label="Stress Regimes")
ax1.set_xlabel("Quantum Register Size (Qubits)", fontsize=12)
ax1.set_ylabel("AUROC", fontsize=12)
ax1.set_title("PI-TQSVDD Qubit Count vs. Detection AUROC", fontsize=13, weight="bold")
ax1.set_xticks([2, 4, 6, 8])
ax1.set_ylim(0.50, 0.85)
ax1.legend()

# Training Time vs Qubits
ax2.plot(qubit_rows["qubits"], qubit_rows["train_time_s"], marker="o", linewidth=2.5, color="#8e44ad", markersize=8)
ax2.set_xlabel("Quantum Register Size (Qubits)", fontsize=12)
ax2.set_ylabel("Training Wall-Clock Time (s)", fontsize=12)
ax2.set_title("PI-TQSVDD Simulation Time vs. Qubit Count", fontsize=13, weight="bold")
ax2.set_xticks([2, 4, 6, 8])
plt.tight_layout()
fig.savefig(FIGS_DIR / "pitqsvdd" / "04_qubit_count_sweep.png")
fig.savefig(FIGS_DIR / "pitqsvdd" / "04_qubit_count_sweep.pdf")
plt.close(fig)
print("Saved 04_qubit_count_sweep (PNG & PDF)")

# -------------------------------------------------------------
# 5. PI-TQSVDD Encodings Comparison
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
enc_models = [
    "PITQSVDD_4Q_Angle_Ref",
    "PITQSVDD_4Q_DataReuploading",
    "PITQSVDD_4Q_Amplitude",
    "PITQSVDD_4Q_IQPZZ",
    "PITQSVDD_4Q_FeatureGroup",
]
enc_rows = df_agg[df_agg["model_id"].isin(enc_models)].sort_values(by="auroc_in_dist", ascending=False)
labels = [m.replace("PITQSVDD_4Q_", "").replace("_Ref", " (Ref)") for m in enc_rows["model_id"]]
bars = ax.bar(labels, enc_rows["auroc_in_dist"], color="#e67e22", edgecolor="black", alpha=0.85, width=0.6)
ax.set_ylim(0.4, 0.85)
ax.set_ylabel("In-Distribution AUROC", fontsize=12)
ax.set_title("PI-TQSVDD Data Encoding Comparison (4 Qubits)", fontsize=14, weight="bold", pad=12)
plt.xticks(rotation=25, ha="right", fontsize=10)
for bar in bars:
    val = bar.get_height()
    if pd.notna(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", fontsize=9, weight="bold")
plt.tight_layout()
fig.savefig(FIGS_DIR / "pitqsvdd" / "05_encoding_comparison.png")
fig.savefig(FIGS_DIR / "pitqsvdd" / "05_encoding_comparison.pdf")
plt.close(fig)
print("Saved 05_encoding_comparison (PNG & PDF)")

# -------------------------------------------------------------
# 6. Objective Ablation: Temporal vs Static SVDD vs Context-Aware
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
abl_models = [
    "PITQSVDD_4Q_Angle_Ref",
    "PITQSVDD_4Q_NoTemporal",
    "StaticQuantumSVDD_4Q",
    "ClassicalDeepSVDD",
]
abl_rows = df_agg[df_agg["model_id"].isin(abl_models)].sort_values(by="auroc_in_dist", ascending=False)
labels = [m.replace("PITQSVDD_4Q_", "").replace("_Ref", " (Full PI-TQSVDD)") for m in abl_rows["model_id"]]
bars = ax.bar(labels, abl_rows["auroc_in_dist"], color="#16a085", edgecolor="black", alpha=0.85, width=0.55)
ax.set_ylim(0.50, 0.80)
ax.set_ylabel("In-Distribution AUROC", fontsize=12)
ax.set_title("PI-TQSVDD Architecture & Temporal Loss Ablation", fontsize=14, weight="bold", pad=12)
plt.xticks(rotation=20, ha="right", fontsize=10)
for bar in bars:
    val = bar.get_height()
    if pd.notna(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", fontsize=9, weight="bold")
plt.tight_layout()
fig.savefig(FIGS_DIR / "pitqsvdd" / "06_temporal_ablation_comparison.png")
fig.savefig(FIGS_DIR / "pitqsvdd" / "06_temporal_ablation_comparison.pdf")
plt.close(fig)
print("Saved 06_temporal_ablation_comparison (PNG & PDF)")

# -------------------------------------------------------------
# 7. Generalization Resilience (In-Dist vs Generalization vs Stress)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
key_models = [
    "PITQSVDD_4Q_Angle_Ref",
    "PITQSVDD_8Q_Angle",
    "PITQSVDD_4Q_Amplitude",
    "IsolationForest_Full",
    "ClassicalDeepSVDD",
    "VQC_4Q",
    "ClassicalAutoencoder",
]
gen_df = df_agg[df_agg["model_id"].isin(key_models)].copy()
x = np.arange(len(gen_df))
width = 0.25

ax.bar(x - width, gen_df["auroc_in_dist"], width, label="In-Distribution", color="#2980b9", edgecolor="black")
ax.bar(x, gen_df["auroc_generalization"], width, label="Generalization (OOD Distance/Eta)", color="#27ae60", edgecolor="black")
ax.bar(x + width, gen_df["auroc_stress"], width, label="Stress (High Loss/Noise)", color="#e74c3c", edgecolor="black")

ax.set_ylabel("AUROC", fontsize=12)
ax.set_title("Generalization Robustness Across Operating Regimes", fontsize=14, weight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels([m.replace("PITQSVDD_", "PI-").replace("_Ref", "") for m in gen_df["model_id"]], rotation=20, ha="right", fontsize=10)
ax.set_ylim(0.4, 0.90)
ax.legend()
plt.tight_layout()
fig.savefig(FIGS_DIR / "generalization" / "07_generalization_robustness.png")
fig.savefig(FIGS_DIR / "generalization" / "07_generalization_robustness.pdf")
plt.close(fig)
print("Saved 07_generalization_robustness (PNG & PDF)")

print("\nSUCCESS: All 7 publication figures generated in PNG and PDF formats!")
