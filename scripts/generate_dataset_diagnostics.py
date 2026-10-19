"""Generate visual diagnostics and distribution figures for QISM research dataset."""

from __future__ import annotations

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qism.datasets.sharding import list_sharded_telemetry_files


def generate_diagnostics(dataset_dir: pathlib.Path) -> None:
    diag_dir = dataset_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating dataset diagnostics for {dataset_dir} ...")

    # Load telemetry across partitions
    dfs = []
    partitions = ["calibration", "train", "validation", "test_in_distribution", "test_generalization", "test_stress"]
    partition_counts = {}

    for part in partitions:
        part_dir = dataset_dir / part
        t_files = list_sharded_telemetry_files(part_dir)
        part_df_list = [pd.read_parquet(f) for f in t_files]
        if part_df_list:
            pdf = pd.concat(part_df_list, ignore_index=True)
            pdf["split"] = part
            dfs.append(pdf)
            n_atk = int(pdf["ground_truth_attack_active"].sum()) if "ground_truth_attack_active" in pdf.columns else 0
            partition_counts[part] = {"normal": len(pdf) - n_atk, "attack": n_atk}

    if not dfs:
        print("No telemetry files found!")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    # 1. Partition Window Counts (Normal vs Attack)
    plt.figure(figsize=(10, 5))
    splits_list = list(partition_counts.keys())
    normals = [partition_counts[s]["normal"] for s in splits_list]
    attacks = [partition_counts[s]["attack"] for s in splits_list]

    x = np.arange(len(splits_list))
    width = 0.4

    plt.bar(x - width/2, normals, width, label="Normal Windows", color="#2b5c8f")
    plt.bar(x + width/2, attacks, width, label="Attack Windows", color="#d95f02")
    plt.xticks(x, splits_list, rotation=15, ha="right", fontsize=9)
    plt.ylabel("Window Count")
    plt.title("QISM Research Dataset — Window Distribution Across Partitions", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(diag_dir / "partition_window_counts.png", dpi=200)
    plt.close()

    # 2. Attack Family Distribution
    if "ground_truth_attack_type" in df_all.columns:
        atk_series = df_all[df_all["ground_truth_attack_active"] == 1]["ground_truth_attack_type"].value_counts()
        plt.figure(figsize=(9, 4.5))
        atk_series.plot(kind="bar", color="#7570b3")
        plt.title("Attack Window Distribution by Physical Family", fontsize=12, fontweight="bold")
        plt.xlabel("Attack Mechanism Family")
        plt.ylabel("Attack Windows")
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(diag_dir / "attack_family_distribution.png", dpi=200)
        plt.close()

    # 3. Normal vs Attack QBER and Count Rate Distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    norm_mask = df_all["ground_truth_attack_active"] == 0
    atk_mask = df_all["ground_truth_attack_active"] == 1

    axes[0].hist(df_all.loc[norm_mask, "qber"].dropna(), bins=40, alpha=0.7, label="Normal", color="#2b5c8f", density=True)
    axes[0].hist(df_all.loc[atk_mask, "qber"].dropna(), bins=40, alpha=0.7, label="Attack", color="#d95f02", density=True)
    axes[0].set_xlabel("Sifted QBER")
    axes[0].set_ylabel("Density")
    axes[0].set_title("QBER Distribution: Normal vs. Attack")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    if "detector_imbalance" in df_all.columns:
        axes[1].hist(df_all.loc[norm_mask, "detector_imbalance"].dropna(), bins=40, alpha=0.7, label="Normal", color="#2b5c8f", density=True)
        axes[1].hist(df_all.loc[atk_mask, "detector_imbalance"].dropna(), bins=40, alpha=0.7, label="Attack", color="#d95f02", density=True)
        axes[1].set_xlabel("Detector Imbalance ((C0 - C1)/(C0 + C1))")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Detector Imbalance: Normal vs. Attack")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(diag_dir / "normal_vs_attack_distributions.png", dpi=200)
    plt.close()

    print(f"Generated diagnostics saved to {diag_dir}")


if __name__ == "__main__":
    import sys
    dataset_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/qism_research_dataset")
    generate_diagnostics(dataset_path)
