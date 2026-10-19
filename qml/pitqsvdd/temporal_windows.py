"""Causal temporal window construction and sequence batching for PI-TQSVDD.

Guarantees:
- Causal ordering: Window X_t = [x_{t-L+1}, ..., x_t] uses strictly past and current observations.
- Anti-leakage: Windows are constructed strictly within individual scenarios.
  No window ever spans across scenario boundaries.
- Future target alignment: For temporal prediction loss, target is next step x_{t+1}
  strictly within the same scenario.
- Label & Metadata mapping: Decision at time t is matched to timestamp/labels at time t.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import torch


class TemporalWindowBuilder:
    """Builds strictly causal temporal sequences within scenario boundaries."""

    def __init__(self, sequence_length: int = 3, stride: int = 1):
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
        self.sequence_length = sequence_length
        self.stride = stride

    def build_scenario_windows(
        self,
        X_scenario: np.ndarray,
        y_scenario_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Build causal windows from a single scenario's features and metadata.

        Args:
            X_scenario: Array of shape (T, D) representing features at each time step.
            y_scenario_df: DataFrame of length T containing metadata and labels.

        Returns:
            Dictionary with:
            - "windows": Array of shape (N_windows, L, D)
            - "targets": Array of shape (N_windows, D) or None (next-step target if available)
            - "has_target": Boolean array of shape (N_windows,)
            - "labels": Array of shape (N_windows,) representing attack_active at time t
            - "metadata": List of metadata dicts corresponding to decision step t
        """
        T, D = X_scenario.shape
        L = self.sequence_length

        if T < L:
            # If scenario is shorter than L, pad with earliest observation (warmup)
            pad_len = L - T
            pad_x = np.repeat(X_scenario[:1], pad_len, axis=0)
            X_padded = np.concatenate([pad_x, X_scenario], axis=0)
            # Create single window
            w = X_padded[-L:][np.newaxis, ...]  # (1, L, D)
            label = int(y_scenario_df["attack_active"].iloc[-1]) if "attack_active" in y_scenario_df.columns else 0
            meta = [y_scenario_df.iloc[-1].to_dict()]
            return {
                "windows": w,
                "targets": np.zeros((1, D)),
                "has_target": np.array([False]),
                "labels": np.array([label]),
                "metadata": meta,
            }

        windows = []
        targets = []
        has_targets = []
        labels = []
        metadata_list = []

        # t is the index of the decision step (last element in window)
        for t in range(L - 1, T, self.stride):
            w = X_scenario[t - L + 1 : t + 1]
            windows.append(w)

            # Check if next step exists in the same scenario
            if t + 1 < T:
                targets.append(X_scenario[t + 1])
                has_targets.append(True)
            else:
                targets.append(np.zeros(D))
                has_targets.append(False)

            row_y = y_scenario_df.iloc[t]
            label = int(row_y.get("attack_active", 0))
            labels.append(label)
            metadata_list.append(row_y.to_dict())

        return {
            "windows": np.array(windows, dtype=np.float32),
            "targets": np.array(targets, dtype=np.float32),
            "has_target": np.array(has_targets, dtype=bool),
            "labels": np.array(labels, dtype=int),
            "metadata": metadata_list,
        }

    def build_dataset_windows(
        self,
        scenario_features: list[np.ndarray],
        scenario_metadata: list[pd.DataFrame],
    ) -> dict[str, Any]:
        """Aggregate windows across all scenarios in a partition.

        Guarantees that scenario boundaries are strictly respected.
        """
        all_windows = []
        all_targets = []
        all_has_targets = []
        all_labels = []
        all_meta = []

        for X_scen, y_scen in zip(scenario_features, scenario_metadata):
            res = self.build_scenario_windows(X_scen, y_scen)
            if len(res["windows"]) > 0:
                all_windows.append(res["windows"])
                all_targets.append(res["targets"])
                all_has_targets.append(res["has_target"])
                all_labels.append(res["labels"])
                all_meta.extend(res["metadata"])

        if not all_windows:
            D = scenario_features[0].shape[1] if scenario_features else 4
            return {
                "windows": torch.zeros((0, self.sequence_length, D), dtype=torch.float32),
                "targets": torch.zeros((0, D), dtype=torch.float32),
                "has_target": torch.zeros((0,), dtype=torch.bool),
                "labels": torch.zeros((0,), dtype=torch.long),
                "metadata": [],
            }

        concat_w = np.concatenate(all_windows, axis=0)
        concat_t = np.concatenate(all_targets, axis=0)
        concat_ht = np.concatenate(all_has_targets, axis=0)
        concat_y = np.concatenate(all_labels, axis=0)

        return {
            "windows": torch.tensor(concat_w, dtype=torch.float32),
            "targets": torch.tensor(concat_t, dtype=torch.float32),
            "has_target": torch.tensor(concat_ht, dtype=torch.bool),
            "labels": torch.tensor(concat_y, dtype=torch.long),
            "metadata": all_meta,
        }
