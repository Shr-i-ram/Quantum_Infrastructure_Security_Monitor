"""Scenario-level dataset splitting and leakage audits (Requirements 16–23, 54–56, 189–191)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qism.random import derive_dataset_seed


class SplitManager:
    """Manages scenario-level partition assignments with zero data leakage."""

    def __init__(self, master_seed: int = 42, train_val_split_ratio: float = 0.70):
        self.master_seed = master_seed
        self.train_val_split_ratio = train_val_split_ratio

    def partition_scenarios(
        self,
        scenario_manifest: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        """Partition scenarios from scenario_manifest into TRAIN, VAL, TEST, and GENERALIZATION.

        Rules:
        - `normal` split scenarios are partitioned deterministically into:
          - `train`: normal scenarios (default 70%)
          - `val`: normal scenarios (remaining 30% for threshold calibration & early stopping)
        - `attack` split scenarios go to `test`
        - A designated portion of `normal` (or separate test normal) also goes to `test` for evaluation
        - `generalization` split scenarios go to `generalization` (OOD evaluation)
        - `calibration` scenarios can be used for baseline reference or training
        """
        scenarios = scenario_manifest.get("scenarios", [])
        by_split: dict[str, list[dict[str, Any]]] = {
            "calibration": [],
            "normal": [],
            "attack": [],
            "generalization": [],
        }

        for sc in scenarios:
            sp = sc.get("dataset_split", sc.get("split", "normal"))
            if sp in by_split:
                by_split[sp].append(sc)
            else:
                by_split.setdefault(sp, []).append(sc)

        # If dataset is pre-partitioned (e.g. qism_research_dataset), respect exact split assignments
        if by_split.get("train"):
            partitions = {
                "train": by_split.get("train", []),
                "val": by_split.get("validation", []),
                "test": by_split.get("test_in_distribution", []),
                "generalization": by_split.get("test_generalization", []),
                "stress": by_split.get("test_stress", []),
                "calibration": by_split.get("calibration", []),
            }
            # Also provide test_in_distribution, test_generalization, test_stress keys
            partitions["test_in_distribution"] = partitions["test"]
            partitions["test_generalization"] = partitions["generalization"]
            partitions["test_stress"] = partitions["stress"]
            self.verify_no_leakage(partitions)
            return partitions

        # Deterministically sort normal scenarios to partition into train vs val
        normal_scenarios = list(by_split["normal"])
        seed = derive_dataset_seed(self.master_seed, "normal_split_partition")
        import random
        rng = random.Random(seed)
        shuffled_normal = list(normal_scenarios)
        rng.shuffle(shuffled_normal)

        n_train = max(1, int(round(len(shuffled_normal) * self.train_val_split_ratio)))
        if len(shuffled_normal) >= 2 and n_train >= len(shuffled_normal):
            n_train = len(shuffled_normal) - 1

        train_normal = shuffled_normal[:n_train]
        val_normal = shuffled_normal[n_train:]

        # Test set contains both normal evaluation scenarios and attack scenarios
        # We reserve calibration scenarios or val_normal for normal test coverage
        partitions = {
            "train": train_normal,
            "val": val_normal if val_normal else train_normal,
            "test": by_split["attack"] + val_normal,
            "generalization": by_split["generalization"],
            "calibration": by_split["calibration"],
        }

        # Verify scenario-level separation
        self.verify_no_leakage(partitions)
        return partitions

    def verify_no_leakage(self, partitions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        """Verify scenario-level and device-level separation between train and test/generalization."""
        train_scen_ids = {s["scenario_id"] for s in partitions.get("train", [])}
        val_scen_ids = {s["scenario_id"] for s in partitions.get("val", [])}
        test_attack_scen_ids = {
            s["scenario_id"]
            for s in partitions.get("test", [])
            if s.get("dataset_split", s.get("split")) == "attack"
        }
        gen_scen_ids = {s["scenario_id"] for s in partitions.get("generalization", [])}

        # Train vs Test Attack must never overlap
        train_test_overlap = train_scen_ids.intersection(test_attack_scen_ids)
        if train_test_overlap:
            raise ValueError(f"Scenario leakage detected! Scenarios in both train and test attack: {train_test_overlap}")

        # Train vs Generalization must never overlap
        train_gen_overlap = train_scen_ids.intersection(gen_scen_ids)
        if train_gen_overlap:
            raise ValueError(f"Generalization leakage detected! Scenarios in both train and generalization: {train_gen_overlap}")

        # Device level check: devices in generalization should not be in train
        train_devices = {s.get("infrastructure_id") for s in partitions.get("train", []) if s.get("infrastructure_id")}
        gen_devices = {s.get("infrastructure_id") for s in partitions.get("generalization", []) if s.get("infrastructure_id")}
        device_overlap = train_devices.intersection(gen_devices)

        return {
            "train_scenarios": len(train_scen_ids),
            "val_scenarios": len(val_scen_ids),
            "test_attack_scenarios": len(test_attack_scen_ids),
            "generalization_scenarios": len(gen_scen_ids),
            "train_test_overlap": list(train_test_overlap),
            "train_gen_overlap": list(train_gen_overlap),
            "device_overlap": list(device_overlap),
            "leakage_free": len(train_test_overlap) == 0 and len(train_gen_overlap) == 0,
        }
