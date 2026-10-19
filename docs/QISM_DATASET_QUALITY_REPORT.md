# QISM Research Dataset Quality Assurance Report

## 1. Quality Assurance Summary

This report documents the rigorous quality assurance (QA), schema adherence, physical plausibility, and zero-leakage verification performed on the **`qism_research_dataset`**.
Auditing was conducted via `scripts/audit_dataset_quality.py` and `scripts/validate_dataset.py --check-split`.

### Overall Verdict: **PASSED (ALL 6 QUALITY AUDIT CHECKS PASSED, 0 FAILURES)**

| Verification Suite | Check Name | Target Criterion | Audit Result | Status |
|---|---|---|---|---|
| **Split Isolation** | Scenario-Level Isolation | No scenario ID appears in multiple partitions | 0 shared scenario IDs across all 15 split pairs | **PASSED** |
| **Split Isolation** | Run-Level Isolation | Independent run instances across partitions | 0 shared `(scenario_id, run_id)` instances | **PASSED** |
| **Leakage Audit** | Ground-Truth Separation | Observable features strictly disjoint from ground truth | Observable $\cap$ Ground-Truth $= \emptyset$ (76 vs 18) | **PASSED** |
| **Physical Plausibility** | QBER Boundedness | Sifted QBER strictly $\in [0.0, 1.0]$ | All observed QBER values $\in [0.0, 1.0]$ | **PASSED** |
| **Physical Plausibility** | Photon Statistics | Fano factor non-negativity ($\text{Var}/\text{Mean} \ge 0$) | Fano factor $\ge 0.0$ for all 10,100 windows | **PASSED** |
| **Label Consistency** | Normal Training Purity | Train partition strictly normal (0 attack flags) | Exactly 0 attack flags in `train` partition | **PASSED** |
| **Provenance** | Scenario Registry | Every scenario mapped with seed and config hash | 200 / 200 scenarios registered in Parquet | **PASSED** |

---

## 2. Partition-Level Verification Metrics

Each of the 6 partitions was independently audited for shard integrity, Parquet schema compliance, and completion markers:

| Partition | Total Shards | Validated Runs | Completed Windows | Raw Event Rows | Completion Marker | Manifest Status |
|---|---|---|---|---|---|---|
| `calibration` | 3 | 40 | 800 | 800,000 | `dataset_complete.marker` OK | COMPLETED |
| `train` | 12 | 180 | 3,600 | 3,600,000 | `dataset_complete.marker` OK | COMPLETED |
| `validation` | 4 | 50 | 1,000 | 1,000,000 | `dataset_complete.marker` OK | COMPLETED |
| `test_in_distribution` | 9 | 135 | 2,700 | 2,700,000 | `dataset_complete.marker` OK | COMPLETED |
| `test_generalization` | 4 | 50 | 1,000 | 1,000,000 | `dataset_complete.marker` OK | COMPLETED |
| `test_stress` | 4 | 50 | 1,000 | 1,000,000 | `dataset_complete.marker` OK | COMPLETED |
| **Total** | **36** | **505** | **10,100** | **10,100,000** | **All 6 Present & Verified** | **COMPLETED** |

---

## 3. Physical & Operational Plausibility Findings

### 3.1 QBER & Sifting Mechanics
- Sifted QBER across the 8,520 normal windows averaged **0.0142 (1.42%)**, consistent with standard decoy-state BB84 under typical optical fiber misalignment (1.0 deg) and detector dark count rates (~1,000 Hz).
- Under active `intercept_resend` attacks, sifted QBER elevated to **0.248–0.261 (24.8%–26.1%)**, matching the theoretical 25% error rate induced by random basis measurement and re-emission.
- Under `channel_loss` attacks, QBER remained within normal bounds while count rate and detection gain plummeted, verifying that the simulator does not conflate channel loss with polarization errors.

### 3.2 Detector Imbalance & Dead-Time Loss
- In normal scenarios, `detector_imbalance` $\Delta_c = (C_0 - C_1)/(C_0 + C_1)$ stayed centered around zero ($\text{mean} = 0.0012, \text{std} = 0.018$), reflecting realistic 1% fabrication mismatch between D0 and D1.
- In `detector_manipulation` scenarios, `detector_imbalance` exhibited dramatic excursions up to $\pm 0.72$, validating that efficiency mismatch and blinding attacks alter click distributions causally through detector physics.
- `dead_time_loss_rate` scaled monotonically with count rate in accordance with the rate-dependent dead-time model:
  $$\tau_{\text{eff}}(\lambda) = \tau_0 + \Delta\tau \frac{\lambda}{\lambda + \lambda_c}$$

### 3.3 Decoy Residuals & PNS Signatures
- Decoy state residuals $R_\mu = Q_{\text{obs},\mu} - Q_{\text{pred},\mu}$ and $R_\nu = Q_{\text{obs},\nu} - Q_{\text{pred},\nu}$ were zero-centered under normal operation.
- In `pns` attack scenarios, selective suppression of single-photon pulses and multi-photon tapping induced pronounced negative residuals in $R_\mu$, providing the exact physical signature expected in decoy-state security theory.

---

## 4. Leakage & Split Isolation Verification

The automated isolation audit verified:
1. **Scenario Disjointness**: The 200 scenarios generated across the 6 partitions have mutually disjoint scenario IDs:
   $$\text{Scenario IDs: } \mathcal{S}_{\text{calib}} \cap \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{in-dist}} \cap \mathcal{S}_{\text{gen}} \cap \mathcal{S}_{\text{stress}} = \emptyset$$
2. **Generalization Parameter Holdout**:
   - Training channel distance range: $[10.0, 45.0] \text{ km}$.
   - Generalization channel distance range: $[50.0, 74.8] \text{ km}$ (strictly held out).
   - Training detector efficiency range: $[0.06, 0.28]$.
   - Generalization detector efficiency range: $[0.30, 0.42]$ (strictly held out).
3. **Purity of Normal Partitions**:
   - `calibration`: 800 windows, 0 attack flags.
   - `train`: 3,600 windows, 0 attack flags.
4. **Observable Set Purity**:
   - `get_observable_feature_names()` yields exactly 76 features.
   - None of the 76 features contain `ground_truth_` prefixes, afterpulse rates, yields, attack IDs, or scenario labels.

---

## 5. Machine-Readable Quality Artifacts

The complete quality audit results are persisted in:
- `data/qism_research_dataset/diagnostics/quality_audit_summary.json`
- `data/qism_research_dataset/diagnostics/distribution_summary.csv`
- `data/qism_research_dataset/diagnostics/parameter_coverage.csv`
