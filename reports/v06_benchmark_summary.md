# QISM V0.6 Benchmark Summary Report

## 1. Consolidated Baseline Model Benchmark
| Model | IID FPR | IID Recall | IID F1 | IID AUROC | IID AUPRC | OOD Recall | OOD FPR | Recall Gap | Median Delay (s) |
|---|---|---|---|---|---|---|---|---|---|
| mahalanobis | 0.389 | 0.864 | 0.792 | 0.689 | 0.646 | 0.667 | 0.750 | +0.197 | 0.0020 |
| robust_covariance | 0.444 | 0.818 | 0.750 | 0.785 | 0.809 | 0.167 | 0.500 | +0.652 | 0.0020 |
| isolation_forest | 0.167 | 0.136 | 0.214 | 0.457 | 0.544 | 0.083 | 0.000 | +0.053 | N/A |
| one_class_svm | 0.444 | 0.864 | 0.776 | 0.682 | 0.648 | 0.667 | 0.750 | +0.197 | 0.0020 |
| lof | 0.444 | 0.864 | 0.776 | 0.672 | 0.638 | 0.667 | 0.750 | +0.197 | 0.0020 |
| autoencoder | 0.444 | 0.864 | 0.776 | 0.679 | 0.647 | 0.667 | 0.750 | +0.197 | 0.0020 |
| temporal_autoencoder | 0.222 | 0.682 | 0.732 | 0.720 | 0.715 | 0.583 | 0.875 | +0.098 | 0.0000 |

## 2. Feature Ablation Study
| Feature Set | Precision | Recall | F1 | FPR | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| qkd_only | 0.500 | 0.045 | 0.083 | 0.056 | 0.477 | 0.575 |
| detector_only | 0.500 | 0.045 | 0.083 | 0.056 | 0.376 | 0.493 |
| environmental_only | 0.667 | 0.091 | 0.160 | 0.056 | 0.369 | 0.550 |
| infrastructure_only | 0.000 | 0.000 | 0.000 | 0.056 | 0.313 | 0.473 |
| full | 0.000 | 0.000 | 0.000 | 0.056 | 0.419 | 0.498 |