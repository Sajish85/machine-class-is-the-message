---
pretty_name: KANVAS GPU Task Failure Features
license: cc-by-4.0
license_name: creative-commons-attribution-4.0
license_link: https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/LICENSE
task_categories:
  - tabular-classification
tags:
  - gpu-cluster
  - failure-prediction
  - interpretability
  - kolmogorov-arnold-network
size_categories:
  - 10K<n<100K
configs:
  - config_name: feature_matrix
    data_files: kanvas_feature_matrix.csv
  - config_name: phi_curves
    data_files: kanvas_phi_curves.csv
  - config_name: benchmark_results
    data_files: kanvas_benchmark_results.csv
---

# KANVAS GPU Task Failure Features

Derived data for the KANVAS project: an interpretable Kolmogorov-Arnold Additive Model (KAAM) that predicts GPU cluster task failure. Code, notebooks and tests: https://github.com/Swapnil007-Curious/KANVAS

## Files

| File | Rows | Contents |
|---|---|---|
| `kanvas_feature_matrix.csv` | 18,890 | 10 scheduling and telemetry features plus the failure label, one row per task |
| `kanvas_phi_curves.csv` | 2,000 | Learned per-feature curves phi_i(x) of the canonical KAAM, long format |
| `kanvas_benchmark_results.csv` | 6 | Test metrics for KAAM, MLP and logistic regression |

Column definitions are in `data_dictionary.md`.

## Source and construction

Built from the Alibaba Cluster Trace GPU 2020 dataset (Weng et al., "MLaaS in the Wild", NSDI 2022; https://github.com/alibaba/clusterdata). Label: Failed = 1, Terminated = 0. Telemetry features average the previous 1 hour of other workers on the same machine, so no information from the task's own run leaks into the features. User and job identifiers are removed. The raw trace is not redistributed here.

## Known limitation

Models trained on this data reach test AUC of about 0.75 with a random split but fall to chance (about 0.45) when the split is grouped by user. One user accounts for 31% of tasks and 54% of failures. Do not use random-split scores from this data as evidence of cross-user generalization.

## Citation

Please cite the original dataset paper (Weng et al., NSDI 2022) and link the GitHub repository above.