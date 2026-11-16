#!/bin/sh
# Re-download the KANVAS GPU-failure dataset files (public, CC-BY-4.0, no account needed).
# Source: https://huggingface.co/datasets/Swapnil007-Curious/kanvas-gpu-failure
# Originally downloaded 16 Nov 2026.
set -e
BASE="https://huggingface.co/datasets/Swapnil007-Curious/kanvas-gpu-failure/resolve/main"
for f in README.md data_dictionary.md kanvas_feature_matrix.csv kanvas_phi_curves.csv kanvas_benchmark_results.csv; do
  curl -sL "$BASE/$f" -o "$f"
done
echo "Downloaded: README.md data_dictionary.md kanvas_feature_matrix.csv kanvas_phi_curves.csv kanvas_benchmark_results.csv"
