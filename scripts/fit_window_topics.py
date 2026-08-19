#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from techtransdetection.dynamic import fit_rolling_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit BERTopic in overlapping moving windows.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-year", required=True, type=int)
    parser.add_argument("--end-year", required=True, type=int)
    parser.add_argument("--window-size", default=20, type=int)
    parser.add_argument("--min-cluster-size", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--save-embeddings", action="store_true")
    args = parser.parse_args()
    fit_rolling_windows(
        args.input,
        args.output,
        args.start_year,
        args.end_year,
        args.window_size,
        args.min_cluster_size,
        args.overwrite,
        args.save_models,
        args.save_embeddings,
    )


if __name__ == "__main__":
    main()
