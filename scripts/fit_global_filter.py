#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from techtransdetection.filtering import fit_global_topics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the corpus-wide BERTopic relevance filter.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-year", required=True, type=int)
    parser.add_argument("--end-year", required=True, type=int)
    parser.add_argument("--min-cluster-size", required=True, type=int)
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()
    fit_global_topics(
        args.input,
        args.output,
        args.start_year,
        args.end_year,
        args.min_cluster_size,
        args.save_model,
    )


if __name__ == "__main__":
    main()
