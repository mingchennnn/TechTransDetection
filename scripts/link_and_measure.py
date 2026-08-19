#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from techtransdetection.linking import link_topics_and_measure


def main() -> None:
    parser = argparse.ArgumentParser(description="Link moving-window topics and compute RTC/ADTM.")
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sector", required=True)
    parser.add_argument("--start-year", required=True, type=int)
    parser.add_argument("--end-year", required=True, type=int)
    parser.add_argument("--window-size", default=20, type=int)
    parser.add_argument("--min-cluster-size", required=True, type=int)
    parser.add_argument("--min-samples", default=2, type=int)
    parser.add_argument("--events-config", type=Path)
    args = parser.parse_args()
    link_topics_and_measure(
        args.windows,
        args.output,
        args.sector,
        args.start_year,
        args.end_year,
        args.window_size,
        args.min_cluster_size,
        args.min_samples,
        args.events_config,
    )


if __name__ == "__main__":
    main()
