#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from techtransdetection.dynamic import fit_rolling_windows
from techtransdetection.linking import link_topics_and_measure


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the published TechTransDetection specifications.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sector", required=True, help="A configured sector or 'all'")
    parser.add_argument(
        "--config", type=Path, default=repository_root / "configs" / "paper_sectors.json"
    )
    parser.add_argument(
        "--events-config", type=Path,
        default=repository_root / "configs" / "benchmark_events.json",
    )
    parser.add_argument("--stage", choices=["all", "first", "second"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--save-embeddings", action="store_true")
    args = parser.parse_args()

    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    sectors = configuration["sectors"]
    selected = list(sectors) if args.sector == "all" else [args.sector]
    unknown = set(selected) - set(sectors)
    if unknown:
        raise ValueError(f"Unknown sectors: {sorted(unknown)}; available: {sorted(sectors)}")

    for sector in selected:
        config = sectors[sector]
        window_size = int(configuration.get("window_size", 20))
        first_size = int(config["first_stage_min_cluster_size"])
        second_size = int(config["second_stage_min_cluster_size"])
        run_root = (
            args.output_root / sector /
            f"window{window_size}_first{first_size}_{config['start_year']}_{config['end_year']}"
        )
        input_path = args.data_root / config["data_folder"] / "df_id_year_document.csv"
        if args.stage in {"all", "first"}:
            fit_rolling_windows(
                input_path,
                run_root,
                int(config["start_year"]),
                int(config["end_year"]),
                window_size,
                first_size,
                args.overwrite,
                args.save_models,
                args.save_embeddings,
            )
        if args.stage in {"all", "second"}:
            link_topics_and_measure(
                run_root,
                run_root / f"final_second{second_size}",
                sector,
                int(config["start_year"]),
                int(config["end_year"]),
                window_size,
                second_size,
                2,
                args.events_config,
            )


if __name__ == "__main__":
    main()
