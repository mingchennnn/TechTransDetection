#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from techtransdetection.filtering import score_topic_relevance


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify topic relevance through four API judgments.")
    parser.add_argument("--topic-info", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sector", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--pause-seconds", default=1.2, type=float)
    args = parser.parse_args()
    score_topic_relevance(
        args.topic_info,
        args.output,
        args.sector,
        args.model,
        args.pause_seconds,
    )


if __name__ == "__main__":
    main()
