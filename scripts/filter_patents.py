#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from techtransdetection.filtering import filter_patents


def main() -> None:
    parser = argparse.ArgumentParser(description="Retain patents assigned to relevant global topics.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--patent-topics", required=True, type=Path)
    parser.add_argument("--topic-scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--company-input", type=Path)
    parser.add_argument("--company-output", type=Path)
    args = parser.parse_args()
    filter_patents(
        args.input,
        args.patent_topics,
        args.topic_scores,
        args.output,
        args.threshold,
        args.company_input,
        args.company_output,
    )


if __name__ == "__main__":
    main()
