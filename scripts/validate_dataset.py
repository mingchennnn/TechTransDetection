#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the processed Zenodo data layout.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--chunk-size", default=200_000, type=int)
    args = parser.parse_args()

    rows = []
    for sector_dir in sorted(path for path in args.data_root.iterdir() if path.is_dir()):
        patent_path = sector_dir / "df_id_year_document.csv"
        company_path = sector_dir / "df_company_id_year.csv"
        if not patent_path.is_file() or not company_path.is_file():
            raise FileNotFoundError(f"Incomplete sector folder: {sector_dir}")
        patent_rows = 0
        years = []
        missing_documents = 0
        for chunk in pd.read_csv(patent_path, chunksize=args.chunk_size):
            required = {"id", "year", "document", "topic"}
            if required - set(chunk):
                raise ValueError(f"{patent_path} is missing {sorted(required - set(chunk))}")
            patent_rows += len(chunk)
            year_values = pd.to_numeric(chunk["year"], errors="coerce").dropna()
            if len(year_values):
                years.extend([int(year_values.min()), int(year_values.max())])
            missing_documents += int(chunk["document"].isna().sum())
        company_rows = sum(
            len(chunk) for chunk in pd.read_csv(company_path, chunksize=args.chunk_size)
        )
        rows.append({
            "sector": sector_dir.name,
            "patent_rows": patent_rows,
            "start_year": min(years),
            "end_year": max(years),
            "missing_documents": missing_documents,
            "company_link_rows": company_rows,
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
