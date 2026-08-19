from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


EXTRA_STOP_WORDS = {
    "first", "second", "third", "sb", "thereof", "include", "method",
    "including", "includes", "solve", "solved", "solution", "jpo", "inpit",
    "ncip", "copyright", "problem", "invention", "innovation", "provide",
    "provides", "provided", "following", "result", "describe", "wherein",
    "left", "right", "purpose", "constitution", "january", "jan", "february",
    "feb", "march", "mar", "april", "may", "june", "july", "august", "aug",
    "september", "sep", "october", "oct", "november", "nov", "december",
    "dec", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
}


def parse_list(value) -> list[str]:
    """Parse a list stored by pandas/BERTopic without evaluating arbitrary code."""
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, (tuple, set)):
        parsed = list(value)
    elif value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
    if not isinstance(parsed, (list, tuple, set)):
        raise ValueError(f"Expected a list-like value, received {type(parsed).__name__}")
    return [str(item).strip() for item in parsed if str(item).strip()]


def stable_union(groups: Iterable[Iterable[str]]) -> list[str]:
    """Combine keyword lists while retaining their first observed order."""
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for value in group:
            item = str(value).strip()
            if item and item not in seen:
                seen.add(item)
                result.append(item)
    return result


def read_patents(path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "year", "document"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Patent file is missing columns: {sorted(missing)}")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame.loc[frame["year"].between(start_year, end_year)].copy()
    frame = frame.dropna(subset=["id", "year"])
    frame["year"] = frame["year"].astype(int)
    frame["document"] = frame["document"].fillna("").astype(str).str.strip()
    frame = frame.loc[frame["document"] != ""].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No non-empty patent documents remain in the requested year range")
    return frame


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
