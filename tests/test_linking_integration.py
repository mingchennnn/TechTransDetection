from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from techtransdetection.linking import link_topics_and_measure


def test_synthetic_second_stage(tmp_path: Path) -> None:
    windows_root = tmp_path / "first_stage"
    topic_rows = []
    for start in range(2000, 2003):
        end = start + 1
        label = f"{start}-{end}"
        window = windows_root / "windows" / f"{start}_{end}"
        window.mkdir(parents=True)
        keyword_groups = {
            0: ["alpha", "beta"],
            1: ["alpha", "beta"],
            2: ["gamma", "delta"],
            3: ["gamma", "delta"],
        }
        for topic, keywords in keyword_groups.items():
            topic_rows.append({
                "topic": topic,
                "keywords": json.dumps(keywords),
                "year_window": label,
            })
        pd.DataFrame({
            "year": [start, end],
            "topic_0": [4.0, 3.0],
            "topic_1": [2.0, 1.0],
            "topic_2": [1.0, 2.0],
            "topic_3": [1.0, 4.0],
        }).to_csv(window / "sum_vectors.csv", index=False)
    pd.DataFrame(topic_rows).to_csv(
        windows_root / "topic_info_all_year_window.csv", index=False
    )
    events = tmp_path / "events.json"
    events.write_text(json.dumps({"synthetic": [2001]}), encoding="utf-8")
    output = tmp_path / "final"
    link_topics_and_measure(
        windows_root,
        output,
        "synthetic",
        2000,
        2003,
        2,
        2,
        2,
        events,
    )
    proportions = pd.read_csv(output / "annual_topic_proportions.csv")
    years = [column for column in proportions if column != "cluster_id"]
    assert np.allclose(proportions[years].sum(axis=0), 1.0)
    audit = json.loads((output / "annual_normalization_audit.json").read_text())
    assert audit["noise_rows_in_saved_proportions"] == 0
    assert (output / "rtc_adtm.csv").is_file()
    assert (output / "rtc_adtm_with_benchmark.png").is_file()
