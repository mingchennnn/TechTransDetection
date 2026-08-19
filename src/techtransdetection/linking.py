from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ensure_directory, parse_list
from .indicators import (
    compute_rtc_adtm,
    normalize_retained_cluster_mass,
    plot_indicators,
    write_normalization_audit,
)


def _quality_metrics(clusterer, labels, distances, binary) -> dict:
    import hdbscan.validity as validity

    valid = sorted(set(labels) - {-1})
    try:
        dbcv = float(
            validity.validity_index(
                distances, labels, metric="precomputed", d=binary.shape[1]
            )
        ) if valid else None
    except Exception:
        dbcv = None
    intra = []
    for cluster_id in valid:
        indices = np.flatnonzero(labels == cluster_id)
        if len(indices) > 1:
            intra.append(float(np.mean(distances[np.ix_(indices, indices)])))
    persistence = getattr(clusterer, "cluster_persistence_", np.asarray([]))
    noise = int(np.sum(labels == -1))
    return {
        "number_valid_clusters": len(valid),
        "number_noise_points": noise,
        "total_window_topics": int(len(labels)),
        "noise_percentage": 100.0 * noise / len(labels),
        "mean_cluster_persistence": float(np.mean(persistence)) if len(persistence) else None,
        "dbcv_score": dbcv,
        "average_intra_cluster_jaccard_distance": float(np.mean(intra)) if intra else 0.0,
    }


def _representative_keywords(topics: pd.DataFrame, labels: np.ndarray) -> dict[int, list[str]]:
    result = {}
    for cluster_id in sorted(set(labels)):
        indices = np.flatnonzero(labels == cluster_id)
        words = [word for row in topics.iloc[indices]["keyword_list"] for word in row]
        result[int(cluster_id)] = [word for word, _ in Counter(words).most_common(30)]
    return result


def _build_cluster_mass(
    windows_dir: Path,
    assignments: pd.DataFrame,
    start_year: int,
    end_year: int,
    window_size: int,
) -> pd.DataFrame:
    all_years = list(range(start_year, end_year + 1))
    mass: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(len(all_years), dtype=float))
    for start in range(start_year, end_year - window_size + 2):
        end = start + window_size - 1
        label = f"{start}-{end}"
        window_dir = windows_dir / "windows" / f"{start}_{end}"
        sums_path = window_dir / "sum_vectors.csv"
        if not sums_path.is_file():
            raise FileNotFoundError(f"Missing annual probability sums: {sums_path}")
        sums = pd.read_csv(sums_path)
        mapping_frame = assignments.loc[
            assignments["year_window"] == label, ["topic", "cluster_id"]
        ]
        mapping = dict(zip(mapping_frame["topic"].astype(int), mapping_frame["cluster_id"].astype(int)))
        for column in sums.columns:
            if not column.startswith("topic_"):
                continue
            topic_id = int(column.removeprefix("topic_"))
            if topic_id not in mapping:
                raise ValueError(f"No second-stage assignment for topic {topic_id} in {label}")
            cluster_id = mapping[topic_id]
            for year, value in zip(sums["year"].astype(int), sums[column].astype(float)):
                if start_year <= year <= end_year:
                    mass[cluster_id][year - start_year] += value
    rows = [
        [cluster_id, *values.tolist()]
        for cluster_id, values in sorted(mass.items())
    ]
    return pd.DataFrame(rows, columns=["cluster_id", *[str(year) for year in all_years]])


def link_topics_and_measure(
    windows_dir: Path,
    output_dir: Path,
    sector: str,
    start_year: int,
    end_year: int,
    window_size: int,
    min_cluster_size: int,
    min_samples: int = 2,
    events_config: Path | None = None,
) -> None:
    import hdbscan
    from sklearn.metrics import pairwise_distances

    aggregate_path = windows_dir / "topic_info_all_year_window.csv"
    if not aggregate_path.is_file():
        raise FileNotFoundError(f"Missing first-stage topic aggregate: {aggregate_path}")
    topics = pd.read_csv(aggregate_path)
    required = {"topic", "keywords", "year_window"}
    missing = required - set(topics.columns)
    if missing:
        raise ValueError(f"Topic aggregate is missing columns: {sorted(missing)}")
    expected_windows = {
        f"{start}-{start + window_size - 1}"
        for start in range(start_year, end_year - window_size + 2)
    }
    observed_windows = set(topics["year_window"].astype(str))
    if expected_windows != observed_windows:
        raise ValueError(
            f"Incomplete or unexpected moving windows; missing={sorted(expected_windows - observed_windows)}, "
            f"extra={sorted(observed_windows - expected_windows)}"
        )
    topics["keyword_list"] = topics["keywords"].apply(parse_list)
    vocabulary = sorted({word for values in topics["keyword_list"] for word in values})
    if not vocabulary:
        raise ValueError("No representative keywords were found")
    word_index = {word: index for index, word in enumerate(vocabulary)}
    binary = np.zeros((len(topics), len(vocabulary)), dtype=bool)
    for row_index, values in enumerate(topics["keyword_list"]):
        binary[row_index, [word_index[word] for word in values]] = True
    distances = pairwise_distances(binary, metric="jaccard").astype(np.float64)
    np.fill_diagonal(distances, 0.0)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        metric="precomputed",
    )
    labels = clusterer.fit_predict(distances)
    representatives = _representative_keywords(topics, labels)
    assignments = topics.drop(columns="keyword_list").copy()
    assignments["cluster_id"] = labels
    assignments["representative_keywords"] = assignments["cluster_id"].map(
        lambda value: json.dumps(representatives[int(value)], ensure_ascii=False)
    )

    output_dir = ensure_directory(output_dir)
    assignments.to_csv(output_dir / "clustered_topics_all_year.csv", index=False)
    metrics = {
        "sector": sector,
        "start_year": start_year,
        "end_year": end_year,
        "window_size": window_size,
        "second_stage_min_cluster_size": min_cluster_size,
        "second_stage_min_samples": min_samples,
        "cluster_selection_method": "eom",
        "metric": "precomputed Jaccard",
        **_quality_metrics(clusterer, labels, distances, binary),
    }
    (output_dir / "clustering_quality_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    cluster_sum = _build_cluster_mass(
        windows_dir, assignments, start_year, end_year, window_size
    )
    retained_sum, proportions, _ = normalize_retained_cluster_mass(cluster_sum)
    retained_sum.to_csv(output_dir / "cluster_sum_mass.csv", index=False)
    proportions.to_csv(output_dir / "annual_topic_proportions.csv", index=False)
    write_normalization_audit(proportions, output_dir / "annual_normalization_audit.json")

    indicators = compute_rtc_adtm(proportions)
    indicators.to_csv(output_dir / "rtc_adtm.csv", index=False)
    plot_indicators(indicators, output_dir / "rtc_adtm.png", sector)
    if events_config is not None:
        event_data = json.loads(events_config.read_text(encoding="utf-8"))
        if sector not in event_data:
            raise KeyError(f"No documented events found for sector {sector!r}")
        plot_indicators(
            indicators,
            output_dir / "rtc_adtm_with_benchmark.png",
            sector,
            event_data[sector],
        )
