from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def centered_moving_average(values: Sequence[float], max_window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if max_window < 1 or max_window % 2 == 0:
        raise ValueError("max_window must be a positive odd integer")
    half = max_window // 2
    return np.asarray([
        np.mean(values[max(0, i - half):min(len(values), i + half + 1)])
        for i in range(len(values))
    ])


def normalize_retained_cluster_mass(
    cluster_sum: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Remove second-stage noise before normalizing every annual topic vector."""
    if "cluster_id" not in cluster_sum:
        raise ValueError("cluster_sum must contain cluster_id")
    years = sorted(
        (column for column in cluster_sum.columns if column != "cluster_id"),
        key=lambda value: int(value),
    )
    retained = cluster_sum.loc[cluster_sum["cluster_id"] != -1, ["cluster_id", *years]].copy()
    totals = retained[years].sum(axis=0)
    if (totals <= 0).any():
        missing = totals.loc[totals <= 0].index.tolist()
        raise ValueError(f"No retained topic mass in years: {missing}")
    proportions = retained.copy()
    proportions[years] = proportions[years].div(totals, axis="columns")
    annual_sums = proportions[years].sum(axis=0)
    if not np.allclose(annual_sums, 1.0):
        raise AssertionError("Retained annual topic vectors are not on the probability simplex")
    return retained, proportions, years


def reorder_by_weighted_mean_year(proportions: pd.DataFrame) -> pd.DataFrame:
    years = sorted(
        (column for column in proportions.columns if column != "cluster_id"),
        key=lambda value: int(value),
    )
    year_values = np.asarray([int(year) for year in years], dtype=float)
    weights = proportions[years].to_numpy(dtype=float)
    totals = weights.sum(axis=1)
    mean_years = np.divide(
        weights @ year_values,
        totals,
        out=np.full(len(proportions), np.inf),
        where=totals > 0,
    )
    ordered = proportions.assign(weighted_mean_year=mean_years).sort_values(
        ["weighted_mean_year", "cluster_id"]
    )
    return ordered.reset_index(drop=True)


def compute_rtc_adtm(proportions: pd.DataFrame, smoothing_window: int = 5) -> pd.DataFrame:
    if (proportions["cluster_id"] == -1).any():
        raise ValueError("Second-stage noise must be removed before computing indicators")
    ordered = reorder_by_weighted_mean_year(proportions)
    year_columns = sorted(
        (column for column in ordered.columns if column not in {"cluster_id", "weighted_mean_year"}),
        key=lambda value: int(value),
    )
    if not np.allclose(ordered[year_columns].sum(axis=0), 1.0):
        raise ValueError("Annual topic proportions must sum to one")

    years = np.asarray([int(year) for year in year_columns], dtype=int)
    ranks = np.arange(len(ordered), dtype=float)
    medians: list[float] = []
    for year in year_columns:
        weights = ordered[year].to_numpy(dtype=float)
        cumulative = np.cumsum(weights) / weights.sum()
        medians.append(float(ranks[np.flatnonzero(cumulative >= 0.5)[0]]))

    rtc_raw = np.diff(np.asarray(medians))
    rtc = centered_moving_average(rtc_raw, smoothing_window)

    states = ordered[year_columns].to_numpy(dtype=float).T
    movements = np.diff(states, axis=0)
    angles: list[float] = []
    for incoming, outgoing in zip(movements[:-1], movements[1:]):
        denominator = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
        if denominator == 0:
            angles.append(0.0)
        else:
            cosine = np.clip(np.dot(incoming, outgoing) / denominator, -1.0, 1.0)
            angles.append(float(np.arccos(cosine)))
    adtm_raw = np.asarray(angles)
    adtm = centered_moving_average(adtm_raw, smoothing_window)

    result = pd.DataFrame({"year": years})
    result["weighted_median_topic_rank"] = medians
    result["rtc_raw"] = np.nan
    result["rtc"] = np.nan
    result.loc[result.index[1:], "rtc_raw"] = rtc_raw
    result.loc[result.index[1:], "rtc"] = rtc
    result["adtm_raw"] = np.nan
    result["adtm"] = np.nan
    if len(result) > 2:
        result.loc[result.index[1:-1], "adtm_raw"] = adtm_raw
        result.loc[result.index[1:-1], "adtm"] = adtm
    return result


def event_intensity(events: Sequence[int | Sequence[int]], start_year: int, end_year: int) -> pd.DataFrame:
    years = np.arange(start_year, end_year + 1)
    intensity = np.zeros(len(years), dtype=float)
    for event in events:
        if isinstance(event, int):
            if start_year <= event <= end_year:
                intensity[event - start_year] += 1.0
        else:
            first, last = (int(value) for value in event)
            weight = 1.0 / (last - first + 1)
            for year in range(first, last + 1):
                if start_year <= year <= end_year:
                    intensity[year - start_year] += weight
    return pd.DataFrame({
        "year": years,
        "event_intensity_raw": intensity,
        "event_intensity": centered_moving_average(intensity, 5),
    })


def _align_twin_axis_reference(
    left_axis,
    right_axis,
    left_reference: float,
    right_reference: float,
    right_upper: float,
) -> None:
    left_first, left_second = left_axis.get_ylim()
    fraction = (left_reference - left_first) / (left_second - left_first)
    if not 0.0 < fraction < 1.0:
        raise ValueError("The left-axis reference must fall inside the visible limits")
    right_lower = (right_reference - fraction * right_upper) / (1.0 - fraction)
    right_axis.set_ylim(right_lower, right_upper)


def plot_indicators(
    indicators: pd.DataFrame,
    output: Path,
    sector: str,
    events: Sequence[int | Sequence[int]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if events is None:
        fig, (rtc_axis, adtm_axis) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        rtc_axis.plot(indicators["year"], indicators["rtc"], color="blue", label="RTC")
        rtc_axis.axhline(0, color="black", linestyle="--", linewidth=1)
        rtc_axis.set_ylabel("RTC")
        adtm_axis.plot(indicators["year"], indicators["adtm"], color="green", label="ADTM")
        adtm_axis.axhline(math.pi / 2, color="black", linestyle="--", linewidth=1)
        adtm_axis.invert_yaxis()
        adtm_axis.set_ylabel("ADTM")
        adtm_axis.set_xlabel("Year")
        for axis in (rtc_axis, adtm_axis):
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper left")
        fig.suptitle(f"{sector}: Rate of Topic Change and Angular Difference in Topic Momentum")
    else:
        benchmark = event_intensity(events, int(indicators.year.min()), int(indicators.year.max()))
        fig, (rtc_axis, adtm_axis) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
        right_axes = []
        for axis, column, color, reference in (
            (rtc_axis, "rtc", "blue", 0.0),
            (adtm_axis, "adtm", "green", math.pi / 2),
        ):
            axis.plot(indicators["year"], indicators[column], color=color, label=column.upper())
            axis.axhline(reference, color="black", linestyle="--", linewidth=1)
            axis.set_ylabel(column.upper())
            axis.grid(True, alpha=0.3)
            right = axis.twinx()
            right_axes.append(right)
            right.plot(
                benchmark["year"], benchmark["event_intensity"], color="red",
                alpha=0.75, label="Documented transition intensity",
            )
            right.set_ylabel("Documented transition intensity", color="red")
            lines1, labels1 = axis.get_legend_handles_labels()
            lines2, labels2 = right.get_legend_handles_labels()
            axis.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        adtm_axis.invert_yaxis()
        upper = max(float(benchmark["event_intensity"].max()) * 1.2, 1.0)
        _align_twin_axis_reference(rtc_axis, right_axes[0], 0.0, 0.0, upper)
        _align_twin_axis_reference(adtm_axis, right_axes[1], math.pi / 2, 0.0, upper)
        adtm_axis.set_xlabel("Year")
        fig.suptitle(f"{sector}: transition indicators and documented-event intensity")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_normalization_audit(proportions: pd.DataFrame, output: Path) -> None:
    years = [column for column in proportions.columns if column != "cluster_id"]
    sums = proportions[years].sum(axis=0)
    payload = {
        "policy": "exclude cluster_id=-1 before annual normalization",
        "noise_excluded_before_normalization": True,
        "noise_rows_in_saved_proportions": int((proportions["cluster_id"] == -1).sum()),
        "number_retained_topics": int(len(proportions)),
        "minimum_annual_normalized_sum": float(sums.min()),
        "maximum_annual_normalized_sum": float(sums.max()),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
