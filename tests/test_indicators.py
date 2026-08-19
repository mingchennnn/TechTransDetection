from __future__ import annotations

import numpy as np
import pandas as pd

from techtransdetection.indicators import (
    centered_moving_average,
    compute_rtc_adtm,
    normalize_retained_cluster_mass,
)


def test_centered_flexible_smoothing() -> None:
    assert np.allclose(centered_moving_average([0, 2, 4], 3), [1, 2, 3])


def test_noise_is_removed_before_normalization() -> None:
    mass = pd.DataFrame({
        "cluster_id": [-1, 10, 20],
        "2000": [2.0, 1.0, 1.0],
        "2001": [1.0, 3.0, 1.0],
    })
    retained, proportions, years = normalize_retained_cluster_mass(mass)
    assert retained["cluster_id"].tolist() == [10, 20]
    assert years == ["2000", "2001"]
    assert np.allclose(proportions[years].sum(axis=0), 1.0)
    assert np.allclose(proportions["2000"], [0.5, 0.5])
    assert np.allclose(proportions["2001"], [0.75, 0.25])


def test_indicator_alignment_and_zero_movement_rule() -> None:
    proportions = pd.DataFrame({
        "cluster_id": [10, 20],
        "2000": [1.0, 0.0],
        "2001": [0.5, 0.5],
        "2002": [0.0, 1.0],
        "2003": [0.0, 1.0],
    })
    result = compute_rtc_adtm(proportions, smoothing_window=3)
    assert result["year"].tolist() == [2000, 2001, 2002, 2003]
    assert np.isnan(result.loc[0, "rtc_raw"])
    assert np.isnan(result.loc[0, "adtm_raw"])
    assert np.isnan(result.loc[3, "adtm_raw"])
    assert result.loc[2, "adtm_raw"] == 0.0
