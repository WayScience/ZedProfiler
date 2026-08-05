"""Accuracy locks and opt-in benchmark scorecards for feature extraction."""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd
import pytest
from benchmarking import (
    dataframe_signature,
    feature_cases,
    real_world_feature_cases,
    scaling_feature_cases,
    time_feature_cases,
)

EXPECTED_SIGNATURES = {
    "intensity": (
        "351f6508dcfc0978c8d5bfc1891847cfe0090c42d8243baf0acb0f528f3061c0"
    ),
    "volume_size_shape": (
        "1fc7482eb490256eca01cc6d54d4b96956d132cdf0a6a27a934eba37dbf83f39"
    ),
    "neighbors": (
        "8f2b18e6023d656ec6fad41ffa0ff9b802f7b3eadb98ea315ea11cfa400644ec"
    ),
    "texture": (
        "e6ae19f7b6bc9e635fb6e199dd45452fca1a17246e027dcb8385478a95e913fa"
    ),
    "granularity": (
        "b46cd8ae17d0d1e8b47d24821c480975dd0464159744bc3913e619616dc94295"
    ),
    "colocalization": (
        "8bf9495cabc617a5743614f8d57c3855283f6cc6aff9715d975fef416bc29d4a"
    ),
}
EXPECTED_OBJECT_ROWS = 2
EXPECTED_SCALING_ROWS = 32
EXPECTED_REAL_WORLD_ROWS = 5


@pytest.mark.parametrize(("feature_name", "run_case"), feature_cases())
def test_feature_outputs_match_current_accuracy_lock(
    feature_name: str,
    run_case: Callable[[], pd.DataFrame],
) -> None:
    """Feature refactors must preserve current deterministic outputs."""
    dataframe = run_case()
    assert dataframe.shape[0] == EXPECTED_OBJECT_ROWS
    assert "Metadata_Object_ObjectID" in dataframe.columns
    assert dataframe_signature(dataframe) == EXPECTED_SIGNATURES[feature_name]


@pytest.mark.benchmark
def test_feature_benchmark_scorecard() -> None:
    """Print an opt-in scorecard for comparing performance passes."""
    scorecard = time_feature_cases(
        [*feature_cases(), *scaling_feature_cases(), *real_world_feature_cases()],
    )
    print("\nZedProfiler feature benchmark scorecard")
    print(json.dumps(scorecard, indent=2, sort_keys=True))

    observed_features = {record["feature"] for record in scorecard}
    assert set(EXPECTED_SIGNATURES).issubset(observed_features)
    for record in scorecard:
        expected_rows = (
            EXPECTED_SCALING_ROWS
            if str(record["feature"]).startswith("scaling_")
            else EXPECTED_OBJECT_ROWS
        )
        if str(record["feature"]).startswith("real_world_"):
            expected_rows = EXPECTED_REAL_WORLD_ROWS
        assert record["rows"] == expected_rows
        assert record["columns"] > 0
        assert record["seconds"] >= 0
        if record["feature"] in EXPECTED_SIGNATURES:
            assert record["signature"] == EXPECTED_SIGNATURES[record["feature"]]
