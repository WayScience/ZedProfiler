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
    "intensity": ("5d1cf592fe0e02f4bfa07becb715d2d924776b8511d0bc071992bf2fdc4b0e94"),
    "volume_size_shape": (
        "7b89edcd8c1427a3a0f50770ad3d97fdb5a919708686b6442db68c9d41795cbd"
    ),
    "neighbors": ("442ba04801300f09ba2796b44256aeed62bf34d1ec646aed4f9deb5ec1aa347b"),
    "texture": ("0e4cfe60a9da0ee358a17cd6c60f05d993b3eb9f229a7ff1ace029789abed8cb"),
    "granularity": ("b66528d9302f4d63d6f24d6c21c147ea9b0f4cd34539ebc66aaac9dd9342eb8f"),
    "colocalization": (
        "304321e87776fce46e6231e4b121bc931d1bd9cc3c8724912dec460dc42b5b8b"
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
