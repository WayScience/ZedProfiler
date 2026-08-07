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
    "intensity": ("570a8f4bcd0a253a7c45d163b5cde172f287b7f089976cc17657f2d44b06917f"),
    "volume_size_shape": (
        "429504786716107bd10ed72c624578c76fc5f26338b7c1556f1f9c6b0dbb4ca2"
    ),
    "neighbors": ("442ba04801300f09ba2796b44256aeed62bf34d1ec646aed4f9deb5ec1aa347b"),
    "texture": ("08eee6b345b04793b2c1cff461015a092a91828c820e27c64d2c6a1b6a52c5dd"),
    "granularity": ("44b746ca088be1f7b8a18282248dc3b7e36f744aa2d646e4c6580fb1218e5c55"),
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
