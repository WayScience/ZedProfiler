"""Tests for zedprofiler.IO.feature_writing_utils."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from pandera.errors import SchemaError

from zedprofiler.IO.feature_writing_utils import (
    FEATURE_NAME_COMPONENT_SCHEMA,
    FEATURE_OUTPUT_SCHEMA,
    FeatureMetadata,
    _coerce_dataframe_column_names_to_strings,
    _coerce_feature_name_components,
    format_morphology_feature_name,
    remove_underscores_from_string,
    save_features_as_parquet,
)

EXPECTED_COMPONENT_COUNT = 4


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("test_string", "test-string"),
        ("test.string", "test-string"),
        ("test string", "test-string"),
        ("test/string", "test-string"),
        ("test_string.with spaces/delimiters", "test-string-with-spaces-delimiters"),
        ("teststring", "teststring"),
        (123, "123"),
        (3.14, "3-14"),
    ],
)
def test_remove_underscores_from_string(raw_value: object, expected: str) -> None:
    """String cleanup should normalize delimiters and coerce non-strings."""
    assert remove_underscores_from_string(raw_value) == expected


@pytest.mark.parametrize(
    ("compartment", "channel", "feature_type", "measurement", "expected"),
    [
        ("nucleus", "dapi", "area", "value", "nucleus_dapi_area_value"),
        (
            "cell_body",
            "gfp.channel",
            "mean intensity",
            "normalized/value",
            "cell-body_gfp-channel_mean-intensity_normalized-value",
        ),
        (1, 2, 3, 4, "1_2_3_4"),
    ],
)
def test_format_morphology_feature_name(
    compartment: object,
    channel: object,
    feature_type: object,
    measurement: object,
    expected: str,
) -> None:
    """Feature name formatting should route through pandera coercion and parsing."""
    result = format_morphology_feature_name(
        compartment,
        channel,
        feature_type,
        measurement,
    )
    assert result == expected
    assert result.count("_") == EXPECTED_COMPONENT_COUNT - 1


def test_feature_name_component_schema_rejects_missing_columns() -> None:
    """Pandera schema should fail when required component columns are missing."""
    component_frame = pd.DataFrame(
        [{"compartment": "nucleus", "channel": "dapi", "feature_type": "area"}],
    )
    with pytest.raises(SchemaError):
        FEATURE_NAME_COMPONENT_SCHEMA.validate(component_frame)


def test_coerce_feature_name_components_uses_cleanup_function() -> None:
    """Component parser should apply cleanup to each naming component."""
    input_df = pd.DataFrame(
        [
            {
                "compartment": "cell_body",
                "channel": "gfp.channel",
                "feature_type": "mean intensity",
                "measurement": "normalized/value",
            },
        ],
    )
    parsed = _coerce_feature_name_components(input_df)
    assert parsed.iloc[0].to_dict() == {
        "compartment": "cell-body",
        "channel": "gfp-channel",
        "feature_type": "mean-intensity",
        "measurement": "normalized-value",
    }


def test_coerce_feature_name_components_skips_missing_columns() -> None:
    """Component parser should leave absent columns untouched."""
    input_df = pd.DataFrame(
        [
            {
                "compartment": "cell_body",
                "measurement": "normalized/value",
            },
        ],
    )

    parsed = _coerce_feature_name_components(input_df)

    assert parsed.iloc[0].to_dict() == {
        "compartment": "cell-body",
        "measurement": "normalized-value",
    }


def test_feature_output_schema_coerces_column_names_to_strings() -> None:
    """Output schema parser should coerce non-string column labels to strings."""
    input_df = pd.DataFrame({0: [1, 2], ("x", "y"): [3, 4]})
    parsed = FEATURE_OUTPUT_SCHEMA.validate(input_df)
    assert list(parsed.columns) == ["0", "('x', 'y')"]


def test_coerce_dataframe_column_names_to_strings_keeps_data() -> None:
    """Column parser should only transform labels and preserve values."""
    input_df = pd.DataFrame({1: [10, 20], 2: [30, 40]})
    parsed = _coerce_dataframe_column_names_to_strings(input_df)
    assert list(parsed.columns) == ["1", "2"]
    pd.testing.assert_frame_equal(parsed, pd.DataFrame({"1": [10, 20], "2": [30, 40]}))


def test_feature_metadata_enforces_runtime_types() -> None:
    """Beartype should validate dataclass init annotations at runtime."""
    with pytest.raises(BeartypeCallHintParamViolation):
        FeatureMetadata(
            compartment=1,
            channel="dapi",
            feature_type="area",
            cpu_or_gpu="cpu",
        )


def test_remove_underscores_value_error() -> None:
    """String cleanup should raise ValueError when conversion fails."""

    class BadString:
        def __str__(self) -> str:
            raise RuntimeError("no string conversion")

    with pytest.raises(ValueError):
        remove_underscores_from_string(BadString())


def test_save_features_as_parquet_writes_file_and_preserves_data(
    tmp_path: Path,
) -> None:
    """Save utility should write parquet and preserve the DataFrame payload."""
    pytest.importorskip("pyarrow")

    parent_path = tmp_path
    original_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    metadata = FeatureMetadata(
        compartment="nucleus",
        channel="dapi",
        feature_type="area",
        cpu_or_gpu="cpu",
    )

    result_path = save_features_as_parquet(parent_path, original_df, metadata)

    assert result_path.exists()
    assert result_path.name == "nucleus_dapi_area_cpu_features.parquet"
    loaded_df = pd.read_parquet(result_path)
    pd.testing.assert_frame_equal(original_df, loaded_df)


def test_save_features_as_parquet_coerces_non_string_column_labels(
    tmp_path: Path,
) -> None:
    """Parquet save path should apply output schema parser before writing."""
    pytest.importorskip("pyarrow")

    parent_path = tmp_path
    df_with_non_string_columns = pd.DataFrame({0: [1], 1: [2]})
    metadata = FeatureMetadata(
        compartment="test",
        channel="ch1",
        feature_type="feat",
        cpu_or_gpu="gpu",
    )

    result_path = save_features_as_parquet(
        parent_path,
        df_with_non_string_columns,
        metadata,
    )
    loaded_df = pd.read_parquet(result_path)

    assert list(loaded_df.columns) == ["0", "1"]


def test_save_features_as_parquet_sanitizes_metadata_in_filename(
    tmp_path: Path,
) -> None:
    """Save utility should normalize metadata delimiters in output filenames."""
    pytest.importorskip("pyarrow")

    metadata = FeatureMetadata(
        compartment="cell_body",
        channel="gfp.channel",
        feature_type="mean intensity",
        cpu_or_gpu="normalized/value",
    )

    result_path = save_features_as_parquet(
        tmp_path,
        pd.DataFrame({"col": [1]}),
        metadata,
    )

    assert (
        result_path.name
        == "cell-body_gfp-channel_mean-intensity_normalized-value_features.parquet"
    )


def test_save_features_as_parquet_requires_path_type() -> None:
    """Beartype should reject non-Path inputs for parent_path."""
    with pytest.raises(BeartypeCallHintParamViolation):
        save_features_as_parquet(
            "not-a-path",
            pd.DataFrame({"col": [1]}),
            FeatureMetadata(
                compartment="nucleus",
                channel="dapi",
                feature_type="area",
                cpu_or_gpu="cpu",
            ),
        )
