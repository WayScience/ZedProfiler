"""Additional integration tests for comprehensive coverage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from zedprofiler.IO.feature_writing_utils import (
    FeatureMetadata,
    format_morphology_feature_name,
    remove_underscores_from_string,
    save_features_as_parquet,
)

# Test constants
LARGE_DATAFRAME_ROWS = 100
LARGE_DATAFRAME_COLUMNS = 10


class TestRobustness:
    """Test robustness and edge cases across modules."""

    def test_format_name_with_all_delimiters(self) -> None:
        """Test formatting with all types of delimiters."""
        result = format_morphology_feature_name(
            "cell_part",
            "channel.name",
            "feature type",
            "measurement/value",
        )
        assert isinstance(result, str)
        assert "_" in result
        assert "-" not in result or "." not in result

    def test_dataframe_with_various_dtypes(self) -> None:
        """Test saving dataframes with multiple data types."""
        pytest.importorskip("pyarrow")
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir)

            # DataFrame with mixed types
            df = pd.DataFrame(
                {
                    "int_col": [1, 2, 3],
                    "float_col": [1.1, 2.2, 3.3],
                    "str_col": ["a", "b", "c"],
                    "bool_col": [True, False, True],
                },
            )

            metadata = FeatureMetadata(
                compartment="test",
                channel="test",
                feature_type="test",
                cpu_or_gpu="cpu",
            )

            result_path = save_features_as_parquet(parent_path, df, metadata)
            loaded = pd.read_parquet(result_path)

            assert loaded.shape == df.shape
            assert list(loaded.columns) == list(df.columns)

    def test_large_dataframe_handling(self) -> None:
        """Test handling of larger dataframes."""
        pytest.importorskip("pyarrow")
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir)

            # Create a larger dataframe
            large_df = pd.DataFrame(
                {
                    f"feature_{i}": range(LARGE_DATAFRAME_ROWS)
                    for i in range(LARGE_DATAFRAME_COLUMNS)
                },
            )

            metadata = FeatureMetadata(
                compartment="large",
                channel="test",
                feature_type="test",
                cpu_or_gpu="cpu",
            )

            result_path = save_features_as_parquet(parent_path, large_df, metadata)
            loaded = pd.read_parquet(result_path)

            assert len(loaded) == LARGE_DATAFRAME_ROWS
            assert len(loaded.columns) == LARGE_DATAFRAME_COLUMNS

    def test_special_string_conversions(self) -> None:
        """Test edge cases in string conversion."""
        # Test None-like behavior
        assert isinstance(remove_underscores_from_string(""), str)

        # Test with numbers and special chars mixed
        result = remove_underscores_from_string("123_456.789/000")
        assert result == "123-456-789-000"

    def test_metadata_attributes_accessible(self) -> None:
        """Test that all FeatureMetadata attributes are accessible."""
        metadata = FeatureMetadata(
            compartment="nuc",
            channel="dapi",
            feature_type="shape",
            cpu_or_gpu="gpu",
        )

        # All attributes should be accessible
        assert metadata.compartment == "nuc"
        assert metadata.channel == "dapi"
        assert metadata.feature_type == "shape"
        assert metadata.cpu_or_gpu == "gpu"

    def test_repeated_delimiter_handling(self) -> None:
        """Test strings with repeated delimiters."""
        result = remove_underscores_from_string("___test___")
        assert result.startswith("-")
        assert result.endswith("-")
        assert "test" in result

    @pytest.mark.parametrize(
        ("raw_value", "expected"),
        [
            ("a", "a"),
            ("_", "-"),
            (".", "-"),
            (" ", "-"),
            ("/", "-"),
        ],
    )
    def test_single_character_strings(
        self,
        raw_value: str,
        expected: str,
    ) -> None:
        """Test single character string handling."""
        assert remove_underscores_from_string(raw_value) == expected
