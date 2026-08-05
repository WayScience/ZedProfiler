"""Comprehensive integration tests for coverage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from zedprofiler.featurization import volumesizeshape
from zedprofiler.IO.feature_writing_utils import (
    FeatureMetadata,
    format_morphology_feature_name,
    remove_underscores_from_string,
    save_features_as_parquet,
)

EXPECTED_COMPONENTS = 4
LONG_NAME_THRESHOLD = 1000


class TestIntegrationWorkflows:
    """Test realistic workflows combining multiple modules."""

    def test_end_to_end_feature_extraction_and_save(self) -> None:
        """Test extracting features and saving to parquet."""
        pytest.importorskip("pyarrow")
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir)

            # Create sample features
            features_df = pd.DataFrame(
                {
                    "Metadata_Object_ObjectID": [1, 2, 3],
                    "volume": [100.5, 200.3, 150.7],
                    "diameter": [10.2, 12.5, 11.8],
                },
            )

            # Create metadata
            metadata = FeatureMetadata(
                compartment="nucleus",
                channel="dapi",
                feature_type="morphology",
                cpu_or_gpu="cpu",
            )

            # Save features
            result_path = save_features_as_parquet(parent_path, features_df, metadata)

            # Verify file exists and contains correct data
            assert result_path.exists()
            loaded_df = pd.read_parquet(result_path)
            pd.testing.assert_frame_equal(features_df, loaded_df)

    def test_feature_naming_consistency_across_modules(self) -> None:
        """Test consistent naming across different feature modules."""
        molecule_names = ["nucleus", "cytoplasm", "membrane"]
        channels = ["dapi", "gfp", "rfp"]
        features = ["area", "volume", "perimeter"]
        measurements = ["mean", "std", "max"]

        results = []
        for mol in molecule_names:
            for ch in channels:
                for feat in features:
                    for meas in measurements:
                        name = format_morphology_feature_name(mol, ch, feat, meas)
                        results.append(name)

                        # Verify all names are unique and properly formed
                        parts = name.split("_")
                        assert len(parts) == EXPECTED_COMPONENTS

        assert len(results) == len(set(results)), "Feature names should be unique"

    @pytest.mark.parametrize(
        ("input_str", "expected"),
        [
            ("single_underscore", "single-underscore"),
            ("multiple.periods.here", "multiple-periods-here"),
            ("mixed_delimiters.here/and here", "mixed-delimiters-here-and-here"),
            ("__leading", "--leading"),
            ("trailing__", "trailing--"),
        ],
    )
    def test_multiple_delimiter_combinations(
        self,
        input_str: str,
        expected: str,
    ) -> None:
        """Test delimiter removal with various combinations."""
        result = remove_underscores_from_string(input_str)
        assert result == expected

    def test_empty_dataframe_save_restore(self) -> None:
        """Test saving and restoring empty dataframes."""
        pytest.importorskip("pyarrow")
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir)

            # Create empty dataframe with proper schema
            empty_df = pd.DataFrame(
                {
                    "Metadata_Object_ObjectID": pd.Series([], dtype="int64"),
                    "feature1": pd.Series([], dtype="float64"),
                    "feature2": pd.Series([], dtype="float64"),
                },
            )

            metadata = FeatureMetadata(
                compartment="test",
                channel="test",
                feature_type="test",
                cpu_or_gpu="cpu",
            )

            result_path = save_features_as_parquet(parent_path, empty_df, metadata)
            loaded_df = pd.read_parquet(result_path)

            assert len(loaded_df) == 0
            assert list(loaded_df.columns) == list(empty_df.columns)

    def test_contract_validation_integration(self) -> None:
        """Test basic feature extraction and formatting workflows."""
        # Test that different methods produce consistent results
        name1 = format_morphology_feature_name("nucleus", "dapi", "area", "mean")
        name2 = format_morphology_feature_name("nucleus", "dapi", "area", "mean")

        assert name1 == name2
        assert isinstance(name1, str)
        assert len(name1) > 0

    def test_volumesizeshape_schema_consistency(self) -> None:
        """Test that volumesizeshape maintains consistent output schema."""
        result1 = volumesizeshape.compute_volume_size_shape()
        result2 = volumesizeshape.compute_volume_size_shape()
        result3 = volumesizeshape.compute_volume_size_shape()

        # All calls should return a DataFrame, matching the other
        # featurization modules' no-data return convention.
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)
        assert isinstance(result3, pd.DataFrame)

        # All calls should return same columns in same order
        assert list(result1.columns) == list(result2.columns)
        assert list(result2.columns) == list(result3.columns)

        # All results should be empty (no rows)
        assert result1.empty
        assert result2.empty
        assert result3.empty


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_unicode_in_feature_names(self) -> None:
        """Test handling of unicode characters in names."""
        # Should successfully convert unicode to string
        result = remove_underscores_from_string("café_résumé")
        assert isinstance(result, str)
        assert "-" in result

    def test_very_long_feature_names(self) -> None:
        """Test handling very long feature names."""
        long_name = "a" * 500 + "_" + "b" * 500
        result = format_morphology_feature_name(long_name, "ch", "feat", "meas")
        assert len(result) > LONG_NAME_THRESHOLD  # Should be very long
        assert "_" in result

    def test_special_characters_in_compartment_names(self) -> None:
        """Test special characters in compartment names."""
        result = format_morphology_feature_name(
            "cell/compartment",
            "ch_1",
            "feat.type",
            "meas",
        )
        assert isinstance(result, str)
        # Should have replaced delimiters
        assert "/" not in result
