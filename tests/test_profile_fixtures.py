"""Test suite demonstrating the usage of test data profile fixtures.

This module shows how to use the available test data profiles in your tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_data_profiles import Profile

EXPECTED_IMAGE_NDIM = 3
EXPECTED_ALL_PROFILES_COUNT = 4
EXPECTED_FEATURE_TYPE_PROFILES_COUNT = 6


class ProfileFixtures:
    """Tests demonstrating profile fixture usage."""

    def test_minimal_profile(self, minimal_profile: Profile) -> None:
        """Verify minimal profile has correct structure."""
        assert isinstance(minimal_profile.image_array, np.ndarray)
        assert minimal_profile.image_array.ndim == EXPECTED_IMAGE_NDIM
        assert len(minimal_profile.features) == 0
        assert len(minimal_profile.metadata) == 0

    @pytest.mark.parametrize(
        ("fixture_name", "expected_shape", "required_feature", "required_metadata_key"),
        [
            (
                "small_image_profile",
                (4, 8, 8),
                "Nuclei_DNA_Intensity_MeanIntensity",
                "Metadata_Object_ObjectID",
            ),
            (
                "medium_image_profile",
                (16, 32, 32),
                "Nuclei_DNA_Intensity_MeanIntensity",
                "Metadata_Object_ObjectID",
            ),
            (
                "large_image_profile",
                (32, 64, 64),
                "Nuclei_DNA_Intensity_MeanIntensity",
                "Metadata_Object_ObjectID",
            ),
        ],
    )
    def test_image_profiles_by_size(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        expected_shape: tuple[int, int, int],
        required_feature: str,
        required_metadata_key: str,
    ) -> None:
        """Verify size-specific profiles via parametrized fixture lookup."""
        profile: Profile = request.getfixturevalue(fixture_name)
        assert profile.image_array.shape == expected_shape
        assert required_feature in profile.features
        assert required_metadata_key in profile.metadata

    @pytest.mark.parametrize(
        ("feature_kind", "expected_ratio", "fixture_name"),
        [
            ("Intensity", 0.7, "intensity_profile"),
            ("Texture", 0.7, "texture_profile"),
            ("Areasizeshape", 0.7, "morphology_profile"),
        ],
    )
    def test_feature_dominance_profiles(
        self,
        request: pytest.FixtureRequest,
        feature_kind: str,
        expected_ratio: float,
        fixture_name: str,
    ) -> None:
        """Verify feature-focused profiles are dominated by their feature kind."""
        profile: Profile = request.getfixturevalue(fixture_name)
        features = profile.features
        feature_count = sum(
            1 for feature_name in features if feature_kind in feature_name
        )
        total_count = len(features)
        assert feature_count > total_count * expected_ratio

    @pytest.mark.parametrize(
        ("fixture_name", "required_token"),
        [
            ("colocalization_profile", "Colocalization"),
            ("granularity_profile", "Granularity"),
            ("neighbors_profile", "Neighbors"),
        ],
    )
    def test_single_kind_feature_profiles(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        required_token: str,
    ) -> None:
        """Verify token-specific profiles contain only their expected token."""
        profile: Profile = request.getfixturevalue(fixture_name)
        features = profile.features
        assert all(required_token in feature_name for feature_name in features)

    def test_complete_profile(self, complete_profile: Profile) -> None:
        """Verify complete profile has comprehensive feature coverage."""
        assert complete_profile.image_array.shape == (24, 64, 64)
        features = complete_profile.features
        assert any("Intensity" in feature_name for feature_name in features)
        assert any("Texture" in feature_name for feature_name in features)
        assert any("Areasizeshape" in feature_name for feature_name in features)
        assert any("Colocalization" in feature_name for feature_name in features)
        assert any("Granularity" in feature_name for feature_name in features)
        assert any("Neighbors" in feature_name for feature_name in features)

    def test_all_profiles_collection(self, all_profiles: list[Profile]) -> None:
        """Verify collection fixture contains expected profiles."""
        assert len(all_profiles) == EXPECTED_ALL_PROFILES_COUNT
        assert all(isinstance(p, Profile) for p in all_profiles)
        # Verify increasing sizes
        sizes = [p.image_array.size for p in all_profiles]
        assert sizes == sorted(sizes)

    def test_all_feature_type_profiles(
        self,
        all_feature_type_profiles: list[Profile],
    ) -> None:
        """Verify feature type collection has all feature types."""
        assert len(all_feature_type_profiles) == EXPECTED_FEATURE_TYPE_PROFILES_COUNT
        assert all(isinstance(p, Profile) for p in all_feature_type_profiles)


class ProfileWithVaryingSize:
    """Tests demonstrating parameterized profile fixtures."""

    def test_varying_profile_sizes(self, profile_with_varying_size: Profile) -> None:
        """Verify profile_with_varying_size fixture produces valid profiles."""
        assert isinstance(profile_with_varying_size.image_array, np.ndarray)
        assert profile_with_varying_size.image_array.ndim == EXPECTED_IMAGE_NDIM
        assert (
            "Nuclei_DNA_Intensity_MeanIntensity" in profile_with_varying_size.features
        )
        assert profile_with_varying_size.metadata["Metadata_Object_ObjectID"] == 1


def test_profile_to_dict() -> None:
    """Verify Profile.to_dict() method works correctly."""
    profile = Profile(
        image_array=np.ones((4, 8, 8)),
        features={"test_feature": 0.5},
        metadata={"test_meta": "value"},
    )

    profile_dict = profile.to_dict()
    assert isinstance(profile_dict, dict)
    assert "image_array" in profile_dict
    assert "features" in profile_dict
    assert "metadata" in profile_dict
    assert np.array_equal(profile_dict["image_array"], profile.image_array)
