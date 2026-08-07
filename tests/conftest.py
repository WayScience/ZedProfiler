"""Pytest fixtures for reusable test profiles and image-size parameters.

This module defines a set of `Profile` fixtures with different image shapes,
feature dictionaries, and metadata patterns used across the test suite.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Import dataclass from test_data_profiles
from test_data_profiles import Profile


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add opt-in benchmark execution flag."""
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="Run opt-in benchmark scorecard tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register local test markers."""
    config.addinivalue_line(
        "markers",
        "benchmark: opt-in performance scorecard tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip benchmark scorecards unless explicitly requested."""
    should_run = config.getoption("--run-benchmarks") or (
        os.environ.get("ZEDPROFILER_RUN_BENCHMARKS") == "1"
    )
    if should_run:
        return

    skip_benchmark = pytest.mark.skip(
        reason="benchmark scorecards require --run-benchmarks",
    )
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)


@pytest.fixture
def my_data() -> str:
    """Provide a basic string fixture.

    Returns
    -------
    str
        A deterministic string value used by tests that require simple text
        input.

    """
    return "Hello, differently!"


@pytest.fixture
def minimal_profile() -> Profile:
    """Create the smallest valid test profile.

    Returns
    -------
    Profile
        A profile containing a valid 3D image array and empty feature and
        metadata mappings.

    """
    return Profile(
        image_array=np.random.rand(8, 16, 16),
        features={},
        metadata={},
    )


@pytest.fixture
def small_image_profile() -> Profile:
    """Create a profile with a small 3D image.

    Returns
    -------
    Profile
        A profile with image shape ``(4, 8, 8)`` plus representative
        intensity features and metadata fields.

    """
    return Profile(
        image_array=np.random.rand(4, 8, 8).astype(np.float32),
        features={
            "Nuclei_DNA_Intensity_MeanIntensity": 0.512,
            "Nuclei_DNA_Intensity_MaxIntensity": 0.987,
            "Nuclei_DNA_Intensity_MinIntensity": 0.015,
        },
        metadata={
            "Metadata_Object_ObjectID": 1,
            "Metadata_Imaging_ExposureTime": 100.0,
        },
    )


@pytest.fixture
def medium_image_profile() -> Profile:
    """Create a profile with a medium 3D image.

    Returns
    -------
    Profile
        A profile with image shape ``(16, 32, 32)`` and a mixed set of
        intensity, texture, and morphology feature values.

    """
    return Profile(
        image_array=np.random.rand(16, 32, 32).astype(np.float32),
        features={
            "Nuclei_DNA_Intensity_MeanIntensity": 0.528,
            "Nuclei_DNA_Intensity_MaxIntensity": 0.992,
            "Nuclei_DNA_Intensity_StdIntensity": 0.125,
            "Nuclei_DNA_Texture_Entropy-256-3": 5.234,
            "Cell_DNA_Areasizeshape_Volume": 512.0,
            "Cell_DNA_Areasizeshape_SurfaceArea": 256.0,
        },
        metadata={
            "Metadata_Object_ObjectID": 42,
            "Metadata_Imaging_ExposureTime": 50.0,
            "Metadata_Microscopy_Magnification": 60.0,
        },
    )


@pytest.fixture
def large_image_profile() -> Profile:
    """Create a profile with a large 3D image.

    Returns
    -------
    Profile
        A profile with image shape ``(32, 64, 64)`` and a richer collection of
        features and metadata fields.

    """
    return Profile(
        image_array=np.random.rand(32, 64, 64).astype(np.float32),
        features={
            "Nuclei_DNA_Intensity_MeanIntensity": 0.435,
            "Nuclei_DNA_Intensity_MaxIntensity": 0.998,
            "Nuclei_DNA_Intensity_MinIntensity": 0.001,
            "Nuclei_DNA_Intensity_StdIntensity": 0.187,
            "Nuclei_DNA_Intensity_MedianIntensity": 0.442,
            "Nuclei_DNA_Texture_Entropy-256-3": 6.145,
            "Nuclei_DNA_Texture_Gabor-3-0": 0.234,
            "Cell_DNA_Areasizeshape_Volume": 2048.0,
            "Cell_DNA_Areasizeshape_SurfaceArea": 1024.0,
            "Cell_DNA_Areasizeshape_Sphericity": 0.856,
            "Cytoplasm_DNA_Intensity_MeanIntensity": 0.312,
        },
        metadata={
            "Metadata_Object_ObjectID": 123,
            "Metadata_Storage_FilePath": "/data/images/cell_001.tif",
            "Metadata_Imaging_ExposureTime": 25.0,
            "Metadata_Microscopy_Magnification": 60.0,
            "Metadata_Biology_CellType": "NPC",
            "Metadata_Experiment_Treatment": "Control",
        },
    )


@pytest.fixture
def intensity_profile() -> Profile:
    """Create a profile focused on intensity features.

    Returns
    -------
    Profile
        A profile emphasizing intensity-derived measurements across multiple
        channels and compartments.

    """
    return Profile(
        image_array=np.random.rand(16, 48, 48).astype(np.float32),
        features={
            "Nuclei_DNA_Intensity_MeanIntensity": 0.654,
            "Nuclei_DNA_Intensity_MaxIntensity": 0.989,
            "Nuclei_DNA_Intensity_StdIntensity": 0.156,
            "Nuclei_Mito_Intensity_MeanIntensity": 0.412,
            "Nuclei_Mito_Intensity_MaxIntensity": 0.956,
            "Nuclei_ER_Intensity_MeanIntensity": 0.378,
            "Cell_DNA_Intensity_MeanIntensity": 0.523,
            "Cell_Mito_Intensity_MeanIntensity": 0.387,
            "Cell_ER_Intensity_MeanIntensity": 0.445,
        },
        metadata={
            "Metadata_Object_ObjectID": 456,
            "Metadata_Imaging_ExposureTime": 30.0,
            "Metadata_Microscopy_Magnification": 100.0,
        },
    )


@pytest.fixture
def texture_profile() -> Profile:
    """Create a profile focused on texture features.

    Returns
    -------
    Profile
        A profile containing entropy, gabor, and contrast style texture
        measurements.

    """
    return Profile(
        image_array=np.random.rand(12, 40, 40).astype(np.float32),
        features={
            "Nuclei_DNA_Texture_Entropy-256-3": 5.678,
            "Nuclei_DNA_Texture_Gabor-3-0": 0.234,
            "Nuclei_DNA_Texture_Gabor-3-45": 0.198,
            "Nuclei_DNA_Texture_Gabor-3-90": 0.212,
            "Nuclei_DNA_Texture_Gabor-3-135": 0.205,
            "Cytoplasm_Mito_Texture_Entropy-256-3": 4.892,
            "Cytoplasm_Mito_Texture_Contrast-3": 0.456,
            "Cell_DNA_Texture_Entropy-256-3": 6.123,
        },
        metadata={
            "Metadata_Object_ObjectID": 789,
            "Metadata_Imaging_ExposureTime": 50.0,
        },
    )


@pytest.fixture
def morphology_profile() -> Profile:
    """Create a profile focused on area-size-shape features.

    Returns
    -------
    Profile
        A profile emphasizing morphology metrics such as volume, surface area,
        and sphericity.

    """
    return Profile(
        image_array=np.random.rand(20, 56, 56).astype(np.float32),
        features={
            "Nuclei_DNA_Areasizeshape_Volume": 1024.0,
            "Nuclei_DNA_Areasizeshape_SurfaceArea": 512.0,
            "Nuclei_DNA_Areasizeshape_Sphericity": 0.892,
            "Nuclei_DNA_Areasizeshape_Solidity": 0.945,
            "Nuclei_DNA_Areasizeshape_Eccentricity": 0.342,
            "Nuclei_DNA_Areasizeshape_EulerCharacteristic": 1.0,
            "Cell_DNA_Areasizeshape_Volume": 4096.0,
            "Cell_DNA_Areasizeshape_SurfaceArea": 2048.0,
            "Cell_DNA_Areasizeshape_Sphericity": 0.756,
        },
        metadata={
            "Metadata_Object_ObjectID": 321,
            "Metadata_Storage_FilePath": "/data/images/cell_002.tif",
            "Metadata_Imaging_ExposureTime": 40.0,
        },
    )


@pytest.fixture
def colocalization_profile() -> Profile:
    """Create a profile focused on colocalization features.

    Returns
    -------
    Profile
        A profile containing correlation and overlap metrics for paired channel
        combinations.

    """
    return Profile(
        image_array=np.random.rand(14, 44, 44).astype(np.float32),
        features={
            "Cell_DNA-Mito_Colocalization_Correlation": 0.623,
            "Cell_DNA-Mito_Colocalization_Overlap": 0.456,
            "Cell_DNA-ER_Colocalization_Correlation": 0.234,
            "Cell_DNA-ER_Colocalization_Overlap": 0.178,
            "Cell_Mito-ER_Colocalization_Correlation": 0.567,
            "Cell_Mito-ER_Colocalization_Overlap": 0.389,
            "Nuclei_DNA-Mito_Colocalization_Correlation": 0.345,
        },
        metadata={
            "Metadata_Object_ObjectID": 654,
            "Metadata_Imaging_ExposureTime": 60.0,
        },
    )


@pytest.fixture
def granularity_profile() -> Profile:
    """Create a profile focused on granularity features.

    Returns
    -------
    Profile
        A profile with a sequence of granularity spectrum measurements.

    """
    return Profile(
        image_array=np.random.rand(16, 48, 48).astype(np.float32),
        features={
            "Nuclei_DNA_Granularity_Spectrum-1": 0.234,
            "Nuclei_DNA_Granularity_Spectrum-2": 0.256,
            "Nuclei_DNA_Granularity_Spectrum-3": 0.289,
            "Nuclei_DNA_Granularity_Spectrum-4": 0.312,
            "Nuclei_DNA_Granularity_Spectrum-5": 0.334,
            "Nuclei_DNA_Granularity_Spectrum-6": 0.345,
            "Nuclei_DNA_Granularity_Spectrum-7": 0.356,
            "Nuclei_DNA_Granularity_Spectrum-8": 0.362,
            "Nuclei_DNA_Granularity_Spectrum-9": 0.365,
            "Nuclei_DNA_Granularity_Spectrum-10": 0.367,
        },
        metadata={
            "Metadata_Object_ObjectID": 987,
            "Metadata_Imaging_ExposureTime": 35.0,
        },
    )


@pytest.fixture
def neighbors_profile() -> Profile:
    """Create a profile focused on neighbor-based features.

    Returns
    -------
    Profile
        A profile with adjacency, neighbor count, and nearest-distance style
        neighborhood features.

    """
    return Profile(
        image_array=np.random.rand(10, 32, 32).astype(np.float32),
        features={
            "Nuclei_NoChannel_Neighbors_AdjacentCount": 6.0,
            "Nuclei_NoChannel_Neighbors_NumberOfNeighbors": 8.0,
            "Nuclei_NoChannel_Neighbors_DistanceClosestNeighbor": 15.234,
            "Nuclei_NoChannel_Neighbors_PercentTouching": 0.35,
            "Cell_NoChannel_Neighbors_AdjacentCount": 12.0,
            "Cell_NoChannel_Neighbors_NumberOfNeighbors": 15.0,
        },
        metadata={
            "Metadata_Object_ObjectID": 110,
            "Metadata_Neighbors_AdjacentCount": 6,
        },
    )


@pytest.fixture
def complete_profile() -> Profile:
    """Create a comprehensive profile spanning feature categories.

    Returns
    -------
    Profile
        A profile that mixes intensity, texture, morphology, granularity,
        colocalization, and neighbor-derived measurements.

    """
    return Profile(
        image_array=np.random.rand(24, 64, 64).astype(np.float32),
        features={
            # Intensity
            "Nuclei_DNA_Intensity_MeanIntensity": 0.543,
            "Nuclei_DNA_Intensity_MaxIntensity": 0.987,
            "Cell_DNA_Intensity_MeanIntensity": 0.421,
            # Texture
            "Nuclei_DNA_Texture_Entropy-256-3": 5.892,
            "Nuclei_DNA_Texture_Gabor-3-0": 0.234,
            # Morphology
            "Nuclei_DNA_Areasizeshape_Volume": 1536.0,
            "Nuclei_DNA_Areasizeshape_Sphericity": 0.865,
            # Granularity
            "Nuclei_DNA_Granularity_Spectrum-5": 0.334,
            # Colocalization
            "Cell_DNA-Mito_Colocalization_Correlation": 0.512,
            # Neighbors
            "Nuclei_NoChannel_Neighbors_AdjacentCount": 5.0,
        },
        metadata={
            "Metadata_Storage_FilePath": "/data/images/sample_001.tif",
            "Metadata_Object_ObjectID": 100,
            "Metadata_Biology_CellType": "Neuron",
            "Metadata_Imaging_ExposureTime": 45.0,
            "Metadata_Microscopy_Magnification": 63.0,
            "Metadata_Experiment_Treatment": "Drug_A",
            "Metadata_Location_CentroidX": 156.5,
            "Metadata_Location_CentroidY": 234.2,
            "Metadata_Location_CentroidZ": 12.8,
        },
    )


# COLLECTION FIXTURES: Groups of profiles


@pytest.fixture
def all_profiles(
    minimal_profile: Profile,
    small_image_profile: Profile,
    medium_image_profile: Profile,
    large_image_profile: Profile,
) -> list[Profile]:
    """Collect profiles spanning multiple image sizes.

    Parameters
    ----------
    minimal_profile : Profile
        Fixture that provides the minimal valid profile.
    small_image_profile : Profile
        Fixture that provides a small image profile.
    medium_image_profile : Profile
        Fixture that provides a medium image profile.
    large_image_profile : Profile
        Fixture that provides a large image profile.

    Returns
    -------
    list[Profile]
        A list of profiles ordered to provide size diversity for parameterized
        tests.

    """
    return [
        small_image_profile,
        minimal_profile,
        medium_image_profile,
        large_image_profile,
    ]


@pytest.fixture
def all_feature_type_profiles(
    request: pytest.FixtureRequest,
) -> list[Profile]:
    """Collect profiles grouped by feature category.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object used to resolve fixtures dynamically by
        name.

    Returns
    -------
    list[Profile]
        Profiles specialized for intensity, texture, morphology,
        colocalization, granularity, and neighbor-based tests.

    """
    fixture_names = [
        "intensity_profile",
        "texture_profile",
        "morphology_profile",
        "colocalization_profile",
        "granularity_profile",
        "neighbors_profile",
    ]
    return [request.getfixturevalue(fixture_name) for fixture_name in fixture_names]


# PARAMETERIZED FIXTURE DATA


@pytest.fixture(
    params=[
        (4, 8, 8),
        (8, 16, 16),
        (12, 24, 24),
        (16, 32, 32),
        (20, 40, 40),
    ],
)
def varying_image_sizes(request: pytest.FixtureRequest) -> tuple[int, int, int]:
    """Provide parameterized image dimensions for profile generation.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request containing the current ``params`` tuple.

    Returns
    -------
    tuple[int, int, int]
        A 3-tuple ``(z, y, x)`` describing image dimensions for test cases.

    """
    return request.param


@pytest.fixture
def profile_with_varying_size(
    varying_image_sizes: tuple[int, int, int],
) -> Profile:
    """Create a profile from the current parameterized image size.

    Parameters
    ----------
    varying_image_sizes : tuple[int, int, int]
        Image dimensions as ``(z, y, x)`` supplied by
        ``varying_image_sizes``.

    Returns
    -------
    Profile
        A profile containing a random 3D float32 image of the requested shape,
        a minimal feature set, and minimal metadata.

    """
    z, y, x = varying_image_sizes
    return Profile(
        image_array=np.random.rand(z, y, x).astype(np.float32),
        features={"Nuclei_DNA_Intensity_MeanIntensity": 0.512},
        metadata={"Metadata_Object_ObjectID": 1},
    )
