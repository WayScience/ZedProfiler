"""Shared deterministic feature cases for accuracy and performance tests."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from zedprofiler.featurization.colocalization import compute_colocalization
from zedprofiler.featurization.granularity import compute_granularity
from zedprofiler.featurization.intensity import compute_intensity
from zedprofiler.featurization.neighbors import compute_neighbors
from zedprofiler.featurization.texture import compute_texture
from zedprofiler.featurization.volumesizeshape import compute_volume_size_shape
from zedprofiler.IO.loading_classes import (
    ImageSetConfig,
    ImageSetLoader,
    ObjectLoader,
    TwoObjectLoader,
)

try:
    import tifffile
except ImportError:  # pragma: no cover - exercised only without optional test dep
    tifffile = None


CELLPROFILER_TUTORIAL_ROOT = (
    Path(__file__).resolve().parent
    / "data"
    / "CP_tutorial_3D_noise_nuclei_segmentation"
)


@dataclass
class BenchmarkImageSet:
    """Minimal image-set loader surface used by feature functions."""

    image_set_name: str = "benchmark-level"
    anisotropy_spacing: tuple[float, float, float] = (10.0, 1.0, 1.0)


@dataclass
class BenchmarkObjectLoader:
    """Minimal object loader for single-channel feature functions."""

    image_set_loader: BenchmarkImageSet
    image: np.ndarray
    label_image: np.ndarray
    object_ids: list[int]
    compartment: str = "Nuclei"
    channel: str = "DNA"


@dataclass
class BenchmarkTwoObjectLoader:
    """Minimal object loader for paired-channel feature functions."""

    image_set_loader: BenchmarkImageSet
    image1: np.ndarray
    image2: np.ndarray
    label_image: np.ndarray
    object_ids: list[int]
    compartment: str = "Nuclei"


FeatureCase = tuple[str, Callable[[], pd.DataFrame]]


def make_benchmark_loaders() -> tuple[BenchmarkObjectLoader, BenchmarkTwoObjectLoader]:
    """Create a deterministic two-object image set for benchmark contracts."""
    shape = (16, 32, 32)
    z, y, x = np.indices(shape)
    image1 = ((z * 11 + y * 7 + x * 5) % 251 + 1).astype(np.uint16)
    image2 = ((image1.astype(np.uint32) * 3 + z * 13 + x * 17) % 251 + 1).astype(
        np.uint16,
    )

    labels = np.zeros(shape, dtype=np.int32)
    labels[2:7, 3:8, 4:9] = 1
    labels[9:14, 20:26, 18:24] = 2
    object_ids = [1, 2]

    image_set = BenchmarkImageSet()
    object_loader = BenchmarkObjectLoader(
        image_set_loader=image_set,
        image=image1,
        label_image=labels,
        object_ids=object_ids,
    )
    two_object_loader = BenchmarkTwoObjectLoader(
        image_set_loader=image_set,
        image1=image1,
        image2=image2,
        label_image=labels,
        object_ids=object_ids,
    )
    return object_loader, two_object_loader


def make_grid_benchmark_loaders(
    *,
    image_set_name: str = "benchmark-grid",
    shape: tuple[int, int, int] = (32, 128, 128),
    object_count: int = 32,
    cube_size: int = 4,
) -> tuple[BenchmarkObjectLoader, BenchmarkTwoObjectLoader]:
    """Create a deterministic many-object image set for scaling scorecards."""
    z, y, x = np.indices(shape)
    image1 = ((z * 11 + y * 7 + x * 5) % 4093 + 1).astype(np.uint16)
    image2 = ((image1.astype(np.uint32) * 3 + z * 13 + x * 17) % 4093 + 1).astype(
        np.uint16,
    )

    labels = np.zeros(shape, dtype=np.int32)
    object_ids: list[int] = []
    object_id = 1
    for z_start in range(1, shape[0] - cube_size, cube_size + 2):
        for y_start in range(1, shape[1] - cube_size, cube_size + 4):
            for x_start in range(1, shape[2] - cube_size, cube_size + 4):
                labels[
                    z_start : z_start + cube_size,
                    y_start : y_start + cube_size,
                    x_start : x_start + cube_size,
                ] = object_id
                object_ids.append(object_id)
                object_id += 1
                if len(object_ids) >= object_count:
                    image_set = BenchmarkImageSet(image_set_name=image_set_name)
                    return (
                        BenchmarkObjectLoader(
                            image_set_loader=image_set,
                            image=image1,
                            label_image=labels,
                            object_ids=object_ids,
                        ),
                        BenchmarkTwoObjectLoader(
                            image_set_loader=image_set,
                            image1=image1,
                            image2=image2,
                            label_image=labels,
                            object_ids=object_ids,
                        ),
                    )

    raise ValueError(
        f"Could not place {object_count} objects in shape {shape} "
        f"with cube size {cube_size}.",
    )


def feature_cases() -> list[FeatureCase]:
    """Return feature computations that should remain result-stable."""
    object_loader, two_object_loader = make_benchmark_loaders()
    return [
        ("intensity", lambda: compute_intensity(object_loader)),
        (
            "volume_size_shape",
            lambda: compute_volume_size_shape(
                image_set_loader=object_loader.image_set_loader,
                object_loader=object_loader,
            ),
        ),
        (
            "neighbors",
            lambda: compute_neighbors(
                object_loader=object_loader,
                distance_threshold=10,
                anisotropy_factor=10,
            ),
        ),
        ("texture", lambda: compute_texture(object_loader, distance=1, grayscale=256)),
        (
            "granularity",
            lambda: compute_granularity(
                object_loader,
                # The production default is 16, but granularity cost scales
                # linearly with spectrum length (one morphology pass per
                # scale). Kept small here so the accuracy lock stays fast and
                # its fingerprint stable; the scaling scorecard below uses the
                # realistic default of 16 for representative timing.
                granular_spectrum_length=3,
                subsample_size=0.5,
                image_sample_size=0.5,
                radius=2,
            ),
        ),
        (
            "colocalization",
            lambda: compute_colocalization(
                two_object_loader,
                fast_costes="Faster",
                channel1="DNA",
                channel2="GFP",
            ),
        ),
    ]


def scaling_feature_cases() -> list[FeatureCase]:
    """Return a many-object benchmark level for opt-in scorecards."""
    object_loader, two_object_loader = make_grid_benchmark_loaders()
    return [
        ("scaling_intensity", lambda: compute_intensity(object_loader)),
        (
            "scaling_volume_size_shape",
            lambda: compute_volume_size_shape(
                image_set_loader=object_loader.image_set_loader,
                object_loader=object_loader,
            ),
        ),
        (
            "scaling_neighbors",
            lambda: compute_neighbors(
                object_loader=object_loader,
                distance_threshold=10,
                anisotropy_factor=10,
            ),
        ),
        (
            "scaling_texture",
            # CellProfiler's default Haralick distance is 3; use it here so
            # the scorecard reflects representative cost. The anisotropy spacing
            # is (10, 1, 1), so 3 voxels in z is 30 physical units while 3 in
            # x/y is 3 — the GLCM distance is in voxel space and does not
            # adjust for anisotropy.
            lambda: compute_texture(object_loader, distance=3, grayscale=256),
        ),
        (
            "scaling_granularity",
            # Use the production default of 16 so the scaling scorecard
            # reflects representative granularity cost, which scales with
            # spectrum length. (No locked fingerprint for scaling cases.)
            lambda: compute_granularity(
                object_loader,
                granular_spectrum_length=16,
                subsample_size=0.5,
                image_sample_size=0.5,
                radius=2,
            ),
        ),
        (
            "scaling_colocalization",
            lambda: compute_colocalization(
                two_object_loader,
                fast_costes="Faster",
                channel1="DNA",
                channel2="GFP",
            ),
        ),
    ]


def _load_real_world_object_loader(
    *,
    image_name: str = "nuclei1_out_c00_dr90_image",
) -> ObjectLoader:
    """Load a representative real-world image/mask pair for scorecards."""
    if tifffile is None:
        raise ModuleNotFoundError("tifffile is required for real-world benchmarks.")

    image = tifffile.imread(CELLPROFILER_TUTORIAL_ROOT / "input" / f"{image_name}.tif")
    label = tifffile.imread(
        CELLPROFILER_TUTORIAL_ROOT
        / "output"
        / "masks"
        / f"{image_name}SegmentationMask.tiff",
    )
    image_set_loader = ImageSetLoader(
        image_set_path=None,
        label_set_path=None,
        image_set_array=image,
        label_set_array=label,
        anisotropy_spacing=(1.0, 1.0, 1.0),
        channel_mapping={
            "DNA": image_name,
            "Nuclei": "SegmentationMask",
        },
        config=ImageSetConfig(
            image_set_name=image_name,
            label_key_name=["Nuclei"],
            raw_image_key_name=["DNA"],
        ),
    )
    return ObjectLoader(
        image_set_loader=image_set_loader,
        channel_name="DNA",
        compartment_name="Nuclei",
    )


def _load_real_world_two_object_loader() -> TwoObjectLoader:
    """Load a representative paired-channel real-world case for scorecards."""
    if tifffile is None:
        raise ModuleNotFoundError("tifffile is required for real-world benchmarks.")

    first_image_name = "nuclei1_out_c00_dr90_image"
    second_image_name = "nuclei2_out_c90_dr90_image"
    label = tifffile.imread(
        CELLPROFILER_TUTORIAL_ROOT
        / "output"
        / "masks"
        / f"{first_image_name}SegmentationMask.tiff",
    )
    object_ids = [int(x) for x in np.unique(label) if x != 0]
    image_set_loader = ImageSetLoader.__new__(ImageSetLoader)
    image_set_loader.image_set_name = "real-world-dr90-c00-c90"
    image_set_loader.image_set_dict = {
        "DNA1": tifffile.imread(
            CELLPROFILER_TUTORIAL_ROOT / "input" / f"{first_image_name}.tif",
        ),
        "DNA2": tifffile.imread(
            CELLPROFILER_TUTORIAL_ROOT / "input" / f"{second_image_name}.tif",
        ),
        "Nuclei": label,
    }
    image_set_loader.unique_compartment_objects = {"Nuclei": object_ids}
    return TwoObjectLoader(
        image_set_loader=image_set_loader,
        compartment="Nuclei",
        channel1="DNA1",
        channel2="DNA2",
    )


def real_world_feature_cases() -> list[FeatureCase]:
    """Return real-world benchmark cases backed by checked-in tutorial data."""
    object_loader = _load_real_world_object_loader()
    two_object_loader = _load_real_world_two_object_loader()
    return [
        ("real_world_intensity", lambda: compute_intensity(object_loader)),
        (
            "real_world_volume_size_shape",
            lambda: compute_volume_size_shape(
                image_set_loader=object_loader.image_set_loader,
                object_loader=object_loader,
            ),
        ),
        (
            "real_world_neighbors",
            lambda: compute_neighbors(
                object_loader=object_loader,
                distance_threshold=50,
                anisotropy_factor=1,
            ),
        ),
        (
            "real_world_texture",
            lambda: compute_texture(object_loader, distance=1, grayscale=256),
        ),
        (
            "real_world_granularity",
            lambda: compute_granularity(
                object_loader,
                radius=1,
                granular_spectrum_length=2,
                subsample_size=1.0,
                image_sample_size=1.0,
            ),
        ),
        (
            "real_world_colocalization",
            lambda: compute_colocalization(
                two_object_loader,
                fast_costes="Faster",
                channel1="DNA1",
                channel2="DNA2",
            ),
        ),
    ]


def _normalize_scalar(value: object, precision: int) -> object:
    """Normalize dataframe scalar values for stable JSON fingerprints."""
    if pd.isna(value):
        return "NaN"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isinf(value):
            return str(value)
        rounded = round(value, precision)
        return 0.0 if rounded == 0 else rounded
    return value


def canonical_records(dataframe: pd.DataFrame, precision: int = 6) -> list[dict]:
    """Return deterministic records independent of dataframe column order."""
    canonical = dataframe.copy()
    if "Metadata_Object_ObjectID" in canonical.columns:
        canonical = canonical.sort_values("Metadata_Object_ObjectID")
    canonical = canonical.reindex(sorted(canonical.columns), axis=1).reset_index(
        drop=True,
    )
    return [
        {column: _normalize_scalar(value, precision) for column, value in row.items()}
        for row in canonical.to_dict(orient="records")
    ]


def dataframe_signature(dataframe: pd.DataFrame, precision: int = 6) -> str:
    """Hash a dataframe after deterministic normalization."""
    payload = json.dumps(
        canonical_records(dataframe, precision=precision),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def time_feature_cases(cases: Iterable[FeatureCase]) -> list[dict[str, object]]:
    """Run feature cases once and return a compact scorecard."""
    scorecard: list[dict[str, object]] = []
    for name, run_case in cases:
        start = time.perf_counter()
        dataframe = run_case()
        elapsed = time.perf_counter() - start
        scorecard.append(
            {
                "feature": name,
                "seconds": round(elapsed, 6),
                "rows": int(dataframe.shape[0]),
                "columns": int(dataframe.shape[1]),
                "signature": dataframe_signature(dataframe),
            },
        )
    return scorecard
