"""Shared deterministic feature cases for accuracy and performance tests."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from zedprofiler.featurization.colocalization import compute_colocalization
from zedprofiler.featurization.granularity import compute_granularity
from zedprofiler.featurization.intensity import compute_intensity
from zedprofiler.featurization.neighbors import compute_neighbors
from zedprofiler.featurization.texture import compute_texture
from zedprofiler.featurization.volumesizeshape import compute_volume_size_shape


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
        {
            column: _normalize_scalar(value, precision)
            for column, value in row.items()
        }
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
