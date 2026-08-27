from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from beartype import beartype
from pydantic import BaseModel, ConfigDict, field_validator

from zedprofiler.featurization.granularity import (
    _labeled_voxel_positions,
    _subsample_3d,
    _upsample_3d,
    compute_granularity,
)

scipy = pytest.importorskip("scipy")

ANISOTROPY_SPACINGS = [
    (1.0, 1.0, 1.0),
    (2.0, 1.0, 1.0),
    (5.0, 1.0, 1.0),
    (10.0, 1.0, 1.0),
]


class ImageSetLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_set_name: str = "gran"
    # mirrors ImageSetLoader.image_id (falls back to image_set_name)
    image_id: str = "gran"
    # mirrors ImageSetLoader.anisotropy_spacing (z, y, x spacing)
    anisotropy_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)


class ObjectLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: np.ndarray
    label_image: np.ndarray
    object_ids: list[int]
    image_set_loader: ImageSetLoaderModel
    compartment: str = "Cell"
    channel: str = "Ch1"

    @field_validator("image", "label_image", mode="before")
    @classmethod
    def ensure_array(_cls, v: object) -> np.ndarray:
        return np.asarray(v)


@beartype
def make_image_and_label(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros(shape, dtype=float)
    label = np.zeros(shape, dtype=int)
    z, y, x = center
    image[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = 10.0
    label[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = 1
    return image, label


@pytest.mark.parametrize("shape,center", [((12, 12, 12), (6, 6, 6))])
@pytest.mark.parametrize("anisotropy_spacing", ANISOTROPY_SPACINGS)
def test_compute_granularity_basic(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
    anisotropy_spacing: tuple[float, float, float],
) -> None:
    img, lab = make_image_and_label(shape, center)
    imgset = ImageSetLoaderModel(anisotropy_spacing=anisotropy_spacing)
    loader = ObjectLoaderModel(
        image=img,
        label_image=lab,
        object_ids=[1],
        image_set_loader=imgset,
    )

    df = compute_granularity(loader, radius=1, granular_spectrum_length=4)
    assert isinstance(df, (pd.DataFrame,))
    # Expect Metadata_Object_ObjectID column
    assert "Metadata_Object_ObjectID" in df.columns


def test_none_image_returns_well_formed_empty_frame() -> None:
    """A degenerate loader with no image must not return a malformed frame.

    ObjectLoader sets ``image`` (and ``label_image``) to None when its channel
    (or compartment) is missing for a given image set. Before the fix,
    compute_granularity returned a bare ``pandas.DataFrame()`` with no columns
    at all in this case, which crashes any downstream merge that expects an
    ID column to key on.
    """
    imgset = SimpleNamespace(image_set_name="gran", image_id="gran")
    loader = SimpleNamespace(
        image=None,
        label_image=None,
        object_ids=[],
        image_set_loader=imgset,
        compartment="Cell",
        channel="Ch1",
    )

    df = compute_granularity(loader, radius=1, granular_spectrum_length=4)

    assert isinstance(df, pd.DataFrame)
    assert "Metadata_Object_ObjectID" in df.columns
    assert len(df) == 0


def test_subsample_and_upsample_roundtrip() -> None:
    data = np.arange(27.0).reshape((3, 3, 3))
    # subsample by factor 0.5 -> larger grid coords division
    subsampled = _subsample_3d(data, np.array([1.5, 1.5, 1.5]), 0.5, order=1)
    assert subsampled.ndim == data.ndim
    # upsample back to original shape
    up = _upsample_3d(subsampled, subsampled.shape, data.shape)
    assert up.shape == data.shape
    # values won't be identical due to interpolation, but structure preserved
    assert up.max() >= data.max() * 0.5


def test_compute_granularity_subsample_size_ge_1_uses_copy() -> None:
    # subsample_size >= 1 returns a copy path (no subsampling)
    shape = (6, 6, 6)
    img = np.zeros(shape, dtype=float)
    lab = np.zeros(shape, dtype=int)
    img[3, 3, 3] = 10.0
    lab[3, 3, 3] = 1

    class Dummy:
        image = img
        label_image = lab
        object_ids: ClassVar[list[int]] = [1]
        image_set_loader = type(
            "ISL",
            (),
            {
                "image_set_name": "s",
                "image_id": "s",
                "anisotropy_spacing": (1.0, 1.0, 1.0),
            },
        )()
        compartment = "Cell"
        channel = "Ch1"

    df = compute_granularity(
        Dummy(),
        radius=1,
        granular_spectrum_length=3,
        subsample_size=1.0,
    )
    assert isinstance(df, pd.DataFrame)
    assert "Metadata_Object_ObjectID" in df.columns


def test_compute_granularity_with_image_sample_size_background_path() -> None:
    # exercise branch where image_sample_size < 1 triggers background subsampling
    shape = (12, 12, 12)
    img = np.zeros(shape, dtype=float)
    lab = np.zeros(shape, dtype=int)
    img[6, 6, 6] = 20.0
    lab[6, 6, 6] = 1

    class Dummy:
        image = img
        label_image = lab
        object_ids: ClassVar[list[int]] = [1]
        image_set_loader = type(
            "ISL",
            (),
            {
                "image_set_name": "s",
                "image_id": "s",
                "anisotropy_spacing": (1.0, 1.0, 1.0),
            },
        )()
        compartment = "Cell"
        channel = "Ch1"

    # small image_sample_size will go through background subsampling branch
    df = compute_granularity(
        Dummy(),
        radius=1,
        granular_spectrum_length=4,
        subsample_size=0.5,
        image_sample_size=0.5,
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] >= 1


def test_compute_granularity_mask_handling_and_zero_volume_skips() -> None:
    # Provide a mask that excludes the object to trigger empty thresholds/path
    shape = (8, 8, 8)
    img = np.zeros(shape, dtype=float)
    lab = np.zeros(shape, dtype=int)
    img[4, 4, 4] = 50.0
    lab[4, 4, 4] = 1

    mask = np.zeros(shape, dtype=bool)  # exclude everything

    class Dummy:
        image = img
        label_image = lab
        object_ids: ClassVar[list[int]] = [1]
        image_set_loader = type(
            "ISL",
            (),
            {
                "image_set_name": "s",
                "image_id": "s",
                "anisotropy_spacing": (1.0, 1.0, 1.0),
            },
        )()
        compartment = "Cell"
        channel = "Ch1"

    # With mask excluding pixels, function should still run and return DataFrame
    df = compute_granularity(
        Dummy(),
        radius=1,
        granular_spectrum_length=3,
        image_mask=mask,
    )
    assert isinstance(df, pd.DataFrame)


def test_upsample_3d_no_division_by_zero_when_dim_is_one() -> None:
    """_upsample_3d must not raise ZeroDivisionError when any dim has size 1 (Bug 7).

    Before the fix, the per-axis scale factor was always computed as
    ``(subsampled_shape[k] - 1) / (original_shape[k] - 1)`` without guarding
    against ``original_shape[k] == 1``, which produces a zero denominator.
    """
    data = np.ones((1, 4, 4), dtype=float)
    try:
        result = _upsample_3d(data, data.shape, (1, 8, 8))
    except ZeroDivisionError as e:
        pytest.fail(f"ZeroDivisionError in _upsample_3d with dim-1 axis: {e}")
    assert result.shape == (1, 8, 8)


def test_granularity_no_crash_on_single_z_slice() -> None:
    """compute_granularity must not crash when the input has only one Z slice (Bug 7).

    The division-by-zero guard must also protect the background upsampling block
    inside compute_granularity, not just _upsample_3d.
    """
    shape = (1, 12, 12)
    img = np.zeros(shape, dtype=float)
    lab = np.zeros(shape, dtype=int)
    img[0, 5, 5] = 10.0
    lab[0, 5, 5] = 1

    class Dummy:
        image = img
        label_image = lab
        object_ids: ClassVar[list[int]] = [1]
        image_set_loader = type(
            "ISL",
            (),
            {
                "image_set_name": "s",
                "image_id": "s",
                "anisotropy_spacing": (1.0, 1.0, 1.0),
            },
        )()
        compartment = "Cell"
        channel = "Ch1"

    try:
        df = compute_granularity(Dummy(), radius=1, granular_spectrum_length=3)
    except ZeroDivisionError as e:
        pytest.fail(
            f"ZeroDivisionError in compute_granularity with single-Z image: {e}"
        )
    assert isinstance(df, pd.DataFrame)


def test_compute_granularity_preserves_sparse_label_ids() -> None:
    # Sparse labels should not be renumbered to 1..n internally.
    shape = (8, 8, 8)
    img = np.zeros(shape, dtype=float)
    lab = np.zeros(shape, dtype=int)
    img[2, 2, 2] = 10.0
    img[5, 5, 5] = 20.0
    lab[2, 2, 2] = 257
    lab[5, 5, 5] = 514

    class Dummy:
        image = img
        label_image = lab
        object_ids: ClassVar[list[int]] = [257, 514]
        image_set_loader = type(
            "ISL",
            (),
            {
                "image_set_name": "s",
                "image_id": "s",
                "anisotropy_spacing": (1.0, 1.0, 1.0),
            },
        )()
        compartment = "Cell"
        channel = "Ch1"

    df = compute_granularity(Dummy(), radius=1, granular_spectrum_length=2)
    assert isinstance(df, pd.DataFrame)
    assert sorted(df["Metadata_Object_ObjectID"].tolist()) == [257, 514]


def test_labeled_voxel_positions_matches_full_array_scan() -> None:
    """Gathering only labeled voxels must match scanning the whole array.

    This pins the core equivalence the granularity per-scale loop relies on
    for its performance optimization: scipy.ndimage.mean(image, labels,
    label_range) run on the full array must equal the same call restricted
    to _labeled_voxel_positions's gathered coordinates/label ids, for
    arbitrary (including sparse, non-contiguous) label placement.
    """
    rng = np.random.default_rng(0)
    shape = (6, 10, 12)
    labels = np.zeros(shape, dtype=int)
    labels[1, 2, 3] = 5
    labels[1, 2, 4] = 5
    labels[4, 8, 9] = 9
    image = rng.uniform(0, 100, size=shape)
    label_range = np.array([5, 9])

    coords, label_ids = _labeled_voxel_positions(labels)

    full_means = scipy.ndimage.mean(image, labels, label_range)
    gathered_means = scipy.ndimage.mean(image[coords], label_ids, label_range)

    np.testing.assert_allclose(full_means, gathered_means)


def test_labeled_voxel_positions_empty_when_no_labels() -> None:
    """An all-background label image yields empty coordinates/labels, not a crash."""
    labels = np.zeros((4, 4, 4), dtype=int)
    coords, label_ids = _labeled_voxel_positions(labels)
    assert label_ids.size == 0
    assert all(c.size == 0 for c in coords)


def test_sparse_upsample_matches_full_array_upsample() -> None:
    """Upsampling only labeled voxels must match upsampling the whole array.

    This pins the other equivalence the granularity per-scale loop's
    optimization relies on (the mean-gathering side is pinned by
    test_labeled_voxel_positions_matches_full_array_scan above): evaluating
    scipy.ndimage.map_coordinates only at the coordinates of labeled voxels,
    scaled into the subsampled array's coordinate space, must equal
    upsampling the *entire* subsampled array with _upsample_3d (the
    pre-optimization approach) and then indexing at those same voxels.
    """
    original_shape = (10, 14, 16)
    subsampled_shape = np.array([5.0, 7.0, 8.0])
    rng = np.random.default_rng(1)
    rec = rng.uniform(0, 100, size=(5, 7, 8))

    labels = np.zeros(original_shape, dtype=int)
    labels[1, 2, 3] = 5
    labels[1, 2, 4] = 5
    labels[8, 12, 14] = 9

    coords, _label_ids = _labeled_voxel_positions(labels)

    # Reference (pre-optimization): upsample the whole subsampled array,
    # then index at the labeled voxels.
    full_upsampled = _upsample_3d(rec, subsampled_shape, original_shape)
    expected = full_upsampled[coords]

    # Optimized: scale only the labeled voxels' coordinates into the
    # subsampled array's space, then map_coordinates just those points.
    k, i, j = (c.astype(float) for c in coords)
    if original_shape[0] > 1:
        k *= float(subsampled_shape[0] - 1) / float(original_shape[0] - 1)
    if original_shape[1] > 1:
        i *= float(subsampled_shape[1] - 1) / float(original_shape[1] - 1)
    if original_shape[2] > 1:
        j *= float(subsampled_shape[2] - 1) / float(original_shape[2] - 1)
    actual = scipy.ndimage.map_coordinates(rec, (k, i, j), order=1)

    np.testing.assert_allclose(actual, expected)


def test_compute_granularity_zero_objects_returns_empty_dataframe() -> None:
    """No labeled objects (nobjects == 0) must not crash the per-scale loop.

    Answers https://github.com/WayScience/ZedProfiler/pull/51#discussion_r3766497646:
    with an all-background label image, every ``nobjects > 0`` branch in
    compute_granularity is skipped, so no per-object measurements are ever
    recorded. The result is a zero-row DataFrame that still carries its
    Metadata_* columns, matching pre-optimization behavior.
    """
    shape = (8, 8, 8)
    img = np.zeros(shape, dtype=float)
    lab = np.zeros(shape, dtype=int)  # no labeled objects

    class Dummy:
        image = img
        label_image = lab
        object_ids: ClassVar[list[int]] = []
        image_set_loader = type(
            "ISL",
            (),
            {
                "image_set_name": "s",
                "image_id": "s",
                "anisotropy_spacing": (1.0, 1.0, 1.0),
            },
        )()
        compartment = "Cell"
        channel = "Ch1"

    df = compute_granularity(Dummy(), radius=1, granular_spectrum_length=3)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "Metadata_Object_ObjectID" in df.columns
    assert "Metadata_Imaging_ImageID" in df.columns
    assert "Metadata_Experiment_ImageSet" in df.columns


@pytest.mark.parametrize("shape,center", [((24, 48, 48), (12, 22, 32))])
@pytest.mark.parametrize("anisotropy_spacing", ANISOTROPY_SPACINGS)
def test_compute_granularity_sparse_object_in_larger_image(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
    anisotropy_spacing: tuple[float, float, float],
) -> None:
    """A small object far from the edges of a much larger, subsampled image.

    This is the scenario the labeled-voxel-only upsampling optimization
    targets: a labeled object occupying a small fraction of the image, with
    subsampling active (the production-default code path). Exercises a
    shape/object-size ratio far more extreme than the small synthetic cases
    elsewhere in this file, where the object is a large fraction of the image.
    """
    img, lab = make_image_and_label(shape, center)
    imgset = ImageSetLoaderModel(anisotropy_spacing=anisotropy_spacing)
    loader = ObjectLoaderModel(
        image=img,
        label_image=lab,
        object_ids=[1],
        image_set_loader=imgset,
    )

    granular_spectrum_length = 4
    df = compute_granularity(
        loader,
        radius=2,
        granular_spectrum_length=granular_spectrum_length,
        subsample_size=0.5,
        image_sample_size=0.5,
    )

    assert len(df) == 1
    value_cols = [
        c
        for c in df.columns
        if c
        not in (
            "Metadata_Object_ObjectID",
            "Metadata_Imaging_ImageID",
            "Metadata_Experiment_ImageSet",
        )
    ]
    # This is an end-to-end smoke test for the sparse-object code path, not a
    # value-equivalence check: erosion/reconstruction spectra don't have a
    # simple closed-form expected value to assert against here. Exact
    # numeric equivalence of the optimization itself (gathering only labeled
    # voxels instead of scanning/upsampling the whole image) is pinned
    # directly by test_labeled_voxel_positions_matches_full_array_scan and
    # test_sparse_upsample_matches_full_array_upsample above. So here we only
    # check that the pipeline produces one granularity column per requested
    # scale (granular_spectrum_length) and that none of them are NaN/inf,
    # which would indicate the sparse-voxel path silently dropped or
    # corrupted a scale's measurement.
    assert len(value_cols) == granular_spectrum_length
    assert np.isfinite(df[value_cols].to_numpy(dtype=float)).all()
