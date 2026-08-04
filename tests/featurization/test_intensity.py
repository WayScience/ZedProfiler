from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from beartype import beartype
from pydantic import BaseModel, ConfigDict, field_validator

from zedprofiler.featurization.intensity import compute_intensity


class ImageSetLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_set_name: str = "intensity"


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
    def to_array(_cls, v: object) -> np.ndarray:
        return np.asarray(v)


@beartype
def make_label_and_image(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros(shape, dtype=float)
    label = np.zeros(shape, dtype=int)
    z, y, x = center
    image[z, y, x] = 50.0
    label[z, y, x] = 1
    return image, label


def test_min_intensity_edge_not_zero_for_bright_cell() -> None:
    """MinIntensityEdge must reflect actual boundary intensities, not 0.

    Regression test for a bug where get_outline used find_boundaries with the
    default mode='thick', which returns both inner (object-side) and outer
    (background-side) boundary pixels. Because the image is zeroed outside the
    cell before the outline is computed, outer boundary pixels always have
    intensity 0, making numpy.min always return 0.

    The fix is to use mode='inner' so only pixels inside the object boundary
    are included in the edge mask.
    """
    shape = (10, 10, 10)
    image = np.zeros(shape, dtype=np.float32)
    label = np.zeros(shape, dtype=np.int32)

    image[3:7, 3:7, 3:7] = 100.0
    label[3:7, 3:7, 3:7] = 1

    imgset = ImageSetLoaderModel()
    loader = ObjectLoaderModel(
        image=image,
        label_image=label,
        object_ids=[1],
        image_set_loader=imgset,
    )

    df = compute_intensity(loader)

    row = df[(df["Metadata_Object_ObjectID"] == 1)]
    min_edge_col = [c for c in df.columns if "MinIntensityEdge" in c]
    assert min_edge_col, "MinIntensityEdge column not found in output"

    min_edge = row[min_edge_col[0]].values[0]

    assert min_edge != 0, (
        "MinIntensityEdge is 0 for a cell with uniform intensity 100. "
        "Likely caused by find_boundaries(mode='thick') including outer "
        "(background) boundary pixels that have been zeroed."
    )
    assert min_edge == pytest.approx(100.0), (
        f"MinIntensityEdge should be 100 (cell intensity) but got {min_edge}"
    )


def test_integrated_intensity_is_per_object_not_global() -> None:
    """IntegratedIntensity must reflect only the target object's voxels (Bug 10).

    Before the fix, scipy.ndimage.sum was called without an explicit ``index``
    argument. With a binarised label array (values 0/1), omitting ``index``
    causes ndimage.sum to return the total over all labelled pixels, which can
    include voxels from other objects if the label array was not fully isolated
    beforehand. Passing ``index=1`` constrains the sum to only the pixels
    belonging to label 1.

    The test constructs an image where one object (label 1, intensity 1.0) is
    isolated from a second brighter region that is masked out. Without the fix,
    the sum bleeds into the second region and the integrated intensity is too
    large.
    """
    shape = (10, 10, 10)
    image = np.zeros(shape, dtype=np.float32)
    label = np.zeros(shape, dtype=np.int32)

    # Object 1: single bright voxel, intensity 1.0
    image[2, 2, 2] = 1.0
    label[2, 2, 2] = 1

    # A second region that is NOT labelled (background) but has high intensity.
    # If scipy.ndimage.sum leaks into it, IntegratedIntensity will be inflated.
    image[7, 7, 7] = 100.0

    imgset = ImageSetLoaderModel()
    loader = ObjectLoaderModel(
        image=image,
        label_image=label,
        object_ids=[1],
        image_set_loader=imgset,
    )

    df = compute_intensity(loader)
    ii_col = [c for c in df.columns if "IntegratedIntensity" in c and "Edge" not in c]
    assert ii_col, "IntegratedIntensity column not found in output"

    row = df[df["Metadata_Object_ObjectID"] == 1]
    ii = float(row[ii_col[0]].values[0])
    expected = 1.0
    assert np.isclose(ii, expected, atol=1e-3), (
        f"IntegratedIntensity = {ii}, expected {expected}. "
        "Likely caused by scipy.ndimage.sum without index=1 summing all pixels."
    )


@pytest.mark.parametrize("shape,center", [((6, 6, 6), (3, 3, 3))])
def test_compute_intensity_basic(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
) -> None:
    img, lab = make_label_and_image(shape, center)
    imgset = ImageSetLoaderModel()
    loader = ObjectLoaderModel(
        image=img,
        label_image=lab,
        object_ids=[1],
        image_set_loader=imgset,
    )

    df = compute_intensity(loader)
    assert isinstance(df, pd.DataFrame)
    assert "Metadata_Object_ObjectID" in df.columns
