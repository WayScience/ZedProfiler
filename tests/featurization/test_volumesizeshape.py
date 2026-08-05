from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from beartype import beartype
from pydantic import BaseModel, ConfigDict, field_validator

from zedprofiler.featurization.volumesizeshape import (
    compute_volume_size_shape,
)


class ImageSetLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    anisotropy_spacing: tuple[float, float, float]
    image_set_name: str = "testset"
    # mirrors ImageSetLoader.image_id (falls back to image_set_name)
    image_id: str = "testset"


class ObjectLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    label_image: np.ndarray
    object_ids: list[int]
    image_set_loader: ImageSetLoaderModel
    compartment: str = "Cell"
    channel: str = "Ch1"

    @field_validator("label_image", mode="before")
    @classmethod
    def ensure_ndarray(_cls, v: object) -> np.ndarray:
        return np.asarray(v)


@beartype
def make_label_image(
    shape: tuple[int, int, int],
    centers: list[tuple[int, int, int]],
) -> np.ndarray:
    lab = np.zeros(shape, dtype=int)
    for i, (z0, y0, x0) in enumerate(centers, start=1):
        z = int(z0)
        y = int(y0)
        x = int(x0)
        # small 3x3x3 cube
        lab[max(0, z - 1) : z + 2, max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = i
    return lab


@pytest.mark.parametrize(
    "shape,centers",
    [
        ((7, 7, 7), [(3, 3, 3)]),
        ((8, 8, 8), [(2, 2, 2), (5, 5, 5)]),
    ],
)
def test_compute_volume_size_shape_returns_dataframe(
    shape: tuple[int, int, int],
    centers: list[tuple[int, int, int]],
) -> None:
    imgset = ImageSetLoaderModel(anisotropy_spacing=(1.0, 1.0, 1.0))
    label = make_label_image(shape, centers)
    obj_ids = sorted(set(label.ravel()) - {0})
    loader = ObjectLoaderModel(
        label_image=label,
        object_ids=obj_ids,
        image_set_loader=imgset,
        compartment="Nucleus",
        channel="DAPI",
    )

    df = compute_volume_size_shape(image_set_loader=imgset, object_loader=loader)

    assert isinstance(df, pd.DataFrame)
    assert "Metadata_Object_ObjectID" in df.columns
    # All object ids present
    returned_ids = sorted(int(x) for x in df["Metadata_Object_ObjectID"].tolist())
    assert returned_ids == obj_ids
