from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import skimage.measure
from beartype import beartype
from pydantic import BaseModel, ConfigDict, field_validator

mahotas = pytest.importorskip("mahotas")

from zedprofiler.featurization.texture import compute_texture, scale_image  # noqa: E402

LABEL_OBJ_1 = 1
LABEL_OBJ_2 = 2


class ImageSetLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_set_name: str = "texture"


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
def make_texture_image(
    shape: tuple[int, int, int], center: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros(shape, dtype=int)
    label = np.zeros(shape, dtype=int)
    z, y, x = center
    image[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = 100
    label[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = 1
    return image, label


@pytest.mark.parametrize("shape,center", [((15, 15, 15), (7, 7, 7))])
def test_compute_texture_basic(
    shape: tuple[int, int, int], center: tuple[int, int, int]
) -> None:
    image, label = make_texture_image(shape, center)
    imgset = ImageSetLoaderModel()
    loader = ObjectLoaderModel(
        image=image, label_image=label, object_ids=[1], image_set_loader=imgset
    )

    df = compute_texture(loader, distance=1, grayscale=256)
    assert isinstance(df, pd.DataFrame)
    assert "Metadata_Object_ObjectID" in df.columns


def test_value_error_in_one_object_does_not_corrupt_others() -> None:
    """When mahotas raises ValueError for one object, other objects must be unaffected.

    Regression test for a bug where the except-ValueError branch executed:
        features = numpy.full(len(feature_names), numpy.nan, dtype=float)
    This replaced the entire 3D features array with a 1D array, destroying all
    previously-computed labels and corrupting the iteration that follows.

    A single-pixel object has no valid pixel pairs at any distance when
    ignore_zeros=True, so it reliably triggers ValueError. Object 1 (a normal
    4x4x4 cube) is processed first and must retain its Haralick values even
    after object 2 (single pixel) raises ValueError.
    """
    shape = (20, 20, 20)
    image = np.zeros(shape, dtype=np.uint8)
    label = np.zeros(shape, dtype=np.int32)

    image[1:5, 1:5, 1:5] = 200
    label[1:5, 1:5, 1:5] = LABEL_OBJ_1

    # single pixel — no neighbour pairs → ValueError from mahotas
    image[15, 15, 15] = 100
    label[15, 15, 15] = LABEL_OBJ_2

    imgset = ImageSetLoaderModel()
    loader = ObjectLoaderModel(
        image=image,
        label_image=label,
        object_ids=[LABEL_OBJ_1, LABEL_OBJ_2],
        image_set_loader=imgset,
    )

    df = compute_texture(loader, distance=1, grayscale=256)

    obj1_rows = df[df["Metadata_Object_ObjectID"] == LABEL_OBJ_1]
    assert not obj1_rows.empty, "Object 1 missing from output after object 2 ValueError"

    asm_cols = [c for c in df.columns if "AngularSecondMoment" in c and "-00-" in c]
    assert asm_cols, "No AngularSecondMoment direction-00 column found"

    obj1_asm = obj1_rows[asm_cols[0]].iloc[0]
    assert not np.isnan(obj1_asm), (
        "Object 1 AngularSecondMoment is NaN after object 2 raised ValueError. "
        "The except branch corrupted the shared features array."
    )

    obj2_rows = df[df["Metadata_Object_ObjectID"] == LABEL_OBJ_2]
    assert not obj2_rows.empty, "Object 2 missing from output"
    obj2_asm = obj2_rows[asm_cols[0]].iloc[0]
    assert np.isnan(obj2_asm), "Object 2 (single pixel) should have NaN texture values"


def test_texture_values_correct_per_object() -> None:
    """Each object must receive its own Haralick values, not another object's.

    Regression test for a bug where features = numpy.empty(...) was called
    inside the per-object loop, resetting the array on every iteration so only
    the last object's values survived. All prior objects received uninitialized
    memory from numpy.empty.

    Two objects with distinct pixel intensities are profiled. The test
    independently computes expected Haralick values per object using mahotas
    directly and asserts the pipeline output matches. It fails on the buggy code
    because object 1 gets garbage values instead of its own.
    """
    # Object 1: bright uniform region (intensity 200)
    # Object 2: dim uniform region (intensity 50)
    # AngularSecondMoment for a uniform region is 1.0; if the array is reset
    # inside the loop, object 1 gets uninitialized values instead.
    shape = (20, 20, 20)
    image = np.zeros(shape, dtype=np.uint8)
    label = np.zeros(shape, dtype=np.int32)

    image[1:5, 1:5, 1:5] = 200
    label[1:5, 1:5, 1:5] = LABEL_OBJ_1

    image[14:18, 14:18, 14:18] = 50
    label[14:18, 14:18, 14:18] = LABEL_OBJ_2

    imgset = ImageSetLoaderModel()
    loader = ObjectLoaderModel(
        image=image,
        label_image=label,
        object_ids=[LABEL_OBJ_1, LABEL_OBJ_2],
        image_set_loader=imgset,
    )

    df = compute_texture(loader, distance=1, grayscale=256)

    # Build per-object lookup: {object_id: {feature_col: value}}
    obj_textures: dict[int, dict[str, float]] = {}
    for _, row in df.iterrows():
        obj_id = int(row["Metadata_Object_ObjectID"])
        obj_textures.setdefault(obj_id, {})
        for col in df.columns:
            if col != "Metadata_Object_ObjectID":
                obj_textures[obj_id][col] = row[col]

    assert LABEL_OBJ_1 in obj_textures, "Object 1 missing from texture output"
    assert LABEL_OBJ_2 in obj_textures, "Object 2 missing from texture output"

    for obj_id in [LABEL_OBJ_1, LABEL_OBJ_2]:
        mask = label == obj_id
        crop_img = image.copy()
        crop_img[~mask] = 0

        props = skimage.measure.regionprops_table(
            mask.astype(np.uint8), properties=["bbox"]
        )
        z0, y0, x0 = (
            int(props["bbox-0"][0]),
            int(props["bbox-1"][0]),
            int(props["bbox-2"][0]),
        )
        z1, y1, x1 = (
            int(props["bbox-3"][0]),
            int(props["bbox-4"][0]),
            int(props["bbox-5"][0]),
        )
        crop_scaled = scale_image(crop_img[z0:z1, y0:y1, x0:x1], num_gray_levels=256)

        expected = mahotas.features.haralick(
            ignore_zeros=True,
            f=crop_scaled,
            distance=1,
            compute_14th_feature=False,
        )
        expected_asm = expected[0, 0]

        asm_cols = [
            c
            for c in obj_textures[obj_id]
            if "AngularSecondMoment" in c and "-00-" in c
        ]
        assert asm_cols, (
            f"No AngularSecondMoment-direction-00 column for object {obj_id}"
        )
        actual_asm = obj_textures[obj_id][asm_cols[0]]

        assert np.isclose(actual_asm, expected_asm, rtol=1e-5), (
            f"Object {obj_id} AngularSecondMoment: "
            f"got {actual_asm}, expected {expected_asm}. "
            f"Likely caused by numpy.empty reset inside the per-object loop."
        )
