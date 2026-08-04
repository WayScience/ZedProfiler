from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from beartype import beartype
from pydantic import BaseModel, ConfigDict, field_validator

from zedprofiler.featurization.neighbors import (
    calculate_centroid,
    compute_neighbors,
    create_results_dataframe,
    crop_3D_image,
    euclidean_distance_from_centroid,
    get_coordinates,
    mahalanobis_distance_from_centroid,
    neighbors_expand_box,
    plot_distance_distributions,
    visualize_organoid_shells,
)


class ImageSetLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_set_name: str = "neighbors"


class ObjectLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    label_image: np.ndarray
    object_ids: list[int]
    image_set_loader: ImageSetLoaderModel
    compartment: str = "Cell"
    channel: str = "Ch1"

    @field_validator("label_image", mode="before")
    @classmethod
    def to_array(_cls, v: object) -> np.ndarray:
        return np.asarray(v)


@beartype
def make_two_labels(
    shape: tuple[int, int, int],
    centers: list[tuple[int, int, int]],
) -> np.ndarray:
    lab = np.zeros(shape, dtype=int)
    for i, (z, y, x) in enumerate(centers, start=1):
        lab[z, y, x] = i
    return lab


@pytest.mark.parametrize(
    "shape,centers",
    [
        ((10, 10, 10), [(3, 3, 3), (6, 6, 6)]),
    ],
)
def test_compute_neighbors_counts(
    shape: tuple[int, int, int],
    centers: list[tuple[int, int, int]],
) -> None:
    lab = make_two_labels(shape, centers)
    imgset = ImageSetLoaderModel()
    obj_ids = sorted(set(lab.ravel()) - {0})
    loader = ObjectLoaderModel(
        label_image=lab,
        object_ids=obj_ids,
        image_set_loader=imgset,
    )

    df = compute_neighbors(loader, distance_threshold=5, anisotropy_factor=1)
    assert isinstance(df, pd.DataFrame)
    assert "Metadata_Object_ObjectID" in df.columns


def test_neighbors_expand_box_bounds() -> None:
    # current_min - expand_by < min_coor -> clipped to min_coor
    min_coord = 0
    max_coord = 10
    a, _b = neighbors_expand_box(min_coord, max_coord, 1, 2, expand_by=5)
    assert a == 0
    # current_max + expand_by > max_coord -> clipped to max_coord
    _a2, b2 = neighbors_expand_box(min_coord, max_coord, 8, 9, expand_by=5)
    assert b2 == max_coord


def test_crop_3d_image_basic() -> None:
    img = np.arange(27).reshape((3, 3, 3))
    cropped = crop_3D_image(img, (1, 0, 0, 3, 2, 2))
    assert cropped.shape == (2, 2, 2)


def test_get_coordinates_and_distances_and_centroid() -> None:
    lab = np.zeros((5, 5, 5), dtype=int)
    lab[1, 1, 1] = 1
    lab[3, 3, 3] = 2

    coords = get_coordinates(lab, object_ids=[1, 2])
    assert list(coords["Metadata_Object_ObjectID"]) == [1, 2]

    centroid = calculate_centroid(coords[["x", "y", "z"]].to_numpy())
    assert centroid.shape == (3,)

    dists = euclidean_distance_from_centroid(
        coords[["x", "y", "z"]].to_numpy(),
        centroid,
    )
    expected_coordinate_count = len(coords)
    assert dists.shape[0] == expected_coordinate_count


def test_mahalanobis_small_and_regularized_and_singular() -> None:
    # small sample -> fallback to euclidean (intentional UserWarning expected)
    coords_small = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    centroid = np.mean(coords_small, axis=0)
    with pytest.warns(UserWarning):
        md_small = mahalanobis_distance_from_centroid(
            coords_small,
            centroid,
            min_cells_threshold=50,
        )
    ed_small = euclidean_distance_from_centroid(coords_small, centroid)
    assert np.allclose(md_small, ed_small)

    # regularized branch with many identical points -> singular covariance
    # -> pseudo-inverse (intentional UserWarning expected)
    repeated_count = 30
    coords_singular = np.tile(np.array([1.0, 2.0, 3.0]), (repeated_count, 1))
    centroid2 = np.mean(coords_singular, axis=0)
    with pytest.warns(UserWarning):
        md_sing = mahalanobis_distance_from_centroid(
            coords_singular,
            centroid2,
            min_cells_threshold=50,
        )
    assert md_sing.shape[0] == repeated_count
    # distances should be zeros because all identical
    assert np.allclose(md_sing, 0.0)


def test_neighbors_count_adjacent_detects_touching_cells() -> None:
    """NeighborsCountAdjacent must be > 0 for two directly touching objects (Bug 5).

    Before the fix, adjacency was detected by cropping to the regionprops bbox
    of each object and counting unique labels inside. Because regionprops uses
    exclusive-max indices (Python slice convention), the boundary voxel of the
    touching neighbour was excluded from the crop, making the count always 0.

    The fix replaces the bbox-crop approach with a 1-voxel morphological dilation
    of the object mask; any label in the dilated region that is not the object
    itself is a true neighbour.
    """
    shape = (10, 10, 10)
    lab = np.zeros(shape, dtype=int)
    # Object 1 occupies [2:5, 2:5, 2:5]; Object 2 occupies [5:8, 2:5, 2:5].
    # They share the plane at z=5, so they are face-adjacent (touching).
    lab[2:5, 2:5, 2:5] = 1
    lab[5:8, 2:5, 2:5] = 2

    imgset = ImageSetLoaderModel()
    loader = ObjectLoaderModel(
        label_image=lab,
        object_ids=[1, 2],
        image_set_loader=imgset,
    )

    df = compute_neighbors(loader, distance_threshold=5, anisotropy_factor=1)
    obj1_row = df[df["Metadata_Object_ObjectID"] == 1]
    adjacent_col = [c for c in df.columns if "NeighborsCountAdjacent" in c]
    assert adjacent_col, "NeighborsCountAdjacent column not found in output"

    n_adj = int(obj1_row[adjacent_col[0]].values[0])
    assert n_adj > 0, (
        "NeighborsCountAdjacent is 0 for two touching cells — "
        "likely caused by bbox-crop excluding the shared boundary voxel"
    )


def test_create_results_dataframe_and_errors_and_plots() -> None:
    # create a simple classification results dict
    results = {
        "Metadata_Object_ObjectID": np.array([1, 2, 3]),
        "ShellAssignments": np.array([0, 1, 1]),
        "DistancesFromCenter": np.array([0.0, 1.0, 2.0]),
        "DistancesFromExterior": np.array([2.0, 1.0, 0.0]),
        "NormalizedDistancesFromCenter": np.array([0.0, 0.5, 1.0]),
        "ShellsUsed": 2,
    }
    df = create_results_dataframe(results)
    assert isinstance(df, pd.DataFrame)

    with pytest.raises(ValueError):
        create_results_dataframe([])

    coords_df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "z": [0, 1, 2]})
    fig1 = visualize_organoid_shells(
        coords_df,
        results,
        centroid=np.array([1.0, 1.0, 1.0]),
    )
    assert hasattr(fig1, "axes")

    fig2 = plot_distance_distributions(results, n_shells=2)
    assert hasattr(fig2, "axes")
