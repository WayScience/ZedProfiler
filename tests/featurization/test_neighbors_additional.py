from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zedprofiler.featurization.neighbors import (
    calculate_centroid,
    classify_cells_into_shells,
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
from zedprofiler.IO.feature_writing_utils import format_morphology_feature_name

scipy = pytest.importorskip("scipy")
skimage = pytest.importorskip("skimage")

ANISOTROPY_FACTORS = [1, 2, 5, 10]


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


@pytest.mark.parametrize("anisotropy_factor", ANISOTROPY_FACTORS)
def test_compute_neighbors_distance_counts(anisotropy_factor: int) -> None:
    # Create a label image with three objects: two nearby, one far
    lab = np.zeros((12, 12, 12), dtype=int)
    lab[2, 2, 2] = 1
    lab[2, 2, 4] = 2
    lab[10, 10, 10] = 3

    class Dummy:
        label_image = lab
        object_ids = (1, 2, 3)
        image_set_loader = type("ISL", (), {"image_set_name": "s", "image_id": "s"})()
        compartment = "Cell"
        channel = "Ch1"

    df = compute_neighbors(
        Dummy(),
        distance_threshold=3,
        anisotropy_factor=anisotropy_factor,
    )
    assert isinstance(df, pd.DataFrame)
    # For object 1 and 2, distance-based neighbors should count each other
    distance_threshold = 3
    col = format_morphology_feature_name(
        compartment="Cell",
        channel="Ch1",
        feature_type="Neighbors",
        measurement=f"NeighborsCountByDistance-{distance_threshold}",
    )
    vals = df[col].tolist()
    # find values for labels in order
    assert vals[0] >= 1
    assert vals[1] >= 1
    # object 3 is far -> zero neighbors by distance
    assert vals[2] == 0


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


def test_classify_cells_into_shells_empty_and_reduction_and_methods() -> None:
    # empty coords
    res, cent = classify_cells_into_shells(
        pd.DataFrame(columns=["Metadata_Object_ObjectID", "x", "y", "z"]),
    )
    assert res["Metadata_Object_ObjectID"] == []
    assert cent is None

    # small set with requested large n_shells -> will reduce (intentional UserWarning)
    coords = {
        "Metadata_Object_ObjectID": [1, 2, 3, 4, 5, 6],
        "x": [0, 1, 2, 3, 4, 5],
        "y": [0, 1, 2, 3, 4, 5],
        "z": [0, 1, 2, 3, 4, 5],
    }
    with pytest.warns(UserWarning):
        results, _centroid = classify_cells_into_shells(
            coords,
            n_shells=10,
            method="euclidean",
            min_cells_per_shell=3,
        )
    # max_shells = max(2, 6//3==2) -> will reduce to 2
    expected_shells = 2
    assert results["ShellsUsed"] == expected_shells


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
