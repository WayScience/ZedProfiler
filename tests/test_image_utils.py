import numpy as np
import pytest

from zedprofiler.image_utils.image_utils import (
    check_for_xy_squareness,
    crop_3D_image,
    expand_box,
    new_crop_border,
    select_objects_from_label,
    single_3D_image_expand_bbox,
    square_off_xy_crop_bbox,
)


def test_select_objects_from_label_filters_values() -> None:
    label_image = np.array([[0, 1, 2], [2, 1, 3]])
    selected = select_objects_from_label(label_image, 2)

    np.testing.assert_array_equal(selected, np.array([[0, 0, 2], [2, 0, 0]]))
    # Ensure the original array is unchanged.
    np.testing.assert_array_equal(label_image, np.array([[0, 1, 2], [2, 1, 3]]))


def test_expand_box_expands_from_min_edge_first() -> None:
    new_min, new_max = expand_box(0, 10, 3, 6, 2)
    assert (new_min, new_max) == (1, 6)


def test_expand_box_raises_value_error_when_impossible() -> None:
    with pytest.raises(ValueError):
        expand_box(0, 5, 1, 4, 3)


def test_new_crop_border_expands_first_bbox_when_second_is_larger() -> None:
    bbox1 = (2, 2, 2, 4, 4, 4)
    bbox2 = (1, 1, 1, 6, 6, 6)
    image = np.zeros((10, 10, 10))

    new_bbox1, new_bbox2 = new_crop_border(bbox1, bbox2, image)

    assert new_bbox2 == bbox2
    assert new_bbox1 == (0, 0, 0, 5, 5, 5)


def test_new_crop_border_expands_second_bbox_when_first_is_larger() -> None:
    bbox1 = (1, 1, 1, 7, 7, 7)
    bbox2 = (2, 2, 2, 4, 4, 4)
    image = np.zeros((10, 10, 10))

    new_bbox1, new_bbox2 = new_crop_border(bbox1, bbox2, image)

    assert new_bbox1 == bbox1
    assert new_bbox2 == (0, 0, 0, 6, 6, 6)


def test_crop_3d_image_returns_expected_subvolume() -> None:
    image = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    cropped = crop_3D_image(image, (1, 2, 1, 3, 5, 5))

    np.testing.assert_array_equal(cropped, image[1:3, 2:5, 1:5])


def test_single_3d_image_expand_bbox_adjusts_for_anisotropy_and_bounds() -> None:
    image = np.zeros((5, 10, 10))
    bbox = (1, 3, 3, 2, 5, 5)

    expanded = single_3D_image_expand_bbox(
        image=image,
        bbox=bbox,
        expand_pixels=4,
        anisotropy_factor=2,
    )

    assert expanded == (0, 0, 0, 4, 9, 9)


def test_check_for_xy_squareness_returns_ratio() -> None:
    ratio = check_for_xy_squareness((0, 10, 20, 5, 30, 30))
    assert ratio == pytest.approx(2.0)


def test_check_for_xy_squareness_raises_for_zero_x_width() -> None:
    with pytest.raises(ValueError, match="zero width"):
        check_for_xy_squareness((0, 1, 5, 2, 6, 5))


def test_square_off_xy_crop_bbox_expands_y_dimension() -> None:
    bbox = (0, 10, 20, 4, 14, 30)
    adjusted = square_off_xy_crop_bbox(bbox)
    assert adjusted == (0, 7, 20, 4, 17, 30)


def test_square_off_xy_crop_bbox_expands_x_dimension() -> None:
    bbox = (0, 10, 20, 4, 20, 24)
    adjusted = square_off_xy_crop_bbox(bbox)
    assert adjusted == (0, 10, 17, 4, 20, 27)


def test_square_off_xy_crop_bbox_keeps_square_bbox_unchanged() -> None:
    bbox = (0, 1, 2, 5, 9, 10)
    adjusted = square_off_xy_crop_bbox(bbox)
    assert adjusted == bbox
