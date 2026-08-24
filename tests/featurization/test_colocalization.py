from __future__ import annotations

import unittest.mock

import numpy as np
import pandas as pd
import pytest
from conftest import make_pair
from pydantic import BaseModel, ConfigDict, field_validator

skimage = pytest.importorskip("skimage")

from zedprofiler.featurization.colocalization import (  # noqa: E402
    bisection_costes_threshold_calculation,
    calculate_colocalization,
    compute_colocalization,
    linear_costes_threshold_calculation,
    prepare_two_images_for_colocalization,
)


class ImageSetLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_set_name: str = "coloc"
    # mirrors ImageSetLoader.image_id (falls back to image_set_name)
    image_id: str = "coloc"


class TwoObjectLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_set_loader: ImageSetLoaderModel
    compartment: str
    image1: np.ndarray
    image2: np.ndarray
    label_image: np.ndarray
    object_ids: list[int]

    @field_validator("image1", "image2", "label_image", mode="before")
    @classmethod
    def to_array(_cls, v: object) -> np.ndarray:
        return np.asarray(v)


@pytest.mark.parametrize("shape,center", [((7, 7, 7), (3, 3, 3))])
def test_compute_colocalization_basic(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
) -> None:
    imgset = ImageSetLoaderModel()
    label, im1, im2 = make_pair(shape, center)
    loader = TwoObjectLoaderModel(
        image_set_loader=imgset,
        compartment="Cell",
        image1=im1,
        image2=im2,
        label_image=label,
        object_ids=[1],
    )

    df = compute_colocalization(loader, channel1="A", channel2="B")

    assert isinstance(df, pd.DataFrame)
    # Expect correlation column with morphology formatting present
    assert any("Colocalization" in c for c in df.columns)


@pytest.mark.parametrize("shape,center", [((7, 7, 7), (3, 3, 3))])
def test_zero_objects_returns_well_formed_empty_frame(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
) -> None:
    """A degenerate loader with zero objects must not return a malformed frame.

    Before the fix, compute_colocalization returned a bare
    ``pandas.DataFrame()`` with no columns at all when no object pairs were
    found, which crashes any downstream merge that expects an ID column to
    key on.
    """
    imgset = ImageSetLoaderModel()
    label, im1, im2 = make_pair(shape, center)
    loader = TwoObjectLoaderModel(
        image_set_loader=imgset,
        compartment="Cell",
        image1=im1,
        image2=im2,
        label_image=label,
        object_ids=[],
    )

    df = compute_colocalization(loader, channel1="A", channel2="B")

    assert isinstance(df, pd.DataFrame)
    assert "Metadata_Object_ObjectID" in df.columns
    assert len(df) == 0


def test_linear_and_bisection_costes_thresholds_basic() -> None:
    # simple linear relationship between channels
    x = np.linspace(1.0, 100.0, 200)
    img1 = x.reshape((200,))
    img2 = (2.0 * x + 5.0).reshape((200,))

    thr_lin = linear_costes_threshold_calculation(img1, img2, scale_max=255)
    thr_bis = bisection_costes_threshold_calculation(img1, img2, scale_max=255)
    expected_threshold_count = 2

    assert isinstance(thr_lin, tuple) and len(thr_lin) == expected_threshold_count
    assert isinstance(thr_bis, tuple) and len(thr_bis) == expected_threshold_count

    for t in (*thr_lin, *thr_bis):
        assert isinstance(t, float)
        assert t >= 0.0


def test_prepare_two_images_for_colocalization_crops() -> None:
    # create two identical label images with one object each and match images
    shape = (7, 7, 7)
    label = np.zeros(shape, dtype=int)
    # 3x3x3 cube in center
    label[2:5, 2:5, 2:5] = 1

    im1 = np.zeros(shape, dtype=float)
    im2 = np.zeros(shape, dtype=float)
    expected_peak_im1 = 10.0
    expected_peak_im2 = 5.0
    im1[3, 3, 3] = expected_peak_im1
    im2[3, 3, 3] = expected_peak_im2

    cropped1, cropped2 = prepare_two_images_for_colocalization(
        label_object1=label,
        label_object2=label,
        image_object1=im1,
        image_object2=im2,
        object_id1=1,
        object_id2=1,
    )

    assert isinstance(cropped1, np.ndarray) and isinstance(cropped2, np.ndarray)
    # crops should be small but non-empty and include the bright voxel
    assert cropped1.size > 0 and cropped2.size > 0
    assert cropped1.max() >= expected_peak_im1
    assert cropped2.max() >= expected_peak_im2


def test_combined_thresh_does_not_raise_unbound_local_error() -> None:
    """combined_thresh must be bound even when images are empty (Bug 2).

    Before the fix, combined_thresh was only assigned inside the try-else block,
    so when numpy.max raised ValueError on an empty crop, the subsequent
    ``if numpy.any(combined_thresh)`` reference raised UnboundLocalError.
    """
    empty = np.zeros((0,), dtype=float)
    try:
        calculate_colocalization(empty, empty, thr=15, fast_costes="Accurate")
    except UnboundLocalError as e:
        pytest.fail(f"UnboundLocalError raised — combined_thresh not initialised: {e}")
    except Exception:
        pass  # any other exception (ValueError, ZeroDivisionError) is acceptable


def test_accurate_mode_calls_linear_not_bisection() -> None:
    """fast_costes='Accurate' must route to linear_costes, not bisection (Bug 3).

    The two algorithms can converge to the same threshold for some inputs, so
    the test uses unittest.mock.patch to spy on which function is actually called
    rather than comparing numeric results.
    """
    rng = np.random.default_rng(42)
    img = rng.uniform(0, 255, (8, 8, 8)).astype(float)

    with (
        unittest.mock.patch(
            "zedprofiler.featurization.colocalization.linear_costes_threshold_calculation",
            wraps=linear_costes_threshold_calculation,
        ) as mock_linear,
        unittest.mock.patch(
            "zedprofiler.featurization.colocalization.bisection_costes_threshold_calculation",
            wraps=bisection_costes_threshold_calculation,
        ) as mock_bisection,
    ):
        calculate_colocalization(img, img, thr=15, fast_costes="Accurate")

    assert mock_linear.called, (
        "linear_costes_threshold_calculation was not called for fast_costes='Accurate'"
    )
    assert not mock_bisection.called, (
        "bisection_costes_threshold_calculation was called for fast_costes='Accurate'"
    )


def test_compute_colocalization_respects_fast_costes_parameter() -> None:
    """compute_colocalization must forward fast_costes to calculate_colocalization.

    Bug B (ZedProfiler-specific):

    Before the fix, the inner call hard-coded fast_costes='Accurate', so the
    caller's value was silently ignored. The test passes fast_costes='Faster' and
    checks via mock that bisection is invoked (the only mode that reaches bisection).
    """
    imgset = ImageSetLoaderModel()
    shape = (7, 7, 7)
    center = (3, 3, 3)
    label, im1, im2 = make_pair(shape, center)
    loader = TwoObjectLoaderModel(
        image_set_loader=imgset,
        compartment="Cell",
        image1=im1,
        image2=im2,
        label_image=label,
        object_ids=[1],
    )

    with unittest.mock.patch(
        "zedprofiler.featurization.colocalization.bisection_costes_threshold_calculation",
        wraps=bisection_costes_threshold_calculation,
    ) as mock_bisection:
        compute_colocalization(loader, channel1="A", channel2="B", fast_costes="Faster")

    assert mock_bisection.called, (
        "bisection_costes_threshold_calculation was not called for "
        "fast_costes='Faster' — compute_colocalization may still be "
        "hard-coding fast_costes='Accurate'"
    )


@pytest.fixture()
def high_contrast_images() -> tuple[np.ndarray, np.ndarray]:
    """Two 1-D images with a bright correlated signal well above background."""
    rng = np.random.default_rng(0)
    background = rng.uniform(0, 50, 300)
    signal_mask = np.zeros(300, dtype=bool)
    signal_mask[100:150] = True
    img1 = background.copy()
    img1[signal_mask] = 200.0
    img2 = background.copy()
    img2[signal_mask] = 180.0
    return img1, img2


def test_all_costes_modes_converge_on_same_threshold(
    high_contrast_images: tuple[np.ndarray, np.ndarray],
) -> None:
    """All three modes must agree to within 15 pixel units on realistic images."""
    img1, img2 = high_contrast_images
    thr_accurate, _ = linear_costes_threshold_calculation(
        first_image=img1, second_image=img2, scale_max=255, fast_costes="Accurate"
    )
    thr_fast, _ = linear_costes_threshold_calculation(
        first_image=img1, second_image=img2, scale_max=255, fast_costes="Fast"
    )
    thr_faster, _ = bisection_costes_threshold_calculation(
        first_image=img1, second_image=img2, scale_max=255
    )
    tolerance = 15
    assert abs(thr_accurate - thr_fast) <= tolerance, (
        f"Accurate ({thr_accurate:.1f}) and Fast ({thr_fast:.1f}) diverged "
        f"by more than {tolerance}"
    )
    assert abs(thr_accurate - thr_faster) <= tolerance, (
        f"Accurate ({thr_accurate:.1f}) and Faster ({thr_faster:.1f}) diverged "
        f"by more than {tolerance}"
    )


def test_costes_threshold_low_for_perfectly_correlated_images() -> None:
    """All three modes must return a low threshold for perfectly correlated images."""
    img = np.linspace(10, 200, 300)
    thr_accurate, _ = linear_costes_threshold_calculation(
        first_image=img, second_image=img, scale_max=255, fast_costes="Accurate"
    )
    thr_fast, _ = linear_costes_threshold_calculation(
        first_image=img, second_image=img, scale_max=255, fast_costes="Fast"
    )
    thr_faster, _ = bisection_costes_threshold_calculation(
        first_image=img, second_image=img, scale_max=255
    )
    max_expected = 15  # near zero in 0-255 space
    assert thr_accurate < max_expected, (
        f"Expected threshold near 0 for fully correlated images, got {thr_accurate:.3f}"
    )
    assert thr_fast < max_expected, (
        f"Expected threshold near 0 for fully correlated images, got {thr_fast:.3f}"
    )
    assert thr_faster < max_expected, (
        f"Expected threshold near 0 for fully correlated images, got {thr_faster:.3f}"
    )


def test_costes_threshold_high_for_anticorrelated_images_linear() -> None:
    """Linear modes must return a threshold near max for anti-correlated images.

    Note: the bisection algorithm's degenerate behaviour for purely anti-correlated
    inputs (returns 0 rather than scale_max) is documented separately in
    test_bisection_degenerate_anticorrelation_returns_zero.
    """
    img1 = np.linspace(0, 255, 300)
    img2 = np.linspace(255, 0, 300)
    thr_accurate, _ = linear_costes_threshold_calculation(
        first_image=img1, second_image=img2, scale_max=255, fast_costes="Accurate"
    )
    thr_fast, _ = linear_costes_threshold_calculation(
        first_image=img1, second_image=img2, scale_max=255, fast_costes="Fast"
    )
    min_expected = 200
    assert thr_accurate > min_expected, (
        "Expected threshold near max for anti-correlated images, "
        f"got {thr_accurate:.3f}"
    )
    assert thr_fast > min_expected, (
        f"Expected threshold near max for anti-correlated images, got {thr_fast:.3f}"
    )


def test_bisection_degenerate_anticorrelation_returns_zero() -> None:
    """Document bisection's edge-case behaviour for purely anti-correlated images.

    When Pearson R is negative for every candidate threshold, valid is never updated
    from its initial value of 1, so the return is valid - 1 = 0. This matches
    CellProfiler's library behaviour and is a known limitation for this degenerate case.
    """
    img1 = np.linspace(0, 255, 300)
    img2 = np.linspace(255, 0, 300)
    thr, _ = bisection_costes_threshold_calculation(
        first_image=img1, second_image=img2, scale_max=255
    )
    assert thr == 0.0, (
        f"Expected bisection to return 0 for fully anti-correlated images "
        f"(degenerate case), got {thr}"
    )


def test_faster_mode_calls_bisection_not_linear() -> None:
    """fast_costes='Faster' must route to bisection, not linear."""
    rng = np.random.default_rng(42)
    img = rng.uniform(0, 255, (8, 8, 8)).astype(float)
    with (
        unittest.mock.patch(
            "zedprofiler.featurization.colocalization.linear_costes_threshold_calculation",
            wraps=linear_costes_threshold_calculation,
        ) as mock_linear,
        unittest.mock.patch(
            "zedprofiler.featurization.colocalization.bisection_costes_threshold_calculation",
            wraps=bisection_costes_threshold_calculation,
        ) as mock_bisection,
    ):
        calculate_colocalization(img, img, thr=15, fast_costes="Faster")
    assert mock_bisection.called, (
        "bisection_costes_threshold_calculation was not called for fast_costes='Faster'"
    )
    assert not mock_linear.called, (
        "linear_costes_threshold_calculation was called for fast_costes='Faster'"
    )


def test_calculate_colocalization_identical_images() -> None:
    # identical images should give high correlation and Manders near 1
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, size=(6, 6, 6)).astype(float)

    results = calculate_colocalization(img, img, thr=10, fast_costes="Accurate")

    # expected keys present and sensible numeric values
    expected_keys = (
        "Correlation",
        "MandersCoeffM1",
        "MandersCoeffM2",
        "OverlapCoeff",
    )
    for k in expected_keys:
        assert k in results
        assert isinstance(results[k], float)

    # identical images -> correlation close to 1
    min_expected_correlation = 0.9
    assert results["Correlation"] > min_expected_correlation
    # Manders should be non-negative
    assert results["MandersCoeffM1"] >= 0.0
    assert results["MandersCoeffM2"] >= 0.0


@pytest.mark.parametrize("shape,center", [((7, 7, 7), (3, 3, 3))])
def test_compute_colocalization_skips_phantom_object_id_without_bbox(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
) -> None:
    """Object ids absent from the label image are skipped via the bbox guard.

    ``compute_colocalization`` looks up each requested object id in the
    ``regionprops`` bbox table and skips ids with no matching region via the
    ``if bbox is None: continue`` guard. Requesting a phantom id (99)
    alongside a real one (1) exercises that guard: only object 1 should appear
    in the output, with no crash.
    """
    imgset = ImageSetLoaderModel()
    label, im1, im2 = make_pair(shape, center)
    loader = TwoObjectLoaderModel(
        image_set_loader=imgset,
        compartment="Cell",
        image1=im1,
        image2=im2,
        label_image=label,
        object_ids=[1, 99],
    )

    df = compute_colocalization(loader, channel1="A", channel2="B")

    returned_ids = sorted(int(x) for x in df["Metadata_Object_ObjectID"].tolist())
    assert returned_ids == [1]
