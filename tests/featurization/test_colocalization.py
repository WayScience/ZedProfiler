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
    shape: tuple[int, int, int], center: tuple[int, int, int]
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
        label, label, im1, im2, 1, 1
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
    caller's value was silently ignored. The test passes fast_costes='Fast' and
    checks via mock that the right algorithm (bisection) is invoked.
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
        compute_colocalization(loader, channel1="A", channel2="B", fast_costes="Fast")

    assert mock_bisection.called, (
        "bisection_costes_threshold_calculation was not called for "
        "fast_costes='Fast' — compute_colocalization may still be "
        "hard-coding fast_costes='Accurate'"
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
