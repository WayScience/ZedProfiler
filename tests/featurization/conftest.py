from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@beartype
def make_pair(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make pairs of images to use for testing
    This is a support function for coloaclization
    Featurization testing

    Parameters
    ----------
    shape : tuple[int, int, int]
        The shape of the images.
    center : tuple[int, int, int]
        The center coordinates of the overlapping blob.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the label image, image 1, and image 2.

    """
    label = np.zeros(shape, dtype=int)
    z, y, x = center
    label[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = 1
    im1 = np.zeros(shape, dtype=float)
    im2 = np.zeros(shape, dtype=float)
    # overlapping bright blob
    im1[z, y, x] = 100.0
    im2[z, y, x] = 80.0
    return label, im1, im2
