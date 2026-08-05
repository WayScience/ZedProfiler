"""Colocalization feature extraction utilities for 3D image objects.

Computes per-object colocalization metrics (Pearson correlation, Manders
coefficients, overlap coefficient, K1/K2 coefficients) between pairs of
fluorescence channels using the Costes automatic thresholding method.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy
import pandas
import scipy.ndimage
import scipy.stats
import skimage

from zedprofiler.contracts import validate_column_name_schema
from zedprofiler.image_utils.image_utils import (
    crop_3D_image,
    new_crop_border,
    select_objects_from_label,
)
from zedprofiler.IO.feature_writing_utils import format_morphology_feature_name

COSTES_R_FAR_THRESHOLD = 0.45
COSTES_R_MID_THRESHOLD = 0.35
COSTES_R_NEAR_THRESHOLD = 0.25
MIN_PEARSON_POINTS = 2
WIDE_BISECTION_WINDOW = 6
UINT8_MAX = 255
UINT16_MAX = 65535


class _SupportsImageSetName(Protocol):
    """Minimal image-set-loader interface for name access."""

    image_set_name: str | None


class SupportsTwoObjectLoader(Protocol):
    """Minimal loader interface required for paired-object colocalization."""

    image_set_loader: _SupportsImageSetName
    compartment: str
    image1: numpy.ndarray
    image2: numpy.ndarray
    label_image: numpy.ndarray
    object_ids: Sequence[int]


def _require_scipy() -> None:
    if scipy is None:
        raise ModuleNotFoundError(
            "scipy is required for colocalization features. "
            "Install zedprofiler with scipy.",
        )


def _require_skimage() -> None:
    if skimage is None:
        raise ModuleNotFoundError(
            "scikit-image is required for colocalization features. "
            "Install zedprofiler with scikit-image.",
        )


def linear_costes_threshold_calculation(  # noqa: C901
    first_image: numpy.ndarray,
    second_image: numpy.ndarray,
    scale_max: int = 255,
    fast_costes: str = "Accurate",
) -> tuple[float, float]:
    """Finds the Costes Automatic Threshold for colocalization using a linear algorithm.
    Candidate thresholds are gradually decreased until Pearson R falls below 0.
    If "Fast" mode is enabled the "steps" between tested thresholds will be increased
    when Pearson R is much greater than 0. The other mode is "Accurate" which
    will always step down by the same amount.

    Parameters
    ----------
    first_image : numpy.ndarray
        The first fluorescence image.
    second_image : numpy.ndarray
        The second fluorescence image.
    scale_max : int, optional
        The maximum value for the image scale, by default 255.
    fast_costes : str, optional
        The mode for the Costes threshold calculation, by default "Accurate".

    Returns
    -------
    Tuple[float, float]
        The calculated thresholds for the first and second images.

    """
    _require_scipy()
    i_step = 1 / scale_max  # Step size for the threshold as a float
    non_zero = (first_image > 0) | (second_image > 0)
    if non_zero.sum() < MIN_PEARSON_POINTS:
        return 0.0, 0.0
    xvar = numpy.var(first_image[non_zero], axis=0, ddof=1)
    yvar = numpy.var(second_image[non_zero], axis=0, ddof=1)

    xmean = numpy.mean(first_image[non_zero], axis=0)
    ymean = numpy.mean(second_image[non_zero], axis=0)

    z = first_image[non_zero] + second_image[non_zero]
    zvar = numpy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    if denom == 0:
        return 0.0, 0.0
    num = (yvar - xvar) + numpy.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar),
    )
    a = num / denom
    b = ymean - a * xmean

    # Start at 1 step above the maximum value
    img_max = max(first_image.max(), second_image.max())
    i = i_step * ((img_max // i_step) + 1)

    num_true = None
    first_image_max = first_image.max()
    second_image_max = second_image.max()

    thr_first_image_c = i
    thr_second_image_c = (a * i) + b
    while i > first_image_max and (a * i) + b > second_image_max:
        i -= i_step
    while i > i_step:
        thr_first_image_c = i
        thr_second_image_c = (a * i) + b
        combt = (first_image < thr_first_image_c) | (second_image < thr_second_image_c)
        try:
            # Only run pearsonr if the input has changed.
            if (positives := numpy.count_nonzero(combt)) != num_true:
                costReg, _ = scipy.stats.pearsonr(
                    first_image[combt],
                    second_image[combt],
                )
                num_true = positives

            if costReg <= 0:
                break
            if fast_costes == "Accurate" or i < i_step * 10:
                i -= i_step
            elif costReg > COSTES_R_FAR_THRESHOLD:
                # We're way off, step down 10x
                i -= i_step * 10
            elif costReg > COSTES_R_MID_THRESHOLD:
                # Still far from 0, step 5x
                i -= i_step * 5
            elif costReg > COSTES_R_NEAR_THRESHOLD:
                # Step 2x
                i -= i_step * 2
            else:
                i -= i_step
        except ValueError:
            break
    return thr_first_image_c, thr_second_image_c


def bisection_costes_threshold_calculation(
    first_image: numpy.ndarray,
    second_image: numpy.ndarray,
    scale_max: int = 255,
) -> tuple[float, float]:
    """Find the Costes Automatic Threshold for colocalization via bisection.
    Candidate thresholds are selected from within a window of possible intensities,
    this window is narrowed based on the R value of each tested candidate.
    We're looking for the first point at 0, and R value can become highly variable
    at lower thresholds in some samples. Therefore the candidate tested in each
    loop is 1/6th of the window size below the maximum value
    (as opposed to the midpoint).

    Parameters
    ----------
    first_image : numpy.ndarray
        The first fluorescence image.
    second_image : numpy.ndarray
        The second fluorescence image.
    scale_max : int, optional
        The maximum value for the image scale, by default 255.

    Returns
    -------
    Tuple[float, float]
        The calculated thresholds for the first and second images.

    """
    _require_scipy()

    non_zero = (first_image > 0) | (second_image > 0)
    if non_zero.sum() < MIN_PEARSON_POINTS:
        return 0.0, 0.0
    xvar = numpy.var(first_image[non_zero], axis=0, ddof=1)
    yvar = numpy.var(second_image[non_zero], axis=0, ddof=1)

    xmean = numpy.mean(first_image[non_zero], axis=0)
    ymean = numpy.mean(second_image[non_zero], axis=0)

    z = first_image[non_zero] + second_image[non_zero]
    zvar = numpy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    if denom == 0:
        return 0.0, 0.0
    num = (yvar - xvar) + numpy.sqrt((yvar - xvar) * (yvar - xvar) + 4 * (covar**2))
    a = num / denom
    b = ymean - a * xmean

    # Initialize variables
    left = 1
    right = scale_max
    mid = (right - left) * 5 // 6 + left
    lastmid = 0
    # Marks the value with the last positive R value.
    valid = 1

    while lastmid != mid:
        # Use raw pixel units (not normalised) so the threshold is comparable
        # with linear_costes_threshold_calculation and with the outer dispatch's
        # `image > thr` comparison. CellProfiler's library has the same
        # mid/scale_max normalisation bug; this is an intentional divergence.
        thr_first_image_c = float(mid)
        thr_second_image_c = (a * thr_first_image_c) + b
        combt = (first_image < thr_first_image_c) | (second_image < thr_second_image_c)
        if numpy.count_nonzero(combt) <= MIN_PEARSON_POINTS:
            # Can't run meaningful Pearson with only a few values.
            left = mid - 1
        else:
            try:
                costReg, _ = scipy.stats.pearsonr(
                    first_image[combt],
                    second_image[combt],
                )
                if costReg < 0:
                    left = mid - 1
                elif costReg >= 0:
                    right = mid + 1
                    valid = mid
            except ValueError:
                # Catch misc Pearson errors with low sample numbers
                left = mid - 1
        lastmid = mid
        if right - left > WIDE_BISECTION_WINDOW:
            mid = (right - left) * 5 // 6 + left
        else:
            mid = ((right - left) // 2) + left

    thr_first_image_c = float(valid - 1)
    thr_second_image_c = (a * thr_first_image_c) + b

    return thr_first_image_c, thr_second_image_c


def prepare_two_images_for_colocalization(  # noqa: PLR0913
    *,
    label_object1: numpy.ndarray,
    label_object2: numpy.ndarray,
    image_object1: numpy.ndarray,
    image_object2: numpy.ndarray,
    object_id1: int,
    object_id2: int,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Prepare two images for colocalization analysis by cropping to object bbox.
    It selects objects from label images, calculates their bounding boxes,
    and crops both images accordingly.

    Parameters
    ----------
    label_object1 : numpy.ndarray
        The segmented label image for the first object.
    label_object2 : numpy.ndarray
        The segmented label image for the second object.
    image_object1 : numpy.ndarray
        The spectral image to crop for the first object.
    image_object2 : numpy.ndarray
        The spectral image to crop for the second object.
    object_id1 : int
        The object index to select from the label image for the first object.
    object_id2 : int
        The object index to select from the label image for the second object.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        The two cropped images for colocalization analysis.

    """
    _require_skimage()
    label_object1 = select_objects_from_label(label_object1, object_id1)
    label_object2 = select_objects_from_label(label_object2, object_id2)
    # get the image bbox
    props_image1 = skimage.measure.regionprops_table(label_object1, properties=["bbox"])
    bbox_image1 = (
        props_image1["bbox-0"][0],  # z min
        props_image1["bbox-1"][0],  # y min
        props_image1["bbox-2"][0],  # x min
        props_image1["bbox-3"][0],  # z max
        props_image1["bbox-4"][0],  # y max
        props_image1["bbox-5"][0],  # x max
    )

    props_image2 = skimage.measure.regionprops_table(label_object2, properties=["bbox"])
    bbox_image2 = (
        props_image2["bbox-0"][0],  # z min
        props_image2["bbox-1"][0],  # y min
        props_image2["bbox-2"][0],  # x min
        props_image2["bbox-3"][0],  # z max
        props_image2["bbox-4"][0],  # y max
        props_image2["bbox-5"][0],  # x max
    )

    new_bbox1, new_bbox2 = new_crop_border(bbox_image1, bbox_image2, image_object1)

    cropped_image_1 = crop_3D_image(image_object1, new_bbox1)
    cropped_image_2 = crop_3D_image(image_object2, new_bbox2)
    return cropped_image_1, cropped_image_2


def calculate_colocalization(  # noqa: PLR0912, PLR0915
    cropped_image_1: numpy.ndarray,
    cropped_image_2: numpy.ndarray,
    thr: int = 15,
    fast_costes: str = "Accurate",
) -> dict[str, float]:
    """This function calculates the colocalization coefficients between two images.
    It computes the correlation coefficient, Manders' coefficients, overlap coefficient,
    and Costes' coefficients. The results are returned as a dictionary.

    Parameters
    ----------
    cropped_image_1 : numpy.ndarray
        The first cropped image.
    cropped_image_2 : numpy.ndarray
        The second cropped image.
    thr : int, optional
        The threshold for the Manders' coefficients, by default 15
    fast_costes : str, optional
        The mode for Costes' threshold calculation, by default "Accurate".
        Options are "Accurate", "Fast", or "Faster" (matching CellProfiler's
        three Costes methods). "Accurate" tests every threshold value using a
        linear scan (slowest, most precise). "Fast" uses the same linear scan
        but skips candidate thresholds when the Pearson R is far from the
        crossing point (faster, slightly less precise). "Faster" uses a
        bisection algorithm and is substantially faster for 16-bit images
        (least precise).

    Returns
    -------
    Dict[str, float]
        The output features for colocalization analysis.

    """
    _require_scipy()
    results = {}
    ################################################################################################
    # Calculate the correlation coefficient between the two images
    # This is the Pearson correlation coefficient
    # Pearson correlation coeffecient = cov(X, Y) / (std(X) * std(Y))
    # where cov(X, Y) is the covariance of X and Y
    # where X and Y are the two images
    # std(X) is the standard deviation of X
    # std(Y) is the standard deviation of Y
    # cov(X, Y) = sum((X - mean(X)) * (Y - mean(Y))) / (N - 1)
    # std(X) = sqrt(sum((X - mean(X)) ** 2) / (N - 1))
    # thus N -1 cancels out in the calculation below
    ################################################################################################
    mean1 = numpy.mean(cropped_image_1)
    mean2 = numpy.mean(cropped_image_2)
    std1 = numpy.sqrt(numpy.sum((cropped_image_1 - mean1) ** 2))
    std2 = numpy.sqrt(numpy.sum((cropped_image_2 - mean2) ** 2))
    x = cropped_image_1 - mean1  # x is not the same as the x dimension here
    y = cropped_image_2 - mean2  # y is not the same as the y dimension here
    denom = std1 * std2
    corr = numpy.sum(x * y) / denom if denom > 0 else 0.0

    ################################################################################################
    # Calculate the Manders' coefficients
    ################################################################################################

    # Threshold as percentage of maximum intensity of objects in each channel
    # Initialise before the try block so these are always bound even when the
    # except branch fires (numpy.max raises ValueError on empty arrays).
    combined_thresh = numpy.zeros_like(cropped_image_1, dtype=bool)
    first_image_thresh = cropped_image_1[combined_thresh]
    second_image_thresh = cropped_image_2[combined_thresh]
    try:
        tff = (thr / 100) * numpy.max(cropped_image_1)
        tss = (thr / 100) * numpy.max(cropped_image_2)
    except ValueError:
        M1, M2 = 0.0, 0.0
    else:
        # get the thresholds
        combined_thresh = (cropped_image_1 >= tff) & (cropped_image_2 >= tss)

        first_image_thresh = cropped_image_1[combined_thresh]
        second_image_thresh = cropped_image_2[combined_thresh]

        tot_first_image_thr = scipy.ndimage.sum(
            cropped_image_1[cropped_image_1 >= tff],
        )
        tot_second_image_thr = scipy.ndimage.sum(
            cropped_image_2[cropped_image_2 >= tss],
        )

        if tot_first_image_thr > 0 and tot_second_image_thr > 0:
            M1 = scipy.ndimage.sum(first_image_thresh) / tot_first_image_thr
            M2 = scipy.ndimage.sum(second_image_thresh) / tot_second_image_thr
        else:
            M1, M2 = 0.0, 0.0
    ################################################################################################
    # Calculate the overlap coefficient
    ################################################################################################

    if numpy.any(combined_thresh):
        fpsq = scipy.ndimage.sum(
            cropped_image_1[combined_thresh] ** 2,
        )
        spsq = scipy.ndimage.sum(
            cropped_image_2[combined_thresh] ** 2,
        )
        pdt = numpy.sqrt(numpy.array(fpsq) * numpy.array(spsq))
        overlap = (
            scipy.ndimage.sum(
                cropped_image_1[combined_thresh] * cropped_image_2[combined_thresh],
            )
            / pdt
        )
        # K1/K2 are computed but not currently exported as features.
        _k1 = scipy.ndimage.sum(
            cropped_image_1[combined_thresh] * cropped_image_2[combined_thresh],
        ) / (numpy.array(fpsq))
        _k2 = scipy.ndimage.sum(
            cropped_image_1[combined_thresh] * cropped_image_2[combined_thresh],
        ) / (numpy.array(spsq))
    else:
        overlap, _k1, _k2 = 0.0, 0.0, 0.0

    # first_pixels, second_pixels = flattened image arrays
    # combined_thresh = boolean mask of pixels above threshold in both channels
    # fi_thresh, si_thresh = thresholded intensities (same shape as pixels)

    # --- Rank computation ---
    # Flatten images for ranking
    img1_flat = cropped_image_1.flatten()
    img2_flat = cropped_image_2.flatten()

    # --- Rank computation ---
    sorted_idx_1 = numpy.argsort(img1_flat)
    sorted_idx_2 = numpy.argsort(img2_flat)

    # Create rank arrays
    rank_1_flat = numpy.empty_like(sorted_idx_1, dtype=float)
    rank_2_flat = numpy.empty_like(sorted_idx_2, dtype=float)
    rank_1_flat[sorted_idx_1] = numpy.arange(len(sorted_idx_1))
    rank_2_flat[sorted_idx_2] = numpy.arange(len(sorted_idx_2))

    # Reshape back to original shape
    rank_im1 = rank_1_flat.reshape(cropped_image_1.shape)
    rank_im2 = rank_2_flat.reshape(cropped_image_2.shape)

    # --- Rank difference weight ---
    R = max(rank_im1.max(), rank_im2.max()) + 1
    Di = numpy.abs(rank_im1 - rank_im2)
    weight = (R - Di) / R

    # Get weights for thresholded pixels
    weight_thresh = weight[combined_thresh]

    # Get thresholded values (no double-thresholding!)
    first_image_thresh_final = first_image_thresh
    second_image_thresh_final = second_image_thresh

    # --- Calculate weighted colocalization ---
    if numpy.any(combined_thresh) and len(first_image_thresh_final) > 0:
        weighted_sum_1 = numpy.sum(first_image_thresh_final * weight_thresh)
        weighted_sum_2 = numpy.sum(second_image_thresh_final * weight_thresh)

        total_1 = numpy.sum(first_image_thresh_final)
        total_2 = numpy.sum(second_image_thresh_final)

        RWC1 = weighted_sum_1 / total_1 if total_1 > 0 else 0.0
        RWC2 = weighted_sum_2 / total_2 if total_2 > 0 else 0.0
    else:
        RWC1, RWC2 = 0.0, 0.0
    ################################################################################################
    # Calculate the Costes' coefficient
    ################################################################################################

    # Orthogonal Regression for Costes' automated threshold
    if numpy.max(cropped_image_1) > UINT8_MAX or numpy.max(cropped_image_2) > UINT8_MAX:
        scale = UINT16_MAX
    else:
        scale = UINT8_MAX

    if fast_costes == "Accurate":
        thr_first_image_c, thr_second_image_c = linear_costes_threshold_calculation(
            first_image=cropped_image_1,
            second_image=cropped_image_2,
            scale_max=scale,
            fast_costes="Accurate",
        )
    elif fast_costes == "Fast":
        thr_first_image_c, thr_second_image_c = linear_costes_threshold_calculation(
            first_image=cropped_image_1,
            second_image=cropped_image_2,
            scale_max=scale,
            fast_costes="Fast",
        )
    else:  # "Faster"
        thr_first_image_c, thr_second_image_c = bisection_costes_threshold_calculation(
            first_image=cropped_image_1,
            second_image=cropped_image_2,
            scale_max=scale,
        )

    # Costes' thershold for entire image is applied to each object
    first_image_above_thr = cropped_image_1 > thr_first_image_c
    second_image_above_thr = cropped_image_2 > thr_second_image_c
    combined_thresh_c = first_image_above_thr & second_image_above_thr
    first_image_thresh_c = cropped_image_1[combined_thresh_c]
    second_image_thresh_c = cropped_image_2[combined_thresh_c]

    tot_first_image_thr_c = scipy.ndimage.sum(
        cropped_image_1[cropped_image_1 >= thr_first_image_c],
    )

    tot_second_image_thr_c = scipy.ndimage.sum(
        cropped_image_2[cropped_image_2 >= thr_second_image_c],
    )
    if tot_first_image_thr_c > 0 and tot_second_image_thr_c > 0:
        C1 = scipy.ndimage.sum(first_image_thresh_c) / tot_first_image_thr_c
        C2 = scipy.ndimage.sum(second_image_thresh_c) / tot_second_image_thr_c
    else:
        C1, C2 = 0.0, 0.0
    ################################################################################################
    # write the results to the output dictionary
    ################################################################################################

    results["Correlation"] = corr
    results["MandersCoeffM1"] = M1
    results["MandersCoeffM2"] = M2
    results["OverlapCoeff"] = overlap
    results["MandersCoeffCostesM1"] = C1
    results["MandersCoeffCostesM2"] = C2
    results["RankWeightedColocalizationCoeff1"] = RWC1
    results["RankWeightedColocalizationCoeff2"] = RWC2

    return results


def compute_colocalization(  # noqa: C901, PLR0912
    two_object_loader: SupportsTwoObjectLoader,
    thr: int = 15,
    fast_costes: str = "Accurate",
    channel1: str | None = None,
    channel2: str | None = None,
) -> pandas.DataFrame:
    """Compute colocalization features for pairs of objects from two channels.

    Parameters
    ----------
    two_object_loader : SupportsTwoObjectLoader
        The loader that provides access to the two channels and their
        corresponding labels.
    thr : int, optional
        The threshold for the Manders' coefficients, by default 15
    fast_costes : str, optional
        The mode for Costes' threshold calculation, by default "Accurate".
        Options are "Accurate", "Fast", or "Faster" (matching CellProfiler's
        three Costes methods). "Accurate" tests every threshold value using a
        linear scan (slowest, most precise). "Fast" uses the same linear scan
        but skips candidate thresholds when the Pearson R is far from the
        crossing point (faster, slightly less precise). "Faster" uses a
        bisection algorithm and is substantially faster for 16-bit images
        (least precise).
    channel1 : str | None, optional
        The name of the first channel, used for feature naming, by default None
    channel2 : str | None, optional
        The name of the second channel, used for feature naming, by default None

    Returns
    -------
    pandas.DataFrame
        Wide-format DataFrame with one row per object pair and one column per
        colocalization metric, plus Metadata columns.

    """
    if channel1 is None or channel2 is None:
        raise ValueError("channel1 and channel2 must be provided for feature naming.")
    list_of_dfs = []
    for object_id in two_object_loader.object_ids:
        cropped_image1, cropped_image2 = prepare_two_images_for_colocalization(
            label_object1=two_object_loader.label_image,
            label_object2=two_object_loader.label_image,
            image_object1=two_object_loader.image1,
            image_object2=two_object_loader.image2,
            object_id1=object_id,
            object_id2=object_id,
        )
        colocalization_features = calculate_colocalization(
            cropped_image_1=cropped_image1,
            cropped_image_2=cropped_image2,
            thr=thr,
            fast_costes=fast_costes,
        )

        # Build a simple dict row (avoid pandas dependency)
        row: dict[str, object] = {}
        for meas_key, meas_val in colocalization_features.items():
            full_name = format_morphology_feature_name(
                compartment=two_object_loader.compartment,
                channel=f"{channel1}-{channel2}",
                feature_type="Colocalization",
                measurement=meas_key,
            )
            # cast numeric values to float32 where appropriate
            if full_name not in (
                "Metadata_Object_ObjectID",
                "Metadata_Experiment_ImageSet",
            ):
                try:
                    row[full_name] = numpy.float32(meas_val)
                except Exception:
                    row[full_name] = meas_val
            else:
                row[full_name] = meas_val

        # ensure object_id and image_set are present and first
        row["Metadata_Object_ObjectID"] = object_id
        row["Metadata_Experiment_ImageSet"] = (
            two_object_loader.image_set_loader.image_set_name
        )
        list_of_dfs.append(row)

    # Convert list of row-dicts into a dict-of-lists with stable ordering
    if not list_of_dfs:
        return pandas.DataFrame()

    # Collect other metric keys preserving first-seen ordering
    other_keys: list[str] = []
    for d in list_of_dfs:
        for k in d:
            if k in ("Metadata_Object_ObjectID", "Metadata_Experiment_ImageSet"):
                continue
            if k not in other_keys:
                other_keys.append(k)

    all_keys = [
        "Metadata_Object_ObjectID",
        "Metadata_Experiment_ImageSet",
        *other_keys,
    ]
    result: dict[str, list[object]] = {
        k: [r.get(k) for r in list_of_dfs] for k in all_keys
    }

    for col in list(result.keys()):
        try:
            validate_column_name_schema(
                column_name=col,
                compartments=[two_object_loader.compartment],
                channels=[f"{channel1}-{channel2}"],
            )
        except ValueError as e:
            raise ValueError(f"Column name {col} does not conform to schema: {e}")

    return pandas.DataFrame(result)
