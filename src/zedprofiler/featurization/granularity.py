"""Calculate the granularity spectrum of a 3D image."""

from __future__ import annotations

import numpy
import pandas
import scipy.ndimage
import skimage.morphology

from zedprofiler.contracts import validate_column_name_schema
from zedprofiler.IO.feature_writing_utils import format_morphology_feature_name
from zedprofiler.IO.loading_classes import ObjectLoader


def _fix_scipy_ndimage_result(result: float | list | numpy.ndarray) -> numpy.ndarray:
    """Convert scipy.ndimage aggregation results to a consistent array.

    Equivalent to centrosome.cpmorphology.fixup_scipy_ndimage_result.
    scipy.ndimage.mean/sum can return a scalar when there's one label,
    or a list otherwise. This ensures we always get a numpy array.

    Parameters
    ----------
    result : scalar, list, or numpy.ndarray
        Output from scipy.ndimage.mean or similar.

    Returns
    -------
    numpy.ndarray
        1-D array of results.

    """
    if numpy.isscalar(result):
        return numpy.array([result])
    return numpy.asarray(result)


def _subsample_3d(
    data: numpy.ndarray,
    new_shape: numpy.ndarray,
    subsample_factor: float,
    order: int = 1,
) -> numpy.ndarray:
    """Subsample a 3D array using map_coordinates, matching CellProfiler.

    CellProfiler generates coordinates for the new shape and divides by
    subsample_factor to map back into the original coordinate space.
    The same scalar factor is used for all three axes.

    Parameters
    ----------
    data : numpy.ndarray
        3D array to subsample.
    new_shape : numpy.ndarray
        Target shape as a float array (coordinate grid extent).
    subsample_factor : float
        The factor used to divide coordinates (same for all axes).
    order : int
        Interpolation order (1 for linear, 0 for nearest-neighbor).

    Returns
    -------
    numpy.ndarray
        Subsampled array.

    """
    if subsample_factor >= 1.0:
        return data.copy()

    k, i, j = (
        numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float)
        / subsample_factor
    )
    return scipy.ndimage.map_coordinates(data, (k, i, j), order=order)


def _labeled_voxel_positions(
    masked_labels: numpy.ndarray,
) -> tuple[tuple[numpy.ndarray, ...], numpy.ndarray]:
    """Return coordinates and label ids for every voxel in a labeled object.

    ``scipy.ndimage.mean(image, masked_labels, label_range)`` only ever reads
    voxels where ``masked_labels`` is nonzero; everything else is discarded.
    Precomputing just those voxel coordinates (and the label id at each one)
    lets a per-scale loop upsample/scan only what will actually be used,
    instead of the whole image, without changing the result.

    Parameters
    ----------
    masked_labels : numpy.ndarray
        Label image with 0 marking background/unlabeled voxels.

    Returns
    -------
    tuple[tuple[numpy.ndarray, ...], numpy.ndarray]
        ``(coords, label_ids)`` where ``coords`` is a per-axis tuple of index
        arrays (as returned by ``numpy.nonzero``) and ``label_ids`` is the
        label value at each of those coordinates, i.e.
        ``masked_labels[coords]``.

    """
    coords = numpy.nonzero(masked_labels)
    return coords, masked_labels[coords]


def _upsample_3d(
    data: numpy.ndarray,
    subsampled_shape: numpy.ndarray,
    original_shape: tuple,
) -> numpy.ndarray:
    """Upsample a 3D array back to original shape using map_coordinates.

    Matches CellProfiler's approach for restoring reconstructed images
    to the original label resolution.

    Parameters
    ----------
    data : numpy.ndarray
        Subsampled 3D array to upsample.
    subsampled_shape : numpy.ndarray
        Shape of the subsampled space (float array, preserves CellProfiler
        precision).
    original_shape : tuple
        Target shape to upsample to.

    Returns
    -------
    numpy.ndarray
        Upsampled array at original_shape resolution.

    """
    k, i, j = numpy.mgrid[
        0 : original_shape[0],
        0 : original_shape[1],
        0 : original_shape[2],
    ].astype(float)
    if original_shape[0] > 1:
        k *= float(subsampled_shape[0] - 1) / float(original_shape[0] - 1)
    if original_shape[1] > 1:
        i *= float(subsampled_shape[1] - 1) / float(original_shape[1] - 1)
    if original_shape[2] > 1:
        j *= float(subsampled_shape[2] - 1) / float(original_shape[2] - 1)
    return scipy.ndimage.map_coordinates(data, (k, i, j), order=1)


def compute_granularity(  # noqa: C901, PLR0912, PLR0913, PLR0915
    object_loader: ObjectLoader,
    *,
    radius: int = 10,
    granular_spectrum_length: int = 16,
    subsample_size: float = 0.25,
    image_sample_size: float = 0.25,
    mask_threshold: float = 0.9,
    verbose: bool = False,
    image_mask: numpy.ndarray | None = None,
) -> pandas.DataFrame:
    """Calculate the granularity spectrum of a 3D image.

    Follows the CellProfiler MeasureGranularity algorithm exactly for 3D:
    1. Subsample the image uniformly (same factor for Z, Y, X).
    2. Further subsample for background tophat removal.
    3. Iteratively erode with ball(1) and reconstruct, measuring
    signal lost at each scale as image-level and per-object values.

    Parameters
    ----------
    object_loader : ObjectLoader
        Loader containing the image and label arrays.
    radius : int
        Radius of the structuring element for background removal.
        Should correspond to texture radius *after* subsampling.
    granular_spectrum_length : int
        Number of granularity scales to measure.
    subsample_size : float
        Subsampling factor for the image (0, 1]. Applied uniformly to Z/Y/X.
    image_sample_size : float
        Subsampling factor for background reduction (0, 1].
        Applied relative to the already-subsampled image.
    mask_threshold : float
        Threshold for converting interpolated masks back to boolean.
    verbose : bool
        Print diagnostic information.
    image_mask : numpy.ndarray or None
        Boolean mask matching the image shape. Corresponds to CellProfiler's
        ``im.mask``. If None (default), all pixels are considered valid
        (all-True mask), matching the typical CellProfiler behavior for
        unmasked images.
    channel : str or None
        Optional channel name for feature naming. If None, channel is not
        included in feature names.
    compartment : str or None
        Optional compartment name for feature naming. If None, compartment is
        not included in feature names.

    Returns
    -------
    pandas.DataFrame
        Wide-format DataFrame with one row per object and one column per
        granularity scale, plus Metadata columns.

    """
    # Validate inputs
    if subsample_size <= 0 or subsample_size > 1:
        raise ValueError(f"subsample_size must be in (0, 1], got {subsample_size}")
    if image_sample_size <= 0 or image_sample_size > 1:
        raise ValueError(
            f"image_sample_size must be in (0, 1], got {image_sample_size}",
        )
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if granular_spectrum_length <= 0:
        raise ValueError(
            "granular_spectrum_length must be positive, "
            f"got {granular_spectrum_length}",
        )

    # Get original data
    if object_loader.image is None or object_loader.label_image is None:
        return pandas.DataFrame()
    original_pixels = object_loader.image
    original_labels = object_loader.label_image
    original_shape = original_pixels.shape

    # Mask: CellProfiler uses im.mask (typically all-True for unmasked images)
    if image_mask is None:
        original_mask = numpy.ones(original_shape, dtype=bool)
    else:
        original_mask = image_mask.astype(bool)

    # ------------------------------------------------------------------
    # Step 1: Subsample image and mask (uniform factor for all axes)
    # CellProfiler: new_shape = shape * subsample_size
    #   coordinates = mgrid[0:new_shape] / subsample_size
    # ------------------------------------------------------------------
    new_shape = numpy.array(original_shape, dtype=float)

    if subsample_size < 1.0:
        new_shape = new_shape * subsample_size

        pixels = _subsample_3d(
            original_pixels,
            new_shape,
            subsample_factor=subsample_size,
            order=1,
        )
        mask = (
            _subsample_3d(
                original_mask.astype(float),
                new_shape,
                subsample_factor=subsample_size,
                order=1,
            )
            > mask_threshold
        )

        if verbose:
            print(
                f"Subsampled image: {original_shape} -> {pixels.shape} "
                f"(factor={subsample_size})",
            )
    else:
        pixels = original_pixels.copy()
        mask = original_mask.copy()

    # ------------------------------------------------------------------
    # Step 2: Background removal via tophat filter
    #
    # CellProfiler 3D BUG (replicated for compatibility):
    #   The 3D branch uses new_shape for grid bounds and subsample_size
    #   for coordinate division, instead of back_shape and
    #   image_sample_size as the 2D branch does. This means:
    #   - back_pixels has the SAME shape as pixels (not smaller)
    #   - Many coordinates are out of bounds → map_coordinates returns 0
    #   We replicate this exactly to match CellProfiler output.
    # ------------------------------------------------------------------
    if image_sample_size < 1.0:
        back_shape = new_shape * image_sample_size

        # CellProfiler 3D: mgrid[0:new_shape] / subsample_size
        # (NOT mgrid[0:back_shape] / image_sample_size as 2D does)
        k, i, j = (
            numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(
                float,
            )
            / subsample_size
        )
        back_pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
        back_mask = (
            scipy.ndimage.map_coordinates(mask.astype(float), (k, i, j))
            > mask_threshold
        )

        if verbose:
            print(
                f"Background subsampled: pixels {pixels.shape} -> "
                f"back_pixels {back_pixels.shape} "
                f"(image_sample_size={image_sample_size})",
            )
    else:
        back_pixels = pixels
        back_mask = mask
        back_shape = new_shape

    # Tophat filter: masked erosion + masked dilation
    footprint_bg = skimage.morphology.ball(radius, dtype=bool)

    back_pixels_masked = numpy.zeros_like(back_pixels)
    back_pixels_masked[back_mask] = back_pixels[back_mask]
    back_pixels = skimage.morphology.erosion(back_pixels_masked, footprint=footprint_bg)

    back_pixels_masked = numpy.zeros_like(back_pixels)
    back_pixels_masked[back_mask] = back_pixels[back_mask]
    back_pixels = skimage.morphology.dilation(
        back_pixels_masked,
        footprint=footprint_bg,
    )

    # Upsample background back to subsampled image size
    if image_sample_size < 1.0:
        # CellProfiler 3D: mgrid[0:new_shape] with coords scaled by
        # (back_shape - 1) / (new_shape - 1)
        k, i, j = numpy.mgrid[
            0 : new_shape[0],
            0 : new_shape[1],
            0 : new_shape[2],
        ].astype(float)
        if new_shape[0] > 1:
            k *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
        if new_shape[1] > 1:
            i *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
        if new_shape[2] > 1:
            j *= float(back_shape[2] - 1) / float(new_shape[2] - 1)
        back_pixels = scipy.ndimage.map_coordinates(back_pixels, (k, i, j), order=1)

    # Subtract background
    pixels = pixels - back_pixels
    pixels[pixels < 0] = 0

    if verbose:
        print("Background removed via tophat filter.")

    # ------------------------------------------------------------------
    # Step 3: Object initialization
    # CellProfiler computes per-object start_mean from the ORIGINAL image
    # (im.pixel_data) using the full-resolution label image, with labels
    # masked by im.mask: labels[~im.mask] = 0.
    # ------------------------------------------------------------------
    object_measurements: dict[str, list] = {
        "Metadata_Object_ObjectID": [],
        "feature": [],
        "value": [],
    }

    label_range = numpy.unique(original_labels[original_labels > 0])
    nobjects = len(label_range)

    if nobjects > 0:
        # CellProfiler: self.labels[~im.mask] = 0
        # When the mask covers the whole image (the common unmasked case,
        # image_mask=None) no labels get zeroed, so skip the full-array copy
        # and reuse original_labels directly. scipy.ndimage.mean does not
        # mutate its label input, so this is safe.
        if original_mask.all():
            masked_labels = original_labels
        else:
            masked_labels = original_labels.copy()
            masked_labels[~original_mask] = 0

        if numpy.any(masked_labels > 0):
            per_object_current_mean = _fix_scipy_ndimage_result(
                scipy.ndimage.mean(original_pixels, masked_labels, label_range),
            )
        else:
            per_object_current_mean = numpy.zeros(len(label_range))
        per_object_start_mean = per_object_current_mean.copy()
    else:
        label_range = numpy.array([], dtype=int)
        masked_labels = original_labels
        per_object_current_mean = numpy.array([])
        per_object_start_mean = numpy.array([])

    # ------------------------------------------------------------------
    # Step 4: Granular spectrum loop
    # CellProfiler computes startmean AFTER background subtraction but
    # BEFORE zeroing pixels outside mask (zeroing is implicit via indexing).
    # ------------------------------------------------------------------
    # Whether the (possibly subsampled) mask covers the whole image. In the
    # common unmasked case (image_mask=None) this is True, and several per-scale
    # operations below become no-ops or can avoid full-array copies: the masked
    # erosion zeroing, the rec[pixels[mask]] mean, and the startmean mean. Compute
    # it once here instead of re-evaluating mask.any()/mask.all() each scale.
    mask_all_true = bool(mask.all())
    startmean = (
        pixels.mean()
        if mask_all_true
        else (numpy.mean(pixels[mask]) if mask.any() else 0.0)
    )
    ero = pixels.copy()
    # Mask the test image so masked pixels have no effect during reconstruction
    ero[~mask] = 0
    currentmean = startmean

    # CellProfiler uses ball(1) for the iterative erosion/reconstruction loop
    footprint = skimage.morphology.ball(1, dtype=bool)

    if verbose:
        print(
            f"Image startmean: {startmean:.6f}, "
            f"Processing {nobjects} objects, "
            f"Spectrum length: {granular_spectrum_length}",
        )

    # Precomputing labeled-voxel positions once (instead of upsampling/
    # scanning the whole image every scale in the loop below) gives identical
    # per-object means for a fraction of the work when labeled objects cover
    # a small part of the image -- the common case. Coordinates depend only
    # on the fixed subsampled/original shapes (not on the per-scale ``rec``),
    # so this is computed once and reused for every ``map_coordinates`` call
    # below. See ``_labeled_voxel_positions`` for why this is equivalent.
    # When nobjects == 0 (no labeled objects at all) this block is skipped,
    # leaving labeled_voxel_coords/labeled_voxel_labels at their empty
    # defaults; upsample_coords below and the per-object branch in the
    # spectrum loop are both also gated on nobjects > 0, so no per-object
    # work is attempted and object_measurements stays empty. This mirrors
    # pre-optimization behavior: the returned DataFrame has zero rows but
    # keeps its Metadata_* columns (see
    # test_compute_granularity_zero_objects_returns_empty_dataframe).
    labeled_voxel_coords: tuple[numpy.ndarray, ...] = ()
    labeled_voxel_labels = numpy.array([], dtype=original_labels.dtype)
    if nobjects > 0:
        labeled_voxel_coords, labeled_voxel_labels = _labeled_voxel_positions(
            masked_labels,
        )
    have_labeled_voxels = labeled_voxel_labels.size > 0

    upsample_coords: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None = None
    if subsample_size < 1.0 and nobjects > 0:
        k = labeled_voxel_coords[0].astype(float)
        i = labeled_voxel_coords[1].astype(float)
        j = labeled_voxel_coords[2].astype(float)
        if original_shape[0] > 1:
            k *= float(new_shape[0] - 1) / float(original_shape[0] - 1)
        if original_shape[1] > 1:
            i *= float(new_shape[1] - 1) / float(original_shape[1] - 1)
        if original_shape[2] > 1:
            j *= float(new_shape[2] - 1) / float(original_shape[2] - 1)
        upsample_coords = (k, i, j)

    for scale in range(1, granular_spectrum_length + 1):
        prevmean = currentmean

        # Masked erosion: zero pixels outside the mask before eroding. When the
        # mask is all-True this is a no-op (ero is already zeroed at ~mask from
        # the prior iteration), so skip the full-array ``numpy.where`` copy and
        # erode ero directly.
        ero_marker = ero if mask_all_true else numpy.where(mask, ero, 0)
        ero = skimage.morphology.erosion(ero_marker, footprint=footprint)

        # Reconstruction
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)

        # Image-level granularity. When the mask is all-True, rec[mask] is all
        # of rec, so rec.mean() avoids the boolean-index copy that rec[mask]
        # would perform.
        currentmean = (
            rec.mean()
            if mask_all_true
            else (numpy.mean(rec[mask]) if mask.any() else 0.0)
        )
        gs = (prevmean - currentmean) * 100 / startmean if startmean > 0 else 0.0

        if verbose and scale == 1:
            print(f"Scale 1 - gs: {gs:.4f}, currentmean: {currentmean:.6f}")

        # ----------------------------------------------------------
        # Per-object granularity: upsample rec to original shape at only the
        # voxels that belong to a labeled object, then compute per-label
        # means using those voxels' label ids. Equivalent to upsampling the
        # whole image and calling scipy.ndimage.mean(rec_full, masked_labels,
        # label_range), since that call already discards everything outside
        # labeled_voxel_coords -- just without doing the discarded work.
        # ----------------------------------------------------------
        if nobjects > 0:
            if upsample_coords is not None:
                rec_at_object_voxels = scipy.ndimage.map_coordinates(
                    rec,
                    upsample_coords,
                    order=1,
                )
            else:
                rec_at_object_voxels = rec[labeled_voxel_coords]

            # Single-pass per-object mean via scipy.ndimage.mean
            if have_labeled_voxels:
                new_object_means = _fix_scipy_ndimage_result(
                    scipy.ndimage.mean(
                        rec_at_object_voxels,
                        labeled_voxel_labels,
                        label_range,
                    ),
                )
            else:
                new_object_means = numpy.zeros(len(label_range))

            # Granular spectrum: (prev - new) * 100 / start, per object
            # Guard against zero start mean — return 0 rather than dividing by eps
            _safe_denom = numpy.where(
                per_object_start_mean > 0,
                per_object_start_mean,
                1.0,
            )
            gss = numpy.where(
                per_object_start_mean > 0,
                (per_object_current_mean - new_object_means) * 100 / _safe_denom,
                0.0,
            )

            per_object_current_mean = new_object_means

            # Record measurements for each object
            for idx in range(len(label_range)):
                object_measurements["Metadata_Object_ObjectID"].append(
                    int(label_range[idx]),
                )
                object_measurements["feature"].append(scale)
                object_measurements["value"].append(float(gss[idx]))

    if verbose:
        n_total = len(object_measurements["Metadata_Object_ObjectID"])
        non_zero = sum(1 for v in object_measurements["value"] if v > 0)
        print(f"Total measurements: {n_total}")
        print(f"Non-zero measurements: {non_zero}")
        if non_zero > 0:
            vals = [v for v in object_measurements["value"] if v > 0]
            print(f"Mean granularity: {numpy.mean(vals):.2f}")

    final_df = pandas.DataFrame(object_measurements)
    final_df = final_df.pivot_table(
        index=["Metadata_Object_ObjectID"],
        columns=["feature"],
        values=["value"],
    )
    final_df.columns = final_df.columns.droplevel()
    final_df = final_df.reset_index()
    # prepend compartment and channel to column names
    final_df.rename(
        columns={
            col: format_morphology_feature_name(
                compartment=object_loader.compartment,
                channel=object_loader.channel,
                feature_type="Granularity",
                measurement=col,
            )
            if col != "Metadata_Object_ObjectID"
            else col
            for col in final_df.columns
        },
        inplace=True,
    )
    final_df.insert(
        0,
        "Metadata_Imaging_ImageID",
        object_loader.image_set_loader.image_id,
    )
    final_df.insert(
        0,
        "Metadata_Experiment_ImageSet",
        object_loader.image_set_loader.image_set_name,
    )
    result = final_df.to_dict(orient="list")
    for col in list(result.keys()):
        try:
            validate_column_name_schema(
                column_name=col,
                compartments=[object_loader.compartment],
                channels=[f"{object_loader.channel}"],
            )
        except ValueError as e:
            raise ValueError(f"Column name {col} does not conform to schema: {e}")

    return final_df
