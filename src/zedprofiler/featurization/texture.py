"""This module generates texture features for each object in the
image using Haralick features.

We do this in a as close to zero-copy way as possible.
We want to make this module fast, memory efficient, and robust to large images
and objects.
We want this module to be python api callable and scalable.
"""

import warnings

import mahotas
import numpy
import pandas
import scipy.ndimage
import skimage
import skimage.measure

from zedprofiler.contracts import (
    validate_anisotropy_factor_with_pydantic,
    validate_column_name_schema,
)
from zedprofiler.IO.feature_writing_utils import format_morphology_feature_name
from zedprofiler.IO.loading_classes import ObjectLoader


def scale_image(image: numpy.ndarray, num_gray_levels: int = 256) -> numpy.ndarray:
    """Scale the image to a specified number of gray levels.
    Example: 1024 gray levels will be scaled to 256 gray levels if
    num_gray_levels=256.
    An image with a pixel value of 0 will be scaled to 0 and a pixel value
    of 1023 will be scaled to 255.

    Parameters
    ----------
    image : numpy.ndarray
        The input image to be scaled. Can be a ndarray of any shape.
    num_gray_levels : int, optional
        The number of gray levels to scale the image to, by default 256

    Returns
    -------
    numpy.ndarray
        The gray level scaled image of any shape.

    """
    outrange_mapping = {
        256: "uint8",
        65536: "uint16",
    }
    out_range = outrange_mapping.get(num_gray_levels)
    if out_range is None:
        raise ValueError(
            f"Unsupported num_gray_levels: {num_gray_levels}. "
            f"Supported values are: {list(outrange_mapping.keys())}",
        )
    # scale the image to the requested gray levels
    return skimage.exposure.rescale_intensity(
        image,
        in_range="image",
        out_range=out_range,
    )


def resample_to_isotropic(
    image: numpy.ndarray,
    anisotropy_factor: float,
    order: int = 3,
) -> numpy.ndarray:
    """Resample a (z, y, x) volume to isotropic voxel spacing along z.
    This function is written to be used for both the signal image
    and the mask image.
    The order parameter controls the interpolation
    for the resampling.
    For the signal image, we use cubic spline interpolation (order=3).
    For the mask image, we use nearest neighbor interpolation (order=0).

    mahotas.features.haralick's ``distance`` parameter is a voxel count, not
    a physical length, and several of its 13 directions step along z. If z
    spacing is coarser than x/y spacing (the common microscopy case), those
    directions sample a larger physical distance than the in-plane
    directions, which biases the resulting Haralick features. Stretching the
    z axis by the anisotropy factor before computing texture makes "1 voxel"
    represent the same physical distance in every direction.

    Parameters
    ----------
    image : numpy.ndarray
        3D array in (z, y, x) order.
    anisotropy_factor : float
        Ratio of z-spacing to x/y-spacing (assumes isotropic x/y spacing).
    order : int, optional
        Interpolation order for the resampling, by default 1 (linear).

    Returns
    -------
    numpy.ndarray
        The volume resampled so that voxels are isotropic in physical space.

    """
    if anisotropy_factor == 1:
        return image

    input_dtype = image.dtype
    if numpy.issubdtype(input_dtype, numpy.integer):
        image = image.astype(numpy.float32)

    return scipy.ndimage.zoom(
        image,
        zoom=(anisotropy_factor, 1.0, 1.0),
        order=order,
    )


def compute_texture(  # noqa: C901, PLR0915
    object_loader: ObjectLoader,
    distance: int = 1,
    grayscale: int = 256,
) -> pandas.DataFrame:
    """Calculate texture features for each object in the image using Haralick features.

    The features are calculated for each object separately and the mean value
    is returned.

    Parameters
    ----------
    object_loader : ObjectLoader
        The object loader containing the image and object information.
    distance : int, optional
        The distance parameter for Haralick features, by default 1
    grayscale : int, optional
        The number of gray levels to scale the image to, by default 256

    Returns
    -------
    pandas.DataFrame
        Wide-format DataFrame with one row per object and one column per
        Haralick texture feature (direction x feature_name x distance x grayscale),
        plus Metadata columns. Feature names follow the pattern:
        ``<compartment>_<channel>_Texture_<feature>-<distance>-<direction>-<grayscale>``

        Haralick features measured per direction:
        AngularSecondMoment, Contrast, Correlation, Variance,
        InverseDifferenceMoment, SumAverage, SumVariance, SumEntropy,
        Entropy, DifferenceVariance, DifferenceEntropy,
        InformationMeasureOfCorrelation1, InformationMeasureOfCorrelation2

    """
    if object_loader.label_image is None or object_loader.image is None:
        return pandas.DataFrame(
            {
                "Metadata_Experiment_ImageSet": [],
                "Metadata_Imaging_ImageID": [],
                "Metadata_Object_ObjectID": [],
            },
        )
    label_object = object_loader.label_image
    labels = object_loader.object_ids
    # Haralick's `distance` is a voxel count, not a physical length, so
    # anisotropic z-spacing must be corrected for before computing texture
    # (see resample_to_isotropic).
    z_spacing, y_spacing, _x_spacing = object_loader.image_set_loader.anisotropy_spacing
    anisotropy_factor = validate_anisotropy_factor_with_pydantic(
        z_spacing / y_spacing,
    ).anisotropy_factor
    feature_names = [
        "AngularSecondMoment",
        "Contrast",
        "Correlation",
        "Variance",
        "InverseDifferenceMoment",
        "SumAverage",
        "SumVariance",
        "SumEntropy",
        "Entropy",
        "DifferenceVariance",
        "DifferenceEntropy",
        "InformationMeasureOfCorrelation1",
        "InformationMeasureOfCorrelation2",
    ]
    # set the number of directions based on the dimensionality of the image
    n_directions = 13

    output_texture_dict: dict[str, list] = {
        "Metadata_Object_ObjectID": [],
        "texture_name": [],
        "texture_value": [],
    }
    # Precompute bboxes for labeled regions to avoid per-object full-array copies.
    props = skimage.measure.regionprops_table(
        label_object,
        properties=["label", "bbox"],
    )
    # Map label id to bbox (z0, y0, x0, z1, y1, x1)
    label_to_bbox = {}
    labels_prop = props.get("label", [])
    for i, lbl in enumerate(labels_prop):
        label_to_bbox[int(lbl)] = (
            int(props["bbox-0"][i]),
            int(props["bbox-1"][i]),
            int(props["bbox-2"][i]),
            int(props["bbox-3"][i]),
            int(props["bbox-4"][i]),
            int(props["bbox-5"][i]),
        )
    # loop through each label and get the bounding box
    # to compute features for the object
    label_to_idx = {int(lbl): i for i, lbl in enumerate(labels)}

    # Allocate once before the loop so each label's slot persists
    features = numpy.full((n_directions, 13, len(labels)), numpy.nan)

    for label in labels:
        if int(label) == 0:
            continue
        idx = label_to_idx[int(label)]
        bbox = label_to_bbox.get(int(label))
        if bbox is None:
            continue

        min_z, min_y, min_x, max_z, max_y, max_x = bbox

        # Crop to the object's bounding box (skimage bboxes are half-open)
        image_object = object_loader.image[min_z:max_z, min_y:max_y, min_x:max_x].copy()
        selected_label_object = label_object[min_z:max_z, min_y:max_y, min_x:max_x]
        object_mask = selected_label_object == label
        if not numpy.any(object_mask):
            continue
        image_object[~object_mask] = 0

        # order of operations here are as follows:
        # 1. resample to isotropic voxel spacing
        # a. this will interpolate the image, which will bleed over
        # the object edges into the background, so we need to remask
        # the image after resampling
        # 2. resample the mask to isotropic
        # voxel spacing (nearest neighbor)
        # 3. remask the resampled image to ensure background is zero
        #    this removes the bleed over from the interpolation
        # of the object edges
        # 4. mahotas can now use the resampled image and mask to
        # compute the Haralick features for this object

        # resample to isotropic after getting the object,
        # this avoid the need to interpolate the mask
        # image interpolation is done with
        # cubic spline interpolation (order=3)
        # mask interpolation is done with nearest neighbor (order=0)
        image_object = resample_to_isotropic(
            image_object,
            anisotropy_factor=anisotropy_factor,
            order=3,  # cubic spline interpolation for image
        )
        resampled_mask = resample_to_isotropic(
            object_mask,
            anisotropy_factor=anisotropy_factor,
            order=0,  # nearest neighbor for mask
        )
        # remask the resampled image to ensure background is zero
        # this removes the bleed over from the interpolation of the object edges
        image_object[~resampled_mask.astype(bool)] = 0

        image_object = scale_image(image_object, num_gray_levels=grayscale)
        try:
            # calculates 13 Haralick features for each direction (13)
            #  and each object, and stores them in a 3D array
            features[:, :, idx] = mahotas.features.haralick(
                ignore_zeros=True,
                f=image_object,
                distance=distance,
                compute_14th_feature=False,
            )
        except ValueError:
            # mahotas cannot compute GLCM features when an object's extent
            # is smaller than the distance parameter; the object's texture
            # values remain NaN in this case.
            warnings.warn(
                f"Object {label} is smaller than distance={distance}; "
                "Texture features are undefined (NaN) for this object.",
                stacklevel=2,
            )
    # iterate through the direction, feature, and object dimensions
    # of the features array to populate the output dictionary
    for direction, direction_features in enumerate(features):
        direction_str = f"{direction:02d}"
        for feature_name, feature in zip(feature_names, direction_features):
            for object_id in labels:
                output_texture_dict["Metadata_Object_ObjectID"].append(object_id)
                output_texture_dict["texture_name"].append(
                    f"{feature_name}-{distance}-{direction_str}-{grayscale}",
                )
                output_texture_dict["texture_value"].append(
                    feature[label_to_idx[int(object_id)]]
                )
    final_df = pandas.DataFrame(output_texture_dict)

    final_df = final_df.pivot(
        index="Metadata_Object_ObjectID",
        columns="texture_name",
        values="texture_value",
    )
    final_df.reset_index(inplace=True)
    final_df.rename(
        columns={
            col: format_morphology_feature_name(
                compartment=object_loader.compartment,
                channel=object_loader.channel,
                feature_type="Texture",
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
    final_df.columns.name = None

    # validate column names against schema
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
