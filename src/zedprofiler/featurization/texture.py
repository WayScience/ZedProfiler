"""This module generates texture features for each object in the
image using Haralick features.

We do this in a as close to zero-copy way as possible.
We want to make this module fast, memory efficient, and robust to large images
and objects.
We want this module to be python api callable and scalable.
"""

import contextlib

import mahotas
import numpy
import pandas
import skimage
import skimage.measure

from zedprofiler.contracts import validate_column_name_schema
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
    try:
        out_range = outrange_mapping.get(num_gray_levels)
    except KeyError:
        out_range = None
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


def compute_texture(  # noqa: C901
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
        return pandas.DataFrame()
    label_object = object_loader.label_image
    labels = object_loader.object_ids
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
        image_object = scale_image(image_object, num_gray_levels=grayscale)
        with contextlib.suppress(ValueError):
            # calculates 13 Haralick features for each direction (13)
            #  and each object, and stores them in a 3D array
            features[:, :, idx] = mahotas.features.haralick(
                ignore_zeros=True,
                f=image_object,
                distance=distance,
                compute_14th_feature=False,
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
