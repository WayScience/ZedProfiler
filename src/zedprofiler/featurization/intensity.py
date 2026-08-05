"""Intensity feature extraction utilities for 3D image objects.

Provides functions to compute intensity statistics (mean, median, min, max,
standard deviation, quartiles), edge-based measurements, center-of-mass
coordinates, and mass displacement for segmented 3D objects.
"""

import numpy
import pandas
import scipy.ndimage
import skimage.measure
import skimage.segmentation

from zedprofiler.contracts import validate_column_name_schema
from zedprofiler.IO.feature_writing_utils import format_morphology_feature_name
from zedprofiler.IO.loading_classes import ObjectLoader


def get_outline(mask: numpy.ndarray) -> numpy.ndarray:
    """Get the outline of a 3D mask.

    Parameters
    ----------
    mask : numpy.ndarray
        The input mask.

    Returns
    -------
    numpy.ndarray
        The outline of the mask.

    """
    outline = numpy.zeros_like(mask)
    for z in range(mask.shape[0]):
        outline[z] = skimage.segmentation.find_boundaries(mask[z], mode="inner")
    return outline


def compute_intensity(  # noqa: PLR0915
    object_loader: ObjectLoader,
) -> pandas.DataFrame:
    """Measure the intensity of objects in a 3D image.

    Parameters
    ----------
    object_loader : ObjectLoader
        The object loader containing the image and label image.

    Returns
    -------
    pandas.DataFrame
        Wide-format DataFrame with one row per object and one column per
        intensity measurement, plus Metadata columns.

    """
    if object_loader.label_image is None or object_loader.image is None:
        return pandas.DataFrame()
    image_object = object_loader.image
    label_object = object_loader.label_image
    labels = object_loader.object_ids

    output_dict: dict[str, list] = {
        "Metadata_Object_ObjectID": [],
        "feature_name": [],
        "channel": [],
        "compartment": [],
        "value": [],
    }

    props = skimage.measure.regionprops_table(
        label_object,
        properties=["label", "bbox"],
    )
    label_to_bbox = {
        int(label): (
            int(props["bbox-0"][index]),
            int(props["bbox-1"][index]),
            int(props["bbox-2"][index]),
            int(props["bbox-3"][index]),
            int(props["bbox-4"][index]),
            int(props["bbox-5"][index]),
        )
        for index, label in enumerate(props.get("label", []))
    }

    # loop through each object and calculate measurements
    for label in labels:
        bbox = label_to_bbox.get(int(label))
        if bbox is None:
            continue
        bbox_min_z, bbox_min_y, bbox_min_x, bbox_max_z, bbox_max_y, bbox_max_x = bbox
        cropped_label_values = label_object[
            bbox_min_z:bbox_max_z,
            bbox_min_y:bbox_max_y,
            bbox_min_x:bbox_max_x,
        ]
        cropped_image_values = image_object[
            bbox_min_z:bbox_max_z,
            bbox_min_y:bbox_max_y,
            bbox_min_x:bbox_max_x,
        ]
        object_mask = cropped_label_values == label
        if not numpy.any(object_mask):
            continue

        object_pixels = cropped_image_values[object_mask]
        non_zero_pixels_object = object_pixels[object_pixels > 0]
        if non_zero_pixels_object.size == 0:
            non_zero_pixels_object = numpy.array([0], dtype=numpy.float32)

        cropped_label = object_mask.astype(numpy.uint8)
        cropped_image = numpy.where(object_mask, cropped_image_values, 0)

        padded_label = numpy.pad(cropped_label, pad_width=1, mode="constant")
        mask_outlines = get_outline(padded_label)[1:-1, 1:-1, 1:-1]

        # Create coordinate grids for the bounding box
        mesh_z, mesh_y, mesh_x = numpy.mgrid[
            bbox_min_z:bbox_max_z,
            bbox_min_y:bbox_max_y,
            bbox_min_x:bbox_max_x,
        ]

        # calculate the integrated intensity
        integrated_intensity = numpy.sum(object_pixels)
        # calculate the volume
        volume = numpy.sum(object_mask)

        # Skip if volume is zero to avoid division by zero
        if volume == 0 or integrated_intensity == 0:
            continue

        # calculate the mean intensity
        mean_intensity = integrated_intensity / volume
        # calculate the standard deviation
        std_intensity = numpy.std(non_zero_pixels_object)
        # min intensity
        min_intensity = numpy.min(non_zero_pixels_object)
        # max intensity
        max_intensity = numpy.max(non_zero_pixels_object)
        # lower quartile
        lower_quartile_intensity = numpy.percentile(non_zero_pixels_object, 25)
        # upper quartile
        upper_quartile_intensity = numpy.percentile(non_zero_pixels_object, 75)
        # median intensity
        median_intensity = numpy.median(non_zero_pixels_object)
        # location of maximum intensity pixel (z, y, x)
        max_position = numpy.unravel_index(
            numpy.argmax(cropped_image),
            cropped_image.shape,
        )
        max_intensity_z = bbox_min_z + max_position[0]
        max_intensity_y = bbox_min_y + max_position[1]
        max_intensity_x = bbox_min_x + max_position[2]

        # Calculate center of mass (geometric center) using cropped arrays
        cm_x = numpy.mean(mesh_x[object_mask])
        cm_y = numpy.mean(mesh_y[object_mask])
        cm_z = numpy.mean(mesh_z[object_mask])

        # Calculate intensity-weighted center of mass using cropped arrays
        intensity_x_coord = cropped_image * mesh_x
        intensity_y_coord = cropped_image * mesh_y
        intensity_z_coord = cropped_image * mesh_z
        i_x = numpy.sum(intensity_x_coord[object_mask])
        i_y = numpy.sum(intensity_y_coord[object_mask])
        i_z = numpy.sum(intensity_z_coord[object_mask])
        # calculate the center of mass
        cmi_x = i_x / integrated_intensity
        cmi_y = i_y / integrated_intensity
        cmi_z = i_z / integrated_intensity
        # calculate the center of mass distance
        diff_x = cm_x - cmi_x
        diff_y = cm_y - cmi_y
        diff_z = cm_z - cmi_z
        # mass displacement
        mass_displacement = numpy.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
        # mean absolute deviation
        mad_intensity = numpy.mean(numpy.abs(non_zero_pixels_object - mean_intensity))
        edge_count = scipy.ndimage.sum(mask_outlines)
        edge_pixels = cropped_image[mask_outlines > 0]
        integrated_intensity_edge = numpy.sum(edge_pixels)
        mean_intensity_edge = integrated_intensity_edge / edge_count
        std_intensity_edge = numpy.std(edge_pixels)
        min_intensity_edge = numpy.min(edge_pixels)
        max_intensity_edge = numpy.max(edge_pixels)
        measurements_dict = {
            "IntegratedIntensity": integrated_intensity,
            "MeanIntensity": mean_intensity,
            "StdIntensity": std_intensity,
            "MinIntensity": min_intensity,
            "MaxIntensity": max_intensity,
            "LowerQuartileIntensity": lower_quartile_intensity,
            "UpperQuartileIntensity": upper_quartile_intensity,
            "MedianIntensity": median_intensity,
            "MassDisplacement": mass_displacement,
            "MeanAbsoluteDeviationIntensity": mad_intensity,
            "IntegratedIntensityEdge": integrated_intensity_edge,
            "MeanIntensityEdge": mean_intensity_edge,
            "StdIntensityEdge": std_intensity_edge,
            "MinIntensityEdge": min_intensity_edge,
            "MaxIntensityEdge": max_intensity_edge,
            "MaxZ": max_intensity_z,
            "MaxY": max_intensity_y,
            "MaxX": max_intensity_x,
            "CMI.X": cmi_x,
            "CMI.Y": cmi_y,
            "CMI.Z": cmi_z,
        }

        for feature_name, measurement_value in measurements_dict.items():
            coerced_value = numpy.float32(measurement_value)
            output_dict["Metadata_Object_ObjectID"].append(numpy.int32(label))
            output_dict["feature_name"].append(feature_name)
            output_dict["channel"].append(object_loader.channel)
            output_dict["compartment"].append(object_loader.compartment)
            output_dict["value"].append(coerced_value)
    final_df = pandas.DataFrame(output_dict)
    # prepend compartment and channel to column names
    final_df = final_df.pivot(
        index=["Metadata_Object_ObjectID"],
        columns="feature_name",
        values="value",
    ).reset_index()
    final_df.rename(
        columns={
            col: format_morphology_feature_name(
                compartment=object_loader.compartment,
                channel=object_loader.channel,
                feature_type="Intensity",
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
        "Metadata_Experiment_ImageSet",
        object_loader.image_set_loader.image_set_name,
    )

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
