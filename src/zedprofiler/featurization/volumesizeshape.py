"""Volume, size, and shape features for 3D objects."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from types import ModuleType
from typing import Protocol

import numpy as np
import pandas

from zedprofiler.contracts import validate_column_name_schema
from zedprofiler.exceptions import ZedProfilerError
from zedprofiler.IO.feature_writing_utils import format_morphology_feature_name


class SupportsImageSetLoader(Protocol):
    """Minimal image-set loader interface required by this module."""

    # voxel size in z,y,x space
    anisotropy_spacing: tuple[float, float, float]


class _SupportsImageSetName(Protocol):
    """Minimal image-set-loader interface for name access."""

    image_set_name: str | None


class SupportsObjectLoader(Protocol):
    """Minimal object loader interface required by this module."""

    # label image the image which contains the labeled objects,
    # where each object is represented by a unique integer label
    # (0 is typically reserved for background)
    label_image: np.ndarray
    object_ids: Sequence[int]
    compartment: str
    channel: str
    image_set_loader: _SupportsImageSetName


def _empty_feature_result() -> dict[str, list[float]]:
    """Return deterministic empty output schema for area/size/shape features."""
    return {
        "Metadata_Object_ObjectID": [],
        "Volume": [],
        "CenterX": [],
        "CenterY": [],
        "CenterZ": [],
        "BboxVolume": [],
        "MinX": [],
        "MaxX": [],
        "MinY": [],
        "MaxY": [],
        "MinZ": [],
        "MaxZ": [],
        "Extent": [],
        "EulerNumber": [],
        "EquivalentDiameter": [],
        "SurfaceArea": [],
    }


def compute_volume_size_shape(
    image_set_loader: SupportsImageSetLoader | None = None,
    object_loader: SupportsObjectLoader | None = None,
) -> pandas.DataFrame:
    """Compute volume/size/shape features for one object loader.

    Supports two invocation modes:

    - no arguments: returns an empty deterministic schema so dispatchers can
      call the function without crashing.
    - both loaders provided: executes feature extraction.
    """
    if image_set_loader is None and object_loader is None:
        return pandas.DataFrame(_empty_feature_result())
    if image_set_loader is None or object_loader is None:
        raise ZedProfilerError(
            "volumesizeshape.compute requires both image_set_loader and "
            "object_loader for execution.",
        )

    return measure_3D_volume_size_shape(
        image_set_loader=image_set_loader,
        object_loader=object_loader,
    )


def _get_skimage_measure() -> ModuleType:
    """Return `skimage.measure` or raise a user-facing dependency error."""
    try:
        return import_module("skimage.measure")
    except ImportError as exc:
        raise ZedProfilerError(
            "volumesizeshape requires scikit-image for area/size/shape computation.",
        ) from exc


def calculate_surface_area(
    label_object: np.ndarray,
    props: dict[str, np.ndarray],
    spacing: tuple[float, float, float],
) -> float:
    """Calculate surface area for one labeled object using marching cubes."""
    measure = _get_skimage_measure()

    volume = label_object[
        max(props["bbox-0"][0], 0) : min(props["bbox-3"][0], label_object.shape[0]),
        max(props["bbox-1"][0], 0) : min(props["bbox-4"][0], label_object.shape[1]),
        max(props["bbox-2"][0], 0) : min(props["bbox-5"][0], label_object.shape[2]),
    ]
    volume_truths = volume > 0
    verts, faces, _normals, _values = measure.marching_cubes(
        volume_truths,
        method="lewiner",
        spacing=spacing,
        level=0,
    )
    return measure.mesh_surface_area(verts, faces)


def measure_3D_volume_size_shape(
    image_set_loader: SupportsImageSetLoader,
    object_loader: SupportsObjectLoader,
) -> pandas.DataFrame:
    """Measure volume/size/shape features for each non-zero label object."""
    measure = _get_skimage_measure()

    label_object = object_loader.label_image
    spacing = image_set_loader.anisotropy_spacing
    unique_objects = object_loader.object_ids

    features_to_record = _empty_feature_result()

    desired_properties = [
        "area",  # for 3D it is volume but skimage uses "area" naming for the property
        "bbox",
        "centroid",
        "bbox_area",
        "extent",
        "euler_number",
        "equivalent_diameter",
    ]
    for label in unique_objects:
        # avoid the 0 index which is the background and not an object,
        if label == 0:
            continue
        subset_lab_object = label_object.copy()
        # subset here means zeroing out all other objects except the
        # one we want to measure, so that we can use
        # skimage's regionprops_table to compute
        # features for that object
        subset_lab_object[subset_lab_object != label] = 0
        props = measure.regionprops_table(
            subset_lab_object,
            properties=desired_properties,
        )

        features_to_record["Metadata_Object_ObjectID"].append(label)
        features_to_record["Volume"].append(props["area"].item())
        features_to_record["CenterX"].append(props["centroid-2"].item())
        features_to_record["CenterY"].append(props["centroid-1"].item())
        features_to_record["CenterZ"].append(props["centroid-0"].item())
        features_to_record["BboxVolume"].append(props["bbox_area"].item())
        features_to_record["MinX"].append(props["bbox-2"].item())
        features_to_record["MaxX"].append(props["bbox-5"].item())
        features_to_record["MinY"].append(props["bbox-1"].item())
        features_to_record["MaxY"].append(props["bbox-4"].item())
        features_to_record["MinZ"].append(props["bbox-0"].item())
        features_to_record["MaxZ"].append(props["bbox-3"].item())
        features_to_record["Extent"].append(props["extent"].item())
        features_to_record["EulerNumber"].append(props["euler_number"].item())
        features_to_record["EquivalentDiameter"].append(
            props["equivalent_diameter"].item(),
        )

        try:
            features_to_record["SurfaceArea"].append(
                calculate_surface_area(
                    label_object=subset_lab_object,
                    props=props,
                    spacing=spacing,
                ),
            )
        except (RuntimeError, ValueError):
            features_to_record["SurfaceArea"].append(np.nan)

    final_df = pandas.DataFrame(features_to_record)

    # prepend compartment and channel to column names
    final_df.rename(
        columns={
            col: format_morphology_feature_name(
                compartment=object_loader.compartment,
                channel=object_loader.channel,
                feature_type="VolumeSizeShape",
                measurement=col,
            )
            if col != "Metadata_Object_ObjectID"
            else col
            for col in final_df.columns
        },
        inplace=True,
    )

    final_df.insert(
        1,
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
