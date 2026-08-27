"""Core data contracts shared across featurizers.

We have contracts for the data inputs and outputs of featurizers,
as well as for column names in the output DataFrame.
These contracts are enforced through a combination of Pydantic models,
Beartypes Pandera schemas.

Where inputs are the image path or arrays.
Outputs are the feature DataFrame and metadata dictionary/Dfs.

The package accepts:
- Single-channel 3D arrays shaped (z, y, x)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandera.pandas as pa
from beartype import beartype
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from zedprofiler.exceptions import ContractError

MIN_ANISOTROPY_FACTOR = 1
EXPECTED_SPATIAL_DIMS = 3
TWO_DIMENSIONAL = 2
FOUR_DIMENSIONAL = 4
FIVE_OR_MORE_DIMENSIONS = 5
NON_METADATA_UNDERSCORE_SEPARATED_PARTS = 4
METADATA_UNDERSCORE_SEPARATED_PARTS = 3
REQUIRED_RETURN_KEYS = ("image_array", "features", "metadata")
FEATURES = [
    "Colocalization",
    "Granularity",
    "Texture",
    "Intensity",
    "Neighbors",
    "VolumeSizeShape",
]

# Pandera schema for validating numpy arrays with expected dimensionality
ImageArraySchema = pa.DataFrameSchema(
    columns={},
    checks=[
        pa.Check(
            lambda x: len(x.shape) in [3, 4, 5],
            error="Array must have 3, 4, or 5 dimensions",
        ),
    ],
    name="ImageArray",
)


class ImageArrayModel(BaseModel):
    """Pydantic model for validating image arrays."""

    array: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("array", mode="before")
    @classmethod
    def validate_array_type(_cls, v: object) -> np.ndarray:
        """Ensure array is a numpy array."""
        if not isinstance(v, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(v)}")
        return v

    @field_validator("array", mode="after")
    @classmethod
    def validate_array_dtype_and_shape(_cls, arr: np.ndarray) -> np.ndarray:
        """Validate array dtype is numeric and shape is valid."""
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"Array dtype must be numeric, got {arr.dtype}")

        arr_shape = arr.shape
        # Check dimensionality
        if len(arr_shape) == TWO_DIMENSIONAL:
            raise ValueError(
                f"Input array has shape {arr_shape} with {TWO_DIMENSIONAL} "
                f"dimensions. Expected {EXPECTED_SPATIAL_DIMS} dimensions.",
            )
        if len(arr_shape) == FOUR_DIMENSIONAL and arr_shape[0] > 1:
            raise ValueError(
                f"Input array has shape {arr_shape} with {FOUR_DIMENSIONAL} "
                "dimensions, but the first dimension (channels) has size "
                f"{arr_shape[0]}. Expected a single-channel 3D array.",
            )
        if (
            len(arr_shape) >= FIVE_OR_MORE_DIMENSIONS
            and arr_shape[0] > 1
            and arr_shape[1] > 1
        ):
            raise ValueError(
                f"Input array has shape {arr_shape} with {len(arr_shape)} "
                f"dimensions. Expected {EXPECTED_SPATIAL_DIMS} dimensions.",
            )

        # Check all dimensions are positive
        for dim_size in arr_shape:
            if dim_size <= 0:
                raise ValueError(
                    f"Input array has shape {arr_shape} with non-positive "
                    "dimension size. All dimensions must have size greater than 0.",
                )

        # Check that the array is not a degenerate single-voxel volume
        if sum(arr_shape) == len(arr_shape):
            raise ValueError(
                f"Input array has shape {arr_shape} with all dimensions equal to 1. "
                "Expected all three dimensions to have size greater than 1.",
            )
        if not validate_image_array_shape_contracts(arr):
            raise ValueError(
                f"Input array with shape {arr_shape} failed shape contract validation.",
            )
        if not validate_image_array_type_contracts(arr):
            raise ValueError(
                f"Input array with dtype {arr.dtype} failed type contract validation.",
            )
        return arr


class AnisotropyFactorModel(BaseModel):
    """Pydantic model for validating the anisotropy factor.

    Feature modules (e.g. texture, neighbors) assume the z-spacing is never
    finer than the x/y-spacing, so ``anisotropy_factor`` (z_spacing /
    y_spacing) must be 1 or greater. ``anisotropy_factor`` collapses to a
    single z/y ratio, so it is only meaningful when the transverse (y, x)
    spacings are equal; ``y_spacing``/``x_spacing`` are optional and, when
    both are provided, are checked for equality.
    """

    anisotropy_factor: float
    y_spacing: float | None = None
    x_spacing: float | None = None

    @field_validator("anisotropy_factor", mode="after")
    @classmethod
    def validate_at_least_one(_cls, value: float) -> float:
        """Ensure the anisotropy factor is finite and 1 or greater."""
        if not math.isfinite(value):
            raise ValueError(
                f"Anisotropy factor must be a finite number, got {value}.",
            )
        if value < MIN_ANISOTROPY_FACTOR:
            raise ValueError(
                f"Anisotropy factor must be {MIN_ANISOTROPY_FACTOR} or greater, "
                f"got {value}.",
            )
        return value

    @model_validator(mode="after")
    def validate_transverse_spacing_equal(self) -> AnisotropyFactorModel:
        """Ensure y/x spacing are equal, when both are provided."""
        if (
            self.y_spacing is not None
            and self.x_spacing is not None
            and self.y_spacing != self.x_spacing
        ):
            raise ValueError(
                "anisotropy_factor is a single z/y ratio and requires equal "
                f"transverse spacing, got y_spacing={self.y_spacing} and "
                f"x_spacing={self.x_spacing}.",
            )
        return self


class FeatureDictModel(BaseModel):
    """Pydantic model for validating feature dictionaries."""

    features: dict[str, Any]

    @field_validator("features", mode="before")
    @classmethod
    def validate_is_dict(_cls, v: object) -> dict[str, Any]:
        """Ensure features is a dictionary."""
        if not isinstance(v, dict):
            raise ValueError(f"Expected dict, got {type(v)}")
        return v


class MetadataDictModel(BaseModel):
    """Pydantic model for validating metadata dictionaries."""

    metadata: dict[str, Any]

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_is_dict(_cls, v: object) -> dict[str, Any]:
        """Ensure metadata is a dictionary."""
        if not isinstance(v, dict):
            raise ValueError(f"Expected dict, got {type(v)}")
        return v


class ReturnSchemaModel(BaseModel):
    """Pydantic model for validating return schema."""

    result: dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("result", mode="before")
    @classmethod
    def validate_is_dict(_cls, v: object) -> dict[str, Any]:
        """Ensure result is a dictionary."""
        if not isinstance(v, dict):
            raise ValueError(f"Expected dict, got {type(v)}")
        return v

    @field_validator("result", mode="after")
    @classmethod
    def validate_keys_and_types(_cls, result: dict[str, Any]) -> dict[str, Any]:
        """Validate result has correct keys in correct order and correct types."""
        actual_keys = tuple(result.keys())
        if actual_keys != REQUIRED_RETURN_KEYS:
            raise ValueError(
                "Return result keys must match required deterministic order "
                f"{REQUIRED_RETURN_KEYS}, got {actual_keys}.",
            )

        if not isinstance(result["image_array"], np.ndarray):
            raise ValueError(
                f"Return result key 'image_array' must be a numpy array, "
                f"got {type(result['image_array'])}",
            )
        if not isinstance(result["features"], dict):
            raise ValueError(
                f"Return result key 'features' must be a dict, "
                f"got {type(result['features'])}",
            )
        if not isinstance(result["metadata"], dict):
            raise ValueError(
                f"Return result key 'metadata' must be a dict, "
                f"got {type(result['metadata'])}",
            )

        return result


class ColumnNameModel(BaseModel):
    """Pydantic model for parsing and validating column names."""

    column_name: str
    compartment: str | None = None
    channel: str | None = None
    feature: str | None = None

    @field_validator("column_name", mode="before")
    @classmethod
    def validate_is_string(_cls, v: object) -> str:
        """Ensure column_name is a string."""
        if not isinstance(v, str):
            raise ValueError(f"Expected string, got {type(v)}")
        return v

    @model_validator(mode="after")
    def parse_column_name(self) -> ColumnNameModel:
        """Parse column name into components."""
        parts = self.column_name.split("_")

        if "Metadata" in self.column_name:
            if len(parts) < METADATA_UNDERSCORE_SEPARATED_PARTS:
                raise ValueError(
                    "Metadata column name must have at least "
                    f"{METADATA_UNDERSCORE_SEPARATED_PARTS} "
                    "parts separated by underscores, "
                    f"got {len(parts)} parts in '{self.column_name}'",
                )
            # Don't parse compartment/channel/feature for metadata columns
            return self

        if len(parts) < NON_METADATA_UNDERSCORE_SEPARATED_PARTS:
            raise ValueError(
                "Column name must have at least "
                f"{NON_METADATA_UNDERSCORE_SEPARATED_PARTS} "
                "parts separated by underscores, "
                f"got {len(parts)} parts in '{self.column_name}'",
            )

        self.compartment = parts[0]
        self.channel = parts[1]
        self.feature = parts[2]

        return self


@beartype
def validate_image_array_shape_contracts(
    arr: np.ndarray,
) -> bool:
    """Validate the input array for dimensionality

    Parameters
    ----------
    arr : np.ndarray
        Input array to validate

    Returns
    -------
    bool
        The status of the validation

    Raises
    ------
    ContractError
        If the input array does not meet the expected contract

    """
    arr_shape = arr.shape
    if len(arr_shape) == TWO_DIMENSIONAL:
        raise ContractError(
            f"Input array has shape {arr_shape} with {TWO_DIMENSIONAL} dimensions. "
            f"Expected {EXPECTED_SPATIAL_DIMS} dimensions.",
        )
    if len(arr_shape) == FOUR_DIMENSIONAL and arr_shape[0] > 1:
        raise ContractError(
            f"Input array has shape {arr_shape} with {FOUR_DIMENSIONAL} dimensions, "
            "but the first dimension (channels) has size "
            f"{arr_shape[0]}. Expected a single-channel 3D array.",
        )
    if (
        len(arr_shape) >= FIVE_OR_MORE_DIMENSIONS
        and arr_shape[0] > 1
        and arr_shape[1] > 1
    ):
        raise ContractError(
            f"Input array has shape {arr_shape} with {len(arr_shape)} dimensions. "
            f"Expected {EXPECTED_SPATIAL_DIMS} dimensions.",
        )

    for dim_size in arr_shape:
        if dim_size <= 0:
            raise ContractError(
                f"Input array has shape {arr_shape} with non-positive dimension size. "
                "All dimensions must have size greater than 0.",
            )
    if sum(arr_shape) == len(arr_shape):
        raise ContractError(
            f"Input array has shape {arr_shape} with all dimensions equal to 1. "
            "Expected all three dimensions to have size greater than 1.",
        )
    return True


@beartype
def validate_image_array_type_contracts(
    arr: np.ndarray,
) -> bool:
    """Validate the input array for type

    Parameters
    ----------
    arr : np.ndarray
        Input array to validate

    Returns
    -------
    bool
        The status of the validation

    Raises
    ------
    ContractError
        If the input array does not meet the expected contract

    """
    if not isinstance(arr, np.ndarray):
        raise ContractError(f"Input is of type {type(arr)}, expected a numpy array.")
    # check for numeric dtype (int or float) in the array
    if not np.issubdtype(arr.dtype, np.number):
        raise ContractError(
            f"Input array has dtype {arr.dtype}, expected a numeric dtype "
            "(int or float).",
        )
    return True


@beartype
def validate_return_with_pydantic(
    result: dict[str, object],
) -> ReturnSchemaModel:
    """Validate return schema using Pydantic model.

    Parameters
    ----------
    result : dict[str, object]
        Return result to validate

    Returns
    -------
    ReturnSchemaModel
        Validated return schema model

    Raises
    ------
    ContractError
        If validation fails

    """
    try:
        return ReturnSchemaModel(result=result)
    except Exception as e:
        msg = (
            "Return schema validation failed. Please ensure that the data "
            f"fit the expected schema: {e}"
        )
        raise ContractError(msg)


def validate_anisotropy_factor_with_pydantic(
    anisotropy_factor: float,
    y_spacing: float | None = None,
    x_spacing: float | None = None,
) -> AnisotropyFactorModel:
    """Validate the anisotropy factor using a Pydantic model.

    Parameters
    ----------
    anisotropy_factor : float
        Ratio of z-spacing to y-spacing to validate.
    y_spacing : float | None, optional
        Y (transverse) spacing. When provided along with ``x_spacing``, they
        are checked for equality since ``anisotropy_factor`` only captures a
        single z/y ratio.
    x_spacing : float | None, optional
        X (transverse) spacing. See ``y_spacing``.

    Returns
    -------
    AnisotropyFactorModel
        Validated anisotropy factor model.

    Raises
    ------
    ContractError
        If the anisotropy factor is less than ``MIN_ANISOTROPY_FACTOR``, is
        not finite, or if ``y_spacing`` and ``x_spacing`` are unequal.

    """
    try:
        return AnisotropyFactorModel(
            anisotropy_factor=anisotropy_factor,
            y_spacing=y_spacing,
            x_spacing=x_spacing,
        )
    except Exception as e:
        msg = (
            "Anisotropy factor validation failed. Please ensure that the "
            "z-spacing is not finer than the y-spacing and that the y- and "
            f"x-spacing are equal: {e}"
        )
        raise ContractError(msg)


def validate_image_with_pydantic(arr: np.ndarray) -> ImageArrayModel:
    """Validate the input image array using Pydantic model.

    Parameters
    ----------
    arr : np.ndarray
        Input image array to validate

    Returns
    -------
    ImageArrayModel
        Validated image array model

    Raises
    ------
    ContractError
        If validation fails

    """
    try:
        return ImageArrayModel(array=arr)
    except Exception as e:
        msg = (
            "Image array validation failed. Please ensure that the input "
            f"array meets the expected contracts: {e}"
        )
        raise ContractError(msg)


@beartype
def validate_column_name_with_pydantic(column_name: str) -> ColumnNameModel:
    """Validate column name using Pydantic model.

    Parameters
    ----------
    column_name : str
        Column name to validate

    Returns
    -------
    ColumnNameModel
        Validated column name model with parsed components

    Raises
    ------
    ContractError
        If validation fails

    """
    try:
        return ColumnNameModel(column_name=column_name)
    except Exception as e:
        raise ContractError(f"Column name validation failed: {e}")


def create_image_array_schema() -> pa.SeriesSchema:
    """Create a Pandera schema for image array validation.

    Returns
    -------
    pa.SeriesSchema
        Pandera schema for numeric arrays

    """
    # Use a single numeric dtype for the series schema; Pandera dtype
    # objects should be instantiated rather than combined with `|`.
    return pa.SeriesSchema(
        dtype=pa.Float64(),
        name="image_array",
        checks=[pa.Check(lambda x: x is not None, error="Value cannot be None")],
    )


@beartype
def validate_return_schema_contract(
    result: dict[str, object],
) -> bool:
    """Validate return schema keys, types, and deterministic key ordering."""
    if not isinstance(result, dict):
        raise ContractError(f"Return result must be a dict, got {type(result)}.")

    actual_keys = tuple(result.keys())
    if actual_keys != REQUIRED_RETURN_KEYS:
        raise ContractError(
            "Return result keys must match required deterministic order "
            f"{REQUIRED_RETURN_KEYS}, got {actual_keys}.",
        )

    try:
        ReturnSchemaModel(result=result)
    except Exception as e:
        raise ContractError(f"Return result validation failed: {e}")

    return True


class ExpectedFeatureNameValues(BaseModel):
    """Pydantic model for expected values in feature naming validation."""

    compartments: list[str] | None = Field(default_factory=list)
    channels: list[str] | None = Field(default_factory=list)
    features: list[str] | None = Field(default_factory=list)
    expected_values_dict: dict[str, list[str]] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        if self.compartments is not None:
            self.compartments = list(set(self.compartments))
        else:
            raise ValueError("Compartments list cannot be None.")
        if self.channels is not None:
            # Add "NoChannel" to channels list
            self.channels = list(set(self.channels) | {"NoChannel"})
        else:
            raise ValueError("Channels list cannot be None.")
        if self.features is not None and len(self.features) > 0:
            self.features = list(set(self.features))
        else:
            self.features = FEATURES

        self.expected_values_dict = {
            "compartments": self.compartments,
            "channels": self.channels,
            "features": self.features,
        }


@beartype
def validate_column_name_schema(
    column_name: str,
    channels: list[str],
    compartments: list[str],
    features: list[str] | None = None,
) -> bool:
    """Validate the column name schema for required fields and types

    Parameters
    ----------
    column_name : str
        The column name to validate
    channels : list[str]
        List of valid channels for feature naming
    compartments : list[str]
        List of valid compartments for feature naming
    features : list[str] | None, optional
        List of valid features for feature naming, by default None

    Returns
    -------
    bool
        The status of the validation

    Raises
    ------
    ContractError
        If the column name does not meet the expected schema

    """
    non_metadata_underscore_separated_parts = NON_METADATA_UNDERSCORE_SEPARATED_PARTS
    metadata_underscore_separated_parts = METADATA_UNDERSCORE_SEPARATED_PARTS

    expected_values = ExpectedFeatureNameValues(
        channels=channels,
        compartments=compartments,
        features=features,
    ).expected_values_dict

    # check if the column name is a string
    if not isinstance(column_name, str):
        raise ContractError(f"Column name must be a string, got {type(column_name)}")

    # check if the column name has at least 4 parts separated by underscores
    parts = column_name.split("_")
    if (
        len(parts) < non_metadata_underscore_separated_parts
        and "Metadata" not in column_name
    ):
        msg = (
            "Column name must have at least "
            f"{non_metadata_underscore_separated_parts} "
            "parts separated by underscores, "
            f"got {len(parts)} parts in '{column_name}'"
        )
        raise ContractError(msg)

    if "Metadata" in column_name:
        if len(parts) < metadata_underscore_separated_parts:
            raise ContractError(
                "Metadata column name must have at least "
                f"{metadata_underscore_separated_parts} "
                "parts separated by "
                f"underscores, got {len(parts)} parts in '{column_name}'",
            )
        return True
    # Validate the non-metadata column's compartment/channel/feature components.
    #
    # This is equivalent to the prior pandera DataFrameSchema check
    # (Check.isin with coerce=True, nullable=False, strict=True applied to
    # parts[0..2]) but expressed as a plain set-membership test. ``parts`` are
    # always strings produced by ``column_name.split("_")``, so the pandera
    # ``coerce`` (str cast), ``nullable`` (no NaN), and ``strict`` (exactly
    # three components) considerations are no-ops here, and pass/fail semantics
    # are identical.
    #
    # The motivation is overhead: this function is called once per output
    # column by every feature module's tail validation loop, and rebuilding a
    # pandas DataFrame plus a pandera DataFrameSchema and running ``.validate``
    # on a one-row frame per column dominated feature extraction time (e.g.
    # ~67% of texture runtime). A direct membership check removes that cost
    # without changing any feature value, column name, or row identity.
    feature_components = {
        "compartment": parts[0],
        "channel": parts[1],
        "feature": parts[2],
    }
    for component, allowed_key in (
        ("compartment", "compartments"),
        ("channel", "channels"),
        ("feature", "features"),
    ):
        allowed = expected_values.get(allowed_key, [])
        if feature_components[component] not in allowed:
            raise ContractError(
                f"Column name schema validation failed: "
                f"{component} '{feature_components[component]}' in "
                f"'{column_name}' is not in allowed {allowed_key} {list(allowed)}.",
            )

    return True
