"""Functions for formatting morphology feature names in a consistent way.

Formats morphology feature names and saves features as parquet files.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib

import pandas
import pandera.pandas as pa
from beartype import beartype


def _coerce_dataframe_column_names_to_strings(
    dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    """Ensure DataFrame column labels are string-typed before writing.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame whose column names should be coerced to strings.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with all column names coerced to strings.

    """
    parsed_dataframe = dataframe.copy()
    parsed_dataframe.columns = [str(column) for column in parsed_dataframe.columns]
    return parsed_dataframe


@beartype
def remove_underscores_from_string(string: object) -> str:
    """Remove unwanted delimiters from a string and replace them with hyphens.

    Parameters
    ----------
    string : str
        The string to remove unwanted delimiters from.

    Returns
    -------
    str
        The string with unwanted delimiters removed and replaced with hyphens.

    """
    if not isinstance(string, str):
        try:
            string = str(string)
        except Exception as e:
            msg = (
                f"Input string must be a string or convertible to a string. "
                f"Received input: {string!r} of type {type(string)}"
            )
            raise ValueError(msg) from e
    string = string.translate(str.maketrans("_. /", "----"))

    return string


def _coerce_feature_name_components(
    dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    """Normalize feature-name components using shared delimiter cleanup.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing feature name components to be normalized.
        Expected to have columns corresponding to FEATURE NAME COMPONENT COLUMNS.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with feature name components normalized by
        removing unwanted delimiters and replacing them with hyphens.

    """
    parsed_dataframe = dataframe.copy()
    for column in FEATURE_NAME_COMPONENT_COLUMNS:
        if column in parsed_dataframe.columns:
            parsed_dataframe[column] = parsed_dataframe[column].map(
                remove_underscores_from_string,
            )
    return parsed_dataframe


# ============================================================================
# Constants
# ============================================================================

FEATURE_NAME_COMPONENT_COLUMNS = (
    "compartment",
    "channel",
    "feature_type",
    "measurement",
)

FEATURE_OUTPUT_SCHEMA = pa.DataFrameSchema(
    columns={},
    strict=False,
    parsers=[pa.Parser(_coerce_dataframe_column_names_to_strings)],
)

FEATURE_NAME_COMPONENT_SCHEMA = pa.DataFrameSchema(
    columns={
        "compartment": pa.Column(object, nullable=False, coerce=True),
        "channel": pa.Column(object, nullable=False, coerce=True),
        "feature_type": pa.Column(object, nullable=False, coerce=True),
        "measurement": pa.Column(object, nullable=False, coerce=True),
    },
    strict=True,
    parsers=[pa.Parser(_coerce_feature_name_components)],
)


@beartype
def format_morphology_feature_name(
    compartment: object,
    channel: object,
    feature_type: object,
    measurement: object,
) -> str:
    """Format a morphology feature name consistently across all morphology features.
    This format follows specification for the following:
    https://github.com/WayScience/NF1_3D_organoid_profiling_pipeline/blob/main/docs/RFC-2119-Feature-Naming-Convention.md

    Parameters
    ----------
    compartment : str
        The compartment name.
    channel : str
        The channel name.
    feature_type : str
        The feature type.
    measurement : str
        The measurement name.

    Returns
    -------
    str
        The formatted feature name.

    """
    component_frame = pandas.DataFrame(
        [
            {
                "compartment": compartment,
                "channel": channel,
                "feature_type": feature_type,
                "measurement": measurement,
            },
        ],
    )
    coerced_components = FEATURE_NAME_COMPONENT_SCHEMA.validate(component_frame)
    parsed_row = coerced_components.iloc[0]
    return (
        f"{parsed_row['compartment']}_{parsed_row['channel']}_"
        f"{parsed_row['feature_type']}_{parsed_row['measurement']}"
    )


@beartype
@dataclasses.dataclass
class FeatureMetadata:
    """Metadata for feature output."""

    compartment: str
    channel: str
    feature_type: str
    cpu_or_gpu: str


@beartype
def save_features_as_parquet(
    parent_path: pathlib.Path,
    df: pandas.DataFrame,
    metadata: FeatureMetadata,
    atomic: bool = False,
) -> pathlib.Path:
    """Save features as parquet files in a consistent way.

    Saves features as parquet files with consistent naming across morphology
    features.

    Parameters
    ----------
    parent_path : pathlib.Path
        The parent path to save the features to.
    df : pandas.DataFrame
        The dataframe containing the features to save.
    metadata : FeatureMetadata
        Metadata for the feature output (compartment, channel, feature_type,
        cpu_or_gpu).
    atomic : bool
        When True, write to a sibling ``.tmp`` file then atomically replace
        the destination via ``os.replace``. This prevents a crashed write
        from leaving a partial parquet file that a restartable caller (using
        ``--skip-existing``) could mistake for a complete one. Default False
        preserves the existing direct-write behavior.

    Returns
    -------
    pathlib.Path

    """
    validated_df = FEATURE_OUTPUT_SCHEMA.validate(df)
    output_prefix = format_morphology_feature_name(
        metadata.compartment,
        metadata.channel,
        metadata.feature_type,
        metadata.cpu_or_gpu,
    )
    save_path = parent_path / f"{output_prefix}_features.parquet"
    if atomic:
        tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        validated_df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, save_path)
    else:
        validated_df.to_parquet(save_path, index=False)
    return save_path
