"""Tests for data contract validation in zedprofiler.contracts."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

# Skip tests if core optional dependencies aren't installed in this environment
pytest.importorskip("pydantic")
pytest.importorskip("pandera")
from pydantic import ValidationError

from zedprofiler.contracts import (
    ColumnNameModel,
    ExpectedFeatureNameValues,
    FeatureDictModel,
    ImageArrayModel,
    MetadataDictModel,
    ReturnSchemaModel,
    create_image_array_schema,
    validate_column_name_schema,
    validate_column_name_with_pydantic,
    validate_image_array_shape_contracts,
    validate_image_array_type_contracts,
    validate_image_with_pydantic,
    validate_return_schema_contract,
    validate_return_with_pydantic,
)
from zedprofiler.exceptions import ContractError


def _load_toml_config(config_path: Path) -> dict[str, list[str]]:
    """Load expected values from TOML file."""
    with open(config_path, "rb") as f:
        toml_data = tomllib.load(f)
    return toml_data.get("expected_values", {})


@pytest.fixture
def expected_values_config_path(tmp_path: Path) -> Path:
    """Create a temporary expected-values TOML configuration file."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[expected_values]
compartments = ["Nuclei", "Cytoplasm", "Cell", "Organoid"]
channels = ["DNA", "AGP", "ER", "Mito"]
""".strip(),
    )
    return config_path


def test_validate_image_array_shape_contracts_accepts_valid_3d_array() -> None:
    arr = np.zeros((8, 16, 16), dtype=float)

    assert validate_image_array_shape_contracts(arr) is True


def test_validate_image_array_shape_contracts_accepts_single_channel_4d_array() -> None:
    arr = np.zeros((1, 8, 16, 16), dtype=float)

    assert validate_image_array_shape_contracts(arr) is True


def test_validate_image_array_shape_contracts_rejects_2d_array() -> None:
    arr = np.zeros((16, 16), dtype=float)

    with pytest.raises(ContractError):
        validate_image_array_shape_contracts(arr)


def test_validate_image_array_shape_contracts_rejects_multichannel_4d_array() -> None:
    arr = np.zeros((2, 8, 16, 16), dtype=float)

    with pytest.raises(ContractError):
        validate_image_array_shape_contracts(arr)


def test_validate_image_array_shape_contracts_rejects_multichannel_5d_array() -> None:
    arr = np.zeros((2, 2, 4, 8, 8), dtype=float)

    with pytest.raises(ContractError):
        validate_image_array_shape_contracts(arr)


def test_validate_image_array_shape_contracts_rejects_all_singleton_3d_array() -> None:
    arr = np.zeros((1, 1, 1), dtype=float)

    with pytest.raises(ContractError):
        validate_image_array_shape_contracts(arr)


def test_validate_image_array_type_contracts_accepts_numeric_array() -> None:
    arr = np.zeros((8, 8, 8), dtype=np.float32)

    assert validate_image_array_type_contracts(arr) is True


def test_validate_image_array_type_contracts_rejects_non_numpy_array() -> None:
    arr = [[1, 2], [3, 4]]

    # Beartype catches type errors before function execution
    with pytest.raises(BeartypeCallHintParamViolation):
        validate_image_array_type_contracts(arr)  # type: ignore[arg-type]


def test_validate_image_array_type_contracts_rejects_non_numeric_dtype() -> None:
    arr = np.array([["a", "b"], ["c", "d"]], dtype=str)

    with pytest.raises(ContractError):
        validate_image_array_type_contracts(arr)


@pytest.mark.parametrize(
    argnames="channels,compartments",
    argvalues=[
        (["DNA", "AGP", "ER"], ["Nuclei", "Cytoplasm", "Cell"]),
    ],
)
def test_expected_values_loads_config_and_adds_nochannel(
    channels: list[str],
    compartments: list[str],
) -> None:
    values = ExpectedFeatureNameValues(
        compartments=compartments,
        channels=channels,
    )
    print(values)

    assert "Nuclei" in values.compartments
    assert "DNA" in values.channels
    assert "NoChannel" in values.channels
    assert "Intensity" in values.features


def test_expected_values_to_dict_returns_expected_keys(
    expected_values_config_path: Path,
) -> None:
    config = _load_toml_config(expected_values_config_path)
    values = ExpectedFeatureNameValues(
        compartments=config["compartments"],
        channels=config["channels"],
    )

    result = values.expected_values_dict
    assert set(result.keys()) == {"compartments", "channels", "features"}


def test_expected_values_accepts_expected_values_dict() -> None:
    expected_values = {
        "compartments": ["Nuclei", "Cytoplasm"],
        "channels": ["DNA", "ER"],
        "features": ["Intensity", "Texture"],
    }

    values = ExpectedFeatureNameValues(
        compartments=expected_values["compartments"],
        channels=expected_values["channels"],
        features=expected_values["features"],
    )

    assert "NoChannel" in values.channels
    assert set(values.compartments) == {"Nuclei", "Cytoplasm"}
    assert "Intensity" in values.features
    assert "Texture" in values.features


def test_validate_column_name_schema_accepts_valid_feature_column(
    expected_values_config_path: Path,
) -> None:
    valid_name = "Nuclei_DNA_Intensity_MeanIntensity"
    config = _load_toml_config(expected_values_config_path)

    assert (
        validate_column_name_schema(
            valid_name,
            channels=config["channels"],
            compartments=config["compartments"],
        )
        is True
    )


def test_validate_column_name_schema_accepts_valid_metadata_column(
    expected_values_config_path: Path,
) -> None:
    valid_name = "Metadata_Storage_FilePath"
    config = _load_toml_config(expected_values_config_path)

    assert (
        validate_column_name_schema(
            valid_name,
            channels=config["channels"],
            compartments=config["compartments"],
        )
        is True
    )


def test_validate_column_name_schema_rejects_non_string_column_name(
    expected_values_config_path: Path,
) -> None:
    config = _load_toml_config(expected_values_config_path)
    # Beartype catches type errors before function execution
    with pytest.raises(BeartypeCallHintParamViolation):
        validate_column_name_schema(
            123,  # type: ignore[arg-type]
            channels=config["channels"],
            compartments=config["compartments"],
        )


def test_validate_column_name_schema_rejects_non_metadata_with_too_few_parts(
    expected_values_config_path: Path,
) -> None:
    invalid_name = "Nuclei_DNA_Intensity"
    config = _load_toml_config(expected_values_config_path)

    with pytest.raises(ContractError):
        validate_column_name_schema(
            invalid_name,
            channels=config["channels"],
            compartments=config["compartments"],
        )


def test_validate_column_name_schema_rejects_metadata_with_too_few_parts(
    expected_values_config_path: Path,
) -> None:
    invalid_name = "Metadata_Storage"
    config = _load_toml_config(expected_values_config_path)

    with pytest.raises(ContractError):
        validate_column_name_schema(
            invalid_name,
            channels=config["channels"],
            compartments=config["compartments"],
        )


def test_validate_column_name_schema_rejects_unknown_compartment(
    expected_values_config_path: Path,
) -> None:
    invalid_name = "Nucleus_DNA_Intensity_MeanIntensity"
    config = _load_toml_config(expected_values_config_path)

    with pytest.raises(ContractError):
        validate_column_name_schema(
            invalid_name,
            channels=config["channels"],
            compartments=config["compartments"],
        )


def test_validate_column_name_schema_rejects_unknown_channel(
    expected_values_config_path: Path,
) -> None:
    invalid_name = "Nuclei_GFP_Intensity_MeanIntensity"
    config = _load_toml_config(expected_values_config_path)

    with pytest.raises(ContractError):
        validate_column_name_schema(
            invalid_name,
            channels=config["channels"],
            compartments=config["compartments"],
        )


def test_validate_column_name_schema_rejects_unknown_feature(
    expected_values_config_path: Path,
) -> None:
    invalid_name = "Nuclei_DNA_UnknownFeature_MeanIntensity"
    config = _load_toml_config(expected_values_config_path)

    with pytest.raises(ContractError):
        validate_column_name_schema(
            invalid_name,
            channels=config["channels"],
            compartments=config["compartments"],
        )


def test_validate_return_schema_contract_accepts_valid_result() -> None:
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": {"Nuclei_DNA_Intensity_MeanIntensity": 0.5},
        "metadata": {"Metadata_Object_ObjectID": 1},
    }

    assert validate_return_schema_contract(result) is True


def test_validate_return_schema_contract_rejects_non_dict_result() -> None:
    # Beartype catches type errors before function execution
    with pytest.raises(BeartypeCallHintParamViolation):
        validate_return_schema_contract(["not", "a", "dict"])  # type: ignore[arg-type]


def test_validate_return_schema_contract_rejects_missing_required_key() -> None:
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": {"Nuclei_DNA_Intensity_MeanIntensity": 0.5},
    }

    with pytest.raises(ContractError):
        validate_return_schema_contract(result)  # type: ignore[arg-type]


def test_validate_return_schema_contract_rejects_extra_key() -> None:
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": {"Nuclei_DNA_Intensity_MeanIntensity": 0.5},
        "metadata": {"Metadata_Object_ObjectID": 1},
        "extra": 123,
    }

    with pytest.raises(ContractError):
        validate_return_schema_contract(result)


def test_validate_return_schema_contract_rejects_wrong_key_order() -> None:
    # Dict insertion order is deterministic in Python 3.7+; this order is intentional.
    result = {
        "features": {"Nuclei_DNA_Intensity_MeanIntensity": 0.5},
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "metadata": {"Metadata_Object_ObjectID": 1},
    }

    with pytest.raises(ContractError):
        validate_return_schema_contract(result)


def test_validate_return_schema_contract_rejects_invalid_image_array_type() -> None:
    result = {
        "image_array": [[1, 2], [3, 4]],
        "features": {"Nuclei_DNA_Intensity_MeanIntensity": 0.5},
        "metadata": {"Metadata_Object_ObjectID": 1},
    }

    with pytest.raises(ContractError):
        validate_return_schema_contract(result)


def test_validate_return_schema_contract_rejects_invalid_features_type() -> None:
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": ["not", "a", "dict"],
        "metadata": {"Metadata_Object_ObjectID": 1},
    }

    with pytest.raises(ContractError):
        validate_return_schema_contract(result)


def test_validate_return_schema_contract_rejects_invalid_metadata_type() -> None:
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": {"Nuclei_DNA_Intensity_MeanIntensity": 0.5},
        "metadata": ["not", "a", "dict"],
    }

    with pytest.raises(ContractError):
        validate_return_schema_contract(result)


# ============================================================================
# Tests for Pydantic Models
# ============================================================================


def test_pydantic_image_array_model_accepts_valid_array() -> None:
    """Test ImageArrayModel validates correct numpy arrays."""
    arr = np.zeros((10, 20, 30), dtype=np.float32)
    model = ImageArrayModel(array=arr)

    assert isinstance(model.array, np.ndarray)
    assert model.array.shape == (10, 20, 30)


def test_pydantic_image_array_model_rejects_non_numpy_array() -> None:
    """Test ImageArrayModel rejects non-numpy objects."""
    with pytest.raises(ValidationError):
        ImageArrayModel(array=[1, 2, 3])  # type: ignore[arg-type]


def test_pydantic_image_array_model_rejects_non_numeric_dtype() -> None:
    """Test ImageArrayModel rejects non-numeric dtypes."""
    arr = np.array(["a", "b", "c"], dtype=str)
    with pytest.raises(ValidationError):
        ImageArrayModel(array=arr)


def test_pydantic_image_array_model_rejects_2d_array() -> None:
    """Test ImageArrayModel rejects 2D arrays."""
    arr = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValidationError):
        ImageArrayModel(array=arr)


def test_pydantic_image_array_model_rejects_multichannel_4d_array() -> None:
    """Test ImageArrayModel rejects multi-channel 4D arrays."""
    arr = np.zeros((2, 8, 16, 16), dtype=np.float32)
    with pytest.raises(ValidationError):
        ImageArrayModel(array=arr)


def test_pydantic_feature_dict_model_accepts_valid_dict() -> None:
    """Test FeatureDictModel validates correct dictionaries."""
    features = {"feature_1": 0.5, "feature_2": 1.2}
    model = FeatureDictModel(features=features)

    assert model.features == features


def test_pydantic_feature_dict_model_accepts_empty_dict() -> None:
    """Test FeatureDictModel accepts empty dictionaries."""
    model = FeatureDictModel(features={})

    assert model.features == {}


def test_pydantic_feature_dict_model_rejects_non_dict() -> None:
    """Test FeatureDictModel rejects non-dictionary objects."""
    with pytest.raises(ValidationError):
        FeatureDictModel(features=["not", "a", "dict"])  # type: ignore[arg-type]


def test_pydantic_metadata_dict_model_accepts_valid_dict() -> None:
    """Test MetadataDictModel validates correct dictionaries."""
    metadata = {"key_1": "value_1", "key_2": "value_2"}
    model = MetadataDictModel(metadata=metadata)

    assert model.metadata == metadata


def test_pydantic_metadata_dict_model_rejects_non_dict() -> None:
    """Test MetadataDictModel rejects non-dictionary objects."""
    with pytest.raises(ValidationError):
        MetadataDictModel(metadata="not a dict")  # type: ignore[arg-type]


def test_pydantic_return_schema_model_accepts_valid_result() -> None:
    """Test ReturnSchemaModel validates correct return dictionaries."""
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": {"feature_1": 0.5},
        "metadata": {"key": "value"},
    }
    model = ReturnSchemaModel(result=result)

    assert isinstance(model.result["image_array"], np.ndarray)
    assert isinstance(model.result["features"], dict)
    assert isinstance(model.result["metadata"], dict)


def test_pydantic_return_schema_model_rejects_wrong_key_order() -> None:
    """Test ReturnSchemaModel enforces key order."""
    result = {
        "features": {"feature_1": 0.5},
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "metadata": {"key": "value"},
    }

    with pytest.raises(ValidationError):
        ReturnSchemaModel(result=result)


def test_pydantic_return_schema_model_rejects_missing_key() -> None:
    """Test ReturnSchemaModel requires all keys."""
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": {"feature_1": 0.5},
    }

    with pytest.raises(ValidationError):
        ReturnSchemaModel(result=result)


def test_pydantic_return_schema_model_rejects_extra_key() -> None:
    """Test ReturnSchemaModel rejects extra keys."""
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": {"feature_1": 0.5},
        "metadata": {"key": "value"},
        "extra": "not allowed",
    }

    with pytest.raises(ValidationError):
        ReturnSchemaModel(result=result)


def test_pydantic_column_name_model_parses_valid_feature_column() -> None:
    """Test ColumnNameModel parses feature column names."""
    model = ColumnNameModel(column_name="Nuclei_DNA_Intensity_MeanIntensity")

    assert model.column_name == "Nuclei_DNA_Intensity_MeanIntensity"
    assert model.compartment == "Nuclei"
    assert model.channel == "DNA"
    assert model.feature == "Intensity"


def test_pydantic_column_name_model_parses_metadata_column() -> None:
    """Test ColumnNameModel parses metadata column names."""
    model = ColumnNameModel(column_name="Metadata_Storage_FilePath")

    assert model.column_name == "Metadata_Storage_FilePath"
    # Metadata columns don't parse compartment/channel/feature
    assert model.compartment is None


def test_pydantic_column_name_model_rejects_non_string() -> None:
    """Test ColumnNameModel rejects non-string column names."""
    with pytest.raises(ValidationError):
        ColumnNameModel(column_name=123)  # type: ignore[arg-type]


def test_pydantic_column_name_model_rejects_too_few_parts() -> None:
    """Test ColumnNameModel rejects names with too few parts."""
    with pytest.raises(ValidationError):
        ColumnNameModel(column_name="Nuclei_DNA_Intensity")


def test_pydantic_column_name_model_rejects_metadata_too_few_parts() -> None:
    """Test ColumnNameModel rejects metadata names with too few parts."""
    with pytest.raises(ValidationError):
        ColumnNameModel(column_name="Metadata_Storage")


# ============================================================================
# Tests for Helper Functions with Pydantic
# ============================================================================


def test_validate_image_with_pydantic_accepts_valid_array() -> None:
    """Test validate_image_with_pydantic helper."""
    arr = np.zeros((10, 20, 30), dtype=np.float32)
    model = validate_image_with_pydantic(arr)

    assert isinstance(model, ImageArrayModel)
    assert model.array.shape == (10, 20, 30)


def test_validate_image_with_pydantic_raises_contract_error() -> None:
    """Test validate_image_with_pydantic raises ContractError on invalid input."""
    arr = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ContractError):
        validate_image_with_pydantic(arr)


def test_validate_return_with_pydantic_accepts_valid_result() -> None:
    """Test validate_return_with_pydantic helper."""
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": {"feature": 0.5},
        "metadata": {"key": "value"},
    }
    model = validate_return_with_pydantic(result)

    assert isinstance(model, ReturnSchemaModel)


def test_validate_return_with_pydantic_raises_contract_error() -> None:
    """Test validate_return_with_pydantic raises ContractError on invalid input."""
    result = {
        "image_array": [[1, 2], [3, 4]],
        "features": {"feature": 0.5},
        "metadata": {"key": "value"},
    }

    with pytest.raises(ContractError):
        validate_return_with_pydantic(result)


def test_validate_column_name_with_pydantic_accepts_valid_name() -> None:
    """Test validate_column_name_with_pydantic helper."""
    model = validate_column_name_with_pydantic("Nuclei_DNA_Intensity_Mean")

    assert isinstance(model, ColumnNameModel)
    assert model.compartment == "Nuclei"


def test_validate_column_name_with_pydantic_raises_contract_error() -> None:
    """Test validate_column_name_with_pydantic raises ContractError."""
    with pytest.raises(ContractError):
        validate_column_name_with_pydantic("Nuclei_DNA_Intensity")


# ============================================================================
# Tests for Pandera Schemas
# ============================================================================


def test_create_image_array_schema_returns_series_schema() -> None:
    """Test create_image_array_schema returns a valid Pandera schema."""
    schema = create_image_array_schema()

    assert schema is not None
    assert hasattr(schema, "validate")


def test_pandera_schema_rejects_non_numeric_array() -> None:
    """Test Pandera schema rejects non-numeric arrays."""
    schema = create_image_array_schema()

    arr = np.array(["a", "b", "c"], dtype=str)

    # Should raise SchemaError
    with pytest.raises(Exception):  # SchemaError from pandera
        schema.validate(arr)


# ============================================================================
# Tests for Beartype Runtime Type Checking
# ============================================================================


def test_beartype_enforces_correct_array_type() -> None:
    """Test beartype validates function argument types."""
    arr = np.zeros((8, 16, 16), dtype=np.float32)

    # Should not raise
    assert validate_image_array_type_contracts(arr) is True


def test_beartype_catches_wrong_return_dict_type() -> None:
    """Test beartype catches wrong dict type in validate_return_schema_contract."""
    # Beartype enforces type hints at runtime, so invalid types are caught
    # before execution
    with pytest.raises(BeartypeCallHintParamViolation):
        validate_return_schema_contract(["not", "a", "dict"])  # type: ignore[arg-type]


def test_beartype_enforces_pathlib_path_type(
    expected_values_config_path: Path,
) -> None:
    """Test beartype validates pathlib.Path argument."""
    # Beartype would reject non-Path objects
    with pytest.raises(BeartypeCallHintParamViolation):
        validate_column_name_schema(
            "Nuclei_DNA_Intensity_Mean",
            "not a path",  # type: ignore[arg-type]
        )


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_validation_pipeline_passes() -> None:
    """Test complete validation pipeline with all components."""
    # Create valid array
    arr = np.zeros((10, 20, 30), dtype=np.float32)

    # Validate shape and type
    assert validate_image_array_shape_contracts(arr) is True
    assert validate_image_array_type_contracts(arr) is True

    # Validate with Pydantic
    img_model = validate_image_with_pydantic(arr)
    assert img_model.array.shape == (10, 20, 30)

    # Create valid return
    result = {
        "image_array": arr,
        "features": {"feature_1": 0.5, "feature_2": 1.2},
        "metadata": {"processor": "test"},
    }

    # Validate return schema
    assert validate_return_schema_contract(result) is True

    # Validate with Pydantic
    return_model = validate_return_with_pydantic(result)
    assert "image_array" in return_model.result
    assert "features" in return_model.result
    assert "metadata" in return_model.result


def test_validation_error_messages_are_descriptive() -> None:
    """Test that validation errors provide clear messages."""
    # Test with 2D array
    arr = np.zeros((16, 16), dtype=np.float32)

    try:
        validate_image_array_shape_contracts(arr)
        pytest.fail("Expected ContractError")
    except ContractError as e:
        assert "dimensions" in str(e)
        assert "2" in str(e)


def test_column_name_validation_with_pydantic_and_schema(
    expected_values_config_path: Path,
) -> None:
    """Test column name validation combines Pydantic and schema checking."""
    valid_name = "Nuclei_DNA_Intensity_MeanIntensity"
    config = _load_toml_config(expected_values_config_path)

    # Pydantic model should parse successfully
    col_model = validate_column_name_with_pydantic(valid_name)
    assert col_model.compartment == "Nuclei"

    # Full schema validation should pass
    assert (
        validate_column_name_schema(
            valid_name,
            channels=config["channels"],
            compartments=config["compartments"],
        )
        is True
    )
