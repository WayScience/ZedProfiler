"""Tests for loading_classes module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from zedprofiler.IO import loading_classes
from zedprofiler.IO.loading_classes import (
    ImageSetConfig,
    ImageSetLoader,
    ObjectLoader,
    TwoObjectLoader,
    _image_loading,
)

ZERO_LABEL = 0
ONE_LABEL = 1
TWO_LABEL = 2
EXPECTED_ANISOTROPY = 2.0
ORIGINAL_DNA_PIXEL = 10


class TestImageSetConfig:
    """Tests for ImageSetConfig dataclass."""

    def test_config_creation_defaults(self) -> None:
        """Test creating ImageSetConfig with defaults."""
        config = ImageSetConfig()
        assert config.image_set_name is None
        assert config.label_key_name == []
        assert config.raw_image_key_name == []

    def test_config_creation_with_values(self) -> None:
        """Test creating ImageSetConfig with explicit values."""
        config = ImageSetConfig(
            image_set_name="test_set",
            label_key_name=["label1", "label2"],
            raw_image_key_name=["raw1"],
        )
        assert config.image_set_name == "test_set"
        assert config.label_key_name == ["label1", "label2"]
        assert config.raw_image_key_name == ["raw1"]

    def test_config_post_init_none_defaults(self) -> None:
        """Test that __post_init__ sets None fields to empty lists."""
        config = ImageSetConfig(image_set_name="test")
        assert config.label_key_name == []
        assert config.raw_image_key_name == []

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"image_set_name": 123}, "image_set_name must be a string or None"),
            (
                {"label_key_name": "labels"},
                "label_key_name must be a list of strings or None",
            ),
            (
                {"raw_image_key_name": "raw"},
                "raw_image_key_name must be a list of strings or None",
            ),
        ],
    )
    def test_config_rejects_invalid_types(
        self,
        kwargs: dict[str, object],
        message: str,
    ) -> None:
        """ImageSetConfig should validate field types during initialization."""
        with pytest.raises(TypeError, match=message):
            ImageSetConfig(**kwargs)


class TestImageSetLoaderMethods:
    """Tests for ImageSetLoader helper methods without filesystem coupling."""

    def test_image_loading_uses_bioio_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Image loading should instantiate BioImage and request ZYX data."""
        calls: dict[str, str] = {}

        class FakeBioImage:
            def __init__(self, path: str) -> None:
                calls["path"] = path

            def get_image_data(self, order: str) -> np.ndarray:
                calls["order"] = order
                return np.ones((2, 2, 2), dtype=np.uint16)

        monkeypatch.setattr(loading_classes.bioio, "BioImage", FakeBioImage)

        image_path = Path("/tmp/fake-image.tif")
        result = _image_loading(image_path)

        assert calls == {"path": str(image_path), "order": "ZYX"}
        assert result.shape == (2, 2, 2)

    def test_image_loading_requires_path_type(self) -> None:
        """Beartype should reject non-Path inputs for _image_loading."""
        with pytest.raises(BeartypeCallHintParamViolation):
            _image_loading("/tmp/fake-image.tif")

    def test_lazy_image_dict_loads_on_access_and_caches(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Lazy dict should defer loading until access and cache loaded array."""
        image_path = tmp_path / "dna_raw.tif"
        image_path.touch()
        call_count = {"count": 0}

        def fake_loader(_path: Path) -> np.ndarray:
            call_count["count"] += 1
            return np.ones((2, 2), dtype=np.int32)

        monkeypatch.setattr(loading_classes, "_image_loading", fake_loader)
        lazy_dict = loading_classes._LazyImageSetDict({"DNA": image_path})

        first = lazy_dict["DNA"]
        second = lazy_dict["DNA"]

        assert np.array_equal(first, second)
        assert call_count["count"] == 1

    def test_lazy_image_dict_get_items_and_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Lazy dict helper methods should expose cached arrays and defaults."""
        image_path = tmp_path / "dna_raw.tif"
        image_path.touch()

        monkeypatch.setattr(
            loading_classes,
            "_image_loading",
            lambda _path: np.array([[1, 2]], dtype=np.int32),
        )
        lazy_dict = loading_classes._LazyImageSetDict({"DNA": image_path})

        default_object = object()

        assert lazy_dict.get("missing", default_object) is default_object
        items = list(lazy_dict.items())
        values = list(lazy_dict.values())

        assert items[0][0] == "DNA"
        assert np.array_equal(items[0][1], np.array([[1, 2]], dtype=np.int32))
        assert len(values) == 1
        assert np.array_equal(values[0], np.array([[1, 2]], dtype=np.int32))

    def test_get_compartments_and_image_names(self) -> None:
        """Compartments are declared label keys; the rest are image names."""
        loader = ImageSetLoader.__new__(ImageSetLoader)
        loader.image_set_dict = {
            "Nuclei_label": np.zeros((2, 2), dtype=np.int32),
            "Cell_label": np.zeros((2, 2), dtype=np.int32),
            "DNA": np.ones((2, 2), dtype=np.int32),
        }
        loader._label_key_names = ["Nuclei_label", "Cell_label"]

        compartments = loader.get_compartments()
        names = loader.get_image_names()

        assert compartments == ["Nuclei_label", "Cell_label"]
        assert names == ["DNA"]

    def test_get_unique_objects_in_compartments_filters_background(self) -> None:
        """Unique compartment objects should exclude background label 0."""
        loader = ImageSetLoader.__new__(ImageSetLoader)
        loader.image_set_dict = {
            "Nuclei_label": np.array(
                [[ZERO_LABEL, ONE_LABEL], [TWO_LABEL, ZERO_LABEL]],
                dtype=np.int32,
            ),
        }
        loader.compartments = ["Nuclei_label"]

        loader.get_unique_objects_in_compartments()

        assert loader.unique_compartment_objects["Nuclei_label"] == [
            ONE_LABEL,
            TWO_LABEL,
        ]

    def test_get_unique_objects_empty_compartments_returns_empty(self) -> None:
        """Empty compartments list returns early."""
        loader = ImageSetLoader.__new__(ImageSetLoader)
        loader.image_set_dict = {}
        loader.compartments = []

        loader.get_unique_objects_in_compartments()
        assert loader.unique_compartment_objects == {}

    def test_get_image_and_get_anisotropy(self) -> None:
        """Simple accessors return the expected image and anisotropy ratio."""
        loader = ImageSetLoader.__new__(ImageSetLoader)
        arr = np.arange(8).reshape((2, 2, 2))
        loader.image_set_dict = {"DNA": arr}
        loader.anisotropy_spacing = (2.0, 1.0, 1.0)

        assert np.array_equal(loader.get_image("DNA"), arr)
        assert loader.get_anisotropy() == EXPECTED_ANISOTROPY


class TestImageSetLoaderInit:
    """Tests that exercise ImageSetLoader __init__ with mocked reads."""

    def test_init_loads_channel_and_label_images(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Initialization should load matching files and build derived attributes."""
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()

        (image_dir / "dna_raw.tif").touch()
        (image_dir / "ignore.txt").touch()
        (label_dir / "nuc_label.tif").touch()

        class FakeBioImage:
            def __init__(self, path: str) -> None:
                self.path = path

            def get_image_data(self, _order: str) -> np.ndarray:
                if "nuc_label" in self.path:
                    return np.array(
                        [[ZERO_LABEL, ONE_LABEL], [TWO_LABEL, TWO_LABEL]],
                        dtype=np.int32,
                    )
                return np.ones((2, 2), dtype=np.int32)

        monkeypatch.setattr(loading_classes.bioio, "BioImage", FakeBioImage)

        loader = ImageSetLoader(
            image_set_path=image_dir,
            label_set_path=label_dir,
            anisotropy_spacing=(2.0, 1.0, 1.0),
            channel_mapping={"DNA": "dna_raw", "Nuclei_label": "nuc_label"},
            config=ImageSetConfig(
                image_set_name="set-01",
                label_key_name=["Nuclei_label"],
                raw_image_key_name=["DNA"],
            ),
        )

        assert loader.image_set_name == "set-01"
        assert loader.anisotropy_factor == EXPECTED_ANISOTROPY
        assert set(loader.image_set_dict.keys()) == {"DNA", "Nuclei_label"}
        assert loader.compartments == ["Nuclei_label"]
        assert loader.image_names == ["DNA"]
        assert loader.unique_compartment_objects["Nuclei_label"] == [
            ONE_LABEL,
            TWO_LABEL,
        ]

    def test_init_from_arrays_populates_image_set_dict(
        self,
    ) -> None:
        """Array-backed initialization should populate image_set_dict directly."""
        image_array = np.ones((2, 2, 2), dtype=np.float32)
        label_array = np.array(
            [
                [[ZERO_LABEL, ONE_LABEL], [TWO_LABEL, TWO_LABEL]],
                [[ZERO_LABEL, ONE_LABEL], [TWO_LABEL, TWO_LABEL]],
            ],
            dtype=np.int32,
        )

        loader = ImageSetLoader(
            image_set_path=None,
            label_set_path=None,
            image_set_array=image_array,
            label_set_array=label_array,
            anisotropy_spacing=(2.0, 1.0, 1.0),
            channel_mapping={
                "DNA": "dna_raw",
                "Nuclei_label": "nuc_label",
            },
            config=ImageSetConfig(
                label_key_name=["Nuclei-label"],
                raw_image_key_name=["DNA"],
            ),
        )

        assert np.array_equal(loader.get_image("DNA"), image_array)
        assert np.array_equal(loader.get_image("Nuclei-label"), label_array)

    def test_init_with_none_image_path_raises_value_error(
        self,
    ) -> None:
        """Missing image inputs should raise ValueError during initialization."""
        with pytest.raises(ValueError):
            ImageSetLoader(
                image_set_path=None,
                label_set_path=None,
                anisotropy_spacing=(1.0, 1.0, 1.0),
                channel_mapping={},
                config=ImageSetConfig(
                    label_key_name=["label"],
                    raw_image_key_name=["raw"],
                ),
            )


class TestObjectLoaders:
    """Tests for object-level loader classes."""

    def test_object_loader_drops_background_id(self) -> None:
        """ObjectLoader should omit the 0 label from object_ids."""
        label_image = np.array(
            [[ZERO_LABEL, ONE_LABEL], [TWO_LABEL, TWO_LABEL]],
            dtype=np.int32,
        )
        image = np.ones((2, 2), dtype=np.float32)
        image_set_loader = ImageSetLoader.__new__(ImageSetLoader)
        image_set_loader.image_set_dict = {
            "DNA": image,
            "Nuclei": label_image,
        }

        obj = ObjectLoader(
            image_set_loader=image_set_loader,
            channel_name="DNA",
            compartment_name="Nuclei",
        )

        assert obj.channel == "DNA"
        assert obj.compartment == "Nuclei"
        assert np.array_equal(obj.image, image)
        assert np.array_equal(obj.label_image, label_image)
        assert obj.object_ids == [ONE_LABEL, TWO_LABEL]

    def test_two_object_loader_loads_images_and_ids(self) -> None:
        """TwoObjectLoader should load the expected arrays and preserve object IDs."""
        # label imabe should be 3D
        label_image = np.array(
            [
                [[ZERO_LABEL, ONE_LABEL], [TWO_LABEL, TWO_LABEL]],
                [[ZERO_LABEL, ONE_LABEL], [TWO_LABEL, TWO_LABEL]],
            ],
            dtype=np.int32,
        )
        # establish this 3D image where each dimension is ZYX
        # 2x2x2 array of ones to match the label image shape for simplicity
        image = np.ones((2, 2, 2), dtype=np.float32)
        image_set_loader = ImageSetLoader(
            image_set_path=None,
            label_set_path=None,
            image_set_array=image,
            label_set_array=label_image,
            anisotropy_spacing=(2.0, 1.0, 1.0),
            channel_mapping={
                "DNA": "dna_raw",
                "AGP": "agp_raw",
                "Nuclei-label": "nuc_label",
            },
            config=ImageSetConfig(
                label_key_name=["Nuclei-label"],
                raw_image_key_name=["DNA", "AGP"],
            ),
        )

        obj = TwoObjectLoader(
            image_set_loader=image_set_loader,
            channel1="DNA",
            channel2="AGP",
            compartment="Nuclei-label",
        )

        assert obj.compartment == "Nuclei-label"
        assert np.array_equal(obj.image1, image)
        assert np.array_equal(obj.image2, image)
        assert np.array_equal(obj.label_image, label_image)
        assert obj.object_ids == [ONE_LABEL, TWO_LABEL]
