"""Data-loading classes for featurization workflows."""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass

import bioio
import numpy
from beartype import beartype

from zedprofiler.contracts import ImageArrayModel
from zedprofiler.identifiers import build_image_id

logging.basicConfig(level=logging.INFO)


@beartype
def _image_loading(image_path: pathlib.Path) -> numpy.ndarray:
    """Internal loader using bioio as a backend

    Parameters
    ----------
    image_path : pathlib.Path
        Path to the image to load

    Returns
    -------
    numpy.ndarray
        Image returned

    """
    image = bioio.BioImage(str(image_path))  # selects the first scene found
    return image.get_image_data("ZYX")


@dataclass
class ImageSetConfig:
    """Configuration options for ImageSetLoader."""

    image_set_name: str | None = None
    label_key_name: list[str] | None = None
    raw_image_key_name: list[str] | None = None
    # Imaging-coordinate identifier fields used to build a deterministic
    # ``Metadata_Imaging_ImageID``. All four must be set for ``image_id`` to be
    # populated; when any is None, ``image_id`` is None and the loader falls
    # back to ``image_set_name`` for the emitted metadata column (see
    # ``ImageSetLoader.image_id``).
    patient_tumor: str | None = None
    plate: str | None = None
    well: str | None = None
    field: int | str | None = None

    # validate the arg types
    def __post_init__(self) -> None:
        """Initialize default values for None fields."""
        if not isinstance(self.image_set_name, (str, type(None))):
            raise TypeError("image_set_name must be a string or None")
        if not isinstance(self.label_key_name, (list, type(None))):
            raise TypeError("label_key_name must be a list of strings or None")
        if not isinstance(self.raw_image_key_name, (list, type(None))):
            raise TypeError("raw_image_key_name must be a list of strings or None")
        if not isinstance(self.patient_tumor, (str, type(None))):
            raise TypeError("patient_tumor must be a string or None")
        if not isinstance(self.plate, (str, type(None))):
            raise TypeError("plate must be a string or None")
        if not isinstance(self.well, (str, type(None))):
            raise TypeError("well must be a string or None")
        if not isinstance(self.field, (int, str, type(None))):
            raise TypeError("field must be an int, str, or None")

        if self.label_key_name is None:
            self.label_key_name = []
        if self.raw_image_key_name is None:
            self.raw_image_key_name = []

    @property
    def image_id(self) -> str | None:
        """Deterministic ``Metadata_Imaging_ImageID`` value, or None if unset.

        Returns ``build_image_id(...)`` when all four coordinate fields are
        set, otherwise ``None`` (the loader then falls back to
        ``image_set_name``).
        """
        if (
            self.patient_tumor is not None
            and self.plate is not None
            and self.well is not None
            and self.field is not None
        ):
            return build_image_id(
                patient_tumor=self.patient_tumor,
                plate=self.plate,
                well=self.well,
                field=self.field,
            )
        return None


class _LazyImageSetDict(dict):  # type: ignore[type-arg]
    """Dictionary that loads image arrays on first access."""

    def __getitem__(self, key: str) -> numpy.ndarray:
        value = super().__getitem__(key)
        if isinstance(value, pathlib.Path):
            value = _image_loading(value)
            super().__setitem__(key, value)
        return value

    def get(  # type: ignore[override]
        self,
        key: str,
        default: pathlib.Path | numpy.ndarray | None = None,
    ) -> pathlib.Path | numpy.ndarray | None:
        if key in self:
            return self[key]
        return default

    def items(self) -> Iterator[tuple[str, numpy.ndarray]]:  # type: ignore[override]
        for key in dict.__iter__(self):
            yield key, self[key]

    def values(self) -> Iterator[numpy.ndarray]:  # type: ignore[override]
        for key in dict.__iter__(self):
            yield self[key]


class ImageSetLoader:
    """ImageSet in this context refers to a set of images that can be
    related to each other via their metadata.
    For example all images coming from the same well, FOV or timepoint
    but different spectral channels and segmentation labels.

    Load an image set consisting of raw z stack images and segmentation labels.

    A class to load an image set consisting of raw z stack images from multiple
    spectral channels and segmentation labels. The images are loaded into a
    dictionary, and various attributes and compartments are extracted from the
    images. The class also provides methods to retrieve images and their attributes.

    Parameters
    ----------
    image_set_path : pathlib.Path
        Path to the image set directory.
    label_set_path : pathlib.Path
        Path to the label set directory.
    anisotropy_spacing : tuple
        The anisotropy spacing of the images in format
        (z_spacing, y_spacing, x_spacing).
    channel_mapping : dict
        A dictionary mapping channel names to their corresponding image file names.
        Example: ``{'nuclei': 'nuclei_', 'cell': 'cell_', 'cytoplasm': 'cytoplasm_'}``

    Attributes
    ----------
    image_set_name : str
        The name of the image set.
    anisotropy_spacing : tuple
        The anisotropy spacing of the images.
    anisotropy_factor : float
        The anisotropy factor calculated from the spacing.
    image_set_dict : dict
        A dictionary containing the loaded images, with keys as channel names.
    unique_label_objects : dict
        A dictionary containing unique object IDs for each label in the image set.
    unique_compartment_objects : dict
        A dictionary containing unique object IDs for each compartment in the image set.
        A compartment is defined as a segmented region in the image (e.g., Cell,
        Cytoplasm, Nuclei, Organoid). The compartments are bounds for measurements.
    image_names : list
        A list of image names in the image set.
    compartments : list
        A list of compartment names in the image set.

    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        anisotropy_spacing: tuple[float, float, float],
        channel_mapping: dict[str, str],
        image_set_path: pathlib.Path | None,
        label_set_path: pathlib.Path | None,
        image_set_array: numpy.ndarray | None = None,
        label_set_array: numpy.ndarray | None = None,
        config: ImageSetConfig | None = None,
    ) -> None:
        """Initialize the ImageSetLoader with paths, spacing, and mapping.

        Parameters
        ----------
        image_set_path : pathlib.Path
            Path to the image set directory.
        label_set_path : pathlib.Path | None
            Path to the label set directory.
        anisotropy_spacing : tuple
            The anisotropy spacing of the images. In format
            (z_spacing, y_spacing, x_spacing).
        channel_mapping : dict
            A dictionary mapping channel names to image file names.
        config : ImageSetConfig | None
            Optional configuration object with image_set_name, label_key_name,
            and raw_image_key_name. If None, defaults are used.

        """
        config = config or ImageSetConfig()
        self._validate_input_sources(
            image_set_path=image_set_path,
            label_set_path=label_set_path,
            image_set_array=image_set_array,
            label_set_array=label_set_array,
        )
        self.image_set_dict = _LazyImageSetDict()
        channel_tokens = [str(value) for value in channel_mapping.values()]
        self.anisotropy_spacing = anisotropy_spacing
        self.anisotropy_factor = self.anisotropy_spacing[0] / self.anisotropy_spacing[1]
        self.image_set_name = config.image_set_name
        self.label_set_path = label_set_path
        # Deterministic imaging identifier for the warehouse join key
        # (``Metadata_Imaging_ImageID``). When identifier fields are not
        # provided (legacy/library use), fall back to the image set name so
        # the emitted metadata column always has a value.
        self.image_id = (
            config.image_id if config.image_id is not None else config.image_set_name
        )
        self._load_path_based_images(
            channel_mapping=channel_mapping,
            channel_tokens=channel_tokens,
            image_set_path=image_set_path,
            label_set_path=label_set_path,
        )
        self._load_array_based_images(
            config=config,
            image_set_array=image_set_array,
            label_set_array=label_set_array,
        )

        self.get_compartments()
        self.get_image_names()
        self.get_unique_objects_in_compartments()

    @classmethod
    def from_image_dict(  # noqa: PLR0913
        cls,
        image_dict: dict[str, numpy.ndarray],
        *,
        anisotropy_spacing: tuple[float, float, float],
        image_set_name: str | None = None,
        label_key_names: list[str] | None = None,
        patient_tumor: str | None = None,
        plate: str | None = None,
        well: str | None = None,
        field: int | str | None = None,
    ) -> ImageSetLoader:
        """Build an ImageSetLoader from an in-memory channel/label dict.

        Existing constructors only accept a directory glob (path-based) or a
        single array (array-based). A well/FOV shard carries multiple channels
        and multiple compartments as distinct arrays, so this classmethod
        builds the ``image_set_dict`` directly from a pre-loaded
        ``{key: ndarray}`` mapping. It formalizes the ``ImageSetLoader.__new__``
        workaround previously used in the colocalization test helper.

        Parameters
        ----------
        image_dict : dict[str, numpy.ndarray]
            Mapping of channel names and compartment names to their arrays.
            Each array is validated through ``ImageArrayModel``.
        anisotropy_spacing : tuple[float, float, float]
            (z_spacing, y_spacing, x_spacing).
        image_set_name : str | None
            Optional image set name (emitted as ``Metadata_Experiment_ImageSet``).
        label_key_names : list[str] | None
            Keys in ``image_dict`` that are compartment labels (not channels).
            Used by ``get_compartments`` to distinguish compartments from
            raw channels.
        patient_tumor, plate, well, field : optional
            Imaging-coordinate identifier fields. When all four are set, the
            loader's ``image_id`` is the deterministic
            ``Metadata_Imaging_ImageID``; otherwise it falls back to
            ``image_set_name``.

        Returns
        -------
        ImageSetLoader
            A fully initialized loader (compartments, image names, and unique
            compartment objects populated).

        """
        self = cls.__new__(cls)
        self.image_set_dict = _LazyImageSetDict()
        for key, array in image_dict.items():
            # Run through pydantic validation to ensure each array is valid,
            # mirroring ``_load_array_based_images``.
            self.image_set_dict[key] = ImageArrayModel(array=array).array
        self._label_key_names = list(label_key_names or [])
        self.anisotropy_spacing = anisotropy_spacing
        self.anisotropy_factor = self.anisotropy_spacing[0] / self.anisotropy_spacing[1]
        self.image_set_name = image_set_name
        self.label_set_path = None
        config = ImageSetConfig(
            image_set_name=image_set_name,
            label_key_name=list(label_key_names or []),
            raw_image_key_name=[
                key for key in image_dict if key not in (label_key_names or [])
            ],
            patient_tumor=patient_tumor,
            plate=plate,
            well=well,
            field=field,
        )
        self.image_id = (
            config.image_id if config.image_id is not None else config.image_set_name
        )
        # Set compartments and image names directly from the declared label
        # keys rather than calling ``get_compartments``/``get_image_names``.
        # Those methods' compartment heuristic differs across repo revisions
        # (it was corrected in a later bugfix commit), but this classmethod
        # already knows which keys are labels, so deriving the split here keeps
        # it self-contained and correct on any base.
        self.compartments = list(self._label_key_names)
        self.image_names = [
            key for key in image_dict if key not in self._label_key_names
        ]
        self.get_unique_objects_in_compartments()
        return self

    @staticmethod
    def _validate_input_sources(
        image_set_path: pathlib.Path | None,
        label_set_path: pathlib.Path | None,
        image_set_array: numpy.ndarray | None,
        label_set_array: numpy.ndarray | None,
    ) -> None:
        """Validate the input sources such that either the image path or the
        array is passed through but not neither and not both.

        Parameters
        ----------
        image_set_path : pathlib.Path | None
            Path to the image set directory.
        label_set_path : pathlib.Path | None
            Path to the label set directory.
        image_set_array : numpy.ndarray | None
            Array containing the image data.
        label_set_array : numpy.ndarray | None
            Array containing the label data.

        Raises
        ------
        ValueError
            If neither image_set_array nor image_set_path is provided, or if
            neither label_set_array nor label_set_path is provided.
        ValueError
            If both image_set_array and image_set_path are provided, or if
            both label_set_array and label_set_path are provided.

        """
        if image_set_array is None and image_set_path is None:
            raise ValueError(
                "Either image_set_array or image_set_path must be provided.",
            )
        if label_set_array is None and label_set_path is None:
            raise ValueError(
                "Either label_set_array or label_set_path must be provided.",
            )
        if image_set_array is not None and image_set_path is not None:
            raise ValueError(
                "Only one of image_set_array or image_set_path should be "
                "provided, not both.",
            )
        if label_set_array is not None and label_set_path is not None:
            raise ValueError(
                "Only one of label_set_array or label_set_path should be "
                "provided, not both.",
            )

    def _load_path_based_images(
        self,
        channel_mapping: dict[str, str],
        channel_tokens: list[str],
        image_set_path: pathlib.Path | None,
        label_set_path: pathlib.Path | None,
    ) -> None:
        """Load the images if a path is given.
        Note that currently we only load tiffs...

        Parameters
        ----------
        channel_mapping : dict[str, str]
            A dictionary mapping channel names to image file name tokens.
        channel_tokens : list[str]
            A list of tokens to look for in file names to identify channels.
        image_set_path : pathlib.Path | None
            Path to the image set directory.
        label_set_path : pathlib.Path | None
            Path to the label set directory.

        """
        if image_set_path is None:
            return

        channel_files = sorted(image_set_path.glob("*"))
        channel_files = [
            f
            for f in channel_files
            if f.suffix in [".tif", ".tiff"]
            and any(token in f.name for token in channel_tokens)
        ]

        label_files = sorted(label_set_path.glob("*")) if label_set_path else []
        label_files = [
            f
            for f in label_files
            if f.suffix in [".tif", ".tiff"]
            and any(token in f.name for token in channel_tokens)
        ]

        for f in channel_files:
            for key, value in channel_mapping.items():
                if str(value) in f.name:
                    self.image_set_dict[key] = f
        for f in label_files:
            for key, value in channel_mapping.items():
                if str(value) in f.name:
                    self.image_set_dict[key] = f

    def _load_array_based_images(
        self,
        config: ImageSetConfig,
        image_set_array: numpy.ndarray | None,
        label_set_array: numpy.ndarray | None,
    ) -> None:
        """Load the array based images.
        These are already in memory and stored as numpy arrays.

        Parameters
        ----------
        config : ImageSetConfig
            Configuration object containing key names for images and labels.
        image_set_array : numpy.ndarray | None
            Array containing the image data.
        label_set_array : numpy.ndarray | None
            Array containing the label data.

        """
        if image_set_array is not None:
            for key in config.raw_image_key_name or []:
                # Run through pydantic validation to ensure the array is valid.
                validated_array = ImageArrayModel(array=image_set_array).array
                self.image_set_dict[key] = validated_array
        if label_set_array is not None:
            for key in config.label_key_name or []:
                # Run through pydantic validation to ensure the array is valid.
                validated_array = ImageArrayModel(array=label_set_array).array
                self.image_set_dict[key] = validated_array

    def get_unique_objects_in_compartments(self) -> None:
        """Populate unique object IDs per compartment.

        Parameters
        ----------
        None
            This method does not take any parameters.

        """
        self.unique_compartment_objects = {}
        if len(self.compartments) == 0:
            return
        for compartment in self.compartments:
            self.unique_compartment_objects[compartment] = numpy.unique(
                self.get_image(compartment),
            )
            # remove the 0 label
            self.unique_compartment_objects[compartment] = [
                x for x in self.unique_compartment_objects[compartment] if x != 0
            ]

    def get_image(self, key: str) -> numpy.ndarray:
        """Return an image array for a given key.

        Parameters
        ----------
        key : str
            Channel or label key.

        Returns
        -------
        numpy.ndarray
            Image array for the requested key.

        """
        return self.image_set_dict[key]

    def get_image_names(self) -> list[str]:
        """Populate image (non-compartment) names.

        Returns
        -------
        list[str]
            List of image names excluding compartment labels.

        """
        compartments = (
            self.compartments
            if self.compartments is not None and isinstance(self.compartments, list)
            else []
        )
        self.image_names = [x for x in self.image_set_dict if x not in compartments]
        return self.image_names

    def get_compartments(self) -> list[str]:
        """Populate compartment names from available keys.

        Returns
        -------
        list[str]
            List of compartment keys.

        """
        self.compartments = [
            x
            for x in self.image_set_dict
            if any(
                channel_mapping_key in x for channel_mapping_key in self.image_set_dict
            )
        ]
        return self.compartments

    def get_anisotropy(self) -> float:
        """Return the anisotropy factor for the image set.

        Returns
        -------
        float
            Ratio of z-spacing to y-spacing.

        """
        return self.anisotropy_spacing[0] / self.anisotropy_spacing[1]


class ObjectLoader:
    """A class to load objects from a labeled image and extract their properties.
    Where an object is defined as a segmented region in the image.
    This could be a cell, a nucleus, or any other compartment segmented.

    Parameters
    ----------
    image : numpy.ndarray
        The image from which to extract objects. Preferably a 3D image -> z, y, x
    label_image : numpy.ndarray
        The labeled image containing the segmented objects.
    channel_name : str
        The name of the channel from which the objects are extracted.
    compartment_name : str
        The name of the compartment from which the objects are extracted.

    Attributes
    ----------
    image_set_loader : ImageSetLoader
        An instance of the ImageSetLoader class containing the image set.
    config : ImageSetConfig
        The configuration object containing image set parameters.

    Methods
    -------
    __init__(image, label_image, channel_name, compartment_name)
        Initializes the ObjectLoader with the image, label image, channel
        name, and compartment name.

    """

    def __init__(
        self,
        image_set_loader: ImageSetLoader,
        channel_name: str,
        compartment_name: str,
    ) -> None:
        """Initialize object loader with image and labels.

        Parameters
        ----------
        image_set_loader : ImageSetLoader
            An instance of the ImageSetLoader class containing the image set.
        channel_name : str
            The name of the channel from which the objects are extracted.
        compartment_name : str
            The name of the compartment from which the objects are extracted.

        """
        self.channel = channel_name
        self.compartment = compartment_name
        self.image = image_set_loader.get_image(self.channel) if self.channel else None
        self.label_image = (
            image_set_loader.get_image(self.compartment) if self.compartment else None
        )
        # get the labeled image objects
        if self.label_image is not None:
            self.object_ids: list[int] = [
                int(x) for x in numpy.unique(self.label_image) if x != 0
            ]
        else:
            self.object_ids = []
        # inherit the image set loader
        self.image_set_loader = image_set_loader


class TwoObjectLoader:
    """A class to load two images and a label image for a specific compartment.
    This class is primarily used for loading images for two-channel
    analysis like co-localization.

    Parameters
    ----------
    image_set_loader : ImageSetLoader
        An instance of the ImageSetLoader class containing the image set.
    compartment : str
        The name of the compartment for which the label image is loaded.
    channel1 : str
        The name of the first channel to be loaded.
    channel2 : str
        The name of the second channel to be loaded.

    Attributes
    ----------
    image_set_loader : ImageSetLoader
        An instance of the ImageSetLoader class containing the image set.
    compartment : str
        The name of the compartment for which the label image is loaded.
    label_image : numpy.ndarray
        The labeled image containing the segmented objects for the
        specified compartment.
    image1 : numpy.ndarray
        The image corresponding to the first channel.
    image2 : numpy.ndarray
        The image corresponding to the second channel.
    object_ids : numpy.ndarray
        The unique object IDs for the segmented objects in the specified compartment.

    Methods
    -------
    __init__(image_set_loader, compartment, channel1, channel2)
        Initializes the TwoObjectLoader with the image set loader,
        compartment, and channel names.

    """

    def __init__(
        self,
        image_set_loader: ImageSetLoader,
        compartment: str,
        channel1: str,
        channel2: str,
    ) -> None:
        """Initialize a two-channel loader for a compartment.

        Parameters
        ----------
        image_set_loader : ImageSetLoader
            Image set loader containing images and labels.
        compartment : str
            Compartment name for the label image.
        channel1 : str
            First channel name to load.
        channel2 : str
            Second channel name to load.

        """
        self.image_set_loader = image_set_loader
        self.compartment = compartment
        self.label_image = self.image_set_loader.get_image(compartment)
        self.image1 = self.image_set_loader.get_image(channel1)
        self.image2 = self.image_set_loader.get_image(channel2)
        self.object_ids = image_set_loader.unique_compartment_objects[compartment]
        # inherit the image set name for downstream use
        self.image_set_name = image_set_loader.image_set_name
