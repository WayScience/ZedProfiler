from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zedprofiler.featurization.colocalization import compute_colocalization
from zedprofiler.featurization.granularity import compute_granularity
from zedprofiler.featurization.intensity import compute_intensity
from zedprofiler.featurization.neighbors import compute_neighbors
from zedprofiler.featurization.texture import compute_texture
from zedprofiler.featurization.volumesizeshape import compute_volume_size_shape
from zedprofiler.IO.loading_classes import (
    ImageSetConfig,
    ImageSetLoader,
    ObjectLoader,
    TwoObjectLoader,
)

tifffile = pytest.importorskip("tifffile")

CELLPROFILER_TUTORIAL_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "CP_tutorial_3D_noise_nuclei_segmentation"
)


@dataclass(frozen=True)
class RealImageCase:
    image_name: str
    image_path: Path
    label_path: Path
    image_set_name: str
    channel: str = "DNA"
    compartment: str = "Nuclei"


@dataclass(frozen=True)
class RealDatasetCase:
    name: str
    expected_shape: tuple[int, int, int]
    expected_object_count: int
    expected_image_dtype: np.dtype
    expected_label_dtype: np.dtype
    image_cases: tuple[RealImageCase, ...]


@dataclass(frozen=True)
class RealColocalizationCase:
    name: str
    first_image_case: RealImageCase
    second_image_case: RealImageCase
    label_image_case: RealImageCase
    image_set_name: str
    first_channel: str = "DNA1"
    second_channel: str = "DNA2"
    compartment: str = "Nuclei"


@dataclass(frozen=True)
class LoadedNucleiCase:
    dataset_case: RealDatasetCase
    image_case: RealImageCase
    image_set_loader: ImageSetLoader
    object_loader: ObjectLoader

    @property
    def object_ids(self) -> list[int]:
        return self.object_loader.object_ids


@dataclass(frozen=True)
class FeatureRunner:
    name: str
    run: Callable[[LoadedNucleiCase], pd.DataFrame]
    expected_column_token: str


def _cellprofiler_tutorial_image_case(image_name: str) -> RealImageCase:
    return RealImageCase(
        image_name=image_name,
        image_path=CELLPROFILER_TUTORIAL_ROOT / "input" / f"{image_name}.tif",
        label_path=(
            CELLPROFILER_TUTORIAL_ROOT
            / "output"
            / "masks"
            / f"{image_name}SegmentationMask.tiff"
        ),
        image_set_name=image_name,
    )


CELLPROFILER_TUTORIAL_IMAGES = (
    _cellprofiler_tutorial_image_case("nuclei1_out_c00_dr90_image"),
    _cellprofiler_tutorial_image_case("nuclei2_out_c90_dr90_image"),
    _cellprofiler_tutorial_image_case("nuclei3_out_c00_dr10_image"),
    _cellprofiler_tutorial_image_case("nuclei4_out_c90_dr10_image"),
)


REAL_DATASETS = (
    RealDatasetCase(
        name="cellprofiler-tutorial",
        expected_shape=(100, 258, 258),
        expected_object_count=5,
        expected_image_dtype=np.dtype("uint16"),
        expected_label_dtype=np.dtype("uint16"),
        image_cases=CELLPROFILER_TUTORIAL_IMAGES,
    ),
)


IMAGE_CASES = tuple(
    (dataset_case, image_case)
    for dataset_case in REAL_DATASETS
    for image_case in dataset_case.image_cases
)

COLOCALIZATION_CASES = (
    (
        REAL_DATASETS[0],
        RealColocalizationCase(
            name="dr90-c00-c90",
            first_image_case=CELLPROFILER_TUTORIAL_IMAGES[0],
            second_image_case=CELLPROFILER_TUTORIAL_IMAGES[1],
            label_image_case=CELLPROFILER_TUTORIAL_IMAGES[0],
            image_set_name="dr90-c00-c90",
        ),
    ),
    (
        REAL_DATASETS[0],
        RealColocalizationCase(
            name="dr10-c00-c90",
            first_image_case=CELLPROFILER_TUTORIAL_IMAGES[2],
            second_image_case=CELLPROFILER_TUTORIAL_IMAGES[3],
            label_image_case=CELLPROFILER_TUTORIAL_IMAGES[2],
            image_set_name="dr10-c00-c90",
        ),
    ),
)


def _feature_runner_id(runner: FeatureRunner) -> str:
    return runner.name


def _dataset_case_id(dataset_case: RealDatasetCase) -> str:
    return dataset_case.name


def _image_case_id(case: tuple[RealDatasetCase, RealImageCase]) -> str:
    dataset_case, image_case = case
    return f"{dataset_case.name}-{image_case.image_name}"


def _colocalization_case_id(
    case: tuple[RealDatasetCase, RealColocalizationCase],
) -> str:
    dataset_case, colocalization_case = case
    return f"{dataset_case.name}-{colocalization_case.name}"


def _load_nuclei_case(
    dataset_case: RealDatasetCase,
    image_case: RealImageCase,
) -> LoadedNucleiCase:
    image = tifffile.imread(image_case.image_path)
    label = tifffile.imread(image_case.label_path)
    image_set_loader = ImageSetLoader(
        image_set_path=None,
        label_set_path=None,
        image_set_array=image,
        label_set_array=label,
        anisotropy_spacing=(1.0, 1.0, 1.0),
        channel_mapping={
            image_case.channel: image_case.image_name,
            image_case.compartment: "SegmentationMask",
        },
        config=ImageSetConfig(
            image_set_name=image_case.image_set_name,
            label_key_name=[image_case.compartment],
            raw_image_key_name=[image_case.channel],
        ),
    )
    object_loader = ObjectLoader(
        image_set_loader=image_set_loader,
        channel_name=image_case.channel,
        compartment_name=image_case.compartment,
    )

    return LoadedNucleiCase(
        dataset_case=dataset_case,
        image_case=image_case,
        image_set_loader=image_set_loader,
        object_loader=object_loader,
    )


def _load_nuclei_case_from_paths(
    dataset_case: RealDatasetCase,
    image_case: RealImageCase,
) -> LoadedNucleiCase:
    image_set_loader = ImageSetLoader(
        image_set_path=image_case.image_path.parent,
        label_set_path=image_case.label_path.parent,
        anisotropy_spacing=(1.0, 1.0, 1.0),
        channel_mapping={
            image_case.channel: image_case.image_name,
            image_case.compartment: f"{image_case.image_name}SegmentationMask",
        },
        config=ImageSetConfig(
            image_set_name=image_case.image_set_name,
            label_key_name=[image_case.compartment],
            raw_image_key_name=[image_case.channel],
        ),
    )
    object_loader = ObjectLoader(
        image_set_loader=image_set_loader,
        channel_name=image_case.channel,
        compartment_name=image_case.compartment,
    )

    return LoadedNucleiCase(
        dataset_case=dataset_case,
        image_case=image_case,
        image_set_loader=image_set_loader,
        object_loader=object_loader,
    )


def _load_colocalization_case(
    colocalization_case: RealColocalizationCase,
) -> TwoObjectLoader:
    label = tifffile.imread(colocalization_case.label_image_case.label_path)
    object_ids = [int(x) for x in np.unique(label) if x != 0]
    image_set_loader = ImageSetLoader.__new__(ImageSetLoader)
    image_set_loader.image_set_name = colocalization_case.image_set_name
    image_set_loader.image_set_dict = {
        colocalization_case.first_channel: tifffile.imread(
            colocalization_case.first_image_case.image_path,
        ),
        colocalization_case.second_channel: tifffile.imread(
            colocalization_case.second_image_case.image_path,
        ),
        colocalization_case.compartment: label,
    }
    image_set_loader.unique_compartment_objects = {
        colocalization_case.compartment: object_ids,
    }

    return TwoObjectLoader(
        image_set_loader=image_set_loader,
        compartment=colocalization_case.compartment,
        channel1=colocalization_case.first_channel,
        channel2=colocalization_case.second_channel,
    )


def _expected_volumes_from_label(label: np.ndarray) -> dict[int, int]:
    return {
        int(object_id): int(np.count_nonzero(label == object_id))
        for object_id in np.unique(label)
        if object_id != 0
    }


def _assert_real_feature_frame_matches_objects(
    df: pd.DataFrame,
    loaded_case: LoadedNucleiCase,
    expected_column_token: str,
) -> None:
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == loaded_case.dataset_case.expected_object_count
    assert "Metadata_Object_ObjectID" in df.columns
    assert "Metadata_Experiment_ImageSet" in df.columns
    assert any(expected_column_token in column for column in df.columns)

    returned_ids = sorted(int(x) for x in df["Metadata_Object_ObjectID"].tolist())
    assert returned_ids == loaded_case.object_ids
    assert set(df["Metadata_Experiment_ImageSet"]) == {
        loaded_case.image_case.image_set_name,
    }

    value_columns = [
        column for column in df.columns if not column.startswith("Metadata_")
    ]
    assert value_columns
    values = df[value_columns].to_numpy(dtype=float)
    assert np.isfinite(values).all()


FEATURE_RUNNERS = (
    FeatureRunner(
        name="volume-size-shape",
        run=lambda loaded_case: compute_volume_size_shape(
            image_set_loader=loaded_case.image_set_loader,
            object_loader=loaded_case.object_loader,
        ),
        expected_column_token="VolumeSizeShape",
    ),
    FeatureRunner(
        name="intensity",
        run=lambda loaded_case: compute_intensity(loaded_case.object_loader),
        expected_column_token="Intensity",
    ),
    FeatureRunner(
        name="neighbors",
        run=lambda loaded_case: compute_neighbors(
            loaded_case.object_loader,
            distance_threshold=50,
            anisotropy_factor=1,
        ),
        expected_column_token="Neighbors",
    ),
    FeatureRunner(
        name="texture",
        run=lambda loaded_case: compute_texture(
            loaded_case.object_loader,
            distance=1,
            grayscale=256,
        ),
        expected_column_token="Texture",
    ),
    FeatureRunner(
        name="granularity",
        run=lambda loaded_case: compute_granularity(
            loaded_case.object_loader,
            radius=1,
            granular_spectrum_length=2,
            subsample_size=1.0,
            image_sample_size=1.0,
        ),
        expected_column_token="Granularity",
    ),
)


@pytest.mark.parametrize("dataset_case", REAL_DATASETS, ids=_dataset_case_id)
def test_real_dataset_files_are_static(dataset_case: RealDatasetCase) -> None:
    for image_case in dataset_case.image_cases:
        image = tifffile.imread(image_case.image_path)
        label = tifffile.imread(image_case.label_path)
        object_ids = [int(x) for x in np.unique(label) if x != 0]

        assert image.shape == dataset_case.expected_shape
        assert label.shape == dataset_case.expected_shape
        assert image.dtype == dataset_case.expected_image_dtype
        assert label.dtype == dataset_case.expected_label_dtype
        assert len(object_ids) == dataset_case.expected_object_count


@pytest.mark.parametrize("case", IMAGE_CASES, ids=_image_case_id)
@pytest.mark.parametrize("feature_runner", FEATURE_RUNNERS, ids=_feature_runner_id)
def test_real_world_nuclei_feature_extractors(
    case: tuple[RealDatasetCase, RealImageCase],
    feature_runner: FeatureRunner,
) -> None:
    dataset_case, image_case = case
    loaded_case = _load_nuclei_case(dataset_case, image_case)

    df = feature_runner.run(loaded_case)

    _assert_real_feature_frame_matches_objects(
        df=df,
        loaded_case=loaded_case,
        expected_column_token=feature_runner.expected_column_token,
    )


@pytest.mark.parametrize("case", IMAGE_CASES, ids=_image_case_id)
def test_real_world_volumes_match_label_voxel_counts(
    case: tuple[RealDatasetCase, RealImageCase],
) -> None:
    dataset_case, image_case = case
    label = tifffile.imread(image_case.label_path)
    loaded_case = _load_nuclei_case(dataset_case, image_case)

    df = compute_volume_size_shape(
        image_set_loader=loaded_case.image_set_loader,
        object_loader=loaded_case.object_loader,
    )

    volume_column = next(
        column for column in df.columns if column.endswith("_VolumeSizeShape_Volume")
    )
    zedprofiler_volumes = {
        int(object_id): int(volume)
        for object_id, volume in zip(
            df["Metadata_Object_ObjectID"],
            df[volume_column],
            strict=True,
        )
    }
    reference_volumes = _expected_volumes_from_label(label)

    assert zedprofiler_volumes == reference_volumes


@pytest.mark.parametrize("case", IMAGE_CASES, ids=_image_case_id)
def test_real_world_path_loaded_images_process_nuclei(
    case: tuple[RealDatasetCase, RealImageCase],
) -> None:
    dataset_case, image_case = case
    label = tifffile.imread(image_case.label_path)
    loaded_case = _load_nuclei_case_from_paths(dataset_case, image_case)

    assert set(loaded_case.image_set_loader.image_set_dict) == {
        image_case.channel,
        image_case.compartment,
    }
    assert loaded_case.image_set_loader.compartments == [image_case.compartment]
    assert loaded_case.image_set_loader.image_names == [image_case.channel]
    assert loaded_case.object_loader.image.shape == dataset_case.expected_shape
    assert loaded_case.object_loader.label_image.shape == dataset_case.expected_shape
    assert sorted(loaded_case.object_ids) == sorted(_expected_volumes_from_label(label))

    df = compute_volume_size_shape(
        image_set_loader=loaded_case.image_set_loader,
        object_loader=loaded_case.object_loader,
    )
    _assert_real_feature_frame_matches_objects(
        df=df,
        loaded_case=loaded_case,
        expected_column_token="VolumeSizeShape",
    )


@pytest.mark.parametrize(
    "case",
    COLOCALIZATION_CASES,
    ids=_colocalization_case_id,
)
def test_real_world_colocalization_feature_extractor(
    case: tuple[RealDatasetCase, RealColocalizationCase],
) -> None:
    dataset_case, colocalization_case = case
    two_object_loader = _load_colocalization_case(colocalization_case)

    df = compute_colocalization(
        two_object_loader,
        channel1=colocalization_case.first_channel,
        channel2=colocalization_case.second_channel,
        fast_costes="Faster",
    )

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == dataset_case.expected_object_count
    assert "Metadata_Object_ObjectID" in df.columns
    assert "Metadata_Experiment_ImageSet" in df.columns
    assert any("Colocalization" in column for column in df.columns)
    assert sorted(int(x) for x in df["Metadata_Object_ObjectID"]) == sorted(
        two_object_loader.object_ids,
    )
    assert set(df["Metadata_Experiment_ImageSet"]) == {
        colocalization_case.image_set_name,
    }

    value_columns = [
        column for column in df.columns if not column.startswith("Metadata_")
    ]
    values = df[value_columns].to_numpy(dtype=float)
    assert np.isfinite(values).all()
