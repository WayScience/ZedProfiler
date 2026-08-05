"""End-to-end and unit tests for the per-shard CLI (``zedprofiler.cli``).

The end-to-end tests exercise ``ZedProfiler run`` against the CellProfiler 3D
tutorial data (the same fixtures used by ``test_real_world_data.py``) and assert
the things that matter to the NF1 pipeline: one Parquet per requested feature
table, a deterministic ``Metadata_Imaging_ImageID``, ``--features`` selection,
``--skip-existing`` idempotency, and two-channel colocalization.

The unit tests cover the request-selection and path-naming helpers directly so
the full default feature matrix (which includes slow Texture/Granularity
runs) does not have to run end-to-end here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from zedprofiler.cli import (
    _auto_requests,
    _output_path,
    _parse_feature_spec,
    _parse_name_path,
    _resolve_requests,
    main,
)
from zedprofiler.featurization import texture
from zedprofiler.identifiers import build_image_id

tifffile = pytest.importorskip("tifffile")

TUTORIAL_ROOT = (
    Path(__file__).resolve().parent
    / "data"
    / "CP_tutorial_3D_noise_nuclei_segmentation"
)
IMAGE1 = TUTORIAL_ROOT / "input" / "nuclei1_out_c00_dr90_image.tif"
IMAGE2 = TUTORIAL_ROOT / "input" / "nuclei2_out_c90_dr90_image.tif"
LABEL1 = (
    TUTORIAL_ROOT
    / "output"
    / "masks"
    / "nuclei1_out_c00_dr90_imageSegmentationMask.tiff"
)

EXPECTED_OBJECT_COUNT = 5
EXPECTED_CHANNEL_COUNT = 2
PATIENT_TUMOR, PLATE, WELL, FIELD = "NF0014_T1", "PLATE01", "A1", "1"
EXPECTED_IMAGE_ID = build_image_id(PATIENT_TUMOR, PLATE, WELL, FIELD)


def test_feature_namespace_import() -> None:
    """Lower-level feature namespace remains importable."""
    assert texture.__name__ == "zedprofiler.featurization.texture"


def _run(argv: list[str]) -> int:
    return main(argv)


# ---------------------------------------------------------------------------
# Unit tests: name/path + feature-spec parsing
# ---------------------------------------------------------------------------


def test_parse_name_path_splits_on_first_equals() -> None:
    """NAME=PATH splits on the first '=' so paths may contain '='."""
    name, path = _parse_name_path("DNA=/tmp/a=b.tif")
    assert name == "DNA"
    assert path == Path("/tmp/a=b.tif")


def test_parse_name_path_rejects_missing_equals() -> None:
    """A token without '=' is a user error, not a silent default."""
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_name_path("DNA.tif")


def test_parse_feature_spec_parses_type_and_overrides() -> None:
    """TYPE is the first comma token; the rest are key=value overrides."""
    request = _parse_feature_spec("Intensity,channel=DNA,compartment=Nuclei")
    assert request == {
        "type": "Intensity",
        "channel": "DNA",
        "compartment": "Nuclei",
    }


def test_parse_feature_spec_rejects_unknown_type() -> None:
    """An unknown feature type raises a friendly argument error."""
    with pytest.raises(argparse.ArgumentTypeError, match="Unknown feature type"):
        _parse_feature_spec("NotAFeature,channel=DNA")


def test_parse_feature_spec_rejects_bad_override() -> None:
    """An override without '=' raises a friendly argument error."""
    with pytest.raises(argparse.ArgumentTypeError, match="key=value"):
        _parse_feature_spec("Intensity,bogus")


def test_output_path_mirrors_save_features_as_parquet_naming() -> None:
    """_output_path must match the path save_features_as_parquet writes to."""
    assert _output_path(Path("/out"), "Nuclei", "DNA", "Intensity") == Path(
        "/out/Nuclei_DNA_Intensity_cpu_features.parquet"
    )
    assert _output_path(Path("/out"), "Nuclei", "DNA1-DNA2", "Colocalization") == Path(
        "/out/Nuclei_DNA1-DNA2_Colocalization_cpu_features.parquet"
    )


# ---------------------------------------------------------------------------
# Unit tests: request selection (no featurizers run)
# ---------------------------------------------------------------------------


def test_auto_requests_single_channel_no_colocalization() -> None:
    """One channel x one compartment yields the 5 single-channel types only."""
    requests = _auto_requests(
        ["DNA"],
        ["Nuclei"],
        ["VolumeSizeShape", "Intensity", "Neighbors", "Texture", "Granularity"],
    )
    types = sorted(str(r["type"]) for r in requests)
    assert types == sorted(
        ["VolumeSizeShape", "Intensity", "Neighbors", "Texture", "Granularity"],
    )
    # Channel-agnostic features use the first (only) channel for naming.
    vol = next(r for r in requests if r["type"] == "VolumeSizeShape")
    assert vol["channel"] == "DNA"


def test_auto_requests_two_channels_adds_colocalization_pairs() -> None:
    """Two channels produce a single ordered colocalization pair x compartment."""
    requests = _auto_requests(
        ["DNA1", "DNA2"],
        ["Nuclei"],
        ["Intensity", "Colocalization"],
    )
    coloc = [r for r in requests if r["type"] == "Colocalization"]
    assert len(coloc) == 1
    assert coloc[0]["channel1"] == "DNA1"
    assert coloc[0]["channel2"] == "DNA2"
    # Intensity runs per channel x compartment.
    assert (
        sum(1 for r in requests if r["type"] == "Intensity") == EXPECTED_CHANNEL_COUNT
    )


def test_auto_requests_colocalization_requires_two_channels() -> None:
    """Requesting Colocalization with one channel is a user error."""
    with pytest.raises(argparse.ArgumentTypeError, match="at least two"):
        _auto_requests(["DNA"], ["Nuclei"], ["Colocalization"])


def test_resolve_requests_default_two_channels_includes_colocalization() -> None:
    """Default selection (no --features, no --feature) adds coloc for >=2 channels."""
    requests = _resolve_requests(["DNA1", "DNA2"], ["Nuclei"], [], None)
    types = {str(r["type"]) for r in requests}
    assert "Colocalization" in types
    assert "Intensity" in types


def test_resolve_requests_features_filter_restricts_types() -> None:
    """--features Intensity selects only Intensity from the cross-product."""
    requests = _resolve_requests(
        ["DNA1", "DNA2"],
        ["Nuclei"],
        [],
        ["Intensity"],
    )
    assert all(r["type"] == "Intensity" for r in requests)
    assert len(requests) == EXPECTED_CHANNEL_COUNT  # 2 channels x 1 compartment


def test_resolve_requests_explicit_specs_validated_against_declared() -> None:
    """An explicit --feature referencing an undeclared channel is rejected."""
    specs = [_parse_feature_spec("Intensity,channel=Ghost,compartment=Nuclei")]
    with pytest.raises(argparse.ArgumentTypeError, match="declared via --image"):
        _resolve_requests(["DNA"], ["Nuclei"], specs, None)


def test_resolve_requests_features_filter_applied_to_explicit_specs() -> None:
    """--features restricts explicit --feature requests by type."""
    specs = [
        _parse_feature_spec("Intensity,channel=DNA,compartment=Nuclei"),
        _parse_feature_spec("Texture,channel=DNA,compartment=Nuclei"),
    ]
    requests = _resolve_requests(["DNA"], ["Nuclei"], specs, ["Intensity"])
    assert len(requests) == 1
    assert requests[0]["type"] == "Intensity"


# ---------------------------------------------------------------------------
# End-to-end tests on the CellProfiler 3D tutorial data
# ---------------------------------------------------------------------------

# The end-to-end tests read the CellProfiler 3D tutorial images/masks, which
# are added by a separate data commit and may be absent on some branches. Skip
# them when the data is not present so the CLI test module stays green
# everywhere; they run in full wherever the tutorial data is available.
requires_tutorial_data = pytest.mark.skipif(
    not TUTORIAL_ROOT.exists(),
    reason="CellProfiler 3D tutorial data not present on this branch",
)


def _base_run_args(out_dir: Path, *extra: str) -> list[str]:
    return [
        "run",
        f"--image=DNA={IMAGE1}",
        f"--label=Nuclei={LABEL1}",
        "--anisotropy-spacing",
        "1.0",
        "1.0",
        "1.0",
        f"--patient-tumor={PATIENT_TUMOR}",
        f"--plate={PLATE}",
        f"--well={WELL}",
        f"--field={FIELD}",
        f"--out-dir={out_dir}",
        *extra,
    ]


@requires_tutorial_data
def test_cli_run_intensity_writes_parquet_with_image_id(
    tmp_path: Path,
) -> None:
    """A single Intensity request writes one Parquet carrying the image id."""
    out_dir = tmp_path / "shard"
    assert _run(_base_run_args(out_dir, "--features=Intensity")) == 0

    parquet = _output_path(out_dir, "Nuclei", "DNA", "Intensity")
    assert parquet.exists()

    df = pd.read_parquet(parquet)
    assert len(df) == EXPECTED_OBJECT_COUNT
    assert "Metadata_Imaging_ImageID" in df.columns
    assert "Metadata_Experiment_ImageSet" in df.columns
    assert set(df["Metadata_Imaging_ImageID"]) == {EXPECTED_IMAGE_ID}
    # No leftover temp files from the atomic write.
    assert not any(p.suffix == ".tmp" for p in out_dir.iterdir())


@requires_tutorial_data
def test_cli_features_selector_restricts_outputs(tmp_path: Path) -> None:
    """--features controls exactly which feature tables are written."""
    out_dir = tmp_path / "shard"
    assert _run(_base_run_args(out_dir, "--features=VolumeSizeShape,Intensity")) == 0
    files = sorted(p.name for p in out_dir.glob("*.parquet"))
    assert files == [
        "Nuclei_DNA_Intensity_cpu_features.parquet",
        "Nuclei_DNA_VolumeSizeShape_cpu_features.parquet",
    ]

    out_dir2 = tmp_path / "shard2"
    assert _run(_base_run_args(out_dir2, "--features=Intensity")) == 0
    assert [p.name for p in out_dir2.glob("*.parquet")] == [
        "Nuclei_DNA_Intensity_cpu_features.parquet",
    ]


@requires_tutorial_data
def test_cli_rerun_is_content_identical(tmp_path: Path) -> None:
    """Re-running without --skip-existing reproduces identical feature content."""
    out_dir = tmp_path / "shard"
    _run(_base_run_args(out_dir, "--features=Intensity"))
    first = pd.read_parquet(_output_path(out_dir, "Nuclei", "DNA", "Intensity"))

    out_dir2 = tmp_path / "shard2"
    _run(_base_run_args(out_dir2, "--features=Intensity"))
    second = pd.read_parquet(_output_path(out_dir2, "Nuclei", "DNA", "Intensity"))

    pd.testing.assert_frame_equal(first, second)


@requires_tutorial_data
def test_cli_skip_existing_skips_recompute(tmp_path: Path) -> None:
    """--skip-existing leaves finished outputs untouched and skips image I/O."""
    out_dir = tmp_path / "shard"
    _run(_base_run_args(out_dir, "--features=Intensity"))
    target = _output_path(out_dir, "Nuclei", "DNA", "Intensity")
    first_bytes = target.read_bytes()
    first_mtime_ns = target.stat().st_mtime_ns

    # Re-run with --skip-existing using *nonexistent* image paths: if the CLI
    # tried to load images it would fail, proving skip happens before I/O.
    assert (
        _run(
            [
                "run",
                "--image=DNA=/does/not/exist.tif",
                "--label=Nuclei=/does/not/exist.tiff",
                "--anisotropy-spacing",
                "1.0",
                "1.0",
                "1.0",
                f"--patient-tumor={PATIENT_TUMOR}",
                f"--plate={PLATE}",
                f"--well={WELL}",
                f"--field={FIELD}",
                f"--out-dir={out_dir}",
                "--features=Intensity",
                "--skip-existing",
            ],
        )
        == 0
    )
    assert target.read_bytes() == first_bytes
    assert target.stat().st_mtime_ns == first_mtime_ns


@requires_tutorial_data
def test_cli_colocalization_two_channels(tmp_path: Path) -> None:
    """An explicit colocalization request writes a DNA1-DNA2 Parquet."""
    out_dir = tmp_path / "shard"
    assert (
        _run(
            [
                "run",
                f"--image=DNA1={IMAGE1}",
                f"--image=DNA2={IMAGE2}",
                f"--label=Nuclei={LABEL1}",
                "--anisotropy-spacing",
                "1.0",
                "1.0",
                "1.0",
                f"--patient-tumor={PATIENT_TUMOR}",
                f"--plate={PLATE}",
                f"--well={WELL}",
                f"--field={FIELD}",
                f"--out-dir={out_dir}",
                "--feature=Colocalization,channel1=DNA1,channel2=DNA2,compartment=Nuclei,fast_costes=Faster",
            ],
        )
        == 0
    )
    target = _output_path(out_dir, "Nuclei", "DNA1-DNA2", "Colocalization")
    assert target.exists()
    df = pd.read_parquet(target)
    assert len(df) == EXPECTED_OBJECT_COUNT
    assert set(df["Metadata_Imaging_ImageID"]) == {EXPECTED_IMAGE_ID}
    assert any("Colocalization" in c for c in df.columns)


def test_cli_missing_required_arg_errors(tmp_path: Path) -> None:
    """A run without --out-dir exits non-zero (argparse error)."""
    argv = [
        "run",
        f"--image=DNA={IMAGE1}",
        f"--label=Nuclei={LABEL1}",
        "--anisotropy-spacing",
        "1.0",
        "1.0",
        "1.0",
        f"--patient-tumor={PATIENT_TUMOR}",
        f"--plate={PLATE}",
        f"--well={WELL}",
        f"--field={FIELD}",
    ]
    with pytest.raises(SystemExit):
        _run(argv)
