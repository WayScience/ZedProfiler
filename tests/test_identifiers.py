"""Tests for the deterministic Metadata_Imaging_ImageID builder."""

from __future__ import annotations

import pytest

from zedprofiler.identifiers import build_image_id


def test_build_image_id_formats_all_fields() -> None:
    """The image id joins patient-tumor, plate, well, and a field-of-view suffix."""
    assert (
        build_image_id("NF0014_T1", "PLATE01", "A1", 1) == "NF0014_T1_PLATE01_A1_fov1"
    )


def test_build_image_id_is_deterministic() -> None:
    """Repeated calls with the same inputs produce the same id."""
    args = ("NF0014_T1", "PLATE01", "A1", "2")
    assert build_image_id(*args) == build_image_id(*args)


def test_build_image_id_field_of_view_accepts_string_or_int() -> None:
    """The field of view may be supplied as either an int or its string form."""
    assert build_image_id("X", "P", "A1", 3) == "X_P_A1_fov3"
    assert build_image_id("X", "P", "A1", "3") == "X_P_A1_fov3"


@pytest.mark.parametrize(
    "patient_tumor,plate,well,field_of_view,expected",
    [
        ("NF0014_T1", "PLATE01", "A1", "1", "NF0014_T1_PLATE01_A1_fov1"),
        ("NF0009_T2", "PLATE02", "B3", "7", "NF0009_T2_PLATE02_B3_fov7"),
    ],
)
def test_build_image_id_parametrized(
    patient_tumor: str,
    plate: str,
    well: str,
    field_of_view: str,
    expected: str,
) -> None:
    """Format holds across distinct imaging coordinates."""
    assert build_image_id(patient_tumor, plate, well, field_of_view) == expected
