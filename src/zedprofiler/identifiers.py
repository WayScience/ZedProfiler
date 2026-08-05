"""Deterministic imaging identifiers for warehouse join keys.

The NF1 bioimage profiling warehouse joins every image, object, feature
table, and annotation via stable identifiers (see the future processing
plan's identifier spec). The central one is ``Metadata_Imaging_ImageID``,
built deterministically from the four imaging coordinates:

    patient-tumor, plate, well, field

Because a shard is dispatched per well/FOV, every feature table emitted by a
shard carries this single image id so downstream tables can rejoin without a
database service.

This module is the single source of truth for the id format. Changing the
format here changes every shard's output ids at once.
"""

from __future__ import annotations

from beartype import beartype


@beartype
def build_image_id(
    patient_tumor: str,
    plate: str,
    well: str,
    field: int | str,
) -> str:
    """Build a deterministic ``Metadata_Imaging_ImageID`` value.

    The id is a stable string assembled from the four imaging coordinates so
    that the same well/FOV always produces the same id across runs, batches,
    and reprocessing. Component order is fixed
    (patient-tumor, plate, well, field) so ids sort and group naturally by
    patient then plate then well then field.

    Parameters
    ----------
    patient_tumor : str
        Patient-tumor identifier (e.g. ``"NF0014_T1"``).
    plate : str
        Plate identifier (e.g. ``"PLATE01"``).
    well : str
        Well identifier (e.g. ``"A1"``).
    field : int | str
        Field-of-view index or identifier (e.g. ``1`` or ``"f1"``).

    Returns
    -------
    str
        The deterministic image id, e.g. ``"NF0014_T1_PLATE01_A1_field1"``.

    """
    return f"{patient_tumor}_{plate}_{well}_field{field}"