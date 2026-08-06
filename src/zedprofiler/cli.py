"""Command-line interface for per-shard feature extraction.

``ZedProfiler run`` is the process a workflow manager (Nextflow via SLURM
``sbatch``) dispatches once per well/FOV shard. It loads one image set from
explicit file paths, runs a selected subset of featurizers, and writes one
Parquet per feature table to an output directory.

Why argparse (not ``fire``): the repo's other CLI surfaces use ``fire``, but
this command needs repeatable flags (``--image``/``--label``) and a
three-value ``--anisotropy-spacing`` flag, which argparse handles cleanly and
``fire`` does not. The ``trigger()`` entry point name is kept for consistency
with the existing ``pyproject.toml`` console script.

Idempotency: the same shard spec always produces the same output paths and
content. ``--skip-existing`` skips a feature request whose output Parquet
already exists, so a re-run fills only missing shards/feature tables without
redoing finished ones. Writes are atomic (temp file + ``os.replace``) so a
crashed shard never leaves a partial file that ``--skip-existing`` would
mistake for a complete one.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from zedprofiler.featurization.colocalization import compute_colocalization
from zedprofiler.featurization.granularity import compute_granularity
from zedprofiler.featurization.intensity import compute_intensity
from zedprofiler.featurization.neighbors import compute_neighbors
from zedprofiler.featurization.texture import compute_texture
from zedprofiler.featurization.volumesizeshape import compute_volume_size_shape
from zedprofiler.identifiers import build_image_id
from zedprofiler.IO.feature_writing_utils import (
    FeatureMetadata,
    format_morphology_feature_name,
    save_features_as_parquet,
)
from zedprofiler.IO.loading_classes import (
    ImageSetLoader,
    ObjectLoader,
    TwoObjectLoader,
    _image_loading,
)

# CPU-backed featurizers; the ``cpu_or_gpu`` component of the output filename.
_CPU_OR_GPU = "cpu"

# Minimum number of channels required for colocalization requests.
_MIN_CHANNELS_FOR_COLOCALIZATION = 2

# Feature types that consume a single channel + compartment via ObjectLoader.
_SINGLE_CHANNEL_TYPES = (
    "VolumeSizeShape",
    "Intensity",
    "Neighbors",
    "Texture",
    "Granularity",
)
# Feature types that require two channels via TwoObjectLoader.
_TWO_CHANNEL_TYPES = ("Colocalization",)
ALL_FEATURE_TYPES = (*_SINGLE_CHANNEL_TYPES, *_TWO_CHANNEL_TYPES)

# Channel-agnostic features: their computation does not use the channel image,
# but like every ZedProfiler feature they are namespaced by a channel for
# warehouse organization, so a channel (for naming only) is still required.
_CHANNEL_AGNOSTIC_TYPES = ("VolumeSizeShape", "Neighbors")


def _parse_name_path(token: str) -> tuple[str, Path]:
    """Parse a ``NAME=PATH`` flag value into a (name, path) pair."""
    if "=" not in token:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=PATH, got {token!r}",
        )
    name, raw_path = token.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"NAME in {token!r} is empty")
    return name, Path(raw_path)


def _parse_feature_spec(token: str) -> dict[str, object]:
    """Parse a ``TYPE[,key=value,...]`` feature request into a dict.

    The first comma-separated token is the feature type; the rest are
    ``key=value`` overrides for that feature's parameters.
    """
    parts = [p.strip() for p in token.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"Empty feature spec: {token!r}")
    feature_type = parts[0]
    if feature_type not in ALL_FEATURE_TYPES:
        valid = ", ".join(ALL_FEATURE_TYPES)
        raise argparse.ArgumentTypeError(
            f"Unknown feature type {feature_type!r}; valid: {valid}",
        )
    request: dict[str, object] = {"type": feature_type}
    for part in parts[1:]:
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"Expected key=value, got {part!r} in {token!r}",
            )
        key, value = part.split("=", 1)
        request[key.strip()] = value.strip()
    return request


def _output_path(
    out_dir: Path,
    compartment: str,
    channel: str,
    feature_type: str,
) -> Path:
    """The Parquet path a feature request will write to.

    Mirrors the naming inside ``save_features_as_parquet`` so the CLI can check
    existence (for ``--skip-existing``) before running a featurizer.
    """
    prefix = format_morphology_feature_name(
        compartment,
        channel,
        feature_type,
        _CPU_OR_GPU,
    )
    return out_dir / f"{prefix}_features.parquet"


def _coerce_param(value: object, cast: type, key: str, spec: str) -> object:
    """Cast a parsed string param, raising a friendly error on failure."""
    try:
        return cast(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Parameter {key}={value!r} in {spec!r} is not a valid {cast.__name__}",
        ) from exc


def _resolve_channel(request: dict[str, object], spec: str) -> str:
    """Return the single channel for a single-channel feature request."""
    channel = request.get("channel")
    if not isinstance(channel, str) or not channel:
        raise argparse.ArgumentTypeError(
            f"Feature spec {spec!r} requires a 'channel' key",
        )
    return channel


def _resolve_compartment(request: dict[str, object], spec: str) -> str:
    """Return the compartment for a feature request."""
    compartment = request.get("compartment")
    if not isinstance(compartment, str) or not compartment:
        raise argparse.ArgumentTypeError(
            f"Feature spec {spec!r} requires a 'compartment' key",
        )
    return compartment


def _run_single_channel(
    image_set_loader: ImageSetLoader,
    request: dict[str, object],
) -> tuple[str, str, object]:
    """Run a single-channel featurizer; return (channel, feature_type, df)."""
    feature_type = str(request["type"])
    spec = ",".join(f"{k}={v}" for k, v in request.items())
    compartment = _resolve_compartment(request, spec)
    channel = _resolve_channel(request, spec)
    object_loader = ObjectLoader(
        image_set_loader=image_set_loader,
        channel_name=channel,
        compartment_name=compartment,
    )
    if feature_type == "VolumeSizeShape":
        df = compute_volume_size_shape(
            image_set_loader=image_set_loader,
            object_loader=object_loader,
        )
    elif feature_type == "Intensity":
        df = compute_intensity(object_loader)
    elif feature_type == "Neighbors":
        df = compute_neighbors(
            object_loader,
            distance_threshold=int(
                request.get("distance_threshold", 10),
            ),
            anisotropy_factor=float(
                request.get(
                    "anisotropy_factor",
                    image_set_loader.anisotropy_factor,
                ),
            ),
        )
    elif feature_type == "Texture":
        df = compute_texture(
            object_loader,
            distance=int(request.get("distance", 1)),
            grayscale=int(request.get("grayscale", 256)),
        )
    elif feature_type == "Granularity":
        df = compute_granularity(
            object_loader,
            radius=int(request.get("radius", 10)),
            granular_spectrum_length=int(
                request.get("granular_spectrum_length", 16),
            ),
            subsample_size=float(request.get("subsample_size", 0.25)),
            image_sample_size=float(request.get("image_sample_size", 0.25)),
        )
    else:  # pragma: no cover - exhaustive dispatch above
        raise ValueError(f"Unhandled single-channel feature type: {feature_type}")
    return channel, feature_type, df


def _run_colocalization(
    image_set_loader: ImageSetLoader,
    request: dict[str, object],
) -> tuple[str, str, object]:
    """Run a two-channel colocalization request; return (channel, type, df)."""
    spec = ",".join(f"{k}={v}" for k, v in request.items())
    compartment = _resolve_compartment(request, spec)
    channel1 = request.get("channel1")
    channel2 = request.get("channel2")
    if not isinstance(channel1, str) or not isinstance(channel2, str):
        raise argparse.ArgumentTypeError(
            f"Colocalization spec {spec!r} requires 'channel1' and 'channel2' keys",
        )
    two_object_loader = TwoObjectLoader(
        image_set_loader=image_set_loader,
        compartment=compartment,
        channel1=channel1,
        channel2=channel2,
    )
    df = compute_colocalization(
        two_object_loader,
        thr=int(request.get("thr", 15)),
        fast_costes=str(request.get("fast_costes", "Accurate")),
        channel1=channel1,
        channel2=channel2,
    )
    return f"{channel1}-{channel2}", "Colocalization", df


def _request_output_identity(
    request: dict[str, object],
) -> tuple[str, str, str]:
    """Return (compartment, channel, feature_type) for a request's output path.

    Pure (no featurizer runs): lets ``run()`` compute the target Parquet path
    for ``--skip-existing`` filtering *before* reading any image from disk, so
    a re-run over finished shards skips image I/O entirely.
    """
    feature_type = str(request["type"])
    if feature_type == "Colocalization":
        channel = f"{request['channel1']}-{request['channel2']}"
    else:
        channel = str(request["channel"])
    return str(request["compartment"]), channel, feature_type


def _execute_request(
    image_set_loader: ImageSetLoader,
    request: dict[str, object],
    out_dir: Path,
    target: Path,
) -> Path:
    """Run one feature request and atomically write its Parquet.

    The caller precomputes ``target`` (via ``_output_path``) and handles
    ``--skip-existing`` filtering, so this always writes.
    """
    feature_type = str(request["type"])
    if feature_type == "Colocalization":
        channel, ran_type, df = _run_colocalization(image_set_loader, request)
    else:
        channel, ran_type, df = _run_single_channel(image_set_loader, request)
    compartment = str(request["compartment"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_features_as_parquet(
        out_dir,
        df,
        FeatureMetadata(
            compartment=compartment,
            channel=channel,
            feature_type=ran_type,
            cpu_or_gpu=_CPU_OR_GPU,
        ),
        atomic=True,
    )
    print(f"wrote: {target}", file=sys.stderr)
    return target


def _colocalization_requests(
    channels: list[str],
    compartments: list[str],
) -> list[dict[str, object]]:
    """One Colocalization request per ordered channel pair x compartment."""
    if len(channels) < _MIN_CHANNELS_FOR_COLOCALIZATION:
        raise argparse.ArgumentTypeError(
            "Colocalization requires at least two --image channels",
        )
    return [
        {
            "type": "Colocalization",
            "channel1": channels[i],
            "channel2": channels[j],
            "compartment": compartment,
        }
        for i in range(len(channels))
        for j in range(i + 1, len(channels))
        for compartment in compartments
    ]


def _channel_agnostic_requests(
    feature_type: str,
    channel: str,
    compartments: list[str],
) -> list[dict[str, object]]:
    """One request per compartment, using ``channel`` for naming only."""
    return [
        {"type": feature_type, "channel": channel, "compartment": compartment}
        for compartment in compartments
    ]


def _per_channel_requests(
    feature_type: str,
    channels: list[str],
    compartments: list[str],
) -> list[dict[str, object]]:
    """One request per channel x compartment (Intensity/Texture/Granularity)."""
    return [
        {"type": feature_type, "channel": channel, "compartment": compartment}
        for channel in channels
        for compartment in compartments
    ]


def _auto_requests(
    channels: list[str],
    compartments: list[str],
    feature_types: list[str],
) -> list[dict[str, object]]:
    """Generate requests over the channel x compartment cross-product.

    Used when ``--features`` is given without explicit ``--feature`` specs.
    Channel-agnostic features (VolumeSizeShape, Neighbors) use the first
    channel for naming only. Colocalization runs on each ordered channel pair
    x compartment.
    """
    if not channels:
        raise argparse.ArgumentTypeError("No --image flags provided")
    if not compartments:
        raise argparse.ArgumentTypeError("No --label flags provided")
    first_channel = channels[0]
    requests: list[dict[str, object]] = []
    for feature_type in feature_types:
        if feature_type == "Colocalization":
            requests.extend(_colocalization_requests(channels, compartments))
        elif feature_type in _CHANNEL_AGNOSTIC_TYPES:
            requests.extend(
                _channel_agnostic_requests(feature_type, first_channel, compartments),
            )
        else:  # Intensity, Texture, Granularity
            requests.extend(_per_channel_requests(feature_type, channels, compartments))
    return requests


def _resolve_requests(
    channels: list[str],
    compartments: list[str],
    feature_specs: list[dict[str, object]],
    features_filter: list[str] | None,
) -> list[dict[str, object]]:
    """Determine the final list of feature requests to run.

    - If explicit ``--feature`` specs are given, use them (filtered by
      ``--features`` if present), validating referenced channels/compartments.
    - Else auto-generate requests from ``--features`` (or all applicable types).
    """
    if feature_specs:
        requests = feature_specs
        if features_filter:
            requests = [r for r in requests if str(r["type"]) in features_filter]
        _validate_request_channels_compartments(requests, channels, compartments)
        return requests
    feature_types = features_filter if features_filter else list(_SINGLE_CHANNEL_TYPES)
    if (features_filter is None) and len(channels) >= _MIN_CHANNELS_FOR_COLOCALIZATION:
        feature_types = [*_SINGLE_CHANNEL_TYPES, "Colocalization"]
    return _auto_requests(channels, compartments, feature_types)


def _validate_request_channels_compartments(
    requests: list[dict[str, object]],
    channels: list[str],
    compartments: list[str],
) -> None:
    """Ensure each explicit request references declared channels/compartments."""
    for request in requests:
        feature_type = str(request["type"])
        compartment = request.get("compartment")
        if compartment not in compartments:
            raise argparse.ArgumentTypeError(
                f"compartment {compartment!r} in {feature_type} request is not "
                f"declared via --label (valid: {compartments})",
            )
        if feature_type == "Colocalization":
            for key in ("channel1", "channel2"):
                ch = request.get(key)
                if ch not in channels:
                    raise argparse.ArgumentTypeError(
                        f"{key} {ch!r} in {feature_type} request is not "
                        f"declared via --image (valid: {channels})",
                    )
        else:
            ch = request.get("channel")
            if ch not in channels:
                raise argparse.ArgumentTypeError(
                    f"channel {ch!r} in {feature_type} request is not "
                    f"declared via --image (valid: {channels})",
                )


def _build_image_set_loader(
    images: list[tuple[str, Path]],
    labels: list[tuple[str, Path]],
    anisotropy_spacing: tuple[float, float, float],
    identifiers: tuple[str, str, str, str],
) -> tuple[ImageSetLoader, list[str], list[str]]:
    """Read images/labels and build an ImageSetLoader with identifiers."""
    image_dict: dict[str, object] = {}
    for name, path in images:
        image_dict[name] = _image_loading(path)
    compartment_names: list[str] = []
    for name, path in labels:
        image_dict[name] = _image_loading(path)
        compartment_names.append(name)
    patient_tumor, plate, well, field_of_view = identifiers
    image_set_name = build_image_id_from_identifiers(identifiers)
    image_set_loader = ImageSetLoader.from_image_dict(
        image_dict,
        anisotropy_spacing=anisotropy_spacing,
        image_set_name=image_set_name,
        label_key_names=compartment_names,
        patient_tumor=patient_tumor,
        plate=plate,
        well=well,
        field_of_view=field_of_view,
    )
    channels = [name for name, _ in images]
    return image_set_loader, channels, compartment_names


def build_image_id_from_identifiers(
    identifiers: tuple[str, str, str, str],
) -> str:
    """Build the deterministic image set name from identifier fields."""
    patient_tumor, plate, well, field_of_view = identifiers
    return build_image_id(patient_tumor, plate, well, field_of_view)


def run(  # noqa: PLR0913, PLR0917
    images: list[tuple[str, Path]],
    labels: list[tuple[str, Path]],
    anisotropy_spacing: tuple[float, float, float],
    identifiers: tuple[str, str, str, str],
    out_dir: Path,
    feature_specs: list[dict[str, object]] | None = None,
    features_filter: list[str] | None = None,
    skip_existing: bool = False,
    force: bool = False,
) -> list[Path]:
    """Run feature extraction for one shard and write Parquet outputs.

    Parameters
    ----------
    images : list[tuple[str, Path]]
        (channel name, path) pairs from ``--image``.
    labels : list[tuple[str, Path]]
        (compartment name, path) pairs from ``--label``.
    anisotropy_spacing : tuple[float, float, float]
        (z, y, x) spacing.
    identifiers : tuple[str, str, str, str]
        (patient_tumor, plate, well, field_of_view) imaging coordinates.
    out_dir : Path
        Shard output directory.
    feature_specs : list[dict[str, object]] | None
        Explicit ``--feature`` requests; None means auto-generate.
    features_filter : list[str] | None
        ``--features`` selector restricting which feature types run.
    skip_existing : bool
        Skip a request whose output Parquet already exists.
    force : bool
        Overwrite even when the output exists (still crash-safe/atomic).

    Returns
    -------
    list[Path]
        Paths written (or skipped) per request.
    """
    channels = [name for name, _ in images]
    compartments = [name for name, _ in labels]
    requests = _resolve_requests(
        channels,
        compartments,
        feature_specs or [],
        features_filter,
    )
    if not requests:
        print("No feature requests to run", file=sys.stderr)
        return []
    # Compute target paths and partition into skipped vs. pending *before*
    # reading any image. When every requested output already exists and
    # --skip-existing is set, image I/O is avoided entirely.
    results: list[Path] = []
    pending: list[tuple[dict[str, object], Path]] = []
    for request in requests:
        compartment, channel, feature_type = _request_output_identity(request)
        target = _output_path(out_dir, compartment, channel, feature_type)
        if target.exists() and skip_existing and not force:
            print(f"skip-existing: {target}", file=sys.stderr)
            results.append(target)
        else:
            pending.append((request, target))
    if not pending:
        print(
            "all requested outputs exist; nothing to do",
            file=sys.stderr,
        )
        return results
    image_set_loader, _, _ = _build_image_set_loader(
        images,
        labels,
        anisotropy_spacing,
        identifiers,
    )
    for request, target in pending:
        results.append(_execute_request(image_set_loader, request, out_dir, target))
    return results


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="ZedProfiler",
        description="Per-shard 3D featurization for the NF1 profiling warehouse.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run feature extraction for one well/FOV shard.",
        description=(
            "Run feature extraction for one well/FOV shard. Loads one image "
            "set from explicit file paths, runs the selected featurizers, and "
            "writes one Parquet per feature table to --out-dir."
        ),
    )
    run_parser.add_argument(
        "--image",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Channel image as NAME=PATH (repeatable; >=1 required).",
    )
    run_parser.add_argument(
        "--label",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Compartment label mask as NAME=PATH (repeatable; >=1 required).",
    )
    run_parser.add_argument(
        "--anisotropy-spacing",
        nargs=3,
        required=True,
        type=float,
        metavar=("Z", "Y", "X"),
        help="Z, Y, X voxel spacing.",
    )
    run_parser.add_argument(
        "--patient-tumor",
        required=True,
        help="Patient-tumor identifier (e.g. NF0014_T1).",
    )
    run_parser.add_argument("--plate", required=True, help="Plate identifier.")
    run_parser.add_argument("--well", required=True, help="Well identifier.")
    run_parser.add_argument(
        "--fov",
        required=True,
        help="Field-of-view index or identifier.",
    )
    run_parser.add_argument(
        "--out-dir",
        required=True,
        help="Shard output directory (created if needed).",
    )
    run_parser.add_argument(
        "--features",
        default=None,
        help=(
            "Comma-separated feature types to run (selector). With no "
            "--feature flags, runs these types over the channel x compartment "
            "cross-product. With --feature flags, restricts those requests by "
            "type. Default: all applicable single-channel types (plus "
            "Colocalization when >=2 channels)."
        ),
    )
    run_parser.add_argument(
        "--feature",
        action="append",
        default=[],
        metavar="TYPE[,key=value,...]",
        help=(
            "Explicit feature request as TYPE[,key=value,...] (repeatable). "
            "Examples: 'Intensity,channel=DNA,compartment=Nuclei'; "
            "'Colocalization,channel1=DNA1,channel2=DNA2,compartment=Nuclei,"
            "fast_costes=Faster'."
        ),
    )
    run_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a feature request whose output Parquet already exists.",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite even when the output exists (writes are still atomic).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":  # pragma: no cover - subparsers required=True
        parser.error("a subcommand is required")
        return 2  # pragma: no cover - parser.error exits

    images = [_parse_name_path(token) for token in args.image]
    labels = [_parse_name_path(token) for token in args.label]
    feature_specs = [_parse_feature_spec(token) for token in args.feature]
    features_filter = (
        [f.strip() for f in args.features.split(",") if f.strip()]
        if args.features
        else None
    )
    identifiers = (
        args.patient_tumor,
        args.plate,
        args.well,
        args.fov,
    )
    run(
        images=images,
        labels=labels,
        anisotropy_spacing=tuple(args.anisotropy_spacing),
        identifiers=identifiers,
        out_dir=Path(args.out_dir),
        feature_specs=feature_specs,
        features_filter=features_filter,
        skip_existing=args.skip_existing,
        force=args.force,
    )
    return 0


def trigger() -> None:
    """Console-script entry point (matches the pyproject ``scripts`` target)."""
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    trigger()
