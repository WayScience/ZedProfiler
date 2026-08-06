# ZEDprofiler [![Documentation](https://img.shields.io/badge/documentation-available-brightgreen)](https://zedprofiler.readthedocs.io/en/latest/) ![License](https://img.shields.io/badge/license-BSD%203--Clause-blue)[![Coverage](https://codecov.io/gh/WayScience/ZedProfiler/branch/main/graph/badge.svg)](https://codecov.io/gh/WayScience/ZedProfiler)

<img height="100" src="https://github.com/WayScience/ZedProfiler/raw/main/logo/with-text-for-light-bg.png" />

CPU-first 3D image feature extraction toolkit for high-content and high-throughput image-based profiling.

This repository is used for image-based feature extraction of objects in 3D microscopy images.
In this use case we extract features from single cells in 3D volumetric microscopy images.
We developed ZEDprofiler to be used on high-content and high-throughput microscopy images, which are often large in size and require efficient processing.
ZEDprofiler is extensible to any fluorescence microscopy image modality, and is designed to be modular.

## Install environment

```bash
uv sync --group dev --group docs --group notebooks
```

## Data Contract

Where:

- `x` is the width of the image in pixels
- `y` is the height of the image in pixels
- `z` is the depth of the image in pixels

Different fields use different dimensions for different meanings.
We use `x` and `y` to refer to the same dimensions captured in a 2D image, and `z` to refer to the "depth" dimension in a 3D image if looking down into the image stack.
The `x`, `y`, and `z` dimensions are less description and more absolute while `depth` is relative to angle of observation.

Accepted image formats (order matters):

- Single channel: `(z, y, x)`

## Command-line interface

`ZedProfiler run` extracts features for a single well/field-of-view (FOV) shard: it loads one image set from explicit file paths, runs a selected subset of featurizers, and writes one Parquet per feature table to an output directory. It is the command a workflow manager (for example Nextflow via SLURM `sbatch`) dispatches once per shard.

After `uv sync` (or `pip install .`) the `ZedProfiler` console script is available; from a checkout you can also use `uv run ZedProfiler run ...`. Run `ZedProfiler run --help` for the authoritative, up-to-date list of arguments.

```bash
ZedProfiler run \
  --image=DNA=/path/to/channel1.tif \
  --label=Nuclei=/path/to/nuclei_mask.tiff \
  --anisotropy-spacing 1.0 1.0 1.0 \
  --patient-tumor NF0014_T1 \
  --plate PLATE01 \
  --well A1 \
  --fov 1 \
  --out-dir ./shard_output \
  --features Intensity
```

### Arguments

| Argument                         | Required              | Description                                                                                                                                                                                                                                                                                   |
| -------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--image=NAME=PATH`              | yes (>=1, repeatable) | A channel image as `NAME=PATH`. Repeat for multi-channel shards.                                                                                                                                                                                                                              |
| `--label=NAME=PATH`              | yes (>=1, repeatable) | A compartment label mask as `NAME=PATH`. Repeat for multiple compartments.                                                                                                                                                                                                                    |
| `--anisotropy-spacing Z Y X`     | yes                   | Z, Y, X voxel spacing (three floats).                                                                                                                                                                                                                                                         |
| `--patient-tumor`                | yes                   | Patient-tumor identifier (e.g. `NF0014_T1`).                                                                                                                                                                                                                                                  |
| `--plate`                        | yes                   | Plate identifier.                                                                                                                                                                                                                                                                             |
| `--well`                         | yes                   | Well identifier (e.g. `A1`).                                                                                                                                                                                                                                                                  |
| `--fov`                          | yes                   | Field-of-view index or identifier.                                                                                                                                                                                                                                                            |
| `--out-dir`                      | yes                   | Shard output directory (created if needed).                                                                                                                                                                                                                                                   |
| `--features`                     | no                    | Comma-separated feature types to run (selector). With no `--feature` flags, runs these types over the channel x compartment cross-product; with `--feature` flags, restricts those requests by type. Default: all single-channel types, plus `Colocalization` when >=2 channels are declared. |
| `--feature=TYPE[,key=value,...]` | no (repeatable)       | An explicit feature request, e.g. `Intensity,channel=DNA,compartment=Nuclei` or `Colocalization,channel1=DNA1,channel2=DNA2,compartment=Nuclei,fast_costes=Faster`.                                                                                                                           |
| `--skip-existing`                | no                    | Skip a feature request whose output Parquet already exists.                                                                                                                                                                                                                                   |
| `--force`                        | no                    | Overwrite even when the output exists (writes are still atomic).                                                                                                                                                                                                                              |

### Feature types

`VolumeSizeShape`, `Intensity`, `Neighbors`, `Texture`, and `Granularity` are single-channel features run per channel x compartment. `Colocalization` is a two-channel feature run per ordered channel pair x compartment.

### Outputs

Each request writes `{compartment}_{channel}_{feature_type}_cpu_features.parquet` into `--out-dir`. Every table carries `Metadata_Imaging_ImageID` (deterministically built from the patient-tumor, plate, well, and FOV coordinates) and `Metadata_Experiment_ImageSet` so downstream tables can rejoin. Writes are atomic (temp file + replace), so a crashed shard never leaves a partial file that `--skip-existing` would mistake for a complete one.

### Two-channel colocalization example

```bash
ZedProfiler run \
  --image=DNA1=/path/to/channel1.tif \
  --image=DNA2=/path/to/channel2.tif \
  --label=Nuclei=/path/to/nuclei_mask.tiff \
  --anisotropy-spacing 1.0 1.0 1.0 \
  --patient-tumor NF0014_T1 --plate PLATE01 --well A1 --fov 1 \
  --out-dir ./shard_output \
  --feature=Colocalization,channel1=DNA1,channel2=DNA2,compartment=Nuclei,fast_costes=Faster
```

## Quality Gates

We lint and format code with our pre-commit configuration.
