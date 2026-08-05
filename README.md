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

Though I have reservations about width, height, and depth being used to describe the dimensions of an image.
Different fields use different dimensions for different meanings.
We use `x` and `y` to refer to the same dimensions captured in a 2D image, and `z` to refer to the "depth" dimension in a 3D image if looking down into the image stack.
The `x`, `y`, and `z` dimensions are less description and more absolute while `depth` is relative to angle of observation.

Accepted image formats (order matters):

- Single channel: `(z, y, x)`

## Quality Gates

We lint and format code with our pre-commit configuration.
