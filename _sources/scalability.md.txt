# ZEDprofiler scalability

ZEDprofiler is designed to be scalable and efficient by use of its felixible API.
This means that ZEDprofiler can be parallelized and distributed across multiple machines, allowing it to handle large datasets and complex profiling tasks.
How ZEDprofiler can be scaled depends on the specific use case and requirements of the profiling task.
Here are some ways ZEDprofiler can be scaled:

1. Serially: ZEDprofiler can be run on a single machine, core, or node, processing the profiling tasks sequentially.
   This is suitable for small datasets or when the profiling tasks are not computationally intensive.
1. Parallel (embarrassingly parallel): ZEDprofiler can be parallelized across multiple cores or nodes, allowing it to process multiple profiling tasks simultaneously.
   This is suitable for larger datasets or when the profiling tasks are computationally intensive.

How the parallelization is implemented depends on the specific use case and requirements of the profiling task.
This depends on the compute available, size of the dataset, and the number of channels and compartments being profiled.

Let us run through a thought experiment to see how ZEDprofiler can be scaled.

An image-set is a single field of view (FOV) of a sample, which is imaged across multiple channels and compartments.
A dataset is a collection of image-sets, which can be stored in a single directory or across multiple directories.

We can run maximally parallel by running each API call for each channel and compartment in parallel across all FOVs, wells, and plates.
We refer to this as scaling horizontal.

Alternatively, we can run each API call for each channel and compartment sequentially across all FOVs, wells, and plates.
We refer to this as scaling vertical (Not truly what vertical means, but it works for visualization purposes).

To compute the most efficient way to run ZEDprofiler, we need to consider the compute available, size of the dataset, and the number of channels and compartments being profiled.
We also need to consider the time it takes to load a single channel and compartment, which is the same regardless of how we scale ZEDprofiler.
For large data where it takes a long time to load a single channel and compartment, it may be more efficient to run ZEDprofiler less horizontally and more vertically, as this allows us to load the data once and run multiple API calls on it before moving on to the next channel and compartment.
For smaller data where it takes a short time to load a single channel and compartment, it may be more efficient to run ZEDprofiler more horizontally, as this allows us to process multiple profiling tasks simultaneously and take advantage of the parallelization.

For those who might have access to very few cores or nodes, it may be more efficient to run ZEDprofiler more vertically, as this allows us to process multiple profiling tasks sequentially and take advantage of the compute available when we get it.
