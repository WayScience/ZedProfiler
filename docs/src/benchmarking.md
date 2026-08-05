# Benchmarking and Accuracy Locks

ZEDprofiler performance work should be treated as a score-improvement loop:
make one optimization, keep the same feature outputs, then compare the scorecard.

The test suite now has two layers for this.

## Accuracy locks

Accuracy locks run with normal pytest:

```bash
uv run pytest tests/test_benchmark_contracts.py
```

These tests run deterministic synthetic image-set cases across the major feature
families and compare each output dataframe to a current-result fingerprint. If an
optimization changes a feature value, column name, row identity, or missing-value
pattern, the fingerprint changes and the test fails.

Only update these fingerprints when the project intentionally changes the
scientific result contract.

## Benchmark scorecard

The benchmark scorecard is opt-in:

```bash
uv run pytest tests/test_benchmark_contracts.py --run-benchmarks -s
```

or:

```bash
ZEDPROFILER_RUN_BENCHMARKS=1 uv run pytest tests/test_benchmark_contracts.py -s
```

The scorecard prints elapsed seconds, row count, column count, and the same
result fingerprint for each feature family. It includes the small accuracy-lock
level, a larger many-object synthetic scaling level, and a representative
real-world data level from checked-in tutorial image data. Timing is
intentionally not a hard gate yet because local machines and shared compute
environments vary. Use the scorecard to compare before and after runs on the
same environment.

## Game loop

Use this loop for each performance pass:

1. Run the accuracy locks.
2. Run the benchmark scorecard and keep the printed output.
3. Optimize one hotspot.
4. Re-run the accuracy locks.
5. Re-run the benchmark scorecard in the same environment.
6. Keep the optimization only when fingerprints stay fixed and scorecard timing
   improves enough to matter.

For larger-scale runs, pair this with workflow-level characterization. Local
scorecards explain feature-kernel cost, while workflow traces explain scheduling,
batching, queue, and filesystem behavior.
