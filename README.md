# Standalone cudarc D3Q19 comparator

This crate implements the matched FP32 D3Q19 Taylor-Green workload directly
through cudarc. It uses NVRTC for a compile-time-specialized fused pull-stream
and BGK collision kernel, cudarc for allocation/launch/synchronization, and NVML
for timestamped NVIDIA board-power telemetry.

The GPU layout is SoA and the Tenstorrent layout is tiled/paged. Both preserve
the same frozen directions, weights, initialization, periodic pull rule, BGK
equations, FP32 precision, and conservation-form rest-population residual.
Mathematical and numerical equivalence - not source-level kernel identity - is
the comparison boundary.

The campaign driver provides:

- independent host-reference parity at one and ten steps;
- symmetric 128/256/512-thread tuning outside admitted windows;
- a 300-second idle board-power trace;
- five synchronized measurement windows of at least 60 seconds;
- exact-boundary trapezoidal NVML integration;
- hardware, driver, clock, temperature, utilization, cost, and teardown evidence;
- a bounded Nsight Compute attempt excluded from energy statistics.

Build the Linux upload binary:

```bash
CUDARC_CUDA_VERSION=12080 cargo zigbuild --release \
  --target x86_64-unknown-linux-gnu
```

The direct Vast driver is dry-run by default and requires `--execute` for a
billable lease. It provisions exactly one offer and never retries automatically.
