<div align="center">
<img src="assets/logo.svg" alt="massively" width="400">

[![Crates.io](https://img.shields.io/crates/v/massively.svg)](https://crates.io/crates/massively)
[![API doc](https://docs.rs/massively/badge.svg)](https://docs.rs/massively)
![CI](https://github.com/akiradeveloper/massively/actions/workflows/ci.yml/badge.svg)

----

**Multi-platform GPU parallel algorithms for Rust.**

</div>

## Concept

`massively` is a Thrust-inspired parallel algorithm layer on top of
[CubeCL](https://github.com/tracel-ai/cubecl).

The library is organized around three public layers.

- `runtime` prepares a CubeCL `Runtime` from a `Runtime::Device`, transfers
  data, allocates device memory, and exposes `Executor`, `DeviceVec`,
  `DeviceSlice`, and `DeviceSliceMut`.
- `algorithm` provides Structure-of-Arrays inputs, massively item/vector
  traits, CubeCL-backed operation traits, and algorithms such as `transform`,
  `reduce`, `scan`, `sort`, and `gather`.
- `random` generates GPU-side pseudo-random columns for uniform integer and
  approximate normal floating-point distributions.

Host/device movement is explicit. Device storage does not own the execution
context; algorithms take an `Executor` so launches, transfers, and ownership
checks stay visible.

## Data Model

GPU data is represented as Structure of Arrays. Each column is stored in a
contiguous `DeviceVec`, and algorithms operate on logical rows assembled with
`SoA1`, `SoA2`, and `SoA3`.

Conceptually:

```text
Scalar  = CubePrimitive + CubeElement
MItem   = tuple of Scalar values
MIter   = SoA view over DeviceSlice columns
MVec    = owned SoA device output
```

## Operations

User-defined operations are CubeCL cube traits. `massively` intentionally keeps
that connection visible: CubeCL is the kernel DSL, while `massively` supplies
the algorithm layer and SoA data model.

## Runtime Setup

`massively` uses CubeCL runtimes directly. Pick a CubeCL runtime type and pass
one of its devices to `Executor::new`.

```rust
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::Executor;

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
```

The same `Executor<R>` type drives device allocation, transfers, algorithms,
and random generation for that runtime.

## Examples

API examples live under `crates/massively/examples`.

- `examples/runtime`: allocation, transfer, initialization, and device copy
- `examples/algorithm`: one small runnable example per algorithm API

Run one with Cargo:

```sh
cargo run -p massively --example transform
cargo run -p massively --example runtime-random
cargo run -p massively --example runtime-tabulate
```

## Recipes

Runnable recipes live under `crates/recipes`. They are small, LeetCode-style
programs that combine the algorithm APIs into practical GPU data-processing
tasks. Each recipe defines a runtime-agnostic `solve` function and uses `main`
only for a compact sample case with assertions.

Run one with Cargo:

```sh
cargo run -p recipes --bin monte_carlo_pi
cargo run -p recipes --bin merge_ranked_feeds
```

See `crates/recipes/README.md` for the full recipe list.
