<div align="center">
<img src="assets/logo.svg" alt="massively" width="400">

[![Crates.io](https://img.shields.io/crates/v/massively.svg)](https://crates.io/crates/massively)
[![API doc](https://docs.rs/massively/badge.svg)](https://docs.rs/massively)
![CI](https://github.com/akiradeveloper/massively/actions/workflows/ci.yml/badge.svg)

----

**Multi-platform GPU parallel algorithms for Rust.**

</div>

## Overview

`massively` is a Thrust-inspired parallel algorithm layer on top of
[CubeCL](https://github.com/tracel-ai/cubecl).

The goal is to provide reusable GPU algorithms with explicit device memory,
typed logical values, and CubeCL operations that stay close to the generated
kernel code.

The public API is built around a few ideas:

- explicit host/device transfer through `Executor`
- contiguous device columns through `DeviceVec`, `DeviceSlice`, and
  `DeviceSliceMut`
- logical row values assembled with `Zip1` through `Zip7`
- CubeCL-backed operations such as `UnaryOp`, `PredicateOp`, and `ReductionOp`
- parallel algorithms such as `transform`, `reduce`, `scan`, `sort`,
  `gather`, `copy_where`, and by-key variants

## Quick Example

This `transform` reads three input columns as `(u32, u32, u32)` and writes two
output columns as `(u32, u32)`.

```rust
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::op::UnaryOp;
use massively::{Executor, Zip2, Zip3, transform};

struct SumProduct3;

#[cubecl::cube]
impl<R> UnaryOp<R, (u32, u32, u32)> for SumProduct3
where
    R: Runtime,
{
    type Output = (u32, u32);

    fn apply(input: (u32, u32, u32)) -> (u32, u32) {
        let sum = input.0 + input.1 + input.2;
        let product = input.0 * input.1 * input.2;
        (sum, product)
    }
}

fn main() -> Result<(), massively::Error> {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let a = exec.to_device(&[1_u32, 2, 3])?;
    let b = exec.to_device(&[4_u32, 5, 6])?;
    let c = exec.to_device(&[7_u32, 8, 9])?;
    let sum_out = exec.to_device(&[0_u32; 3])?;
    let product_out = exec.to_device(&[0_u32; 3])?;

    // Zip3 is read as (u32, u32, u32), and Zip2 is written as (u32, u32).
    transform(
        &exec,
        Zip3(a.slice(..), b.slice(..), c.slice(..)),
        SumProduct3,
        Zip2(sum_out.slice_mut(..), product_out.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&sum_out)?, vec![12, 15, 18]);
    assert_eq!(exec.to_host(&product_out)?, vec![28, 80, 162]);
    Ok(())
}
```

## Core Model

### Runtime And Memory

`massively` uses CubeCL runtimes directly. Pick a CubeCL runtime type and pass
one of its devices to `Executor::new`.

```rust
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::Executor;

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
```

The same `Executor<R>` type drives device allocation, transfers, algorithms,
and random generation for that runtime.

Host/device movement is explicit. Device storage does not own the execution
context; algorithms take an `Executor` so launches, transfers, and ownership
checks stay visible.

### Zip Values

Each device column is stored in a contiguous `DeviceVec`. Algorithms do not
take Rust slices directly; they take device slices and Zip views.

`ZipN` is the boundary between physical columns and logical values. For
example, `Zip3(account_id, amount, risk_score)` is read by algorithms as an
item of type `(u32, f32, u32)`.

Conceptually:

```text
Scalar   = CubePrimitive + CubeElement
Column   = DeviceVec<T> / DeviceSlice<T> / DeviceSliceMut<T>
ZipN     = logical row view over N device columns
MItem    = value seen by CubeCL operations
MIter    = read-only algorithm input
MIterMut = explicit algorithm output
```

Today the built-in Zip arity is `1..=7`. Nested logical values such as
`(a, (b, c, d))` are part of the intended value model, but the current storage
and dispatch implementations are still centered on flat `Zip1..Zip7` shapes.

### Operations

User-defined operations are CubeCL cube traits. `massively` intentionally keeps
that connection visible: CubeCL is the kernel DSL, while `massively` supplies
the algorithm layer and Zip data model.

## Design Notes

The implementation favors reusable primitives such as scan, selection,
permutation, and segmented control over one-off algorithm kernels. This keeps
the API surface compact and lets improvements to core primitives benefit many
algorithms.

Multi-column support is a first-class requirement. The code avoids
single-column-only shortcuts and avoids arity explosion by separating control
generation from payload movement where possible, especially in by-key
algorithms.

## Further Reading

### API Examples

API examples live under `crates/massively/examples`.

- `examples/runtime`: allocation, transfer, initialization, and device copy
- `examples/util`: utility examples such as random generation
- `examples/algorithm`: one small runnable example per algorithm API

Run one with Cargo:

```sh
cargo run -p massively --example transform
cargo run -p massively --example util-random
cargo run -p massively --example runtime-counting
```

### Recipes

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

### Real-world Examples

These repositories show `massively` used outside the small example programs in
this workspace. They are useful references for project structure, runtime
setup, and composing multiple algorithms into an application.

- [bph-gpu](https://github.com/akiradeveloper/bph-gpu): an application-sized
  GPU project that uses massively-style algorithm composition.
- [pi-monte-carlo](https://github.com/akiradeveloper/pi-monte-carlo):
  Monte Carlo pi estimation using GPU-side random values and reductions.
