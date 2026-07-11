<div align="center">
<img src="assets/logo.svg" alt="massively" width="400">

[![Crates.io](https://img.shields.io/crates/v/massively.svg)](https://crates.io/crates/massively)
[![API doc](https://docs.rs/massively/badge.svg)](https://docs.rs/massively)
![CI](https://github.com/akiradeveloper/massively/actions/workflows/ci.yml/badge.svg)

----

**Multi-platform GPU parallel algorithms for Rust.**

</div>

## Overview

`massively` provides Thrust-style parallel algorithms for device-resident data
on top of [CubeCL](https://github.com/tracel-ai/cubecl). The same algorithm API
can run on WGPU, CUDA, or HIP through the corresponding CubeCL runtime.

Memory movement is explicit, outputs are preallocated, and user-defined
operations are compiled into GPU kernels. Lazy transforms, permutations, and
reversed views can be consumed without first materializing an intermediate
buffer.

The public API is built around a few ideas:

- explicit host/device transfer through `Executor`
- owning device storage through `DeviceVec` and zero-copy views through
  `DeviceSlice` and `DeviceSliceMut`
- logical row values assembled with `zip2` through `zip7`
- CubeCL-backed operations such as `UnaryOp`, `PredicateOp`, and `ReductionOp`
- parallel algorithms such as `transform`, `reduce`, `inclusive_scan`, `sort`,
  `gather`, `copy_where`, and by-key variants

## Setup

The default backend is WGPU. `cubecl` is also a direct dependency because its
runtime types and `#[cubecl::cube]` macro are part of user-defined operations.
Use the same CubeCL release as the one selected by `massively`.

```sh
cargo add massively
cargo add cubecl --no-default-features --features std,stdlib,wgpu
```

For another backend, disable `massively`'s default features and enable `cuda`
or `hip`, together with the matching CubeCL feature.

## Quick Example

This example doubles a device vector and writes the result into preallocated
output storage.

```rust
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, UnaryOp, transform};

struct Double;

#[cubecl::cube]
impl UnaryOp<u32> for Double {
    type Output = u32;

    fn apply(value: u32) -> u32 {
        value * 2
    }
}

fn main() -> Result<(), massively::Error> {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[1_u32, 2, 3, 4]);
    let output = exec.alloc::<u32>(input.len());

    transform(&exec, input.slice(..), Double, output.slice_mut(..))?;

    assert_eq!(exec.to_host(&output)?, vec![2, 4, 6, 8]);
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

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
```

The same `Executor<R>` drives device allocation, transfers, synchronization,
and algorithms for that runtime. An attempt to use storage with a different
executor is rejected with `Error::ForeignExecutor`.

Algorithms are ordinary functions. Their public constraints describe only
logical iterators, outputs, and operations; kernel lowering and dispatch remain
private implementation details.

```rust
transform(&exec, input, op, output)?;
let sum = reduce(&exec, input, zero, sum_op)?;
```

Host/device movement is explicit. Algorithms take an `Executor`, so launches,
transfers, and ownership checks remain visible at the call site.

### Device Storage And Slices

`DeviceVec<R, T>` owns one contiguous device allocation. Algorithms read a
`DeviceSlice<T>` returned by `DeviceVec::slice`, and write a `DeviceSliceMut<T>`
returned by `DeviceVec::slice_mut`. Slices are zero-copy views and can be sliced
again.

### Multi-column Values

The `zip2` through `zip7` helpers combine slices or lazy iterators into one
logical row stream. For example, `zip3(a, b, c)` has item type
`Tuple3<A, B, C>`.
The matching `Tuple2` through `Tuple7` aliases and `tuple2` through `tuple7`
constructors create fixed-arity values. `flatten3` through `flatten7` destructure
them without exposing the internal binary-tree representation. These helpers
can be called directly from a user-defined CubeCL operation.

Conceptually:

```text
DeviceSlice<T>                         = MIter<Item = T>
zip2(a, b)                             = MIter<Item = Tuple2<A, B>>
zip3(a, b, c)                          = MIter<Item = Tuple3<A, B, C>>
zip2(out_a, out_b)                     = MIterMut<Item = Tuple2<A, B>>
tuple3(a, b, c)                        = Tuple3<A, B, C>
flatten3(value)                        = (A, B, C)
lazy::transform(input, op)             = fused lazy computation
lazy::permute(values, indices)         = lazy indexed view
lazy::reverse(input)                    = lazy reversed view
```

General input and output items support up to seven columns. By-key algorithms
support up to three key columns and seven value columns. Output iterators are
always created before an algorithm runs. If the ordered leaf types match,
`WriteFrom` lets an algorithm write a compatible tuple value without the user
manually matching its internal association.

### Lazy Iterators

`lazy::constant`, `lazy::counting`, `lazy::transform`, `lazy::permute`, and
`lazy::reverse` produce `MIter` values without allocating result storage. Their
expressions are evaluated by the consuming algorithm, allowing operations to be
composed while keeping intermediate values off device memory.

### Operations

User-defined operations are CubeCL cube traits. `massively` intentionally keeps
that connection visible: CubeCL is the kernel DSL, while `massively` supplies
the algorithm and iterator layer.

- `UnaryOp<Input>` maps one item to another.
- `PredicateOp<Item>` tests one item.
- `BinaryPredicateOp<Item>` compares two items.
- `ReductionOp<Item>` combines two items.

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

### Correctness Examples

Every public algorithm has a runnable, single-column example in the
[API documentation](https://docs.rs/massively). The integration tests under
`crates/massively/tests` show lazy and multi-column usage. The oracle tests
compare each public function against the CPU AoS reference and cover the full
transform input/output arity matrix.

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

See the [recipe list](crates/recipes/README.md) for all runnable programs.
