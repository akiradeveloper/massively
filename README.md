<div align="center">
<img src="assets/logo.svg" alt="massively" width="400">

[![Crates.io](https://img.shields.io/crates/v/massively.svg)](https://crates.io/crates/massively)
[![API doc](https://docs.rs/massively/badge.svg)](https://docs.rs/massively)
![CI](https://github.com/akiradeveloper/massively/actions/workflows/ci.yml/badge.svg)

----

**Multi-platform GPU parallel and graph algorithms for Rust.**

</div>

## Overview

`massively` provides Thrust-style parallel algorithms for device-resident data
on top of [CubeCL](https://github.com/tracel-ai/cubecl). The same algorithm API
can run on WGPU, CUDA, or HIP through the corresponding CubeCL runtime.

The algorithms are organized into three complementary families:

- vector algorithms for transform, scan, reduction, sorting, selection, and
  indexed movement
- segment algorithms that apply map, scan, reduction, ordering, and selection
  independently to offset-delimited regions
- graph algorithms expressed through edge computation, aggregation, state
  updates, frontier relaxation, and batched adjacency operations

Memory movement is explicit, outputs are preallocated, and user-defined
operations are compiled into GPU kernels. Lazy transforms, permutations, and
reversed views can be consumed without first materializing an intermediate
buffer.

The public API is built around a few ideas:

- explicit host/device transfer through `Executor`
- owning device storage through `DeviceVec` and zero-copy views through
  `DeviceSlice` and `DeviceSliceMut`
- logical row values assembled with `zip2` through `zip7`
- CubeCL-backed operations under `massively::op`, such as `UnaryOp`,
  `PredicateOp`, and `ReductionOp`
- parallel algorithms under `massively::vector`, such as `transform`, `reduce`,
  `inclusive_scan`, `sort`, `gather`, `copy_where`, and by-key variants
- CSR graph traversal through source, destination, and edge expressions followed
  by explicit emit, reduction, or state-update terminals

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

This example doubles a device vector and returns owned device storage.

```rust
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op::UnaryOp, vector::transform};

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
    let output = transform(&exec, input.slice(..), Double)?;

    assert_eq!(exec.to_host(&output)?, vec![2, 4, 6, 8]);
    Ok(())
}
```

## Graph Algorithms

`massively::graph` is a graph programming layer built from the same device
storage, lazy expressions, multi-column values, and parallel primitives as the
vector API. It is part of the main crate rather than a separate framework.

Graph programs describe the meaning of an edge computation instead of choosing
a low-level expansion strategy. A traversal selects outgoing CSR edges from a
frontier, edge-context expressions select the data to read, and a terminal
defines where results go.

| Edge expression | Meaning |
| --- | --- |
| `source(values)` | Read a value at each edge's source vertex |
| `destination(values)` | Read a value at each edge's destination vertex |
| `edge(values)` | Read a value at the edge's CSR position |
| `source_id()` / `destination_id()` / `edge_id()` | Read topology identities |

Expressions compose with the ordinary `zip2` through `zip7` helpers. Terminals
then give the edge stream an explicit interpretation:

- `emit` writes one result per traversed edge.
- `reduce_by_source` reduces each selected CSR row.
- `reduce_by_destination` resolves colliding destination proposals.
- `update_by_destination` combines proposals into existing vertex state.
- `relax_min_by_destination` updates minimum state and produces the next
  frontier from vertices that actually changed.
- `intersect_count` performs batched intersections of sorted adjacency rows.

For example, CSR sparse matrix-vector multiplication is the edge expression
`edge(weight) * destination(vector)`, reduced by source row:

```rust
use cubecl::prelude::*;
use massively::{Executor, op::ReductionOp, op::UnaryOp, graph, zip2};

struct Multiply;

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for Multiply {
    type Output = f32;

    fn apply(input: (f32, f32)) -> f32 {
        input.0 * input.1
    }
}

struct Sum;

#[cubecl::cube]
impl ReductionOp<f32> for Sum {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

// destinations and offsets form a CSR topology; frontier contains its rows.
let output = exec.alloc::<f32>(frontier.len());
graph::traverse(
    &exec,
    graph::Csr::new(destinations.slice(..), offsets.slice(..)),
    frontier.slice(..),
)?
.map(
    zip2(
        graph::edge(weights.slice(..)),
        graph::destination(vector.slice(..)),
    ),
    Multiply,
)
.reduce_by_source(&exec, 0.0, Sum, output.slice_mut(..))?;
```

Traversal planning and temporary expansion data are private. The current
lowering composes GPU gather, transform, segmented reduction, sorting, and
scatter-reduce primitives. Because programs expose semantics rather than that
lowering, future implementations can fuse edge expressions into terminals and
choose atomic, sort-reduce, hierarchical, push/pull, or degree-aware
intersection strategies without changing graph programs.

The `graph-algorithms` crate includes breadth-first search, single-source shortest paths,
PageRank, personalized PageRank, HITS, graph coloring, k-core decomposition,
minimum spanning tree, Boolean SpGEMM, SpMV, triangle counting, betweenness
centrality, Forman–Ricci curvature, and graph-based geolocation. See
[`crates/graph-algorithms`](crates/graph-algorithms) for the tested compositions,
independent CPU references, and generated graph property tests.

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

Algorithms are ordinary functions. Their public constraints describe logical
iterators and operations; kernel lowering and dispatch remain private
implementation details. Algorithms that naturally produce a new sequence return
an `MVec`. Algorithms whose semantics require an existing destination, such as
`scatter`, take that destination as an argument.

```rust
let output = transform(&exec, input, op)?;
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

The same principle applies to graph traversal: the public model names topology,
edge-context values, and terminal semantics, while frontier expansion and
conflict resolution remain lowering decisions. This separation is what lets
graph algorithms share improvements to Massively's underlying primitives.

## Further Reading

### Correctness Examples

Every public algorithm has a runnable, single-column example in the
[API documentation](https://docs.rs/massively). Integration tests are grouped
under `crates/massively/tests/vector` and `crates/massively/tests/seg`. Their
oracle tests compare public functions against CPU AoS references and cover the
full transform input/output arity matrix. Complete graph algorithm oracles live
under `crates/graph-algorithms/tests` and compare every algorithm with
independent CPU implementations on generated CSR graphs. Tests in `massively`
itself cover the graph traversal primitives.

### Graph Algorithms

Complete algorithms written with the graph traversal algebra, together with
generated property tests and end-to-end benchmarks, live in the separate
`graph-algorithms` crate.

```sh
cargo nextest run -p graph-algorithms --test oracle
cargo bench -p graph-algorithms --bench algorithms
```
