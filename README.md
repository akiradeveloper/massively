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

- vector algorithms for map, scan, reduction, sorting, selection, and
  indexed movement
- segment algorithms that apply map, scan, reduction, ordering, and selection
  independently to offset-delimited regions
- graph algorithms expressed with Traversal Algebra through edge computation,
  aggregation, state updates, frontier relaxation, and batched adjacency
  operations

Memory movement is explicit, outputs are preallocated, and user-defined
operations are compiled into GPU kernels. Lazy maps, permutations, and
reversed views can be consumed without first materializing an intermediate
buffer.

The public API is built around a few ideas:

- explicit host/device transfer through `Executor`
- owning device storage through `DeviceVec` and zero-copy views through
  `DeviceSlice` and `DeviceSliceMut`
- logical row values assembled with `zip2` through `zip7`
- CubeCL-backed operations under `massively::op`, such as `UnaryOp`,
  `ExpandOp`, `PredicateOp`, and `ReductionOp`
- parallel algorithms under `massively::vector`, such as `map`, `reduce`,
  `flat_map`, `inclusive_scan`, `sort`, `gather`, `copy_where`, and by-key
  variants
- CSR graph traversal through source, destination, and edge expressions followed
  by explicit emit, reduction, or state-update terminals

## Setup

`massively` is runtime-agnostic: algorithms are generic over
`R: cubecl::Runtime`, and the application selects a backend through its direct
`cubecl` dependency. `cubecl` is also needed because its runtime types and
`#[cubecl::cube]` macro are part of user-defined operations. Use the same CubeCL
release as the one selected by `massively`.

```sh
cargo add massively
cargo add cubecl --no-default-features --features std,stdlib,wgpu
```

For another backend, enable `cuda` or `hip` on `cubecl` instead of `wgpu` and
construct `Executor` with the corresponding CubeCL runtime.

## Quick Example

This example doubles a device vector and returns owned device storage.

```rust
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op::UnaryOp, vector::map};

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
    let output = map(&exec, input.slice(..), Double)?;

    assert_eq!(exec.to_host(&output)?, vec![2, 4, 6, 8]);
    Ok(())
}
```

## Graph Algorithms

`massively::graph` introduces **Traversal Algebra**, a compositional way to
describe frontier-based graph algorithms. A program says which edges are
active, which source, destination, or edge values to read, how to transform
them, and how to emit or combine the results. It does not bake a particular
GPU expansion or conflict-resolution strategy into the algorithm.

The basic flow is:

```text
frontier -> traverse -> pull values -> map -> push/reduce -> update -> next frontier
```

Traversal Algebra has a machine-checked mathematical foundation. For the
precisely defined finite frontier/BSP fragment, Lean proves that the algebra
and its reference BSP model can represent one another without changing program
results. A second checked lowering carries typed BSP programs through Core TA
to a denotational basis of the exported graph terminals and vector map
and filtering operations. The proof also gives a separate Rust-shaped grammar
for the six edge-expression leaves, one map, and the public terminal contracts.
It proves operation contracts, not that arbitrary Rust/CubeCL code or hardware
refines the Lean model. The implementation is checked separately against an
independent sequential CPU oracle with generated property tests.

The non-specialist summary, exact proof boundary, Lean development, and oracle
are collected in the
[Traversal Algebra artifact](verification/traversal-algebra/).

The graph layer uses the same device storage, lazy expressions, multi-column
values, and parallel primitives as the vector API. It is part of the main crate
rather than a separate framework.

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
graph::traverse(
    &exec,
    graph::Csr::new(destinations.slice(..), offsets.slice(..)),
    frontier.slice(..),
    destinations.len(),
)?
.map(
    zip2(
        graph::edge(weights.slice(..)),
        graph::destination(vector.slice(..)),
    ),
    Multiply,
)
.reduce_by_source(&exec, 0.0, Sum)?;
```

Traversal planning and temporary expansion data are private. The current
lowering composes GPU gather, map, segmented reduction, sorting, and
scatter-reduce primitives. Because programs expose semantics rather than that
lowering, future implementations can fuse edge expressions into terminals and
choose atomic, sort-reduce, hierarchical, push/pull, or degree-aware
intersection strategies without changing graph programs.

The `graph-algorithms` crate implements all 12 applications that Gunrock's
published table marks as supported by current Gunrock, plus all four priority
future ports: connected components, Louvain modularity,
random walk, and subgraph matching. It also includes personalized PageRank,
Boolean SpGEMM, and Forman–Ricci curvature. See the
[`Gunrock comparison table`](verification/traversal-algebra/graph-algorithms#gunrock-comparison)
for the exact scope, tested compositions, independent CPU references, and
generated graph property tests.

## Core Completeness Artifact

The [Massively Core Lean artifact](verification/massively/) treats a conventional
finite-control priority-CRCW PRAM as an external expressiveness benchmark and
Massively Core as a separate bulk-synchronous target machine. Lean checks the
instruction-machine normalization, compilation to pull/map/proposal
compaction/deterministic reduction/controlled scatter, and preservation of
every finite execution. The artifact records the precise current model, its
symbolic schedule costs, and the Rust/CubeCL refinement work that is not yet
part of this theorem.

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
let output = map(&exec, input, op)?;
let sum = reduce(&exec, input, zero, sum_op)?;
```

Host/device movement is explicit. Algorithms take an `Executor`, so launches,
transfers, and ownership checks remain visible at the call site.

### Scalar And Length Boundaries

The public API returns scalar results as ordinary host-visible values:
`reduce` returns its item, predicates return `bool`, and indices and lengths
use `MIndex` (an alias of `u32`). Algorithms with data-dependent output
lengths, such as `copy_where` and `reduce_by_key`, return storage whose exact
length is already available through `len()`.

These calls are synchronous at the return boundary when a GPU-produced scalar
or exact output length must be observed. Length-preserving algorithms and
algorithms writing into caller-provided fixed storage do not need that scalar
readback. Internally, device scalars and logical extents are propagated between
GPU stages without intermediate CPU transfers, so a public operation performs
at most the synchronization required by its return contract. `Executor::to_host`
remains the explicit boundary for copying result data.

CubeCL booleans are register values rather than storage elements. Device-resident
boolean summaries therefore use `seg::BoolVec`: its backing flags are private,
while its iterator item and `Executor::to_host` result are ordinary `bool` values.

### Device Storage And Slices

`DeviceVec<R, T>` owns one contiguous device allocation. Algorithms read a
`DeviceSlice<T>` returned by `DeviceVec::slice`, and write a `DeviceSliceMut<T>`
returned by `DeviceVec::slice_mut`. Slices are zero-copy views and can be sliced
again. Slice bounds use `MIndex`.

### Multi-column Values

The `zip2` through `zip12` helpers combine slices or lazy iterators into one
logical row stream. For example, `zip3(a, b, c)` has item type
`(A, B, C)`. `zip` is associative at the schema level: both
`zip2(zip2(a, b), c)` and
`zip2(a, zip2(b, c))` expose the same flat item type. The internal storage tree
is not part of the public contract.

For an owned multi-column `MVec`, `MStorage::into_columns` returns a native flat
tuple of owning `DeviceVec` columns without copying or reallocating device data.
Tuple types, literals, and destructuring use Rust's native tuple syntax directly,
including inside a user-defined CubeCL operation.

Conceptually:

```text
DeviceSlice<T>                         = MIter<Item = T>
zip2(a, b)                             = MIter<Item = (A, B)>
zip3(a, b, c)                          = MIter<Item = (A, B, C)>
zip2(zip2(a, b), c)                    = MIter<Item = (A, B, C)>
zip2(a, zip2(b, c))                    = MIter<Item = (A, B, C)>
zip2(out_a, out_b)                     = MIterMut<Item = (A, B)>
MStorage::into_columns(output3)        = (DeviceVec<A>, DeviceVec<B>, DeviceVec<C>)
lazy::map(input, op)                   = fused lazy computation
lazy::permute(values, indices)         = lazy indexed view
lazy::reverse(input)                    = lazy reversed view
```

Input and output items support up to twelve columns. Keys passed to by-key
algorithms are limited to three columns; their value items retain the full
twelve-column limit. Output iterators are always created before an algorithm
runs. An operation that intentionally changes a row schema expresses that
conversion explicitly with `map`.

### Lazy Iterators

`lazy::constant`, `lazy::counting`, `lazy::map`, `lazy::permute`, and
`lazy::reverse` produce `MIter` values without allocating result storage. Their
expressions are evaluated by the consuming algorithm, allowing operations to be
composed while keeping intermediate values off device memory.

### Operations

User-defined operations are CubeCL cube traits. `massively` intentionally keeps
that connection visible: CubeCL is the kernel DSL, while `massively` supplies
the algorithm and iterator layer.

- `UnaryOp<Input>` maps one item to another.
- `ExpandOp<Input>` expands one item into zero or more items.
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
under `massively/tests/vector` and `massively/tests/seg`.
Their
oracle tests compare public functions against CPU AoS references and cover the
full map input/output arity matrix. Complete graph algorithm oracles live
under `verification/traversal-algebra/graph-algorithms/tests` and compare every algorithm with
independent CPU implementations on generated CSR graphs. Tests in `massively`
itself cover the graph traversal primitives.

### Graph Algorithms

Complete algorithms written with Traversal Algebra, together with generated
property tests and end-to-end benchmarks, live in the `graph-algorithms` crate
inside the Traversal Algebra artifact.

```sh
cargo nextest run -p graph-algorithms --test oracle
cargo bench -p graph-algorithms --bench algorithms
```
