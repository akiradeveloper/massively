<div align="center">
<img src="assets/logo.svg" alt="massively" width="400">

[![Crates.io](https://img.shields.io/crates/v/massively.svg)](https://crates.io/crates/massively)
[![API doc](https://docs.rs/massively/badge.svg)](https://docs.rs/massively)
![CI](https://github.com/akiradeveloper/massively/actions/workflows/ci.yml/badge.svg)

---- 

**Multi-platform GPU parallel algorithms for Rust.**

</div>

## TL;DR

`massively` is a GPU parallel algorithm library for Rust.
It uses [CubeCL](https://github.com/tracel-ai/cubecl) as the backend layer so the same API can target multiple runtimes.

## Motivation

GPGPU programming is powerful, but it is still difficult to write, tune, and
maintain. Even common data-parallel operations require careful kernel code,
explicit memory movement, backend-specific knowledge, and a clear distinction
between host data and device-resident data.

Portability matters just as much. GPU code should not force an application to
commit to one vendor runtime or maintain separate implementations for CUDA, HIP,
and WGPU.

`massively` aims to provide parallel algorithms for Rust on top of
CubeCL. It keeps memory movement explicit while letting client code express
parallel operations as ordinary Rust API calls over GPU-resident data.

## Data Model

`massively` keeps host data and device-resident data separate.

`DeviceVec<T>` is an owned column on the GPU. Moving data into a `DeviceVec`
with `policy.to_device(...)` is an explicit host-to-device transfer, and reading
it back with `to_vec()` is an explicit device-to-host transfer.

Multi-column data uses Structure of Arrays rather than Array of Structures.
For GPU algorithms this keeps each field in its own contiguous device buffer,
which improves coalesced memory access and lets algorithms reuse or move columns
independently.

- `DeviceVec<T>` is a one-column device vector.
- `DeviceVec::slice(..)` creates a borrowed `DeviceSlice` input whose logical
  item is `(T,)`.
- `(a.slice(..), b.slice(..))` combines borrowed slices into a wider SoA input
  whose logical item is `(A, B)`.
- `DeviceSlice` is lowered as a logical view of the original device buffer.
  Core input paths such as transform, scan, reduce, gather/scatter, search,
  sort, and single-key sort-by-key avoid copying the slice into a temporary
  `DeviceVec` before launching kernels.
- Algorithms return owned device storage directly: `DeviceVec<T>` for one
  output column, or a tuple of `DeviceVec` columns for multi-column output.
- By-key algorithms currently use a single key column. Their values may be
  multi-column SoA inputs. Compound keys should be normalized into one key
  column before calling by-key algorithms.
- Stencil-style algorithms use one `u32` flag column, where `0` is false and any
  non-zero value is true. One stencil column may select or flag multi-column
  values.

## v0.6 API Shapes

Most algorithms read borrowed `DeviceSlice` inputs and return newly owned
`DeviceVec` outputs. Stencil algorithms take a single `u32` flag column rather
than a predicate marker:

```rust
use massively::{CubeWgpu, copy_if};

fn main() -> Result<(), massively::Error> {
    let policy = CubeWgpu::cpu();
    let x = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;
    let tag = policy.to_device(&[10_u32, 20, 30, 40])?;
    let keep = policy.to_device(&[1_u32, 0, 1, 0])?;

    let (x, tag) = copy_if((x.slice(..), tag.slice(..)), (keep.slice(..),))?;

    assert_eq!(x.to_vec()?, vec![1.0, 3.0]);
    assert_eq!(tag.to_vec()?, vec![10, 30]);
    Ok(())
}
```

Predicate markers are still used by predicate-query and partition-style
algorithms such as `remove_if`, `count_if`, `find_if`, and `partition`.

By-key algorithms take one key column and may carry multiple value columns:

```rust
use cubecl::prelude::*;
use massively::{CubeWgpu, sort_by_key};

struct Less;
#[cubecl::cube]
impl massively::op::BinaryPredicateOp<(u32,)> for Less {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

fn main() -> Result<(), massively::Error> {
    let policy = CubeWgpu::cpu();
    let key = policy.to_device(&[2_u32, 0, 1])?;
    let x = policy.to_device(&[20.0_f32, 0.0, 10.0])?;
    let tag = policy.to_device(&[200_u32, 0, 100])?;

    let ((key,), (x, tag)) =
        sort_by_key((key.slice(..),), (x.slice(..), tag.slice(..)), Less)?;

    assert_eq!(key.to_vec()?, vec![0, 1, 2]);
    assert_eq!(x.to_vec()?, vec![0.0, 10.0, 20.0]);
    assert_eq!(tag.to_vec()?, vec![0, 100, 200]);
    Ok(())
}
```

## Example

```rust
use cubecl::prelude::*;
use massively::{CubeWgpu, reduce, transform};

struct Sum;
#[cubecl::cube]
impl massively::op::BinaryOp<(f32,)> for Sum {
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

struct KineticEnergy;
#[cubecl::cube]
impl massively::op::UnaryOp<(f32, f32, f32)> for KineticEnergy {
    type Output = (f32,);

    fn apply(input: (f32, f32, f32)) -> (f32,) {
        (0.5 * (input.0 * input.0 + input.1 * input.1 + input.2 * input.2),)
    }
}

fn main() -> Result<(), massively::Error> {
    let policy = CubeWgpu::cpu();

    let vx = policy.to_device(&[1.0_f32, 0.0, 2.0])?;
    let vy = policy.to_device(&[0.0_f32, 2.0, 0.0])?;
    let vz = policy.to_device(&[0.0_f32, 0.0, 2.0])?;

    let (energy,) = transform((vx.slice(..), vy.slice(..), vz.slice(..)), KineticEnergy)?;
    let sum = reduce((energy.slice(..),), (0.0,), Sum)?;

    assert_eq!(energy.to_vec()?, vec![0.5, 2.0, 4.0]);
    assert_eq!(sum, (6.5,));

    Ok(())
}
```
