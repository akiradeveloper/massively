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
- `&DeviceVec<T>` is a one-column input whose logical item is `(T,)`.
- `(&a, &b)` combines borrowed columns into a wider SoA input whose logical
  item is `(A, B)`.
- Algorithms return owned device storage directly: `DeviceVec<T>` for one
  output column, or a tuple of `DeviceVec` columns for multi-column output.
- By-key algorithms currently use a single key column. Their values may be
  multi-column SoA inputs.

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

    let (energy,) = transform((&vx, &vy, &vz), KineticEnergy)?;
    let sum = reduce((&energy,), (0.0,), Sum)?;

    assert_eq!(energy.to_vec()?, vec![0.5, 2.0, 4.0]);
    assert_eq!(sum, (6.5,));

    Ok(())
}
```
