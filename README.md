<div align="center">
<img src="assets/logo.svg" alt="massively" width="400">

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

`massively` keeps host data, owned device storage, and read-only device input
separate.

`DeviceVec<T>` is an owned column on the GPU. Moving data into a `DeviceVec`
with `policy.to_device(...)` is an explicit host-to-device transfer, and reading
it back with `to_vec()` is an explicit device-to-host transfer.

Multi-column data uses Structure of Arrays rather than Array of Structures.
For GPU algorithms this keeps each field in its own contiguous device buffer,
which improves coalesced memory access and lets algorithms reuse or move columns
independently.

- `DeviceVec<T>` is a one-column owned SoA.
- `zip(a, b)` combines owned SoA columns into a wider owned SoA.
- `unzip(soa)` consumes an owned SoA and returns its `DeviceVec` columns.

Read-only algorithm inputs use a separate virtual view of columns. A borrowed
`&DeviceVec<T>` is a one-column read-only view, and `vzip(&a, &b)` combines
multiple read-only columns into a wider read-only view. Internally and in the
design docs, this read-only Structure of Virtual Arrays is called `SoVA`.

Read-only algorithms such as `transform`, `reduce`, and `gather` accept these
read-only views and return owned SoA outputs when they materialize data.
Consuming algorithms such as `sort`, `reverse`, and `remove_if` take owned SoA
inputs. The two concepts are intentionally separate: `zip(...)` is for owned
storage, while `vzip(...)` is for read-only input.

## Example

```rust
use massively::{CubeWgpu, reduce, transform, unzip, vzip3};

struct Sum;
#[cubecl::cube]
impl massively::op::BinaryOp<f32> for Sum {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

struct KineticEnergy;
#[cubecl::cube]
impl massively::op::UnaryOp<(f32, f32, f32)> for KineticEnergy {
    type Output = f32;

    fn apply(input: (f32, f32, f32)) -> f32 {
        0.5 * (input.0 * input.0 + input.1 * input.1 + input.2 * input.2)
    }
}

fn main() -> Result<(), massively::Error> {
    let policy = CubeWgpu::new();

    let vx = policy.to_device(&[1.0_f32, 0.0, 2.0])?;
    let vy = policy.to_device(&[0.0_f32, 2.0, 0.0])?;
    let vz = policy.to_device(&[0.0_f32, 0.0, 2.0])?;

    let velocity = vzip3(&vx, &vy, &vz);
    let energy = unzip(transform(velocity, KineticEnergy)?)?;
    let sum = reduce(&energy, 0.0, Sum)?;

    assert_eq!(energy.to_vec()?, vec![0.5, 2.0, 4.0]);
    assert_eq!(sum, 6.5);

    Ok(())
}
```

