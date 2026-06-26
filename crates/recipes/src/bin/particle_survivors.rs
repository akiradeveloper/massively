//! # Problem
//!
//! Advance particles by one fixed time step and keep particles inside the box.
//!
//! # Task
//!
//! Implement `solve(x, y, vx) -> surviving particles`.
//!
//! # GPU Algorithm
//!
//! 1. Transform each row to the next position.
//! 2. Build a stencil for rows outside the box.
//! 3. Remove rows with a non-zero stencil.

mod common;

use cubecl::prelude::*;
use massively::op::UnaryOp;
use massively::{DeviceVec, Executor, SoA3, remove_where, transform};

struct AdvanceParticle;

#[cubecl::cube]
impl<B> UnaryOp<B, (f32, f32, f32)> for AdvanceParticle
where
    B: cubecl::prelude::Runtime,
{
    type Output = (f32, f32, f32);

    fn apply(input: (f32, f32, f32)) -> (f32, f32, f32) {
        let x = input.0 + input.2;
        let y = input.1 + 0.5;
        (x, y, input.2)
    }
}

struct OutsideParticleBox;

#[cubecl::cube]
impl<B> UnaryOp<B, (f32, f32, f32)> for OutsideParticleBox
where
    B: cubecl::prelude::Runtime,
{
    type Output = (u32,);

    fn apply(input: (f32, f32, f32)) -> (u32,) {
        if input.0 < 0.0 || input.0 > 10.0 || input.1 < 0.0 || input.1 > 10.0 {
            (1_u32,)
        } else {
            (0_u32,)
        }
    }
}

struct Output<B: cubecl::prelude::Runtime> {
    x: DeviceVec<B, f32>,
    y: DeviceVec<B, f32>,
    vx: DeviceVec<B, f32>,
}

fn solve<B>(
    exec: &Executor<B>,
    x: DeviceVec<B, f32>,
    y: DeviceVec<B, f32>,
    vx: DeviceVec<B, f32>,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let (x, y, vx) = transform(
        exec,
        SoA3(x.slice(..), y.slice(..), vx.slice(..)),
        AdvanceParticle,
    )?;
    let (stencil,) = transform(
        exec,
        SoA3(x.slice(..), y.slice(..), vx.slice(..)),
        OutsideParticleBox,
    )?;
    let (x, y, vx) = remove_where(
        exec,
        SoA3(x.slice(..), y.slice(..), vx.slice(..)),
        stencil.slice(..),
    )?;
    Ok(Output { x, y, vx })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[0.0, 9.5, 4.0])?,
        exec.to_device(&[0.0, 9.8, 1.0])?,
        exec.to_device(&[1.0, 1.0, -2.0])?,
    )?;
    assert_eq!(exec.to_host(&output.x)?, vec![1.0, 2.0]);
    assert_eq!(exec.to_host(&output.y)?, vec![0.5, 1.5]);
    assert_eq!(exec.to_host(&output.vx)?, vec![1.0, -2.0]);
    Ok(())
}
