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
use massively::prelude::*;
use massively::{DeviceVec, Executor, SoA3, remove_where, transform};

struct AdvanceParticle;

#[cubecl::cube]
impl<B> UnaryOp<B, (f32, f32, f32)> for AdvanceParticle
where
    B: cubecl::prelude::Runtime,
{
    type Env = ();
    type Output = (f32, f32, f32);

    fn apply(_env: (), input: (f32, f32, f32)) -> (f32, f32, f32) {
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
    type Env = ();
    type Output = (u32,);

    fn apply(_env: (), input: (f32, f32, f32)) -> (u32,) {
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
    let next_x = exec.constant(x.len(), 0.0_f32)?;
    let next_y = exec.constant(y.len(), 0.0_f32)?;
    let next_vx = exec.constant(vx.len(), 0.0_f32)?;
    transform(
        exec,
        SoA3(x.slice(..), y.slice(..), vx.slice(..)),
        AdvanceParticle,
        (),
        SoA3(
            next_x.slice_mut(..),
            next_y.slice_mut(..),
            next_vx.slice_mut(..),
        ),
    )?;
    let stencil = exec.constant(next_x.len(), 0_u32)?;
    transform(
        exec,
        SoA3(next_x.slice(..), next_y.slice(..), next_vx.slice(..)),
        OutsideParticleBox,
        (),
        SoA1(stencil.slice_mut(..)),
    )?;
    let x = exec.constant(next_x.len(), 0.0_f32)?;
    let y = exec.constant(next_y.len(), 0.0_f32)?;
    let vx = exec.constant(next_vx.len(), 0.0_f32)?;
    let len = remove_where(
        exec,
        SoA3(next_x.slice(..), next_y.slice(..), next_vx.slice(..)),
        stencil.slice(..),
        SoA3(x.slice_mut(..), y.slice_mut(..), vx.slice_mut(..)),
    )?;
    Ok(Output {
        x: exec.to_device(&exec.to_host(&x.slice(..len))?)?,
        y: exec.to_device(&exec.to_host(&y.slice(..len))?)?,
        vx: exec.to_device(&exec.to_host(&vx.slice(..len))?)?,
    })
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
