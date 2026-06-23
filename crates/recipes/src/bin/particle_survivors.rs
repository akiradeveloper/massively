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
//! 2. Remove rows outside the box.

mod common;

use cubecl::prelude::*;
use massively::op::{PredicateOp1, UnaryOp};
use massively::{DeviceVec, Executor, SoA3, Wgpu, remove_if, transform};

struct AdvanceParticle;

#[cubecl::cube]
impl<B> UnaryOp<B, (f32, f32, f32)> for AdvanceParticle
where
    B: massively::Backend,
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
impl<B> PredicateOp1<B, (f32, f32, f32)> for OutsideParticleBox
where
    B: massively::Backend,
{
    fn apply(input: (f32, f32, f32)) -> bool {
        input.0 < 0.0 || input.0 > 10.0 || input.1 < 0.0 || input.1 > 10.0
    }
}

struct Output {
    x: DeviceVec<Wgpu, f32>,
    y: DeviceVec<Wgpu, f32>,
    vx: DeviceVec<Wgpu, f32>,
}

fn solve(
    exec: &Executor<Wgpu>,
    x: DeviceVec<Wgpu, f32>,
    y: DeviceVec<Wgpu, f32>,
    vx: DeviceVec<Wgpu, f32>,
) -> common::Result<Output> {
    let (x, y, vx) = transform(
        exec,
        SoA3(x.slice(..), y.slice(..), vx.slice(..)),
        AdvanceParticle,
    )?;
    let (x, y, vx) = remove_if(
        exec,
        SoA3(x.slice(..), y.slice(..), vx.slice(..)),
        OutsideParticleBox,
    )?;
    Ok(Output { x, y, vx })
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
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
