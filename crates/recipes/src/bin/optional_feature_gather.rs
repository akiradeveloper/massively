//! # Problem
//!
//! Given feature values, requested row indices, and availability flags, gather
//! requested values only when available and use a default otherwise.
//!
//! # Task
//!
//! Implement `solve(value, index, available) -> gathered`.
//!
//! # GPU Algorithm
//!
//! 1. Use `gather_if` with availability flags as the stencil.

mod common;

use massively::{DeviceVec, Executor, SoA1, Wgpu, gather_if};

fn solve(
    exec: &Executor<Wgpu>,
    value: DeviceVec<Wgpu, f32>,
    index: DeviceVec<Wgpu, u32>,
    available: DeviceVec<Wgpu, u32>,
) -> common::Result<DeviceVec<Wgpu, f32>> {
    let (out,) = gather_if(
        exec,
        SoA1(value.slice(..)),
        index.slice(..),
        (-1.0_f32,),
        available.slice(..),
    )?;
    Ok(out)
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let out = solve(
        &exec,
        exec.to_device(&[10.0, 20.0, 30.0])?,
        exec.to_device(&[2, 0, 1])?,
        exec.to_device(&[1, 0, 1])?,
    )?;
    assert_eq!(exec.to_host(&out)?, vec![30.0, -1.0, 20.0]);
    Ok(())
}
