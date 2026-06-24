//! # Problem
//!
//! Given a dense feature table and requested row indices, gather the requested
//! rows in request order.
//!
//! # Task
//!
//! Implement `solve(age, score, row_index) -> gathered features`.
//!
//! # GPU Algorithm
//!
//! 1. Treat feature columns as one SoA row.
//! 2. Gather rows by index.

mod common;

use massively::{DeviceVec, Executor, SoA2, gather};

struct Output<B: cubecl::prelude::Runtime> {
    age: DeviceVec<B, u32>,
    score: DeviceVec<B, f32>,
}

fn solve<B>(
    exec: &Executor<B>,
    age: DeviceVec<B, u32>,
    score: DeviceVec<B, f32>,
    row_index: DeviceVec<B, u32>,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let (age, score) = gather(
        exec,
        SoA2(age.slice(..), score.slice(..)),
        row_index.slice(..),
    )?;
    Ok(Output { age, score })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[21, 35, 50])?,
        exec.to_device(&[0.2, 0.8, 0.4])?,
        exec.to_device(&[2, 0, 2, 1])?,
    )?;
    assert_eq!(exec.to_host(&output.age)?, vec![50, 21, 50, 35]);
    assert_eq!(exec.to_host(&output.score)?, vec![0.4, 0.2, 0.4, 0.8]);
    Ok(())
}
