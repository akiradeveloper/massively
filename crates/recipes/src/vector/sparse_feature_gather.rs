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
//! 1. Treat feature columns as one Zip row.
//! 2. Gather rows by index.

use super::common;

use massively::{DeviceVec, Executor, MIndex, vector::gather, zip2};

struct Output<B: cubecl::prelude::Runtime> {
    age: DeviceVec<B, u32>,
    score: DeviceVec<B, f32>,
}

fn solve<B>(
    exec: &Executor<B>,
    age: DeviceVec<B, u32>,
    score: DeviceVec<B, f32>,
    row_index: DeviceVec<B, MIndex>,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let gathered_age = exec.full(row_index.len(), 0_u32)?;
    let gathered_score = exec.full(row_index.len(), 0.0_f32)?;
    gather(
        &exec,
        zip2(age.slice(..), score.slice(..)),
        row_index.slice(..),
        zip2(gathered_age.slice_mut(..), gathered_score.slice_mut(..)),
    )?;
    Ok(Output {
        age: gathered_age,
        score: gathered_score,
    })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[21, 35, 50]),
        exec.to_device(&[0.2, 0.8, 0.4]),
        exec.to_device(&[2, 0, 2, 1]),
    )?;
    assert_eq!(exec.to_host(&output.age)?, vec![50, 21, 50, 35]);
    assert_eq!(exec.to_host(&output.score)?, vec![0.4, 0.2, 0.4, 0.8]);
    Ok(())
}
