//! # Problem
//!
//! Given sorted scores, return the range containing all entries equal to a
//! target score.
//!
//! # Task
//!
//! Implement `solve(score, target) -> (lower, upper)`.
//!
//! # GPU Algorithm
//!
//! 1. Lower-bound and upper-bound the target score.

mod common;

use massively::{DeviceVec, Executor, MIndex, lower_bound, upper_bound};

fn solve<B>(
    exec: &Executor<B>,
    score: DeviceVec<B, u32>,
    target: u32,
) -> common::Result<(MIndex, MIndex)>
where
    B: cubecl::prelude::Runtime,
{
    let query = exec.to_device(&[target]);
    let lower = exec.full(1, 0_u32)?;
    lower_bound(
        &exec,
        score.slice(..),
        query.slice(..),
        common::LessU32,
        lower.slice_mut(..),
    )?;
    let upper = exec.full(1, 0_u32)?;
    upper_bound(
        &exec,
        score.slice(..),
        query.slice(..),
        common::LessU32,
        upper.slice_mut(..),
    )?;
    let lower = exec.to_host(&lower)?;
    let upper = exec.to_host(&upper)?;
    Ok((lower[0], upper[0]))
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let range = solve(&exec, exec.to_device(&[10, 20, 20, 20, 30]), 20)?;
    assert_eq!(range, (1, 4));
    Ok(())
}
