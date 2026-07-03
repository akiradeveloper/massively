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

use massively::{DeviceVec, Executor, MIndex, SoA1, lower_bound, upper_bound};

fn solve<B>(
    exec: &Executor<B>,
    score: DeviceVec<B, u32>,
    target: u32,
) -> common::Result<(MIndex, MIndex)>
where
    B: cubecl::prelude::Runtime,
{
    let query = exec.to_device(&[target])?;
    let lower = lower_bound(
        exec,
        SoA1(score.slice(..)),
        SoA1(query.slice(..)),
        common::LessU32,
    )?;
    let upper = upper_bound(
        exec,
        SoA1(score.slice(..)),
        SoA1(query.slice(..)),
        common::LessU32,
    )?;
    let lower = exec.to_host(&lower)?;
    let upper = exec.to_host(&upper)?;
    Ok((lower[0], upper[0]))
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let range = solve(&exec, exec.to_device(&[10, 20, 20, 20, 30])?, 20)?;
    assert_eq!(range, (1, 4));
    Ok(())
}
