//! # Problem
//!
//! Given sorted timestamps and a half-open query interval `[start, end)`, return
//! the index range containing timestamps inside the interval.
//!
//! # Task
//!
//! Implement `solve(timestamp, start, end) -> (lower, upper)`.
//!
//! # GPU Algorithm
//!
//! 1. Lower-bound `[start, end]` in parallel.

mod common;

use massively::{DeviceVec, Executor, SoA1, lower_bound};

fn solve<B>(
    exec: &Executor<B>,
    timestamp: DeviceVec<B, u32>,
    start: u32,
    end: u32,
) -> common::Result<(usize, usize)>
where
    B: cubecl::prelude::Runtime,
{
    let queries = exec.to_device(&[start, end])?;
    let indices = lower_bound(
        exec,
        SoA1(timestamp.slice(..)),
        SoA1(queries.slice(..)),
        common::LessU32,
    )?;
    let indices = exec.to_host(&indices)?;
    Ok((indices[0] as usize, indices[1] as usize))
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let range = solve(&exec, exec.to_device(&[10, 20, 30, 40, 50])?, 20, 45)?;
    assert_eq!(range, (1, 4));
    Ok(())
}
