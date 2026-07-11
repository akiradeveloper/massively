//! # Problem
//!
//! Given event timestamps, report where sorted order first breaks.
//!
//! # Task
//!
//! Implement `solve(timestamp) -> sorted_until_index`.
//!
//! # GPU Algorithm
//!
//! 1. Use `is_sorted_until` with an ascending timestamp comparator.

use super::common;

use massively::{DeviceVec, Executor, MIndex, vector::is_sorted_until};

fn solve<B>(exec: &Executor<B>, timestamp: DeviceVec<B, u32>) -> common::Result<MIndex>
where
    B: cubecl::prelude::Runtime,
{
    is_sorted_until(&exec, timestamp.slice(..), common::LessU32)
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let index = solve(&exec, exec.to_device(&[10, 20, 30, 25, 40]))?;
    assert_eq!(index, 3);
    Ok(())
}
