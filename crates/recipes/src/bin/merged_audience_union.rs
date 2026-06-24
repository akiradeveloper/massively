//! # Problem
//!
//! Given two sorted audience id lists, return their sorted union.
//!
//! # Task
//!
//! Implement `solve(left, right) -> union`.
//!
//! # GPU Algorithm
//!
//! 1. Use `set_union` on sorted ranges.

mod common;

use massively::{DeviceVec, Executor, SoA1, set_union};

fn solve<B>(
    exec: &Executor<B>,
    left: DeviceVec<B, u32>,
    right: DeviceVec<B, u32>,
) -> common::Result<DeviceVec<B, u32>>
where
    B: cubecl::prelude::Runtime,
{
    let (out,) = set_union(
        exec,
        SoA1(left.slice(..)),
        SoA1(right.slice(..)),
        common::LessU32,
    )?;
    Ok(out)
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let out = solve(
        &exec,
        exec.to_device(&[1, 2, 4, 8])?,
        exec.to_device(&[2, 3, 4, 9])?,
    )?;
    assert_eq!(exec.to_host(&out)?, vec![1, 2, 3, 4, 8, 9]);
    Ok(())
}
