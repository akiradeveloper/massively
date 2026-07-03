//! # Problem
//!
//! Given two configuration snapshots represented as numeric option ids, find
//! the first differing option.
//!
//! # Task
//!
//! Implement `solve(expected, actual) -> Option<MIndex>`.
//!
//! # GPU Algorithm
//!
//! 1. Use `mismatch` with equality over option ids.

mod common;

use massively::{DeviceVec, Executor, MIndex, SoA1, mismatch};

fn solve<B>(
    exec: &Executor<B>,
    expected: DeviceVec<B, u32>,
    actual: DeviceVec<B, u32>,
) -> common::Result<Option<MIndex>>
where
    B: cubecl::prelude::Runtime,
{
    mismatch(
        exec,
        SoA1(expected.slice(..)),
        SoA1(actual.slice(..)),
        common::EqualU32,
    )
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let index = solve(
        &exec,
        exec.to_device(&[1, 4, 9, 16])?,
        exec.to_device(&[1, 4, 8, 16])?,
    )?;
    assert_eq!(index, Some(2));
    Ok(())
}
