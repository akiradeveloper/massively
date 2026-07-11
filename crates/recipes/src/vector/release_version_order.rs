//! # Problem
//!
//! Given two release version vectors, decide whether the left version is
//! lexicographically older than the right version.
//!
//! # Task
//!
//! Implement `solve(left, right) -> bool`.
//!
//! # GPU Algorithm
//!
//! 1. Use `lexicographical_compare`.

use super::common;

use massively::{DeviceVec, Executor, vector::lexicographical_compare};

fn solve<B>(
    exec: &Executor<B>,
    left: DeviceVec<B, u32>,
    right: DeviceVec<B, u32>,
) -> common::Result<bool>
where
    B: cubecl::prelude::Runtime,
{
    lexicographical_compare(&exec, left.slice(..), right.slice(..), common::LessU32)
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let older = solve(
        &exec,
        exec.to_device(&[1, 4, 9]),
        exec.to_device(&[1, 5, 0]),
    )?;
    assert!(older);
    Ok(())
}
