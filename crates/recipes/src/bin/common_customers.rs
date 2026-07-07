//! # Problem
//!
//! Given two sorted campaign customer lists, return customers present in both.
//!
//! # Task
//!
//! Implement `solve(a, b) -> intersection`.
//!
//! # GPU Algorithm
//!
//! 1. Use `set_intersection` on sorted ranges.

mod common;

use massively::{DeviceVec, Executor, Zip1, set_intersection};

fn solve<B>(
    exec: &Executor<B>,
    a: DeviceVec<B, u32>,
    b: DeviceVec<B, u32>,
) -> common::Result<DeviceVec<B, u32>>
where
    B: cubecl::prelude::Runtime,
{
    let out = exec.full(a.len().min(b.len()), 0_u32)?;
    let len = set_intersection(
        exec,
        Zip1(a.slice(..)),
        Zip1(b.slice(..)),
        common::LessU32,
        Zip1(out.slice_mut(..)),
    )?;
    Ok(exec.to_device(&exec.to_host(&out.slice(..len))?)?)
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let out = solve(
        &exec,
        exec.to_device(&[1, 2, 4, 8])?,
        exec.to_device(&[2, 3, 4])?,
    )?;
    assert_eq!(exec.to_host(&out)?, vec![2, 4]);
    Ok(())
}
