//! # Problem
//!
//! Given a sorted allowlist and a sorted banlist, return allowed users after
//! removing banned users.
//!
//! # Task
//!
//! Implement `solve(allowlist, banlist) -> filtered_allowlist`.
//!
//! # GPU Algorithm
//!
//! 1. Use `set_difference(allowlist, banlist)`.

mod common;

use massively::{DeviceVec, Executor, SoA1, set_difference};

fn solve<B>(
    exec: &Executor<B>,
    allowlist: DeviceVec<B, u32>,
    banlist: DeviceVec<B, u32>,
) -> common::Result<DeviceVec<B, u32>>
where
    B: cubecl::prelude::Runtime,
{
    let SoA1(out) = set_difference(
        exec,
        SoA1(allowlist.slice(..)),
        SoA1(banlist.slice(..)),
        common::LessU32,
    )?;
    Ok(out)
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let out = solve(
        &exec,
        exec.to_device(&[1, 2, 4, 8])?,
        exec.to_device(&[2, 7])?,
    )?;
    assert_eq!(exec.to_host(&out)?, vec![1, 4, 8]);
    Ok(())
}
