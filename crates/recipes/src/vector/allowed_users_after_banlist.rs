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

use super::common;

use massively::{DeviceVec, Executor, vector::set_difference};

fn solve<B>(
    exec: &Executor<B>,
    allowlist: DeviceVec<B, u32>,
    banlist: DeviceVec<B, u32>,
) -> common::Result<DeviceVec<B, u32>>
where
    B: cubecl::prelude::Runtime,
{
    let out = exec.full(allowlist.len(), 0_u32)?;
    let len = set_difference(
        &exec,
        allowlist.slice(..),
        banlist.slice(..),
        common::LessU32,
        out.slice_mut(..),
    )?;
    Ok(exec.to_device(&exec.to_host(&out.slice(..len as usize))?))
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let out = solve(
        &exec,
        exec.to_device(&[1, 2, 4, 8]),
        exec.to_device(&[2, 7]),
    )?;
    assert_eq!(exec.to_host(&out)?, vec![1, 4, 8]);
    Ok(())
}
