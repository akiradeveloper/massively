//! # Problem
//!
//! Given approved flags and target slots, scatter approved item ids into slots
//! while leaving unapproved slots at a default value.
//!
//! # Task
//!
//! Implement `solve(item_id, slot, approved, len) -> slots`.
//!
//! # GPU Algorithm
//!
//! 1. Use `scatter_where` with approved flags as the stencil.

mod common;

use massively::{DeviceVec, Executor, MIndex, Zip1, scatter_where};

fn solve<B>(
    exec: &Executor<B>,
    item_id: DeviceVec<B, u32>,
    slot: DeviceVec<B, MIndex>,
    approved: DeviceVec<B, u32>,
    len: MIndex,
) -> common::Result<DeviceVec<B, u32>>
where
    B: cubecl::prelude::Runtime,
{
    let out = exec.constant(len, 0_u32)?;
    scatter_where(
        exec,
        Zip1(item_id.slice(..)),
        slot.slice(..),
        approved.slice(..),
        Zip1(out.slice_mut(..)),
    )?;
    Ok(out)
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let out = solve(
        &exec,
        exec.to_device(&[10, 20, 30])?,
        exec.to_device(&[2, 0, 1])?,
        exec.to_device(&[1, 0, 1])?,
        4,
    )?;
    assert_eq!(exec.to_host(&out)?, vec![0, 30, 10, 0]);
    Ok(())
}
