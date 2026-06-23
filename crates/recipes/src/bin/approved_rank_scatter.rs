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
//! 1. Use `scatter_if` with approved flags as the stencil.

mod common;

use massively::{DeviceVec, Executor, SoA1, Wgpu, scatter_if};

fn solve(
    exec: &Executor<Wgpu>,
    item_id: DeviceVec<Wgpu, u32>,
    slot: DeviceVec<Wgpu, u32>,
    approved: DeviceVec<Wgpu, u32>,
    len: usize,
) -> common::Result<DeviceVec<Wgpu, u32>> {
    let (out,) = scatter_if(
        exec,
        SoA1(item_id.slice(..)),
        slot.slice(..),
        len,
        (0_u32,),
        approved.slice(..),
    )?;
    Ok(out)
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
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
