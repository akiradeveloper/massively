//! # Problem
//!
//! Merge two timestamp-sorted event feeds.
//!
//! # Task
//!
//! Implement `solve(left, right) -> merged_feed`.
//!
//! # GPU Algorithm
//!
//! 1. Keep timestamps sorted.
//! 2. Use `merge_by_key` and carry event ids as values.

mod common;

use massively::{DeviceVec, Executor, merge_by_key};

struct Output<B: cubecl::prelude::Runtime> {
    timestamp: DeviceVec<B, u32>,
    event_id: DeviceVec<B, u32>,
}

fn solve<B>(
    exec: &Executor<B>,
    left_timestamp: DeviceVec<B, u32>,
    left_event_id: DeviceVec<B, u32>,
    right_timestamp: DeviceVec<B, u32>,
    right_event_id: DeviceVec<B, u32>,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let len = left_timestamp.len() + right_timestamp.len();
    let timestamp = exec.full(len as usize, 0_u32)?;
    let event_id = exec.full(len as usize, 0_u32)?;
    merge_by_key(
        &exec,
        left_timestamp.slice(..),
        left_event_id.slice(..),
        right_timestamp.slice(..),
        right_event_id.slice(..),
        common::LessU32,
        timestamp.slice_mut(..),
        event_id.slice_mut(..),
    )?;
    Ok(Output {
        timestamp,
        event_id,
    })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[1, 4, 8]),
        exec.to_device(&[10, 40, 80]),
        exec.to_device(&[2, 3, 9]),
        exec.to_device(&[20, 30, 90]),
    )?;
    assert_eq!(exec.to_host(&output.timestamp)?, vec![1, 2, 3, 4, 8, 9]);
    assert_eq!(
        exec.to_host(&output.event_id)?,
        vec![10, 20, 30, 40, 80, 90]
    );
    Ok(())
}
