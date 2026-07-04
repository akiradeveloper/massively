//! # Problem
//!
//! Given transactions sorted by account id and time, compute the running
//! balance within each account.
//!
//! # Task
//!
//! Implement `solve(account_id, amount_delta) -> running_balance`.
//!
//! # GPU Algorithm
//!
//! 1. Use `inclusive_scan_by_key` with account id as the segment key.

mod common;

use massively::{DeviceVec, Executor, SoA1, inclusive_scan_by_key};

fn solve<B>(
    exec: &Executor<B>,
    account_id: DeviceVec<B, u32>,
    amount_delta: DeviceVec<B, f32>,
) -> common::Result<DeviceVec<B, f32>>
where
    B: cubecl::prelude::Runtime,
{
    let balance = exec.constant(amount_delta.len(), 0.0_f32)?;
    inclusive_scan_by_key(
        exec,
        SoA1(account_id.slice(..)),
        SoA1(amount_delta.slice(..)),
        common::EqualU32,
        common::SumF32,
        SoA1(balance.slice_mut(..)),
    )?;
    Ok(balance)
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let balance = solve(
        &exec,
        exec.to_device(&[1, 1, 2, 2, 2])?,
        exec.to_device(&[10.0, -3.0, 5.0, 7.0, -2.0])?,
    )?;
    assert_eq!(exec.to_host(&balance)?, vec![10.0, 7.0, 5.0, 12.0, 10.0]);
    Ok(())
}
