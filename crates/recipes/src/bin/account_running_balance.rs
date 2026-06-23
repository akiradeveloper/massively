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

use massively::{DeviceVec, Executor, SoA1, Wgpu, inclusive_scan_by_key};

fn solve(
    exec: &Executor<Wgpu>,
    account_id: DeviceVec<Wgpu, u32>,
    amount_delta: DeviceVec<Wgpu, f32>,
) -> common::Result<DeviceVec<Wgpu, f32>> {
    let (balance,) = inclusive_scan_by_key(
        exec,
        SoA1(account_id.slice(..)),
        SoA1(amount_delta.slice(..)),
        common::EqualU32,
        common::SumF32,
    )?;
    Ok(balance)
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let balance = solve(
        &exec,
        exec.to_device(&[1, 1, 2, 2, 2])?,
        exec.to_device(&[10.0, -3.0, 5.0, 7.0, -2.0])?,
    )?;
    assert_eq!(exec.to_host(&balance)?, vec![10.0, 7.0, 5.0, 12.0, 10.0]);
    Ok(())
}
