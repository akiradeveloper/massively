//! # Problem
//!
//! Given event timestamps, report where sorted order first breaks.
//!
//! # Task
//!
//! Implement `solve(timestamp) -> sorted_until_index`.
//!
//! # GPU Algorithm
//!
//! 1. Use `is_sorted_until` with an ascending timestamp comparator.

mod common;

use massively::{DeviceVec, Executor, SoA1, Wgpu, is_sorted_until};

fn solve(exec: &Executor<Wgpu>, timestamp: DeviceVec<Wgpu, u32>) -> common::Result<usize> {
    is_sorted_until(exec, SoA1(timestamp.slice(..)), common::LessU32)
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let index = solve(&exec, exec.to_device(&[10, 20, 30, 25, 40])?)?;
    assert_eq!(index, 3);
    Ok(())
}
