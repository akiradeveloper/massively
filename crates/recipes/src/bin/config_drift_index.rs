//! # Problem
//!
//! Given two configuration snapshots represented as numeric option ids, find
//! the first differing option.
//!
//! # Task
//!
//! Implement `solve(expected, actual) -> Option<usize>`.
//!
//! # GPU Algorithm
//!
//! 1. Use `mismatch` with equality over option ids.

mod common;

use massively::{DeviceVec, Executor, SoA1, Wgpu, mismatch};

fn solve<B>(
    exec: &Executor<B>,
    expected: DeviceVec<B, u32>,
    actual: DeviceVec<B, u32>,
) -> common::Result<Option<usize>>
where
    B: massively::Backend,
{
    mismatch(
        exec,
        SoA1(expected.slice(..)),
        SoA1(actual.slice(..)),
        common::EqualU32,
    )
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let index = solve(
        &exec,
        exec.to_device(&[1, 4, 9, 16])?,
        exec.to_device(&[1, 4, 8, 16])?,
    )?;
    assert_eq!(index, Some(2));
    Ok(())
}
