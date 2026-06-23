//! # Problem
//!
//! Given sorted scores, return the range containing all entries equal to a
//! target score.
//!
//! # Task
//!
//! Implement `solve(score, target) -> (lower, upper)`.
//!
//! # GPU Algorithm
//!
//! 1. Use `equal_range` with an ascending score comparator.

mod common;

use massively::{DeviceVec, Executor, SoA1, Wgpu, equal_range};

fn solve(
    exec: &Executor<Wgpu>,
    score: DeviceVec<Wgpu, u32>,
    target: u32,
) -> common::Result<(usize, usize)> {
    equal_range(exec, SoA1(score.slice(..)), (target,), common::LessU32)
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let range = solve(&exec, exec.to_device(&[10, 20, 20, 20, 30])?, 20)?;
    assert_eq!(range, (1, 4));
    Ok(())
}
