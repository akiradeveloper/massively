//! # Problem
//!
//! Given sorted timestamps and a half-open query interval `[start, end)`, return
//! the index range containing timestamps inside the interval.
//!
//! # Task
//!
//! Implement `solve(timestamp, start, end) -> (lower, upper)`.
//!
//! # GPU Algorithm
//!
//! 1. Lower-bound `start`.
//! 2. Lower-bound `end`.

mod common;

use massively::{DeviceVec, Executor, SoA1, Wgpu, lower_bound};

fn solve(
    exec: &Executor<Wgpu>,
    timestamp: DeviceVec<Wgpu, u32>,
    start: u32,
    end: u32,
) -> common::Result<(usize, usize)> {
    let lower = lower_bound(exec, SoA1(timestamp.slice(..)), (start,), common::LessU32)?;
    let upper = lower_bound(exec, SoA1(timestamp.slice(..)), (end,), common::LessU32)?;
    Ok((lower, upper))
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let range = solve(&exec, exec.to_device(&[10, 20, 30, 40, 50])?, 20, 45)?;
    assert_eq!(range, (1, 4));
    Ok(())
}
