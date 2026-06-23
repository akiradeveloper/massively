//! # Problem
//!
//! Given exam scores, count how many students passed with score at least 60.
//!
//! # Task
//!
//! Implement `solve(score) -> pass_count`.
//!
//! # GPU Algorithm
//!
//! 1. Use `count_if` with a passing-score predicate.

mod common;

use cubecl::prelude::*;
use massively::op::PredicateOp1;
use massively::{DeviceVec, Executor, SoA1, Wgpu, count_if};

struct PassingScore;

#[cubecl::cube]
impl<B> PredicateOp1<B, (u32,)> for PassingScore
where
    B: massively::Backend,
{
    fn apply(input: (u32,)) -> bool {
        input.0 >= 60_u32
    }
}

fn solve(exec: &Executor<Wgpu>, score: DeviceVec<Wgpu, u32>) -> common::Result<usize> {
    count_if(exec, SoA1(score.slice(..)), PassingScore)
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let count = solve(&exec, exec.to_device(&[95, 40, 60, 59, 80])?)?;
    assert_eq!(count, 3);
    Ok(())
}
