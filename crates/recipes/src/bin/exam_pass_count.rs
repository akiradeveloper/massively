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
use massively::op::PredicateOp;
use massively::{DeviceVec, Executor, MIndex, SoA1, count_if};

struct PassingScore;

#[cubecl::cube]
impl<B> PredicateOp<B, (u32,)> for PassingScore
where
    B: cubecl::prelude::Runtime,
{
    type Env = ();

    fn apply(_env: (), input: (u32,)) -> bool {
        input.0 >= 60_u32
    }
}

fn solve<B>(exec: &Executor<B>, score: DeviceVec<B, u32>) -> common::Result<MIndex>
where
    B: cubecl::prelude::Runtime,
{
    count_if(exec, SoA1(score.slice(..)), PassingScore, ())
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let count = solve(&exec, exec.to_device(&[95, 40, 60, 59, 80])?)?;
    assert_eq!(count, 3);
    Ok(())
}
