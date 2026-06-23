//! # Problem
//!
//! Find the first adjacent temperature pair where the increase is greater than
//! five degrees.
//!
//! # Task
//!
//! Implement `solve(temperature) -> Option<usize>`.
//!
//! # GPU Algorithm
//!
//! 1. Use `adjacent_find` with an adjacent-pair predicate.

mod common;

use cubecl::prelude::*;
use massively::op::BinaryPredicateOp;
use massively::{DeviceVec, Executor, SoA1, Wgpu, adjacent_find};

struct TemperatureSpike;

#[cubecl::cube]
impl<B> BinaryPredicateOp<B, (f32,)> for TemperatureSpike
where
    B: massively::Backend,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        rhs.0 > lhs.0 + 5.0
    }
}

fn solve<B>(exec: &Executor<B>, temperature: DeviceVec<B, f32>) -> common::Result<Option<usize>>
where
    B: massively::Backend,
{
    adjacent_find(exec, SoA1(temperature.slice(..)), TemperatureSpike)
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let index = solve(&exec, exec.to_device(&[20.0, 21.0, 30.0, 31.0])?)?;
    assert_eq!(index, Some(1));
    Ok(())
}
