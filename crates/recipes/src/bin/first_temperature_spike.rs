//! # Problem
//!
//! Find the first adjacent temperature pair where the increase is greater than
//! five degrees.
//!
//! # Task
//!
//! Implement `solve(temperature) -> Option<MIndex>`.
//!
//! # GPU Algorithm
//!
//! 1. Use `adjacent_find` with an adjacent-pair predicate.

mod common;

use cubecl::prelude::*;
use massively::op::BinaryPredicateOp;
use massively::{DeviceVec, Executor, MIndex, adjacent_find};

struct TemperatureSpike;

#[cubecl::cube]
impl BinaryPredicateOp<f32> for TemperatureSpike {
    fn apply(lhs: f32, rhs: f32) -> bool {
        rhs > lhs + 5.0
    }
}

fn solve<B>(exec: &Executor<B>, temperature: DeviceVec<B, f32>) -> common::Result<Option<MIndex>>
where
    B: cubecl::prelude::Runtime,
{
    adjacent_find(&exec, temperature.slice(..), TemperatureSpike)
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let index = solve(&exec, exec.to_device(&[20.0, 21.0, 30.0, 31.0]))?;
    assert_eq!(index, Some(1));
    Ok(())
}
