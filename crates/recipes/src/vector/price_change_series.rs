//! # Problem
//!
//! Given daily closing prices, compute day-over-day price changes.
//!
//! # Task
//!
//! Implement `solve(price) -> delta`.
//!
//! # GPU Algorithm
//!
//! 1. Use `adjacent_difference` with subtraction.

use super::common;

use cubecl::prelude::*;
use massively::op::ReductionOp;
use massively::{DeviceVec, Executor, vector::adjacent_difference};

struct PriceDelta;

#[cubecl::cube]
impl ReductionOp<f32> for PriceDelta {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs - rhs
    }
}

fn solve<B>(exec: &Executor<B>, price: DeviceVec<B, f32>) -> common::Result<DeviceVec<B, f32>>
where
    B: cubecl::prelude::Runtime,
{
    let delta = exec.full(price.len(), 0.0_f32)?;
    adjacent_difference(&exec, price.slice(..), PriceDelta, delta.slice_mut(..))?;
    Ok(delta)
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let delta = solve(&exec, exec.to_device(&[10.0, 13.0, 12.0, 18.0]))?;
    assert_eq!(exec.to_host(&delta)?, vec![10.0, 3.0, -1.0, 6.0]);
    Ok(())
}
