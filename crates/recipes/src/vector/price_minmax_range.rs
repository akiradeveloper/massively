//! # Problem
//!
//! Given product prices, return the indices of the cheapest and most expensive
//! products.
//!
//! # Task
//!
//! Implement `solve(price) -> Option<(min_index, max_index)>`.
//!
//! # GPU Algorithm
//!
//! 1. Use `minmax_element` with an ascending price comparator.

use super::common;

use massively::{DeviceVec, Executor, MIndex, vector::minmax_element};

fn solve<B>(
    exec: &Executor<B>,
    price: DeviceVec<B, f32>,
) -> common::Result<Option<(MIndex, MIndex)>>
where
    B: cubecl::prelude::Runtime,
{
    minmax_element(&exec, price.slice(..), common::LessF32)
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let indices = solve(&exec, exec.to_device(&[9.0, 3.5, 12.0, 7.0]))?;
    assert_eq!(indices, Some((1, 2)));
    Ok(())
}
