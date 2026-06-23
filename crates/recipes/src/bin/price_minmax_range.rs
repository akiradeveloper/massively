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

mod common;

use massively::{DeviceVec, Executor, SoA1, Wgpu, minmax_element};

fn solve<B>(exec: &Executor<B>, price: DeviceVec<B, f32>) -> common::Result<Option<(usize, usize)>>
where
    B: massively::Backend,
{
    minmax_element(exec, SoA1(price.slice(..)), common::LessF32)
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let indices = solve(&exec, exec.to_device(&[9.0, 3.5, 12.0, 7.0])?)?;
    assert_eq!(indices, Some((1, 2)));
    Ok(())
}
