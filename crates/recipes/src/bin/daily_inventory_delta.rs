//! # Problem
//!
//! Merge two sorted warehouse delta streams and compute total delta per sku.
//!
//! # Task
//!
//! Implement `solve(left, right) -> total delta per sku`.
//!
//! # GPU Algorithm
//!
//! 1. Merge both streams by sku.
//! 2. Reduce merged deltas by sku.

mod common;

use massively::{DeviceVec, Executor, SoA1, Wgpu, merge_by_key, reduce_by_key};

struct Output {
    sku: DeviceVec<Wgpu, u32>,
    delta: DeviceVec<Wgpu, f32>,
}

fn solve(
    exec: &Executor<Wgpu>,
    left_sku: DeviceVec<Wgpu, u32>,
    left_delta: DeviceVec<Wgpu, f32>,
    right_sku: DeviceVec<Wgpu, u32>,
    right_delta: DeviceVec<Wgpu, f32>,
) -> common::Result<Output> {
    let ((sku,), (delta,)) = merge_by_key(
        exec,
        SoA1(left_sku.slice(..)),
        SoA1(left_delta.slice(..)),
        SoA1(right_sku.slice(..)),
        SoA1(right_delta.slice(..)),
        common::LessU32,
    )?;
    let ((sku,), (delta,)) = reduce_by_key(
        exec,
        SoA1(sku.slice(..)),
        SoA1(delta.slice(..)),
        common::EqualU32,
        (0.0_f32,),
        common::SumF32,
    )?;
    Ok(Output { sku, delta })
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let output = solve(
        &exec,
        exec.to_device(&[1, 2, 4])?,
        exec.to_device(&[5.0, -1.0, 9.0])?,
        exec.to_device(&[2, 3, 4])?,
        exec.to_device(&[2.0, 7.0, -3.0])?,
    )?;
    assert_eq!(exec.to_host(&output.sku)?, vec![1, 2, 3, 4]);
    assert_eq!(exec.to_host(&output.delta)?, vec![5.0, 1.0, 7.0, 6.0]);
    Ok(())
}
