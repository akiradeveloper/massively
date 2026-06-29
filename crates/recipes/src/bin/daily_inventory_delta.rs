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

use massively::{DeviceVec, Executor, SoA1, merge_by_key, reduce_by_key};

struct Output<B: cubecl::prelude::Runtime> {
    sku: DeviceVec<B, u32>,
    delta: DeviceVec<B, f32>,
}

fn solve<B>(
    exec: &Executor<B>,
    left_sku: DeviceVec<B, u32>,
    left_delta: DeviceVec<B, f32>,
    right_sku: DeviceVec<B, u32>,
    right_delta: DeviceVec<B, f32>,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let (SoA1(sku), SoA1(delta)) = merge_by_key(
        exec,
        SoA1(left_sku.slice(..)),
        SoA1(left_delta.slice(..)),
        SoA1(right_sku.slice(..)),
        SoA1(right_delta.slice(..)),
        common::LessU32,
    )?;
    let (SoA1(sku), SoA1(delta)) = reduce_by_key(
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
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
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
