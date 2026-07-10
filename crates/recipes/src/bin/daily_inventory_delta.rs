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

use massively::{DeviceVec, Executor, MIndex, merge_by_key, reduce_by_key};

struct Output<B: cubecl::prelude::Runtime> {
    sku: DeviceVec<B, u32>,
    delta: DeviceVec<B, f32>,
    len: MIndex,
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
    let merged_len = (left_sku.len() + right_sku.len()) as usize;
    let sku = exec.to_device(&vec![0_u32; merged_len]);
    let delta = exec.to_device(&vec![0.0_f32; merged_len]);
    merge_by_key(
        &exec,
        left_sku.slice(..),
        left_delta.slice(..),
        right_sku.slice(..),
        right_delta.slice(..),
        common::LessU32,
        sku.slice_mut(..),
        delta.slice_mut(..),
    )?;
    let out_sku = exec.to_device(&vec![0_u32; merged_len]);
    let out_delta = exec.to_device(&vec![0.0_f32; merged_len]);
    let len = reduce_by_key(
        &exec,
        sku.slice(..),
        delta.slice(..),
        common::EqualU32,
        0.0_f32,
        common::SumF32,
        out_sku.slice_mut(..),
        out_delta.slice_mut(..),
    )?;
    Ok(Output {
        sku: out_sku,
        delta: out_delta,
        len,
    })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[1, 2, 4]),
        exec.to_device(&[5.0, -1.0, 9.0]),
        exec.to_device(&[2, 3, 4]),
        exec.to_device(&[2.0, 7.0, -3.0]),
    )?;
    assert_eq!(
        exec.to_host(&output.sku.slice(..output.len as usize))?,
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        exec.to_host(&output.delta.slice(..output.len as usize))?,
        vec![5.0, 1.0, 7.0, 6.0]
    );
    Ok(())
}
