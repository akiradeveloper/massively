//! # Problem
//!
//! Count how many times each category id appears.
//!
//! # Task
//!
//! Implement `solve(category_id) -> (category_id, count)`.
//!
//! # GPU Algorithm
//!
//! 1. Sort category ids.
//! 2. Create a column of ones.
//! 3. Reduce by category id.

use super::common;

use massively::{DeviceVec, Executor, MIndex, vector::reduce_by_key, vector::sort};

struct Output<B: cubecl::prelude::Runtime> {
    category_id: DeviceVec<B, u32>,
    count: DeviceVec<B, u32>,
    len: MIndex,
}

fn solve<B>(exec: &Executor<B>, category_id: DeviceVec<B, u32>) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let len = category_id.len() as usize;
    let sorted = exec.to_device(&vec![0_u32; len]);
    sort(
        &exec,
        category_id.slice(..),
        common::LessU32,
        sorted.slice_mut(..),
    )?;
    let ones = exec.full(sorted.len(), 1_u32)?;
    let category_id = exec.to_device(&vec![0_u32; len]);
    let count = exec.to_device(&vec![0_u32; len]);
    let len = reduce_by_key(
        &exec,
        sorted.slice(..),
        ones.slice(..),
        common::EqualU32,
        0_u32,
        common::SumU32,
        category_id.slice_mut(..),
        count.slice_mut(..),
    )?;
    Ok(Output {
        category_id,
        count,
        len,
    })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(&exec, exec.to_device(&[2, 1, 2, 3, 1, 2]))?;
    assert_eq!(
        exec.to_host(&output.category_id.slice(..output.len as usize))?,
        vec![1, 2, 3]
    );
    assert_eq!(
        exec.to_host(&output.count.slice(..output.len as usize))?,
        vec![2, 3, 1]
    );
    Ok(())
}
