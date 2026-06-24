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

mod common;

use massively::{DeviceVec, Executor, SoA1, reduce_by_key, sort};

struct Output<B: cubecl::prelude::Runtime> {
    category_id: DeviceVec<B, u32>,
    count: DeviceVec<B, u32>,
}

fn solve<B>(exec: &Executor<B>, category_id: DeviceVec<B, u32>) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let (sorted,) = sort(exec, SoA1(category_id.slice(..)), common::LessU32)?;
    let ones = exec.filled(sorted.len(), 1_u32)?;
    let ((category_id,), (count,)) = reduce_by_key(
        exec,
        SoA1(sorted.slice(..)),
        SoA1(ones.slice(..)),
        common::EqualU32,
        (0_u32,),
        common::SumU32,
    )?;
    Ok(Output { category_id, count })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(&exec, exec.to_device(&[2, 1, 2, 3, 1, 2])?)?;
    assert_eq!(exec.to_host(&output.category_id)?, vec![1, 2, 3]);
    assert_eq!(exec.to_host(&output.count)?, vec![2, 3, 1]);
    Ok(())
}
