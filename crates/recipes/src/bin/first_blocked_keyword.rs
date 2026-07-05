//! # Problem
//!
//! Given token ids and blocked keyword ids, find the first blocked token.
//!
//! # Task
//!
//! Implement `solve(token_id, blocked_keyword_id) -> Option<MIndex>`.
//!
//! # GPU Algorithm
//!
//! 1. Use `find_first_of`.

mod common;

use massively::{DeviceVec, Executor, MIndex, Zip1, find_first_of};

fn solve<B>(
    exec: &Executor<B>,
    token_id: DeviceVec<B, u32>,
    blocked_keyword_id: DeviceVec<B, u32>,
) -> common::Result<Option<MIndex>>
where
    B: cubecl::prelude::Runtime,
{
    find_first_of(
        exec,
        Zip1(token_id.slice(..)),
        Zip1(blocked_keyword_id.slice(..)),
        common::EqualU32,
    )
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let index = solve(
        &exec,
        exec.to_device(&[7, 9, 11, 13])?,
        exec.to_device(&[4, 11])?,
    )?;
    assert_eq!(index, Some(2));
    Ok(())
}
