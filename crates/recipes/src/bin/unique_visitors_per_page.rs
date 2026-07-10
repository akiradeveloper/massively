//! # Problem
//!
//! Given visit rows `(page_id, user_id)`, count unique users per page.
//!
//! # Task
//!
//! Implement `solve(page_id, user_id) -> (page_id, unique_count)`.
//!
//! # GPU Algorithm
//!
//! 1. Sort tuple rows `(page_id, user_id)`.
//! 2. Remove adjacent duplicate pairs.
//! 3. Reduce page ids with a column of ones.

mod common;

use cubecl::prelude::*;
use massively::op::BinaryPredicateOp;
use massively::{DeviceVec, Executor, reduce_by_key, sort, unique, zip2};

struct LessVisitPair;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32)> for LessVisitPair {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
    }
}

struct EqualVisitPair;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32)> for EqualVisitPair {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1
    }
}

struct Output<B: cubecl::prelude::Runtime> {
    page_id: DeviceVec<B, u32>,
    unique_count: DeviceVec<B, u32>,
}

fn solve<B>(
    exec: &Executor<B>,
    page_id: DeviceVec<B, u32>,
    user_id: DeviceVec<B, u32>,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let len = page_id.len();
    let sorted_page_id = exec.full(len as usize, 0_u32)?;
    let sorted_user_id = exec.full(len as usize, 0_u32)?;
    sort(
        &exec,
        zip2(page_id.slice(..), user_id.slice(..)),
        LessVisitPair,
        zip2(sorted_page_id.slice_mut(..), sorted_user_id.slice_mut(..)),
    )?;
    let page_id = exec.full(len as usize, 0_u32)?;
    let user_id = exec.full(len as usize, 0_u32)?;
    let unique_len = unique(
        &exec,
        zip2(sorted_page_id.slice(..), sorted_user_id.slice(..)),
        EqualVisitPair,
        zip2(page_id.slice_mut(..), user_id.slice_mut(..)),
    )?;
    let ones = exec.full(unique_len as usize, 1_u32)?;
    let out_page_id = exec.full(unique_len as usize, 0_u32)?;
    let unique_count = exec.full(unique_len as usize, 0_u32)?;
    let len = reduce_by_key(
        &exec,
        page_id.slice(..unique_len as usize),
        ones.slice(..),
        common::EqualU32,
        0_u32,
        common::SumU32,
        out_page_id.slice_mut(..),
        unique_count.slice_mut(..),
    )?;
    Ok(Output {
        page_id: exec.to_device(&exec.to_host(&out_page_id.slice(..len as usize))?),
        unique_count: exec.to_device(&exec.to_host(&unique_count.slice(..len as usize))?),
    })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[2, 1, 1, 2, 1, 2]),
        exec.to_device(&[7, 5, 5, 7, 8, 9]),
    )?;
    assert_eq!(exec.to_host(&output.page_id)?, vec![1, 2]);
    assert_eq!(exec.to_host(&output.unique_count)?, vec![2, 2]);
    Ok(())
}
