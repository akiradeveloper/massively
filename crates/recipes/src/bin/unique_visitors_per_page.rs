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
use massively::{DeviceVec, Executor, SoA1, SoA2, Wgpu, reduce_by_key, sort, unique};

struct LessVisitPair;

#[cubecl::cube]
impl<B> BinaryPredicateOp<B, (u32, u32)> for LessVisitPair
where
    B: massively::Backend,
{
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
    }
}

struct EqualVisitPair;

#[cubecl::cube]
impl<B> BinaryPredicateOp<B, (u32, u32)> for EqualVisitPair
where
    B: massively::Backend,
{
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1
    }
}

struct Output<B: massively::Backend> {
    page_id: DeviceVec<B, u32>,
    unique_count: DeviceVec<B, u32>,
}

fn solve<B>(
    exec: &Executor<B>,
    page_id: DeviceVec<B, u32>,
    user_id: DeviceVec<B, u32>,
) -> common::Result<Output<B>>
where
    B: massively::Backend,
{
    let (page_id, user_id) = sort(
        exec,
        SoA2(page_id.slice(..), user_id.slice(..)),
        LessVisitPair,
    )?;
    let (page_id, _user_id) = unique(
        exec,
        SoA2(page_id.slice(..), user_id.slice(..)),
        EqualVisitPair,
    )?;
    let ones = exec.filled(page_id.len(), 1_u32)?;
    let ((page_id,), (unique_count,)) = reduce_by_key(
        exec,
        SoA1(page_id.slice(..)),
        SoA1(ones.slice(..)),
        common::EqualU32,
        (0_u32,),
        common::SumU32,
    )?;
    Ok(Output {
        page_id,
        unique_count,
    })
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let output = solve(
        &exec,
        exec.to_device(&[2, 1, 1, 2, 1, 2])?,
        exec.to_device(&[7, 5, 5, 7, 8, 9])?,
    )?;
    assert_eq!(exec.to_host(&output.page_id)?, vec![1, 2]);
    assert_eq!(exec.to_host(&output.unique_count)?, vec![2, 2]);
    Ok(())
}
