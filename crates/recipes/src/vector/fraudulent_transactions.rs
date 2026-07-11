//! # Problem
//!
//! Return suspicious transactions and partition them into high-risk and
//! review-needed groups.
//!
//! # Task
//!
//! Implement `solve(account_id, amount, risk_score) -> groups`.
//!
//! # GPU Algorithm
//!
//! 1. Build a lazy suspicious stencil.
//! 2. Compact suspicious rows.
//! 3. Partition suspicious rows by the high-risk predicate.

use super::common;

use cubecl::prelude::*;
use massively::op::{PredicateOp, UnaryOp};
use massively::{DeviceVec, Executor, lazy, vector::copy_where, vector::partition, zip3};

struct SuspiciousTransaction;

#[cubecl::cube]
impl UnaryOp<((u32, f32), u32)> for SuspiciousTransaction {
    type Output = u32;

    fn apply(input: ((u32, f32), u32)) -> u32 {
        if input.0.1 >= 100.0 || input.1 >= 80_u32 {
            1u32
        } else {
            0u32
        }
    }
}

struct HighRiskTransaction;

#[cubecl::cube]
impl PredicateOp<((u32, f32), u32)> for HighRiskTransaction {
    fn apply(input: ((u32, f32), u32)) -> bool {
        input.0.1 >= 200.0 || input.1 >= 90_u32
    }
}

struct Group<B: cubecl::prelude::Runtime> {
    account_id: DeviceVec<B, u32>,
    amount: DeviceVec<B, f32>,
    risk_score: DeviceVec<B, u32>,
}

struct Output<B: cubecl::prelude::Runtime> {
    high_risk: Group<B>,
    review_needed: Group<B>,
}

fn solve<B>(
    exec: &Executor<B>,
    account_id: DeviceVec<B, u32>,
    amount: DeviceVec<B, f32>,
    risk_score: DeviceVec<B, u32>,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let suspicious_account_id = exec.full(account_id.len(), 0_u32)?;
    let suspicious_amount = exec.full(amount.len(), 0.0_f32)?;
    let suspicious_risk_score = exec.full(risk_score.len(), 0_u32)?;
    let suspicious_len = copy_where(
        &exec,
        zip3(account_id.slice(..), amount.slice(..), risk_score.slice(..)),
        lazy::transform(
            zip3(account_id.slice(..), amount.slice(..), risk_score.slice(..)),
            SuspiciousTransaction,
        ),
        zip3(
            suspicious_account_id.slice_mut(..),
            suspicious_amount.slice_mut(..),
            suspicious_risk_score.slice_mut(..),
        ),
    )?;
    let account_id = exec.full(suspicious_len as usize, 0_u32)?;
    let amount = exec.full(suspicious_len as usize, 0.0_f32)?;
    let risk_score = exec.full(suspicious_len as usize, 0_u32)?;
    let split = partition(
        &exec,
        zip3(
            suspicious_account_id.slice(..suspicious_len as usize),
            suspicious_amount.slice(..suspicious_len as usize),
            suspicious_risk_score.slice(..suspicious_len as usize),
        ),
        HighRiskTransaction,
        zip3(
            account_id.slice_mut(..),
            amount.slice_mut(..),
            risk_score.slice_mut(..),
        ),
    )?;
    Ok(Output {
        high_risk: Group {
            account_id: exec.to_device(&exec.to_host(&account_id.slice(..split as usize))?),
            amount: exec.to_device(&exec.to_host(&amount.slice(..split as usize))?),
            risk_score: exec.to_device(&exec.to_host(&risk_score.slice(..split as usize))?),
        },
        review_needed: Group {
            account_id: exec.to_device(
                &exec.to_host(&account_id.slice(split as usize..suspicious_len as usize))?,
            ),
            amount: exec
                .to_device(&exec.to_host(&amount.slice(split as usize..suspicious_len as usize))?),
            risk_score: exec.to_device(
                &exec.to_host(&risk_score.slice(split as usize..suspicious_len as usize))?,
            ),
        },
    })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[1, 2, 3, 4]),
        exec.to_device(&[50.0, 150.0, 250.0, 40.0]),
        exec.to_device(&[20, 85, 70, 95]),
    )?;
    assert_eq!(exec.to_host(&output.high_risk.account_id)?, vec![3, 4]);
    assert_eq!(exec.to_host(&output.high_risk.amount)?, vec![250.0, 40.0]);
    assert_eq!(exec.to_host(&output.high_risk.risk_score)?, vec![70, 95]);
    assert_eq!(exec.to_host(&output.review_needed.account_id)?, vec![2]);
    assert_eq!(exec.to_host(&output.review_needed.amount)?, vec![150.0]);
    assert_eq!(exec.to_host(&output.review_needed.risk_score)?, vec![85]);
    Ok(())
}
