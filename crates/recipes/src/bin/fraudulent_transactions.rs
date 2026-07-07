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
//! 1. Transform transaction rows into suspicious flags.
//! 2. Compact suspicious rows.
//! 3. Partition suspicious rows by the high-risk predicate.

mod common;

use cubecl::prelude::*;
use massively::op::{PredicateOp, UnaryOp};
use massively::prelude::*;
use massively::{DeviceVec, Executor, Zip3, copy_where, partition, transform};

struct SuspiciousTransaction;

#[cubecl::cube]
impl<B> UnaryOp<B, (u32, f32, u32)> for SuspiciousTransaction
where
    B: cubecl::prelude::Runtime,
{
    type Output = (u32,);

    fn apply(input: (u32, f32, u32)) -> (u32,) {
        if input.1 >= 100.0 || input.2 >= 80_u32 {
            (1_u32,)
        } else {
            (0_u32,)
        }
    }
}

struct HighRiskTransaction;

#[cubecl::cube]
impl<B> PredicateOp<B, (u32, f32, u32)> for HighRiskTransaction
where
    B: cubecl::prelude::Runtime,
{
    fn apply(input: (u32, f32, u32)) -> bool {
        input.1 >= 200.0 || input.2 >= 90_u32
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
    let flag = exec.full(account_id.len(), 0_u32)?;
    transform(
        exec,
        Zip3(account_id.slice(..), amount.slice(..), risk_score.slice(..)),
        SuspiciousTransaction,
        Zip1(flag.slice_mut(..)),
    )?;
    let suspicious_account_id = exec.full(account_id.len(), 0_u32)?;
    let suspicious_amount = exec.full(amount.len(), 0.0_f32)?;
    let suspicious_risk_score = exec.full(risk_score.len(), 0_u32)?;
    let suspicious_len = copy_where(
        exec,
        Zip3(account_id.slice(..), amount.slice(..), risk_score.slice(..)),
        flag.slice(..),
        Zip3(
            suspicious_account_id.slice_mut(..),
            suspicious_amount.slice_mut(..),
            suspicious_risk_score.slice_mut(..),
        ),
    )?;
    let account_id = exec.full(suspicious_len, 0_u32)?;
    let amount = exec.full(suspicious_len, 0.0_f32)?;
    let risk_score = exec.full(suspicious_len, 0_u32)?;
    let split = partition(
        exec,
        Zip3(
            suspicious_account_id.slice(..suspicious_len),
            suspicious_amount.slice(..suspicious_len),
            suspicious_risk_score.slice(..suspicious_len),
        ),
        HighRiskTransaction,
        Zip3(
            account_id.slice_mut(..),
            amount.slice_mut(..),
            risk_score.slice_mut(..),
        ),
    )?;
    Ok(Output {
        high_risk: Group {
            account_id: exec.to_device(&exec.to_host(&account_id.slice(..split))?)?,
            amount: exec.to_device(&exec.to_host(&amount.slice(..split))?)?,
            risk_score: exec.to_device(&exec.to_host(&risk_score.slice(..split))?)?,
        },
        review_needed: Group {
            account_id: exec.to_device(&exec.to_host(&account_id.slice(split..suspicious_len))?)?,
            amount: exec.to_device(&exec.to_host(&amount.slice(split..suspicious_len))?)?,
            risk_score: exec.to_device(&exec.to_host(&risk_score.slice(split..suspicious_len))?)?,
        },
    })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[1, 2, 3, 4])?,
        exec.to_device(&[50.0, 150.0, 250.0, 40.0])?,
        exec.to_device(&[20, 85, 70, 95])?,
    )?;
    assert_eq!(exec.to_host(&output.high_risk.account_id)?, vec![3, 4]);
    assert_eq!(exec.to_host(&output.high_risk.amount)?, vec![250.0, 40.0]);
    assert_eq!(exec.to_host(&output.high_risk.risk_score)?, vec![70, 95]);
    assert_eq!(exec.to_host(&output.review_needed.account_id)?, vec![2]);
    assert_eq!(exec.to_host(&output.review_needed.amount)?, vec![150.0]);
    assert_eq!(exec.to_host(&output.review_needed.risk_score)?, vec![85]);
    Ok(())
}
