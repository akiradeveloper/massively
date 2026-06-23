//! # Problem
//!
//! Given item rows and a target rank for each row, place each item into its
//! ranked output slot.
//!
//! # Task
//!
//! Implement `solve(item_id, score, rank_index, len) -> ranked rows`.
//!
//! # GPU Algorithm
//!
//! 1. Treat item columns as carried SoA values.
//! 2. Scatter rows by rank index.

mod common;

use massively::{DeviceVec, Executor, SoA2, Wgpu, scatter};

struct Output<B: massively::Backend> {
    item_id: DeviceVec<B, u32>,
    score: DeviceVec<B, f32>,
}

fn solve<B>(
    exec: &Executor<B>,
    item_id: DeviceVec<B, u32>,
    score: DeviceVec<B, f32>,
    rank_index: DeviceVec<B, u32>,
    len: usize,
) -> common::Result<Output<B>>
where
    B: massively::Backend,
{
    let (item_id, score) = scatter(
        exec,
        SoA2(item_id.slice(..), score.slice(..)),
        rank_index.slice(..),
        len,
        (0_u32, 0.0_f32),
    )?;
    Ok(Output { item_id, score })
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let output = solve(
        &exec,
        exec.to_device(&[10, 20, 30])?,
        exec.to_device(&[1.5, 9.0, 3.0])?,
        exec.to_device(&[2, 0, 1])?,
        4,
    )?;
    assert_eq!(exec.to_host(&output.item_id)?, vec![20, 30, 10, 0]);
    assert_eq!(exec.to_host(&output.score)?, vec![9.0, 3.0, 1.5, 0.0]);
    Ok(())
}
