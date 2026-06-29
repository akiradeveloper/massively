//! # Problem
//!
//! Given player scores, return the top `k` rows by descending score.
//!
//! # Task
//!
//! Implement `solve(player_id, score, k) -> top rows`.
//!
//! # GPU Algorithm
//!
//! 1. Sort `(score, player_id)` by score ascending.
//! 2. Reverse both columns.
//! 3. Read the first `k` rows.

mod common;

use massively::{DeviceVec, Executor, SoA1, reverse, sort_by_key};

struct Output<B: cubecl::prelude::Runtime> {
    player_id: DeviceVec<B, u32>,
    score: DeviceVec<B, f32>,
}

fn solve<B>(
    exec: &Executor<B>,
    player_id: DeviceVec<B, u32>,
    score: DeviceVec<B, f32>,
    k: usize,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let (SoA1(score), SoA1(player_id)) = sort_by_key(
        exec,
        SoA1(score.slice(..)),
        SoA1(player_id.slice(..)),
        common::LessF32,
    )?;
    let SoA1(score) = reverse(exec, SoA1(score.slice(..)))?;
    let SoA1(player_id) = reverse(exec, SoA1(player_id.slice(..)))?;
    Ok(Output {
        player_id: exec.to_device(&exec.to_host(&player_id.slice(..k))?)?,
        score: exec.to_device(&exec.to_host(&score.slice(..k))?)?,
    })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[10, 20, 30, 40])?,
        exec.to_device(&[7.0, 10.0, 3.0, 9.0])?,
        2,
    )?;
    assert_eq!(exec.to_host(&output.player_id)?, vec![20, 40]);
    assert_eq!(exec.to_host(&output.score)?, vec![10.0, 9.0]);
    Ok(())
}
