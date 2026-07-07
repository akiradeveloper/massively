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

use massively::{DeviceVec, Executor, MIndex, Zip1, reverse, sort_by_key};

struct Output<B: cubecl::prelude::Runtime> {
    player_id: DeviceVec<B, u32>,
    score: DeviceVec<B, f32>,
}

fn solve<B>(
    exec: &Executor<B>,
    player_id: DeviceVec<B, u32>,
    score: DeviceVec<B, f32>,
    k: MIndex,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let len = player_id.len();
    let sorted_score = exec.full(len, 0.0_f32)?;
    let sorted_player_id = exec.full(len, 0_u32)?;
    sort_by_key(
        exec,
        Zip1(score.slice(..)),
        Zip1(player_id.slice(..)),
        common::LessF32,
        Zip1(sorted_score.slice_mut(..)),
        Zip1(sorted_player_id.slice_mut(..)),
    )?;
    let score = exec.full(len, 0.0_f32)?;
    let player_id = exec.full(len, 0_u32)?;
    reverse(
        exec,
        Zip1(sorted_score.slice(..)),
        Zip1(score.slice_mut(..)),
    )?;
    reverse(
        exec,
        Zip1(sorted_player_id.slice(..)),
        Zip1(player_id.slice_mut(..)),
    )?;
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
