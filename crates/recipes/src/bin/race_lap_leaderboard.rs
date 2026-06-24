//! # Problem
//!
//! Compute total lap time per racer, then rank racers from fastest to slowest.
//!
//! # Task
//!
//! Implement `solve(racer_id, lap_time_ms) -> leaderboard`.
//!
//! # GPU Algorithm
//!
//! 1. Sort lap rows by racer id.
//! 2. Reduce lap times by racer id.
//! 3. Sort racers by total time.

mod common;

use massively::{DeviceVec, Executor, SoA1, reduce_by_key, sort_by_key};

struct Output<B: cubecl::prelude::Runtime> {
    racer_id: DeviceVec<B, u32>,
    total_time_ms: DeviceVec<B, u32>,
}

fn solve<B>(
    exec: &Executor<B>,
    racer_id: DeviceVec<B, u32>,
    lap_time_ms: DeviceVec<B, u32>,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let ((racer_id,), (lap_time_ms,)) = sort_by_key(
        exec,
        SoA1(racer_id.slice(..)),
        SoA1(lap_time_ms.slice(..)),
        common::LessU32,
    )?;
    let ((racer_id,), (total_time_ms,)) = reduce_by_key(
        exec,
        SoA1(racer_id.slice(..)),
        SoA1(lap_time_ms.slice(..)),
        common::EqualU32,
        (0_u32,),
        common::SumU32,
    )?;
    let ((total_time_ms,), (racer_id,)) = sort_by_key(
        exec,
        SoA1(total_time_ms.slice(..)),
        SoA1(racer_id.slice(..)),
        common::LessU32,
    )?;
    Ok(Output {
        racer_id,
        total_time_ms,
    })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[2, 1, 2, 1, 3])?,
        exec.to_device(&[50, 40, 45, 42, 100])?,
    )?;
    assert_eq!(exec.to_host(&output.racer_id)?, vec![1, 2, 3]);
    assert_eq!(exec.to_host(&output.total_time_ms)?, vec![82, 95, 100]);
    Ok(())
}
