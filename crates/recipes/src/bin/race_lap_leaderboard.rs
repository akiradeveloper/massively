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

use massively::{DeviceVec, Executor, MIndex, Zip1, reduce_by_key, sort_by_key};

struct Output<B: cubecl::prelude::Runtime> {
    racer_id: DeviceVec<B, u32>,
    total_time_ms: DeviceVec<B, u32>,
    len: MIndex,
}

fn solve<B>(
    exec: &Executor<B>,
    racer_id: DeviceVec<B, u32>,
    lap_time_ms: DeviceVec<B, u32>,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let len = racer_id.len() as usize;
    let sorted_racer_id = exec.to_device(&vec![0_u32; len])?;
    let sorted_lap_time_ms = exec.to_device(&vec![0_u32; len])?;
    sort_by_key(
        exec,
        Zip1(racer_id.slice(..)),
        Zip1(lap_time_ms.slice(..)),
        common::LessU32,
        Zip1(sorted_racer_id.slice_mut(..)),
        Zip1(sorted_lap_time_ms.slice_mut(..)),
    )?;
    let racer_id = exec.to_device(&vec![0_u32; len])?;
    let total_time_ms = exec.to_device(&vec![0_u32; len])?;
    let len = reduce_by_key(
        exec,
        Zip1(sorted_racer_id.slice(..)),
        Zip1(sorted_lap_time_ms.slice(..)),
        common::EqualU32,
        (0_u32,),
        common::SumU32,
        Zip1(racer_id.slice_mut(..)),
        Zip1(total_time_ms.slice_mut(..)),
    )?;
    let ranked_total_time_ms = exec.to_device(&vec![0_u32; len as usize])?;
    let ranked_racer_id = exec.to_device(&vec![0_u32; len as usize])?;
    sort_by_key(
        exec,
        Zip1(total_time_ms.slice(..len)),
        Zip1(racer_id.slice(..len)),
        common::LessU32,
        Zip1(ranked_total_time_ms.slice_mut(..)),
        Zip1(ranked_racer_id.slice_mut(..)),
    )?;
    Ok(Output {
        racer_id: ranked_racer_id,
        total_time_ms: ranked_total_time_ms,
        len,
    })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[2, 1, 2, 1, 3])?,
        exec.to_device(&[50, 40, 45, 42, 100])?,
    )?;
    assert_eq!(
        exec.to_host(&output.racer_id.slice(..output.len))?,
        vec![1, 2, 3]
    );
    assert_eq!(
        exec.to_host(&output.total_time_ms.slice(..output.len))?,
        vec![82, 95, 100]
    );
    Ok(())
}
