//! # Problem
//!
//! Given support tickets with required slot counts, compute each ticket's
//! starting offset in a flattened work queue.
//!
//! # Task
//!
//! Implement `solve(team_id, slot_count) -> offsets and total_slots`.
//!
//! # GPU Algorithm
//!
//! 1. Exclusive-scan slot counts into offsets.
//! 2. Reduce slot counts into the total allocation size.

mod common;

use massively::{DeviceVec, Executor, Zip1, exclusive_scan, reduce};

struct Output<B: cubecl::prelude::Runtime> {
    offset: DeviceVec<B, u32>,
    total_slots: u32,
}

fn solve<B>(
    exec: &Executor<B>,
    _team_id: DeviceVec<B, u32>,
    slot_count: DeviceVec<B, u32>,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let offset = exec.full(slot_count.len(), 0_u32)?;
    exclusive_scan(
        exec,
        Zip1(slot_count.slice(..)),
        (0_u32,),
        common::SumU32,
        Zip1(offset.slice_mut(..)),
    )?;
    let (total_slots,) = reduce(exec, Zip1(slot_count.slice(..)), (0_u32,), common::SumU32)?;
    Ok(Output {
        offset,
        total_slots,
    })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[0, 0, 1, 1])?,
        exec.to_device(&[3, 2, 5, 1])?,
    )?;
    assert_eq!(exec.to_host(&output.offset)?, vec![0, 3, 5, 10]);
    assert_eq!(output.total_slots, 11);
    Ok(())
}
