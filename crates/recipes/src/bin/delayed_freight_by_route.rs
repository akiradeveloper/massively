//! # Problem
//!
//! Given package rows `(route_id, weight, delayed)`, compute total delayed
//! weight per route.
//!
//! # Task
//!
//! Implement `solve(route_id, weight, delayed) -> (route_id, delayed_weight)`.
//!
//! # GPU Algorithm
//!
//! 1. Compact delayed rows with `copy_where`.
//! 2. Sort delayed weights by route id with `sort_by_key`.
//! 3. Sum weights per route with `reduce_by_key`.

mod common;

use massively::{DeviceVec, Executor, Zip1, Zip2, copy_where, reduce_by_key, sort_by_key};

struct Output<B: cubecl::prelude::Runtime> {
    route_id: DeviceVec<B, u32>,
    delayed_weight: DeviceVec<B, f32>,
}

fn solve<B>(
    exec: &Executor<B>,
    route_id: DeviceVec<B, u32>,
    weight: DeviceVec<B, f32>,
    delayed: DeviceVec<B, u32>,
) -> common::Result<Output<B>>
where
    B: cubecl::prelude::Runtime,
{
    let delayed_route = exec.full(route_id.len(), 0_u32)?;
    let delayed_weight = exec.full(weight.len(), 0.0_f32)?;
    let delayed_len = copy_where(
        exec,
        Zip2(route_id.slice(..), weight.slice(..)),
        delayed.slice(..),
        Zip2(delayed_route.slice_mut(..), delayed_weight.slice_mut(..)),
    )?;
    let sorted_route = exec.full(delayed_len, 0_u32)?;
    let sorted_weight = exec.full(delayed_len, 0.0_f32)?;
    sort_by_key(
        exec,
        Zip1(delayed_route.slice(..delayed_len)),
        Zip1(delayed_weight.slice(..delayed_len)),
        common::LessU32,
        Zip1(sorted_route.slice_mut(..)),
        Zip1(sorted_weight.slice_mut(..)),
    )?;
    let route_id = exec.full(delayed_len, 0_u32)?;
    let delayed_weight = exec.full(delayed_len, 0.0_f32)?;
    let len = reduce_by_key(
        exec,
        Zip1(sorted_route.slice(..)),
        Zip1(sorted_weight.slice(..)),
        common::EqualU32,
        (0.0_f32,),
        common::SumF32,
        Zip1(route_id.slice_mut(..)),
        Zip1(delayed_weight.slice_mut(..)),
    )?;
    Ok(Output {
        route_id: exec.to_device(&exec.to_host(&route_id.slice(..len))?)?,
        delayed_weight: exec.to_device(&exec.to_host(&delayed_weight.slice(..len))?)?,
    })
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let output = solve(
        &exec,
        exec.to_device(&[2, 1, 2, 1, 3])?,
        exec.to_device(&[5.0, 7.0, 11.0, 13.0, 17.0])?,
        exec.to_device(&[1, 0, 1, 1, 0])?,
    )?;
    assert_eq!(exec.to_host(&output.route_id)?, vec![1, 2]);
    assert_eq!(exec.to_host(&output.delayed_weight)?, vec![13.0, 16.0]);
    Ok(())
}
