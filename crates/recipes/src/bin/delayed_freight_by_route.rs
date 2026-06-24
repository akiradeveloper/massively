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
//! 1. Compact delayed rows with `copy_if`.
//! 2. Sort delayed weights by route id with `sort_by_key`.
//! 3. Sum weights per route with `reduce_by_key`.

mod common;

use massively::{DeviceVec, Executor, SoA1, SoA2, copy_if, reduce_by_key, sort_by_key};

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
    let (delayed_route, delayed_weight) = copy_if(
        exec,
        SoA2(route_id.slice(..), weight.slice(..)),
        delayed.slice(..),
    )?;
    let ((sorted_route,), (sorted_weight,)) = sort_by_key(
        exec,
        SoA1(delayed_route.slice(..)),
        SoA1(delayed_weight.slice(..)),
        common::LessU32,
    )?;
    let ((route_id,), (delayed_weight,)) = reduce_by_key(
        exec,
        SoA1(sorted_route.slice(..)),
        SoA1(sorted_weight.slice(..)),
        common::EqualU32,
        (0.0_f32,),
        common::SumF32,
    )?;
    Ok(Output {
        route_id,
        delayed_weight,
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
