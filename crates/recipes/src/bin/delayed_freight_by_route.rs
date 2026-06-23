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

use massively::{DeviceVec, Executor, SoA1, SoA2, Wgpu, copy_if, reduce_by_key, sort_by_key};

struct Output {
    route_id: DeviceVec<Wgpu, u32>,
    delayed_weight: DeviceVec<Wgpu, f32>,
}

fn solve(
    exec: &Executor<Wgpu>,
    route_id: DeviceVec<Wgpu, u32>,
    weight: DeviceVec<Wgpu, f32>,
    delayed: DeviceVec<Wgpu, u32>,
) -> common::Result<Output> {
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
    let exec = Executor::<Wgpu>::cpu();
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
