//! # Problem
//!
//! Estimate pi by throwing deterministic pseudo-random points into the unit
//! square and counting how many land inside the unit quarter-circle.
//!
//! # Task
//!
//! Implement `solve(samples) -> pi_estimate`.
//!
//! # GPU Algorithm
//!
//! 1. Generate deterministic pseudo-random x/y columns on the GPU.
//! 2. Map each point to an inside flag.
//! 3. Reduce the flags.
//! 4. Convert the count to a pi estimate on the host.

use super::common;

use cubecl::prelude::*;
use massively::op::UnaryOp;
use massively::{Executor, MIndex, vector::reduce, vector::transform, zip2};

const SCALE: u32 = 1_000_000;

struct InsideQuarterCircle;
#[cubecl::cube]
impl UnaryOp<(u32, u32)> for InsideQuarterCircle {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        let x = (input.0 as f32) / SCALE as f32;
        let y = (input.1 as f32) / SCALE as f32;
        if x * x + y * y <= 1.0 { 1_u32 } else { 0_u32 }
    }
}

fn solve<B>(exec: &Executor<B>, samples: MIndex) -> common::Result<f32>
where
    B: cubecl::prelude::Runtime,
{
    let pseudo_random = |seed: u32| {
        (0..samples)
            .map(|index| {
                index
                    .wrapping_mul(1_664_525)
                    .wrapping_add(seed)
                    .wrapping_add(1_013_904_223)
                    % SCALE
            })
            .collect::<Vec<_>>()
    };
    let x = exec.to_device(&pseudo_random(0x1234_5678));
    let y = exec.to_device(&pseudo_random(0x8765_4321));

    let inside = exec.full(samples as usize, 0_u32)?;
    transform(
        &exec,
        zip2(x.slice(..), y.slice(..)),
        InsideQuarterCircle,
        inside.slice_mut(..),
    )?;
    let hits = reduce(&exec, inside.slice(..), 0_u32, common::SumU32)?;
    Ok(4.0 * hits as f32 / samples as f32)
}

fn main() -> common::Result {
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::Cpu);
    let pi = solve(&exec, 10_000)?;
    common::assert_f32_near(pi, 3.14, 0.12);
    Ok(())
}
