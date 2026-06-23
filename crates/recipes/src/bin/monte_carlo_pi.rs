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
//! 1. Generate sample indices.
//! 2. Map each index to a deterministic point and then to an inside flag.
//! 3. Reduce the flags.
//! 4. Convert the count to a pi estimate on the host.

mod common;

use cubecl::prelude::*;
use massively::op::UnaryOp;
use massively::runtime::op::TabulateOp;
use massively::{Executor, SoA1, Wgpu, reduce, transform};

struct Index;

#[cubecl::cube]
impl TabulateOp<Wgpu, u32> for Index {
    fn apply(index: u32) -> u32 {
        index
    }
}

struct InsideQuarterCircle;

#[cubecl::cube]
impl<B> UnaryOp<B, (u32,)> for InsideQuarterCircle
where
    B: massively::Backend,
{
    type Output = (u32,);

    fn apply(input: (u32,)) -> (u32,) {
        let a = (input.0 * 37_u32 + 17_u32) % 100_u32;
        let b = (input.0 * 57_u32 + 31_u32) % 100_u32;
        let x = (a as f32) / 100.0;
        let y = (b as f32) / 100.0;
        if x * x + y * y <= 1.0 {
            (1_u32,)
        } else {
            (0_u32,)
        }
    }
}

fn solve(exec: &Executor<Wgpu>, samples: usize) -> common::Result<f32> {
    let indices = exec.tabulate(samples, Index)?;
    let (inside,) = transform(exec, SoA1(indices.slice(..)), InsideQuarterCircle)?;
    let (hits,) = reduce(exec, SoA1(inside.slice(..)), (0_u32,), common::SumU32)?;
    Ok(4.0 * hits as f32 / samples as f32)
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let pi = solve(&exec, 10_000)?;
    common::assert_f32_near(pi, 3.14, 0.12);
    Ok(())
}
