//! # Problem
//!
//! Given token ids and blocked keyword ids, find the first blocked token.
//!
//! # Task
//!
//! Implement `solve(token_id, blocked_keyword_id) -> Option<usize>`.
//!
//! # GPU Algorithm
//!
//! 1. Use `find_first_of`.

mod common;

use massively::{DeviceVec, Executor, SoA1, Wgpu, find_first_of};

fn solve(
    exec: &Executor<Wgpu>,
    token_id: DeviceVec<Wgpu, u32>,
    blocked_keyword_id: DeviceVec<Wgpu, u32>,
) -> common::Result<Option<usize>> {
    find_first_of(
        exec,
        SoA1(token_id.slice(..)),
        SoA1(blocked_keyword_id.slice(..)),
        common::EqualU32,
    )
}

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let index = solve(
        &exec,
        exec.to_device(&[7, 9, 11, 13])?,
        exec.to_device(&[4, 11])?,
    )?;
    assert_eq!(index, Some(2));
    Ok(())
}
