use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{BinaryPredicateOp, Executor, PredicateOp, ReductionOp, UnaryOp};
use oracle_ref as oracle;

const DEFAULT_SCALE_LEN: usize = 10_000_000;

pub struct MaxU32;
pub struct NonZero;
pub struct IdentityU32;
pub struct EqualU32;
pub struct LessU32;

#[cubecl::cube]
impl ReductionOp<u32> for MaxU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs.max(rhs)
    }
}

impl oracle::op::ReductionOp<u32> for MaxU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs.max(rhs)
    }
}

#[cubecl::cube]
impl PredicateOp<u32> for NonZero {
    fn apply(input: u32) -> bool {
        input != 0u32
    }
}

impl oracle::op::PredicateOp<u32> for NonZero {
    fn apply(input: u32) -> bool {
        input != 0
    }
}

#[cubecl::cube]
impl UnaryOp<u32> for IdentityU32 {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        input
    }
}

impl oracle::op::UnaryOp<u32> for IdentityU32 {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        input
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

impl oracle::op::BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

impl oracle::op::BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

pub fn exec() -> Executor<WgpuRuntime> {
    Executor::new(WgpuDevice::DefaultDevice)
}

pub fn lazify<Input>(
    input: Input,
) -> massively::lazy::Transform<
    massively::lazy::Permute<Input, massively::lazy::Taken<massively::lazy::Counting>>,
    massively::op::Identity,
>
where
    Input: massively::MIter<WgpuRuntime>,
{
    let len = input.len().unwrap();
    massively::lazy::identity(massively::lazy::permute(
        input,
        massively::lazy::counting(0).take(len),
    ))
}

pub fn scale_len() -> usize {
    std::env::var("MASSIVELY_SCALE_LEN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SCALE_LEN)
}

pub fn scale_input() -> Vec<u32> {
    (0..scale_len())
        .map(|index| ((index as u64 * 1_103_515_245 + 12_345) % 1_000_003) as u32)
        .collect()
}

pub fn scale_other() -> Vec<u32> {
    (0..scale_len())
        .map(|index| ((index as u64 * 1_664_525 + 1_013_904_223) % 1_000_003) as u32)
        .collect()
}

pub fn flags_for(input: &[u32]) -> Vec<u32> {
    input.iter().map(|value| value & 1).collect()
}

pub fn indices_for(len: usize) -> Vec<u32> {
    (0..len).rev().map(|index| index as u32).collect()
}
