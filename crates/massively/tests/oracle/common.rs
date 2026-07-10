use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{BinaryPredicateOp, Executor, PredicateOp, ReductionOp, UnaryOp};
use oracle_ref as oracle;

pub const CASES: u32 = 6;
pub const MAX_LEN: usize = 96;

pub struct AddOne;
pub struct Sum;
pub struct Even;
pub struct Equal;
pub struct Less;

#[cubecl::cube]
impl UnaryOp<u32> for AddOne {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        input + 1u32
    }
}

impl oracle::op::UnaryOp<u32> for AddOne {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        input + 1
    }
}

#[cubecl::cube]
impl ReductionOp<u32> for Sum {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

impl oracle::op::ReductionOp<u32> for Sum {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

#[cubecl::cube]
impl PredicateOp<u32> for Even {
    fn apply(input: u32) -> bool {
        input % 2u32 == 0u32
    }
}

impl oracle::op::PredicateOp<u32> for Even {
    fn apply(input: u32) -> bool {
        input % 2 == 0
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

impl oracle::op::BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

impl oracle::op::BinaryPredicateOp<u32> for Less {
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

pub fn flags_for(input: &[u32]) -> Vec<u32> {
    input
        .iter()
        .enumerate()
        .map(|(index, value)| ((index as u32 + value) % 3 != 0) as u32)
        .collect()
}

pub fn indices_for(len: usize) -> Vec<u32> {
    (0..len).rev().map(|index| index as u32).collect()
}
