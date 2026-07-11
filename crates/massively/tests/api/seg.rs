use cubecl::prelude::*;
use massively::{
    BinaryPredicateOp, Error, Executor, MIter, MIterMut, ReductionOp, UnaryOp,
    seg::{
        ForEachSegment, Map, MapLikeExecutable, Reduce, ReduceLikeExecutable, Segmented,
        SegmentedMut, Unique, UniqueLikeExecutable,
    },
};

struct Transform;

#[cubecl::cube]
impl UnaryOp<u32> for Transform {
    type Output = u64;

    fn apply(value: u32) -> u64 {
        u64::cast_from(value)
    }
}

struct Equal;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

struct Add;

#[cubecl::cube]
impl ReductionOp<u32> for Add {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

#[allow(dead_code)]
fn map_interface<R, Input, InputOffsets, Output>(
    exec: &Executor<R>,
    input: Segmented<Input, InputOffsets>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R, Item = u32>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R, Item = u64>,
{
    ForEachSegment(Map(Transform)).run(exec, input, output)
}

#[allow(dead_code)]
fn unique_interface<R, Input, InputOffsets, Output, OutputOffsets>(
    exec: &Executor<R>,
    input: Segmented<Input, InputOffsets>,
    output: SegmentedMut<Output, OutputOffsets>,
) -> Result<u32, Error>
where
    R: Runtime,
    Input: MIter<R, Item = u32>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R, Item = u32>,
    OutputOffsets: MIterMut<R, Item = u32>,
{
    ForEachSegment(Unique(Equal)).run(exec, input, output)
}

#[allow(dead_code)]
fn reduce_interface<R, Input, InputOffsets, Output>(
    exec: &Executor<R>,
    input: Segmented<Input, InputOffsets>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R, Item = u32>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R, Item = u32>,
{
    ForEachSegment(Reduce(Add, 0)).run(exec, input, output)
}

#[test]
fn segmented_wrappers_expose_their_parts() {
    let input = Segmented::new([1_u32, 2], [0_u32, 2]);
    assert_eq!(input.values(), &[1, 2]);
    assert_eq!(input.offsets(), &[0, 2]);
    assert_eq!(input.into_parts(), ([1, 2], [0, 2]));

    let output = SegmentedMut::new([0_u32; 2], [0_u32; 2]);
    assert_eq!(output.values(), &[0, 0]);
    assert_eq!(output.offsets(), &[0, 0]);
    assert_eq!(output.into_parts(), ([0, 0], [0, 0]));
}
