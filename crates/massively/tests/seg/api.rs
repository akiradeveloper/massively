use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor,
    op::BinaryPredicateOp,
    op::ReductionOp,
    op::UnaryOp,
    seg::{
        ForEachSegment, Map, MapLikeExecutable, Reduce, ReduceLikeExecutable, Segmented, Unique,
        UniqueLikeExecutable,
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

#[test]
fn segmented_wrappers_expose_their_parts() {
    let input = Segmented::new([1_u32, 2], [0_u32, 2]);
    assert_eq!(input.values(), &[1, 2]);
    assert_eq!(input.offsets(), &[0, 2]);
    assert_eq!(input.into_parts(), ([1, 2], [0, 2]));
}

#[test]
fn segmented_algorithms_return_owned_device_results() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[1_u32, 1, 2, 3, 3]);
    let offsets = exec.to_device(&[0_u32, 3, 5]);

    let mapped = ForEachSegment(Map(Transform))
        .run(&exec, Segmented::new(values.slice(..), offsets.slice(..)))
        .unwrap();
    assert_eq!(exec.to_host(&mapped).unwrap(), vec![1_u64, 1, 2, 3, 3]);

    let unique = ForEachSegment(Unique(Equal))
        .run(&exec, Segmented::new(values.slice(..), offsets.slice(..)))
        .unwrap();
    assert_eq!(exec.to_host(unique.values()).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(unique.offsets()).unwrap(), vec![0, 2, 3]);

    let reduced = ForEachSegment(Reduce(Add, 0))
        .run(&exec, Segmented::new(values.slice(..), offsets.slice(..)))
        .unwrap();
    assert_eq!(exec.to_host(&reduced).unwrap(), vec![4, 6]);
}
