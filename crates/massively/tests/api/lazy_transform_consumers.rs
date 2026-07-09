use crate::common::*;

struct PairGreaterAsU32;
struct PairGreaterAsBool;
struct U32ToTuple1;
struct U32NonZeroScalar;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32, f32)> for PairGreaterAsU32 {
    type Output = u32;

    fn apply(input: (f32, f32)) -> u32 {
        if input.0 > input.1 { 1_u32 } else { 0_u32 }
    }
}

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32, f32)> for PairGreaterAsBool {
    type Output = bool;

    fn apply(input: (f32, f32)) -> bool {
        input.0 > input.1
    }
}

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for U32ToTuple1 {
    type Output = (u32,);

    fn apply(input: u32) -> (u32,) {
        (input,)
    }
}

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, u32> for U32NonZeroScalar {
    fn apply(input: u32) -> bool {
        input != 0
    }
}

#[test]
fn transform_materializes_lazy_tuple_transform_to_scalar_items() {
    let exec = exec();
    let left = exec.to_device(&[0.0_f32, 2.0, 3.0, 1.0]).unwrap();
    let right = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0]).unwrap();
    let output = exec.to_device(&[99_u32; 4]).unwrap();
    let source = massively::lazy::transform(
        massively::Zip2(left.slice(..), right.slice(..)),
        PairGreaterAsU32,
    );

    transform(
        &exec,
        source,
        U32ToTuple1,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 1, 0]);
}

#[test]
fn predicate_queries_accept_lazy_tuple_transform_to_scalar_items() {
    let exec = exec();
    let left = exec.to_device(&[0.0_f32, 2.0, 3.0, 1.0]).unwrap();
    let right = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0]).unwrap();

    let source = || {
        massively::lazy::transform(
            massively::Zip2(left.slice(..), right.slice(..)),
            PairGreaterAsU32,
        )
    };

    assert_eq!(count_if(&exec, source(), U32NonZeroScalar).unwrap(), 2);
    assert!(!massively::all_of(&exec, source(), U32NonZeroScalar).unwrap());
    assert!(massively::any_of(&exec, source(), U32NonZeroScalar).unwrap());
    assert!(!massively::none_of(&exec, source(), U32NonZeroScalar).unwrap());
    assert_eq!(find_if(&exec, source(), U32NonZeroScalar).unwrap(), Some(1));
}

#[test]
fn is_partitioned_accepts_lazy_tuple_transform_to_scalar_items() {
    let exec = exec();
    let left = exec.to_device(&[3.0_f32, 2.0, 1.0, 0.0]).unwrap();
    let right = exec.to_device(&[2.0_f32, 1.0, 1.0, 1.0]).unwrap();
    let source = massively::lazy::transform(
        massively::Zip2(left.slice(..), right.slice(..)),
        PairGreaterAsU32,
    );

    assert!(is_partitioned(&exec, source, U32NonZeroScalar).unwrap());
}

#[test]
fn search_queries_accept_lazy_tuple_transform_to_scalar_items() {
    let exec = exec();
    let left = exec.to_device(&[0.0_f32, 2.0, 3.0, 1.0]).unwrap();
    let right = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0]).unwrap();
    let expected = exec.to_device(&[0_u32, 1, 1, 0]).unwrap();
    let mismatch_at_two = exec.to_device(&[0_u32, 1, 0, 0]).unwrap();
    let needle = exec.to_device(&[1_u32]).unwrap();

    let source = || {
        massively::lazy::transform(
            massively::Zip2(left.slice(..), right.slice(..)),
            PairGreaterAsU32,
        )
    };

    assert!(equal(&exec, source(), expected.slice(..), EqualU32).unwrap());
    assert_eq!(
        mismatch(&exec, source(), mismatch_at_two.slice(..), EqualU32).unwrap(),
        Some(2)
    );
    assert_eq!(
        find_first_of(&exec, source(), needle.slice(..), EqualU32).unwrap(),
        Some(1)
    );
    assert_eq!(adjacent_find(&exec, source(), EqualU32).unwrap(), Some(1));
}

#[test]
fn ordering_queries_accept_lazy_tuple_transform_to_scalar_items() {
    let exec = exec();
    let left = exec.to_device(&[0.0_f32, 1.0, 3.0, 4.0]).unwrap();
    let right = exec.to_device(&[1.0_f32, 2.0, 2.0, 3.0]).unwrap();
    let lex_rhs = exec.to_device(&[0_u32, 1, 0, 0]).unwrap();

    let source = || {
        massively::lazy::transform(
            massively::Zip2(left.slice(..), right.slice(..)),
            PairGreaterAsU32,
        )
    };

    assert!(is_sorted(&exec, source(), LessU32).unwrap());
    assert_eq!(is_sorted_until(&exec, source(), LessU32).unwrap(), 4);
    assert!(lexicographical_compare(&exec, source(), lex_rhs.slice(..), LessU32).unwrap());
}

#[test]
fn selection_stencils_accept_lazy_tuple_transform_to_bool_items() {
    let exec = exec();
    let left = exec.to_device(&[0.0_f32, 2.0, 3.0, 1.0]).unwrap();
    let right = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0]).unwrap();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let copied = exec.to_device(&[0_u32; 4]).unwrap();

    let stencil = || {
        massively::lazy::transform(
            massively::Zip2(left.slice(..), right.slice(..)),
            PairGreaterAsBool,
        )
    };

    let len = copy_where(
        &exec,
        massively::Zip1(values.slice(..)),
        stencil(),
        massively::Zip1(copied.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(len, 2);
    assert_eq!(exec.to_host(&copied.slice(..len)).unwrap(), vec![20, 30]);

    let indices = exec.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let gathered = exec.to_device(&[99_u32; 4]).unwrap();
    gather_where(
        &exec,
        massively::Zip1(values.slice(..)),
        indices.slice(..),
        stencil(),
        massively::Zip1(gathered.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&gathered).unwrap(), vec![99, 30, 20, 99]);

    let scattered = exec.to_device(&[99_u32; 4]).unwrap();
    scatter_where(
        &exec,
        massively::Zip1(values.slice(..)),
        indices.slice(..),
        stencil(),
        massively::Zip1(scattered.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&scattered).unwrap(), vec![99, 30, 20, 99]);
}
