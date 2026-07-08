use crate::common::*;

fn set_intersection_with_generic_right<Left, Right, Less, Output>(
    exec: &Executor<WgpuRuntime>,
    left: Left,
    right: Right,
    less: Less,
    out: Output,
) -> Result<massively::MIndex, massively::Error>
where
    Left: massively::iter::MIter<WgpuRuntime>,
    Left::Item: massively::MAlloc<WgpuRuntime>,
    Right: massively::iter::MIter<WgpuRuntime, Item = Left::Item>,
    Less: BinaryPredicateOp<WgpuRuntime, Left::Item>,
    Output: massively::MIterMut<WgpuRuntime, Item = Left::Item>,
{
    set_intersection(exec, left, right, less, out)
}

#[test]
fn set_intersection_accepts_generic_right_without_inner_equality_bound() {
    let exec = exec();
    let left_a = exec.to_device(&[1.0_f32, 2.0, 4.0]).unwrap();
    let left_b = exec.to_device(&[10_u32, 20, 40]).unwrap();
    let right_a = exec.to_device(&[2.0_f32, 3.0, 4.0]).unwrap();
    let right_b = exec.to_device(&[20_u32, 30, 40]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();

    let len = set_intersection_with_generic_right(
        &exec,
        massively::Zip2(left_a.slice(..), left_b.slice(..)),
        massively::Zip2(right_a.slice(..), right_b.slice(..)),
        MixedTupleLess,
        massively::Zip2(out_a.slice_mut(..), out_b.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a.slice(..len)).unwrap(), vec![2.0, 4.0]);
    assert_eq!(exec.to_host(&out_b.slice(..len)).unwrap(), vec![20, 40]);
}

#[test]
fn set_intersection_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let left_a = exec.to_device(&[1.0_f32, 2.0, 4.0]).unwrap();
    let left_b = exec.to_device(&[10_u32, 20, 40]).unwrap();
    let right_a = exec.to_device(&[2.0_f32, 3.0, 4.0]).unwrap();
    let right_b = exec.to_device(&[20_u32, 30, 40]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();

    let len = set_intersection(
        &exec,
        massively::Zip2(left_a.slice(..), left_b.slice(..)),
        massively::Zip2(right_a.slice(..), right_b.slice(..)),
        MixedTupleLess,
        massively::Zip2(out_a.slice_mut(..), out_b.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a.slice(..len)).unwrap(), vec![2.0, 4.0]);
    assert_eq!(exec.to_host(&out_b.slice(..len)).unwrap(), vec![20, 40]);
}

#[test]
fn set_intersection_uses_sorted_multiset_semantics_for_duplicates() {
    let exec = exec();
    let left = exec.to_device(&[1_u32, 2, 2, 4]).unwrap();
    let right = exec.to_device(&[2_u32, 2, 2, 3]).unwrap();
    let out = exec.to_device(&[0_u32; 4]).unwrap();

    let len = set_intersection(
        &exec,
        massively::Zip1(left.slice(..)),
        massively::Zip1(right.slice(..)),
        LessU32,
        massively::Zip1(out.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out.slice(..len)).unwrap(), vec![2, 2]);
}

#[test]
fn set_intersection_uses_sorted_multiset_semantics_for_tuple_duplicates() {
    let exec = exec();
    let left_a = exec.to_device(&[1.0_f32, 2.0, 2.0, 4.0]).unwrap();
    let left_b = exec.to_device(&[10_u32, 20, 20, 40]).unwrap();
    let right_a = exec.to_device(&[2.0_f32, 2.0, 2.0, 3.0]).unwrap();
    let right_b = exec.to_device(&[20_u32, 20, 20, 30]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_b = exec.to_device(&[0_u32; 4]).unwrap();

    let len = set_intersection(
        &exec,
        massively::Zip2(left_a.slice(..), left_b.slice(..)),
        massively::Zip2(right_a.slice(..), right_b.slice(..)),
        MixedTupleLess,
        massively::Zip2(out_a.slice_mut(..), out_b.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a.slice(..len)).unwrap(), vec![2.0, 2.0]);
    assert_eq!(exec.to_host(&out_b.slice(..len)).unwrap(), vec![20, 20]);
}
