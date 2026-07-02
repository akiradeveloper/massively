use crate::common::*;

fn set_difference_with_generic_right<Left, Right, Less>(
    exec: &Executor<WgpuRuntime>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<<Left::Item as massively::MItem<WgpuRuntime>>::Vec, massively::Error>
where
    Left: massively::MIter<WgpuRuntime>,
    Right: massively::MIter<WgpuRuntime, Item = Left::Item>,
    Less: BinaryPredicateOp<WgpuRuntime, Left::Item>,
{
    set_difference(exec, left, right, less)
}

#[test]
fn set_difference_accepts_generic_right_without_inner_equality_bound() {
    let exec = exec();
    let left_a = exec.to_device(&[1.0_f32, 2.0, 4.0]).unwrap();
    let left_b = exec.to_device(&[10_u32, 20, 40]).unwrap();
    let right_a = exec.to_device(&[2.0_f32, 3.0]).unwrap();
    let right_b = exec.to_device(&[20_u32, 30]).unwrap();

    let massively::SoA2(a, b) = set_difference_with_generic_right(
        &exec,
        massively::SoA2(left_a.slice(..), left_b.slice(..)),
        massively::SoA2(right_a.slice(..), right_b.slice(..)),
        MixedTupleLess,
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 4.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 40]);
}

#[test]
fn set_difference_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let left_a = exec.to_device(&[1.0_f32, 2.0, 4.0]).unwrap();
    let left_b = exec.to_device(&[10_u32, 20, 40]).unwrap();
    let right_a = exec.to_device(&[2.0_f32, 3.0]).unwrap();
    let right_b = exec.to_device(&[20_u32, 30]).unwrap();

    let output = set_difference(
        &exec,
        massively::SoA2(left_a.slice(..), left_b.slice(..)),
        massively::SoA2(right_a.slice(..), right_b.slice(..)),
        MixedTupleLess,
    )
    .unwrap();
    let massively::SoA2(a, b) = output;

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 4.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 40]);
}
