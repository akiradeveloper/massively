use crate::common::*;

fn find_first_of_with_generic_needles<Input, Needles, Eq>(
    exec: &Executor<WgpuRuntime>,
    input: Input,
    needles: Needles,
    eq: Eq,
) -> Option<usize>
where
    Input: massively::MIter<WgpuRuntime>,
    Needles: massively::MIter<WgpuRuntime, Item = Input::Item>,
    Eq: BinaryPredicateOp<WgpuRuntime, Input::Item>,
{
    find_first_of(exec, input, needles, eq).unwrap()
}

#[test]
fn find_first_of_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let needle_a = exec.to_device(&[9.0_f32, 3.0]).unwrap();
    let needle_b = exec.to_device(&[90_u32, 30]).unwrap();

    assert_eq!(
        find_first_of(
            &exec,
            massively::SoA2(a.slice(..), b.slice(..)),
            massively::SoA2(needle_a.slice(..), needle_b.slice(..)),
            MixedTupleEqual
        )
        .unwrap(),
        Some(2)
    );
}

#[test]
fn find_first_of_accepts_generic_needles_without_inner_equality_bound() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let needle_a = exec.to_device(&[9.0_f32, 3.0]).unwrap();
    let needle_b = exec.to_device(&[90_u32, 30]).unwrap();

    assert_eq!(
        find_first_of_with_generic_needles(
            &exec,
            massively::SoA2(a.slice(..), b.slice(..)),
            massively::SoA2(needle_a.slice(..), needle_b.slice(..)),
            MixedTupleEqual,
        ),
        Some(2)
    );
}
