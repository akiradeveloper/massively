use crate::common::*;

fn mismatch_with_generic_right<Left, Right, Eq>(
    exec: &Executor<WgpuRuntime>,
    left: Left,
    right: Right,
    eq: Eq,
) -> Option<usize>
where
    Left: massively::MIter<WgpuRuntime>,
    Right: massively::MIter<WgpuRuntime, Item = Left::Item>,
    Eq: BinaryPredicateOp<WgpuRuntime, Left::Item>,
{
    mismatch(exec, left, right, eq).unwrap()
}

#[test]
fn mismatch_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[1.0_f32, 2.0, 4.0]).unwrap();
    let d = exec.to_device(&[10_u32, 20, 40]).unwrap();

    assert_eq!(
        mismatch(
            &exec,
            massively::SoA2(a.slice(..), b.slice(..)),
            massively::SoA2(c.slice(..), d.slice(..)),
            MixedTupleEqual
        )
        .unwrap(),
        Some(2)
    );
}

#[test]
fn mismatch_accepts_generic_right_without_inner_equality_bound() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[1.0_f32, 2.0, 4.0]).unwrap();
    let d = exec.to_device(&[10_u32, 20, 40]).unwrap();

    assert_eq!(
        mismatch_with_generic_right(
            &exec,
            massively::SoA2(a.slice(..), b.slice(..)),
            massively::SoA2(c.slice(..), d.slice(..)),
            MixedTupleEqual,
        ),
        Some(2)
    );
}
