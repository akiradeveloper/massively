use crate::common::*;

#[test]
fn mismatch_accepts_borrowed_tuple_columns() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[1.0_f32, 2.0, 4.0]).unwrap();
    let d = policy.to_device(&[10_u32, 20, 40]).unwrap();

    assert_eq!(
        mismatch(
            (a.slice(..), b.slice(..)),
            (c.slice(..), d.slice(..)),
            MixedTupleEqual
        )
        .unwrap(),
        Some(2)
    );
}
