use crate::common::*;

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
            (a.slice(..), b.slice(..)),
            (c.slice(..), d.slice(..)),
            MixedTupleEqual
        )
        .unwrap(),
        Some(2)
    );
}
