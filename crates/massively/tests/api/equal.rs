use crate::common::*;

#[test]
fn equal_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20]).unwrap();
    let c = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let d = exec.to_device(&[10_u32, 20]).unwrap();

    assert!(
        equal(
            &exec,
            massively::Zip2(a.slice(..), b.slice(..)),
            massively::Zip2(c.slice(..), d.slice(..)),
            MixedTupleEqual
        )
        .unwrap()
    );
}
