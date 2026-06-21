use crate::common::*;

#[test]
fn is_sorted_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let k = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let l = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    assert!(is_sorted(&exec, (k.slice(..), l.slice(..)), MixedTupleLess).unwrap());
}
