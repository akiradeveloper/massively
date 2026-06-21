use crate::common::*;

#[test]
fn equal_range_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let k = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let l = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let input = (k.slice(..), l.slice(..));
    assert_eq!(
        equal_range(&exec, input, (3.0_f32, 30_u32), MixedTupleLess).unwrap(),
        (2, 3)
    );
}
