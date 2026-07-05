use crate::common::*;

#[test]
fn is_sorted_until_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let k = exec.to_device(&[1.0_f32, 3.0, 2.0, 4.0]).unwrap();
    let l = exec.to_device(&[10_u32, 30, 20, 40]).unwrap();
    assert_eq!(
        is_sorted_until(
            &exec,
            massively::Zip2(k.slice(..), l.slice(..)),
            MixedTupleLess
        )
        .unwrap(),
        2
    );
}
