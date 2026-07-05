use crate::common::*;

#[test]
fn lexicographical_compare_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let left_a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_b = exec.to_device(&[10_u32, 20]).unwrap();
    let right_a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let right_b = exec.to_device(&[10_u32, 25]).unwrap();

    assert!(
        lexicographical_compare(
            &exec,
            massively::Zip2(left_a.slice(..), left_b.slice(..)),
            massively::Zip2(right_a.slice(..), right_b.slice(..)),
            MixedTupleLess
        )
        .unwrap()
    );
    assert!(
        !lexicographical_compare(
            &exec,
            massively::Zip2(right_a.slice(..), right_b.slice(..)),
            massively::Zip2(left_a.slice(..), left_b.slice(..)),
            MixedTupleLess
        )
        .unwrap()
    );
}
