use crate::common::*;

#[test]
fn is_sorted_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let k = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let l = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    assert!(
        is_sorted(
            &exec,
            massively::SoA2(k.slice(..), l.slice(..)),
            MixedTupleLess
        )
        .unwrap()
    );
}

#[test]
fn scalar_device_slice_sorted_queries_read_scalar_items() {
    let exec = exec();
    let sorted = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let unsorted = exec.to_device(&[1.0_f32, 3.0, 2.0, 4.0]).unwrap();

    assert!(is_sorted(&exec, sorted.slice(..), Less).unwrap());
    assert_eq!(is_sorted_until(&exec, unsorted.slice(..), Less).unwrap(), 2);
}
