use crate::common::*;

#[test]
fn scalar_device_slice_minmax_queries_read_scalar_items() {
    let exec = exec();
    let values = exec.to_device(&[3.0_f32, 1.0, 4.0, 2.0]).unwrap();

    assert_eq!(min_element(&exec, values.slice(..), Less).unwrap(), Some(1));
    assert_eq!(max_element(&exec, values.slice(..), Less).unwrap(), Some(2));
    assert_eq!(
        minmax_element(&exec, values.slice(..), Less).unwrap(),
        Some((1, 2))
    );
}

#[test]
fn minmax_queries_have_deterministic_duplicate_policy() {
    let exec = exec();
    let values = exec.to_device(&[3_u32, 1, 4, 1, 4, 2]).unwrap();

    assert_eq!(min_element(&exec, values.slice(..), LessU32).unwrap(), Some(1));
    assert_eq!(max_element(&exec, values.slice(..), LessU32).unwrap(), Some(2));
    assert_eq!(
        minmax_element(&exec, values.slice(..), LessU32).unwrap(),
        Some((1, 2))
    );
}
