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
