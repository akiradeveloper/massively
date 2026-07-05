use crate::common::*;

#[test]
fn scalar_device_slice_adjacent_find_reads_scalar_items() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 2.0, 3.0]).unwrap();

    assert_eq!(
        adjacent_find(&exec, values.slice(..), EqualF32).unwrap(),
        Some(1)
    );
}
