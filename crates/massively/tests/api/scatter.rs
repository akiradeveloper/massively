use crate::common::*;

#[test]
fn scatter_accepts_heterogeneous_columns() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = exec.to_device(&[2_u32, 0, 3]).unwrap();

    let out_values = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_ids = exec.to_device(&[0_u32; 4]).unwrap();
    scatter(
        &exec,
        massively::Zip2(values.slice(..), ids.slice(..)),
        indices.slice(..),
        massively::Zip2(out_values.slice_mut(..), out_ids.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![2.0, 0.0, 1.0, 3.0]);
    assert_eq!(exec.to_host(&out_ids).unwrap(), vec![20, 0, 10, 30]);
}
