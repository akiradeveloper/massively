mod common;

use massively::{CubeWgpu, reduce_by_key, unzip};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let keys = policy.to_device(&[0_u32, 0, 1, 1, 1])?;
    let values = policy.to_device(&[1.0_f32, 2.0, 10.0, 20.0, 30.0])?;

    let (out_keys, out_values) = reduce_by_key(&values, &keys, 0.0, common::SumF32)?;
    let out_keys = unzip(out_keys)?;
    let out_values = unzip(out_values)?;

    assert_eq!(out_keys.to_vec()?, vec![0, 1]);
    assert_eq!(out_values.to_vec()?, vec![3.0, 60.0]);
    Ok(())
}
