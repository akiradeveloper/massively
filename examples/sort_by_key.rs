mod common;

use massively::{CubeWgpu, sort_by_key, unzip};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let keys = policy.to_device(&[2_u32, 0, 1])?;
    let values = policy.to_device(&[20.0_f32, 0.0, 10.0])?;

    let (keys, values) = sort_by_key(keys, values, common::LessU32)?;
    let keys = unzip(keys)?;
    let values = unzip(values)?;

    assert_eq!(keys.to_vec()?, vec![0, 1, 2]);
    assert_eq!(values.to_vec()?, vec![0.0, 10.0, 20.0]);
    Ok(())
}
