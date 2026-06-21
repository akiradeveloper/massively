mod common;

use massively::{CubeWgpu, unique_by_key};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let keys = policy.to_device(&[0_u32, 0, 1, 1, 2])?;
    let values = policy.to_device(&[0.0_f32, 1.0, 10.0, 11.0, 20.0])?;

    let ((keys,), (values,)) =
        unique_by_key((keys.slice(..),), (values.slice(..),), common::EqualU32)?;

    assert_eq!(keys.to_vec()?, vec![0, 1, 2]);
    assert_eq!(values.to_vec()?, vec![0.0, 10.0, 20.0]);
    Ok(())
}
