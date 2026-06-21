mod common;

use massively::{CubeWgpu, stable_sort_by_key};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let keys = policy.to_device(&[2_u32, 0, 1])?;
    let values = policy.to_device(&[20.0_f32, 0.0, 10.0])?;

    let ((keys,), (values,)) =
        stable_sort_by_key((keys.slice(..),), (values.slice(..),), common::LessU32)?;

    assert_eq!(keys.to_vec()?, vec![0, 1, 2]);
    assert_eq!(values.to_vec()?, vec![0.0, 10.0, 20.0]);
    Ok(())
}
