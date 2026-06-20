mod common;

use massively::{CubeWgpu, merge_by_key};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let left_keys = policy.to_device(&[0_u32, 2])?;
    let left_values = policy.to_device(&[0.0_f32, 20.0])?;
    let right_keys = policy.to_device(&[1_u32, 3])?;
    let right_values = policy.to_device(&[10.0_f32, 30.0])?;

    let ((keys,), (values,)) = merge_by_key(
        (&left_keys,),
        (&left_values,),
        (&right_keys,),
        (&right_values,),
        common::LessU32,
    )?;

    assert_eq!(keys.to_vec()?, vec![0, 1, 2, 3]);
    assert_eq!(values.to_vec()?, vec![0.0, 10.0, 20.0, 30.0]);
    Ok(())
}
