mod common;

use massively::{CubeWgpu, merge, unzip};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let left = policy.to_device(&[1.0_f32, 3.0])?;
    let right = policy.to_device(&[2.0_f32, 4.0])?;

    let output = unzip(merge(&left, &right, common::LessF32)?)?;

    assert_eq!(output.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}
