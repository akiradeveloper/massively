mod common;

use massively::{CubeWgpu, equal};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let left = policy.to_device(&[1.0_f32, 2.0, 3.0])?;
    let right = policy.to_device(&[1.0_f32, 2.0, 3.0])?;

    let result = equal(&left, &right, common::EqualF32)?;

    assert!(result);
    Ok(())
}
