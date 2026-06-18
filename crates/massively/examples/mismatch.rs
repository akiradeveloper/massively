mod common;

use massively::{CubeWgpu, mismatch};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let left = policy.to_device(&[1.0_f32, 2.0, 3.0])?;
    let right = policy.to_device(&[1.0_f32, 9.0, 3.0])?;

    let index = mismatch(&left, &right, common::EqualF32)?;

    assert_eq!(index, Some(1));
    Ok(())
}
