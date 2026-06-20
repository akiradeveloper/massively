mod common;

use massively::{CubeWgpu, lexicographical_compare};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let left = policy.to_device(&[1.0_f32, 2.0])?;
    let right = policy.to_device(&[1.0_f32, 3.0])?;

    let result = lexicographical_compare((&left,), (&right,), common::LessF32)?;

    assert!(result);
    Ok(())
}
