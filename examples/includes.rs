mod common;

use massively::{CubeWgpu, includes};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;
    let subset = policy.to_device(&[2.0_f32, 4.0])?;

    let result = includes(&input, &subset, common::LessF32)?;

    assert!(result);
    Ok(())
}
