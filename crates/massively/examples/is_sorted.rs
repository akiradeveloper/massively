mod common;

use massively::{CubeWgpu, is_sorted};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0])?;

    let result = is_sorted((&values,), common::LessF32)?;

    assert!(result);
    Ok(())
}
