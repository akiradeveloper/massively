mod common;

use massively::{CubeWgpu, min_element};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[3.0_f32, 1.0, 2.0])?;

    let index = min_element(&values, common::LessF32)?;

    assert_eq!(index, Some(1));
    Ok(())
}
