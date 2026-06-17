mod common;

use massively::{CubeWgpu, find_end};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0, 2.0, 3.0])?;
    let pattern = policy.to_device(&[2.0_f32, 3.0])?;

    let index = find_end(&input, &pattern, common::EqualF32)?;

    assert_eq!(index, Some(3));
    Ok(())
}
