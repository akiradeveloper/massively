mod common;

use massively::{CubeWgpu, all_of};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0])?;

    let result = all_of(&values, common::Positive)?;

    assert!(result);
    Ok(())
}
