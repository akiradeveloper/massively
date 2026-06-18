mod common;

use massively::{CubeWgpu, is_partitioned};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[2.0_f32, 4.0, -1.0, -3.0])?;

    let result = is_partitioned(&values, common::Positive)?;

    assert!(result);
    Ok(())
}
