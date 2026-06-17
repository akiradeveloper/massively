mod common;

use massively::{CubeWgpu, find_if_not};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[1.0_f32, 2.0, -3.0])?;

    let index = find_if_not(&values, common::Positive)?;

    assert_eq!(index, Some(2));
    Ok(())
}
