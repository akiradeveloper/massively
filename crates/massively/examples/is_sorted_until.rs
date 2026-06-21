mod common;

use massively::{CubeWgpu, is_sorted_until};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[1.0_f32, 2.0, 4.0, 3.0])?;

    let index = is_sorted_until((values.slice(..),), common::LessF32)?;

    assert_eq!(index, 3);
    Ok(())
}
