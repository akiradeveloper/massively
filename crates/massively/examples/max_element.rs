mod common;

use massively::{CubeWgpu, max_element};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[3.0_f32, 1.0, 2.0])?;

    let index = max_element((values.slice(..),), common::LessF32)?;

    assert_eq!(index, Some(0));
    Ok(())
}
