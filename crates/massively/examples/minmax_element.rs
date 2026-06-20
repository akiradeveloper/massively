mod common;

use massively::{CubeWgpu, minmax_element};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[3.0_f32, 1.0, 2.0])?;

    let indices = minmax_element((&values,), common::LessF32)?;

    assert_eq!(indices, Some((1, 0)));
    Ok(())
}
