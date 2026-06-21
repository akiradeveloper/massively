mod common;

use massively::{CubeWgpu, find_first_of};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;
    let needles = policy.to_device(&[3.0_f32, 9.0])?;

    let index = find_first_of((input.slice(..),), (needles.slice(..),), common::EqualF32)?;

    assert_eq!(index, Some(2));
    Ok(())
}
