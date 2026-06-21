mod common;

use massively::{CubeWgpu, stable_sort};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[3.0_f32, 1.0, 2.0])?;

    let (output,) = stable_sort((values.slice(..),), common::LessF32)?;

    assert_eq!(output.to_vec()?, vec![1.0, 2.0, 3.0]);
    Ok(())
}
