mod common;

use massively::{CubeWgpu, sort};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[3.0_f32, 1.0, 2.0])?;

    let output = sort(&values, common::LessF32)?;

    assert_eq!(output.to_vec()?, vec![1.0, 2.0, 3.0]);
    Ok(())
}
