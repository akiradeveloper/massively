mod common;

use massively::{CubeWgpu, unique};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0])?;

    let (output,) = unique(&values, common::EqualF32)?;

    assert_eq!(output.to_vec()?, vec![1.0, 2.0, 3.0]);
    Ok(())
}
