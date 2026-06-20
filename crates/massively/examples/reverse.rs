mod common;

use massively::{CubeWgpu, reverse};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0])?;

    let (output,) = reverse((&values,))?;

    assert_eq!(output.to_vec()?, vec![3.0, 2.0, 1.0]);
    Ok(())
}
