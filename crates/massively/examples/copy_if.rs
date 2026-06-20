mod common;

use massively::{CubeWgpu, copy_if};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;

    let (output,) = copy_if((&values,), (&values,), common::Positive)?;

    assert_eq!(output.to_vec()?, vec![2.0, 4.0]);
    Ok(())
}
