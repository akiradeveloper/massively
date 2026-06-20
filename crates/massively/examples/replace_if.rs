mod common;

use massively::{CubeWgpu, replace_if};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;

    let (output,) = replace_if((&values,), (0.0,), (&values,), common::Positive)?;

    assert_eq!(output.to_vec()?, vec![-1.0, 0.0, -3.0, 0.0]);
    Ok(())
}
