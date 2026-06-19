mod common;

use massively::{CubeWgpu, inclusive_scan};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;

    let (output,) = inclusive_scan((&values,), common::TupleSumF32)?;

    assert_eq!(output.to_vec()?, vec![1.0, 3.0, 6.0, 10.0]);
    Ok(())
}
