mod common;

use massively::{CubeWgpu, exclusive_scan};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;

    let (output,) = exclusive_scan((values.slice(..),), (0.0,), common::TupleSumF32)?;

    assert_eq!(output.to_vec()?, vec![0.0, 1.0, 3.0, 6.0]);
    Ok(())
}
