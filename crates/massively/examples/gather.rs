mod common;

use massively::{CubeWgpu, gather};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[10.0_f32, 20.0, 30.0])?;
    let indices = policy.to_device(&[2_u32, 0, 1])?;

    let (output,) = gather((values.slice(..),), (indices.slice(..),))?;

    assert_eq!(output.to_vec()?, vec![30.0, 10.0, 20.0]);
    Ok(())
}
