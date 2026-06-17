mod common;

use massively::{CubeWgpu, scatter_if, unzip};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[10.0_f32, 20.0, 30.0])?;
    let indices = policy.to_device(&[2_u32, 0, 1])?;
    let stencil = policy.to_device(&[1.0_f32, -1.0, 1.0])?;
    let initial = policy.device_filled(3, 0.0_f32)?;

    let output = unzip(scatter_if(
        &values,
        &indices,
        &stencil,
        initial,
        common::Positive,
    )?)?;

    assert_eq!(output.to_vec()?, vec![0.0, 30.0, 10.0]);
    Ok(())
}
