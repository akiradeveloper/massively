mod common;

use massively::{CubeWgpu, scatter_if};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[10.0_f32, -20.0, 30.0])?;
    let indices = policy.to_device(&[2_u32, 0, 1])?;
    let stencil = policy.to_device(&[1_u32, 0, 1])?;

    let (output,) = scatter_if(
        (values.slice(..),),
        (indices.slice(..),),
        3,
        (0.0_f32,),
        (stencil.slice(..),),
    )?;

    assert_eq!(output.to_vec()?, vec![0.0, 30.0, 10.0]);
    Ok(())
}
