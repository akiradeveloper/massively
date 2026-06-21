mod common;

use massively::{CubeWgpu, gather_if};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[10.0_f32, -20.0, 30.0])?;
    let indices = policy.to_device(&[2_u32, 0, 1])?;
    let stencil = policy.to_device(&[1_u32, 1, 0])?;

    let (output,) = gather_if(
        (values.slice(..),),
        (indices.slice(..),),
        (0.0_f32,),
        (stencil.slice(..),),
    )?;

    assert_eq!(output.to_vec()?, vec![30.0, 10.0, 0.0]);
    Ok(())
}
