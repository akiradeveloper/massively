mod common;

use massively::{Executor, Wgpu, gather_if};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[10.0_f32, -20.0, 30.0])?;
    let indices = exec.to_device(&[2_u32, 0, 1])?;
    let stencil = exec.to_device(&[1_u32, 1, 0])?;

    let (output,) = gather_if(
        &exec,
        massively::SoA1(values.slice(..)),
        indices.slice(..),
        (0.0_f32,),
        stencil.slice(..),
    )?;

    assert_eq!(exec.to_host(&output)?, vec![30.0, 10.0, 0.0]);
    Ok(())
}
