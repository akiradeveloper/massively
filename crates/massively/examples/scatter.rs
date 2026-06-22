mod common;

use massively::{Executor, Wgpu, scatter};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[10.0_f32, 20.0, 30.0])?;
    let indices = exec.to_device(&[2_u32, 0, 1])?;

    let (output,) = scatter(
        &exec,
        massively::SoA1(values.slice(..)),
        indices.slice(..),
        3,
        (0.0_f32,),
    )?;

    assert_eq!(exec.to_host(&output)?, vec![20.0, 30.0, 10.0]);
    Ok(())
}
