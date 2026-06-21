mod common;

use massively::{Executor, Wgpu, copy_if};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;
    let stencil = exec.to_device(&[0_u32, 1, 0, 1])?;

    let (output,) = copy_if(&exec, (values.slice(..),), (stencil.slice(..),))?;

    assert_eq!(exec.to_host(&output)?, vec![2.0, 4.0]);
    Ok(())
}
