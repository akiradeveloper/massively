mod common;

use massively::{Executor, Wgpu, gather};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[10.0_f32, 20.0, 30.0])?;
    let indices = exec.to_device(&[2_u32, 0, 1])?;

    let (output,) = gather(&exec, massively::SoA1(values.slice(..)), indices.slice(..))?;

    assert_eq!(exec.to_host(&output)?, vec![30.0, 10.0, 20.0]);
    Ok(())
}
