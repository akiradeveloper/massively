mod common;

use massively::{Executor, Wgpu, unique};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0])?;

    let (output,) = unique(&exec, massively::SoA1(values.slice(..)), common::EqualF32)?;

    assert_eq!(exec.to_host(&output)?, vec![1.0, 2.0, 3.0]);
    Ok(())
}
