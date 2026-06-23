mod common;

use massively::{Executor, Wgpu, sort};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[3.0_f32, 1.0, 2.0])?;

    let (output,) = sort(&exec, massively::SoA1(values.slice(..)), common::LessF32)?;

    assert_eq!(exec.to_host(&output)?, vec![1.0, 2.0, 3.0]);
    Ok(())
}
