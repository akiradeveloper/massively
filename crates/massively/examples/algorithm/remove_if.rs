mod common;

use massively::{Executor, Wgpu, remove_if};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;

    let (output,) = remove_if(&exec, massively::SoA1(values.slice(..)), common::Positive)?;

    assert_eq!(exec.to_host(&output)?, vec![-1.0, -3.0]);
    Ok(())
}
