mod common;

use massively::{Executor, Wgpu, reverse};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0])?;

    let (output,) = reverse(&exec, (values.slice(..),))?;

    assert_eq!(exec.to_host(&output)?, vec![3.0, 2.0, 1.0]);
    Ok(())
}
