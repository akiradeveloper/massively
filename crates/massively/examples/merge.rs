mod common;

use massively::{Executor, Wgpu, merge};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let left = exec.to_device(&[1.0_f32, 3.0])?;
    let right = exec.to_device(&[2.0_f32, 4.0])?;

    let (output,) = merge(
        &exec,
        (left.slice(..),),
        (right.slice(..),),
        common::LessF32,
    )?;

    assert_eq!(exec.to_host(&output)?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}
