mod common;

use massively::{Executor, Wgpu, mismatch};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let left = exec.to_device(&[1.0_f32, 2.0, 3.0])?;
    let right = exec.to_device(&[1.0_f32, 9.0, 3.0])?;

    let index = mismatch(
        &exec,
        (left.slice(..),),
        (right.slice(..),),
        common::EqualF32,
    )?;

    assert_eq!(index, Some(1));
    Ok(())
}
