mod common;

use massively::{Executor, Wgpu, lower_bound};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let sorted = exec.to_device(&[1.0_f32, 2.0, 2.0, 4.0])?;

    let index = lower_bound(&exec, (sorted.slice(..),), (2.0,), common::LessF32)?;

    assert_eq!(index, 1);
    Ok(())
}
