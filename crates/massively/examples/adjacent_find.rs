mod common;

use massively::{Executor, Wgpu, adjacent_find};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[1.0_f32, 2.0, 2.0, 3.0])?;

    let index = adjacent_find(&exec, (values.slice(..),), common::EqualF32)?;

    assert_eq!(index, Some(1));
    Ok(())
}
