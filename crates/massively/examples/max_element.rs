mod common;

use massively::{Executor, Wgpu, max_element};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[3.0_f32, 1.0, 2.0])?;

    let index = max_element(&exec, (values.slice(..),), common::LessF32)?;

    assert_eq!(index, Some(0));
    Ok(())
}
