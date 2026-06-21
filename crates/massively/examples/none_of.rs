mod common;

use massively::{Executor, Wgpu, none_of};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[-1.0_f32, -2.0, -3.0])?;

    let result = none_of(&exec, (values.slice(..),), common::Positive)?;

    assert!(result);
    Ok(())
}
