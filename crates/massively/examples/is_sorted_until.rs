mod common;

use massively::{Executor, Wgpu, is_sorted_until};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[1.0_f32, 2.0, 4.0, 3.0])?;

    let index = is_sorted_until(&exec, (values.slice(..),), common::LessF32)?;

    assert_eq!(index, 3);
    Ok(())
}
