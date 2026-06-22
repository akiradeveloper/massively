mod common;

use massively::{Executor, Wgpu, minmax_element};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[3.0_f32, 1.0, 2.0])?;

    let indices = minmax_element(&exec, massively::SoA1(values.slice(..)), common::LessF32)?;

    assert_eq!(indices, Some((1, 0)));
    Ok(())
}
