mod common;

use massively::{Executor, Wgpu, is_sorted};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0])?;

    let result = is_sorted(&exec, massively::SoA1(values.slice(..)), common::LessF32)?;

    assert!(result);
    Ok(())
}
