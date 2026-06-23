mod common;

use massively::{Executor, Wgpu, lexicographical_compare};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let left = exec.to_device(&[1.0_f32, 2.0])?;
    let right = exec.to_device(&[1.0_f32, 3.0])?;

    let result = lexicographical_compare(
        &exec,
        massively::SoA1(left.slice(..)),
        massively::SoA1(right.slice(..)),
        common::LessF32,
    )?;

    assert!(result);
    Ok(())
}
