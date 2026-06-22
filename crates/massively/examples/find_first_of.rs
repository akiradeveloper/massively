mod common;

use massively::{Executor, Wgpu, find_first_of};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let input = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;
    let needles = exec.to_device(&[3.0_f32, 9.0])?;

    let index = find_first_of(
        &exec,
        massively::SoA1(input.slice(..)),
        massively::SoA1(needles.slice(..)),
        common::EqualF32,
    )?;

    assert_eq!(index, Some(2));
    Ok(())
}
