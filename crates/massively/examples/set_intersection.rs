mod common;

use massively::{Executor, Wgpu, set_intersection};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let left = exec.to_device(&[1.0_f32, 2.0, 4.0])?;
    let right = exec.to_device(&[2.0_f32, 3.0])?;

    let (output,) = set_intersection(
        &exec,
        massively::SoA1(left.slice(..)),
        massively::SoA1(right.slice(..)),
        common::LessF32,
    )?;

    println!("set intersection: {:?}", exec.to_host(&output)?);
    Ok(())
}
