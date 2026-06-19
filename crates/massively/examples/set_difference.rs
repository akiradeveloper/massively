mod common;

use massively::{CubeWgpu, set_difference};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let left = policy.to_device(&[1.0_f32, 2.0, 4.0])?;
    let right = policy.to_device(&[2.0_f32, 3.0])?;

    let (output,) = set_difference(&left, &right, common::LessF32)?;

    println!("set difference: {:?}", output.to_vec()?);
    Ok(())
}
