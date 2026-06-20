mod common;

use massively::{CubeWgpu, adjacent_difference};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[1.0_f32, 3.0, 6.0, 10.0])?;

    let (output,) = adjacent_difference((&values,), common::SumF32)?;

    println!("adjacent_difference with SumF32: {:?}", output.to_vec()?);
    Ok(())
}
