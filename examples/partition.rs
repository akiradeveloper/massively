mod common;

use massively::{CubeWgpu, partition};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;

    let output = partition(&values, common::Positive)?;

    println!("partitioned positives first: {:?}", output.to_vec()?);
    Ok(())
}
