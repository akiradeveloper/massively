mod common;

use massively::{CubeWgpu, count_if};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;

    let count = count_if((&values,), common::Positive)?;

    assert_eq!(count, 2);
    Ok(())
}
