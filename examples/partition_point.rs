mod common;

use massively::{CubeWgpu, partition_point};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[2.0_f32, 4.0, -1.0, -3.0])?;

    let point = partition_point(&values, common::Positive)?;

    assert_eq!(point, 2);
    Ok(())
}
