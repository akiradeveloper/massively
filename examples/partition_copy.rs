mod common;

use massively::{CubeWgpu, partition_copy, unzip};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;

    let (matching, failing) = partition_copy(&values, common::Positive)?;
    let matching = unzip(matching)?;
    let failing = unzip(failing)?;

    assert_eq!(matching.to_vec()?, vec![2.0, 4.0]);
    assert_eq!(failing.to_vec()?, vec![-1.0, -3.0]);
    Ok(())
}
