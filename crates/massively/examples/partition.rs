mod common;

use massively::{CubeWgpu, partition};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let values = policy.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;

    let ((positives,), (non_positives,)) = partition((values.slice(..),), common::Positive)?;

    assert_eq!(positives.to_vec()?, vec![2.0, 4.0]);
    assert_eq!(non_positives.to_vec()?, vec![-1.0, -3.0]);
    Ok(())
}
