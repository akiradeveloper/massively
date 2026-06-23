mod common;

use massively::{Executor, Wgpu, partition};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;

    let ((positives,), (non_positives,)) =
        partition(&exec, massively::SoA1(values.slice(..)), common::Positive)?;

    assert_eq!(exec.to_host(&positives)?, vec![2.0, 4.0]);
    assert_eq!(exec.to_host(&non_positives)?, vec![-1.0, -3.0]);
    Ok(())
}
