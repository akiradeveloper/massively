mod common;

use massively::{Executor, Wgpu, is_partitioned};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[2.0_f32, 4.0, -1.0, -3.0])?;

    let result = is_partitioned(&exec, massively::SoA1(values.slice(..)), common::Positive)?;

    assert!(result);
    Ok(())
}
