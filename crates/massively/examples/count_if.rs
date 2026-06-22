mod common;

use massively::{Executor, Wgpu, count_if};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;

    let count = count_if(&exec, massively::SoA1(values.slice(..)), common::Positive)?;

    assert_eq!(count, 2);
    Ok(())
}
