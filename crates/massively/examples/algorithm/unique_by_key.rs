mod common;

use massively::{Executor, Wgpu, unique_by_key};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let keys = exec.to_device(&[0_u32, 0, 1, 1, 2])?;
    let values = exec.to_device(&[0.0_f32, 1.0, 10.0, 11.0, 20.0])?;

    let ((keys,), (values,)) = unique_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        common::EqualU32,
    )?;

    assert_eq!(exec.to_host(&keys)?, vec![0, 1, 2]);
    assert_eq!(exec.to_host(&values)?, vec![0.0, 10.0, 20.0]);
    Ok(())
}
