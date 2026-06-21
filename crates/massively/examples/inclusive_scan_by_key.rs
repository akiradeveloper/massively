mod common;

use massively::{Executor, Wgpu, inclusive_scan_by_key};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let keys = exec.to_device(&[0_u32, 0, 1, 1])?;
    let values = exec.to_device(&[1.0_f32, 2.0, 10.0, 20.0])?;

    let (output,) = inclusive_scan_by_key(
        &exec,
        (keys.slice(..),),
        (values.slice(..),),
        common::EqualU32,
        common::SumF32,
    )?;

    assert_eq!(exec.to_host(&output)?, vec![1.0, 3.0, 10.0, 30.0]);
    Ok(())
}
