mod common;

use massively::{CubeWgpu, inclusive_scan_by_key};

fn main() -> common::Result {
    let policy = CubeWgpu::cpu();
    let keys = policy.to_device(&[0_u32, 0, 1, 1])?;
    let values = policy.to_device(&[1.0_f32, 2.0, 10.0, 20.0])?;

    let (output,) = inclusive_scan_by_key(
        (keys.slice(..),),
        (values.slice(..),),
        common::EqualU32,
        common::SumF32,
    )?;

    assert_eq!(output.to_vec()?, vec![1.0, 3.0, 10.0, 30.0]);
    Ok(())
}
