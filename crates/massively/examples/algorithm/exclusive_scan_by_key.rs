mod common;

use massively::{Executor, Wgpu, exclusive_scan_by_key};

fn main() -> common::Result {
    let exec = Executor::<Wgpu>::cpu();
    let keys = exec.to_device(&[0_u32, 0, 1, 1])?;
    let values = exec.to_device(&[1.0_f32, 2.0, 10.0, 20.0])?;

    let (output,) = exclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        common::EqualU32,
        (0.0,),
        common::SumF32,
    )?;

    assert_eq!(exec.to_host(&output)?, vec![0.0, 1.0, 0.0, 10.0]);
    Ok(())
}
