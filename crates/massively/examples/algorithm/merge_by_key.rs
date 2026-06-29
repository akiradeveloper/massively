use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, merge_by_key};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let left_keys = exec.to_device(&[0_u32, 2])?;
    let left_values = exec.to_device(&[0.0_f32, 20.0])?;
    let right_keys = exec.to_device(&[1_u32, 3])?;
    let right_values = exec.to_device(&[10.0_f32, 30.0])?;

    let (SoA1(keys), SoA1(values)) = merge_by_key(
        &exec,
        SoA1(left_keys.slice(..)),
        SoA1(left_values.slice(..)),
        SoA1(right_keys.slice(..)),
        SoA1(right_values.slice(..)),
        common::LessU32,
    )?;

    assert_eq!(exec.to_host(&keys)?, vec![0, 1, 2, 3]);
    assert_eq!(exec.to_host(&values)?, vec![0.0, 10.0, 20.0, 30.0]);
    Ok(())
}
