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

    let keys = exec.to_device(&[0_u32; 4])?;
    let values = exec.to_device(&[0.0_f32; 4])?;
    merge_by_key(
        &exec,
        Zip1(left_keys.slice(..)),
        Zip1(left_values.slice(..)),
        Zip1(right_keys.slice(..)),
        Zip1(right_values.slice(..)),
        common::LessU32,
        Zip1(keys.slice_mut(..)),
        Zip1(values.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&keys)?, vec![0, 1, 2, 3]);
    assert_eq!(exec.to_host(&values)?, vec![0.0, 10.0, 20.0, 30.0]);
    Ok(())
}
