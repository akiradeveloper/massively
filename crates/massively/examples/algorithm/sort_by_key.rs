use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, sort_by_key};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let keys = exec.to_device(&[2_u32, 0, 1])?;
    let values = exec.to_device(&[20.0_f32, 0.0, 10.0])?;

    let out_keys = exec.to_device(&[0_u32; 3])?;
    let out_values = exec.to_device(&[0.0_f32; 3])?;
    sort_by_key(
        &exec,
        SoA1(keys.slice(..)),
        SoA1(values.slice(..)),
        common::LessU32,
        SoA1(out_keys.slice_mut(..)),
        SoA1(out_values.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&out_keys)?, vec![0, 1, 2]);
    assert_eq!(exec.to_host(&out_values)?, vec![0.0, 10.0, 20.0]);
    Ok(())
}
