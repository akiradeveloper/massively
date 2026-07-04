use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, inclusive_scan_by_key};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let keys = exec.to_device(&[0_u32, 0, 1, 1])?;
    let values = exec.to_device(&[1.0_f32, 2.0, 10.0, 20.0])?;

    let output = exec.to_device(&[0.0_f32; 4])?;
    inclusive_scan_by_key(
        &exec,
        SoA1(keys.slice(..)),
        SoA1(values.slice(..)),
        common::EqualU32,
        common::SumF32,
        SoA1(output.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&output)?, vec![1.0, 3.0, 10.0, 30.0]);
    Ok(())
}
