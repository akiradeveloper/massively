use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, exclusive_scan_by_key};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let keys = exec.to_device(&[0_u32, 0, 1, 1])?;
    let values = exec.to_device(&[1.0_f32, 2.0, 10.0, 20.0])?;

    let output = exec.to_device(&[0.0_f32; 4])?;
    exclusive_scan_by_key(
        &exec,
        Zip1(keys.slice(..)),
        Zip1(values.slice(..)),
        common::EqualU32,
        (0.0,),
        common::SumF32,
        Zip1(output.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&output)?, vec![0.0, 1.0, 0.0, 10.0]);
    Ok(())
}
