use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, inclusive_scan};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;

    let output = exec.to_device(&[0.0_f32; 4])?;
    inclusive_scan(
        &exec,
        SoA1(values.slice(..)),
        common::TupleSumF32,
        SoA1(output.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&output)?, vec![1.0, 3.0, 6.0, 10.0]);
    Ok(())
}
