use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, sort};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[3.0_f32, 1.0, 2.0])?;

    let output = exec.to_device(&[0.0_f32; 3])?;
    sort(
        &exec,
        SoA1(values.slice(..)),
        common::LessF32,
        SoA1(output.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&output)?, vec![1.0, 2.0, 3.0]);
    Ok(())
}
