use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, unique};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0])?;

    let output = exec.to_device(&[0.0_f32; 5])?;
    let len = unique(
        &exec,
        Zip1(values.slice(..)),
        common::EqualF32,
        Zip1(output.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&output.slice(..len))?, vec![1.0, 2.0, 3.0]);
    Ok(())
}
