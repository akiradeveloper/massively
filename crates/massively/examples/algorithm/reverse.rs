use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, reverse};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0])?;

    let output = exec.to_device(&[0.0_f32; 3])?;
    reverse(&exec, Zip1(values.slice(..)), Zip1(output.slice_mut(..)))?;

    assert_eq!(exec.to_host(&output)?, vec![3.0, 2.0, 1.0]);
    Ok(())
}
