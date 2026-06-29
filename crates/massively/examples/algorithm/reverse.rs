use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, reverse};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0])?;

    let SoA1(output) = reverse(&exec, SoA1(values.slice(..)))?;

    assert_eq!(exec.to_host(&output)?, vec![3.0, 2.0, 1.0]);
    Ok(())
}
