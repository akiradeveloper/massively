use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, fill};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;

    fill(&exec, (0.0,), SoA1(values.slice_mut(1..3)))?;

    assert_eq!(exec.to_host(&values)?, vec![1.0, 0.0, 0.0, 4.0]);
    Ok(())
}
