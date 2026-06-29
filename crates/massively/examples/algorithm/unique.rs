use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, unique};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0])?;

    let SoA1(output) = unique(&exec, SoA1(values.slice(..)), common::EqualF32)?;

    assert_eq!(exec.to_host(&output)?, vec![1.0, 2.0, 3.0]);
    Ok(())
}
