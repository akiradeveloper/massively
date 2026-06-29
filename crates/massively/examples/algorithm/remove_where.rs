use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, remove_where};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;
    let stencil = exec.to_device(&[0_u32, 1, 0, 1])?;

    let SoA1(output) = remove_where(&exec, SoA1(values.slice(..)), stencil.slice(..))?;

    assert_eq!(exec.to_host(&output)?, vec![-1.0, -3.0]);
    Ok(())
}
