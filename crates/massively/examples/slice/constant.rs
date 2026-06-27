use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::slice::constant_slice;
use massively::{Executor, copy_where};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;

    let stencil = constant_slice(4, 1);
    let (output,) = copy_where(&exec, massively::SoA1(values.slice(..)), stencil)?;

    assert_eq!(exec.to_host(&output)?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}
