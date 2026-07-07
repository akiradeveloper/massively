use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, copy_where};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;
    let stencil = common::bool_stencil(4, common::IndexOdd);

    let output = exec.to_device(&[0.0_f32; 4])?;
    let len = copy_where(
        &exec,
        Zip1(values.slice(..)),
        stencil,
        Zip1(output.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&output.slice(..len))?, vec![2.0, 4.0]);
    Ok(())
}
