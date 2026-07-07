use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, gather_where};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[10.0_f32, -20.0, 30.0])?;
    let indices = exec.to_device(&[2_u32, 0, 1])?;
    let stencil = common::bool_stencil(3, common::IndexLt2);

    let output = exec.to_device(&[0.0_f32; 3])?;
    gather_where(
        &exec,
        Zip1(values.slice(..)),
        indices.slice(..),
        stencil,
        Zip1(output.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&output)?, vec![30.0, 10.0, 0.0]);
    Ok(())
}
