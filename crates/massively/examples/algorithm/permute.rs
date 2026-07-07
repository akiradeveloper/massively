use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, gather, transform};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[10.0_f32, 20.0, 30.0])?;
    let indices = exec.to_device(&[2_u32, 0, 1])?;

    let permuted = exec.to_device(&[0.0_f32; 3])?;
    gather(
        &exec,
        Zip1(values.slice(..)),
        indices.slice(..),
        Zip1(permuted.slice_mut(..)),
    )?;
    let output = exec.to_device(&[0.0_f32; 3])?;
    transform(
        &exec,
        Zip1(permuted.slice(..)),
        common::AddOne,
        Zip1(output.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&output)?, vec![31.0, 11.0, 21.0]);
    Ok(())
}
