use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, partition};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;

    let output = exec.to_device(&[0.0_f32; 4])?;
    let split = partition(
        &exec,
        SoA1(values.slice(..)),
        common::Positive,
        (),
        SoA1(output.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&output.slice(..split))?, vec![2.0, 4.0]);
    assert_eq!(exec.to_host(&output.slice(split..4))?, vec![-1.0, -3.0]);
    Ok(())
}
