use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, adjacent_difference};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 3.0, 6.0, 10.0])?;

    let output = exec.to_device(&[0.0_f32; 4])?;
    adjacent_difference(
        &exec,
        Zip1(values.slice(..)),
        common::SumF32,
        Zip1(output.slice_mut(..)),
    )?;

    println!(
        "adjacent_difference with SumF32: {:?}",
        exec.to_host(&output)?
    );
    Ok(())
}
