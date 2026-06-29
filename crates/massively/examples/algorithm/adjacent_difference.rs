use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, adjacent_difference};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 3.0, 6.0, 10.0])?;

    let SoA1(output) = adjacent_difference(&exec, SoA1(values.slice(..)), common::SumF32)?;

    println!(
        "adjacent_difference with SumF32: {:?}",
        exec.to_host(&output)?
    );
    Ok(())
}
