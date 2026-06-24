use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::{Executor, none_of};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[-1.0_f32, -2.0, -3.0])?;

    let result = none_of(&exec, massively::SoA1(values.slice(..)), common::Positive)?;

    assert!(result);
    Ok(())
}
