use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, any_of};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[-1.0_f32, -2.0, 3.0])?;

    let result = any_of(&exec, SoA1(values.slice(..)), common::Positive, ())?;

    assert!(result);
    Ok(())
}
