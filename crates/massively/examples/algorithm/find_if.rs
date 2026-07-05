use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, find_if};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[-1.0_f32, -2.0, 3.0])?;

    let index = find_if(&exec, Zip1(values.slice(..)), common::Positive, ())?;

    assert_eq!(index, Some(2));
    Ok(())
}
