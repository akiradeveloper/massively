use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, is_partitioned};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[2.0_f32, 4.0, -1.0, -3.0])?;

    let result = is_partitioned(&exec, Zip1(values.slice(..)), common::Positive, ())?;

    assert!(result);
    Ok(())
}
