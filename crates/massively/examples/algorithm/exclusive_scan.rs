use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, exclusive_scan};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;

    let SoA1(output) = exclusive_scan(&exec, SoA1(values.slice(..)), (0.0,), common::TupleSumF32)?;

    assert_eq!(exec.to_host(&output)?, vec![0.0, 1.0, 3.0, 6.0]);
    Ok(())
}
