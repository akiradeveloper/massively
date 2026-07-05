use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, reduce};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0])?;

    let total = reduce(&exec, Zip1(values.slice(..)), (0.0,), common::TupleSumF32)?;

    assert_eq!(total, (10.0,));
    Ok(())
}
