use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, replace_where};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;
    let stencil = common::bool_stencil(4, common::IndexOdd);

    replace_where(&exec, (0.0,), stencil, Zip1(values.slice_mut(..)))?;

    assert_eq!(exec.to_host(&values)?, vec![-1.0, 0.0, -3.0, 0.0]);
    Ok(())
}
