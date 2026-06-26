use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::{Executor, replace_where};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let mut values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;
    let stencil = exec.to_device(&[0_u32, 1, 0, 1])?;

    replace_where(
        &exec,
        (0.0,),
        stencil.slice(..),
        massively::SoA1(values.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&values)?, vec![-1.0, 0.0, -3.0, 0.0]);
    Ok(())
}
