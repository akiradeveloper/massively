use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::Executor;

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let source = exec.to_device(&[1_u32, 2, 3, 4])?;
    let destination = exec.filled(6, 0_u32)?;

    exec.copy(source.slice(1..4), destination.slice_mut(2..5))?;

    assert_eq!(exec.to_host(&destination)?, vec![0, 0, 2, 3, 4, 0]);
    Ok(())
}
