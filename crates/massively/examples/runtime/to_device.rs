use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::Executor;

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1_u32, 2, 3, 4])?;

    assert_eq!(exec.to_host(&values)?, vec![1, 2, 3, 4]);
    Ok(())
}
