use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::Executor;

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.full(4, 7_u32)?;

    assert_eq!(exec.to_host(&values)?, vec![7, 7, 7, 7]);
    Ok(())
}
