use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::Executor;

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.tabulate(4)?;

    assert_eq!(exec.to_host(&values)?, vec![0, 1, 2, 3]);
    Ok(())
}
