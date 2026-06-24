use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::{Executor, partition};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0])?;

    let ((positives,), (non_positives,)) =
        partition(&exec, massively::SoA1(values.slice(..)), common::Positive)?;

    assert_eq!(exec.to_host(&positives)?, vec![2.0, 4.0]);
    assert_eq!(exec.to_host(&non_positives)?, vec![-1.0, -3.0]);
    Ok(())
}
