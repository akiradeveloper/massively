use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::{Executor, is_sorted_until};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[1.0_f32, 2.0, 4.0, 3.0])?;

    let index = is_sorted_until(&exec, massively::SoA1(values.slice(..)), common::LessF32)?;

    assert_eq!(index, 3);
    Ok(())
}
