use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::{Executor, lower_bound};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let sorted = exec.to_device(&[1.0_f32, 2.0, 2.0, 4.0])?;

    let index = lower_bound(
        &exec,
        massively::SoA1(sorted.slice(..)),
        (2.0,),
        common::LessF32,
    )?;

    assert_eq!(index, 1);
    Ok(())
}
