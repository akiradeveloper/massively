use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, reduce_by_key};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let keys = exec.to_device(&[0_u32, 0, 1, 1, 1])?;
    let values = exec.to_device(&[1.0_f32, 2.0, 10.0, 20.0, 30.0])?;

    let out_keys = exec.to_device(&[0_u32; 5])?;
    let out_values = exec.to_device(&[0.0_f32; 5])?;
    let len = reduce_by_key(
        &exec,
        SoA1(keys.slice(..)),
        SoA1(values.slice(..)),
        common::EqualU32,
        (0.0,),
        common::SumF32,
        SoA1(out_keys.slice_mut(..)),
        SoA1(out_values.slice_mut(..)),
    )?;

    assert_eq!(exec.to_host(&out_keys.slice(..len))?, vec![0, 1]);
    assert_eq!(exec.to_host(&out_values.slice(..len))?, vec![3.0, 60.0]);
    Ok(())
}
