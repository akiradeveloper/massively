use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, reduce_by_key};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let keys = exec.to_device(&[0_u32, 0, 1, 1, 1])?;
    let values = exec.to_device(&[1.0_f32, 2.0, 10.0, 20.0, 30.0])?;

    let (SoA1(out_keys), SoA1(out_values)) = reduce_by_key(
        &exec,
        SoA1(keys.slice(..)),
        SoA1(values.slice(..)),
        common::EqualU32,
        (0.0,),
        common::SumF32,
    )?;

    assert_eq!(exec.to_host(&out_keys)?, vec![0, 1]);
    assert_eq!(exec.to_host(&out_values)?, vec![3.0, 60.0]);
    Ok(())
}
