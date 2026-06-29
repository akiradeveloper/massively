use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, inner_product};

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let x = exec.to_device(&[1.0_f32, 2.0, 3.0])?;
    let y = exec.to_device(&[10.0_f32, 20.0, 30.0])?;

    let dot = inner_product(
        &exec,
        SoA1(x.slice(..)),
        SoA1(y.slice(..)),
        common::PairProduct,
        (0.0,),
        common::SumF32,
    )?;

    assert_eq!(dot, (140.0,));
    Ok(())
}
