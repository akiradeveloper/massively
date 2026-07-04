use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
mod common;

use massively::prelude::*;
use massively::{Executor, scatter};

fn scatter_to_new<R, Input>(
    exec: &Executor<R>,
    source: Input,
    indices: DeviceSlice<'_, R, MIndex>,
    len: MIndex,
) -> Result<<Input::Item as MAlloc<R>>::Storage, massively::Error>
where
    R: cubecl::prelude::Runtime,
    Input: MIter<R>,
    Input::Item: MAlloc<R>,
    <Input::Item as MAlloc<R>>::Storage: ToSliceMut,
    for<'a> <<Input::Item as MAlloc<R>>::Storage as ToSliceMut>::SliceMut<'a>:
        MIterMut<R, Item = Input::Item>,
{
    let out = exec.alloc::<Input::Item>(len)?;
    scatter(exec, source, indices, out.slice_mut(..))?;
    Ok(out)
}

fn main() -> common::Result {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let scores = exec.to_device(&[10.0_f32, 20.0, 30.0])?;
    let ids = exec.to_device(&[100_u32, 200, 300])?;
    let indices = exec.to_device(&[2_u32, 0, 1])?;

    let SoA2(out_scores, out_ids) = scatter_to_new(
        &exec,
        SoA2(scores.slice(..), ids.slice(..)),
        indices.slice(..),
        3,
    )?;

    assert_eq!(exec.to_host(&out_scores)?, vec![20.0, 30.0, 10.0]);
    assert_eq!(exec.to_host(&out_ids)?, vec![200, 300, 100]);
    Ok(())
}
