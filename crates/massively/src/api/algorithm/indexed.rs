use cubecl::prelude::Runtime;

use crate::{Error, Executor, MIndex, MIter, MIterMut, WriteFrom};

/// Gathers `values[indices[i]]` into preallocated output storage.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::gather};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let output = exec.alloc::<u32>(3);
///
/// gather(
///     &exec,
///     values.slice(..),
///     indices.slice(..),
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![30, 10, 20]);
/// ```
pub fn gather<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Values::Item>,
{
    let indices = indices.materialize_u32(exec)?;
    values.indexed_with(exec, indices.column(), None, false, output)
}

/// Gathers selected rows while preserving other output rows.
///
/// A zero stencil leaves the corresponding output position unchanged.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::gather_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let stencil = exec.to_device(&[1_u32, 0, 1]);
/// let output = exec.to_device(&[99_u32, 99, 99]);
///
/// gather_where(
///     &exec,
///     values.slice(..),
///     indices.slice(..),
///     stencil.slice(..),
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![30, 99, 20]);
/// ```
pub fn gather_where<R, Values, Indices, Stencil, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: MIter<R, Item = MIndex>,
    Stencil: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Values::Item>,
{
    let indices = indices.materialize_u32(exec)?;
    let stencil = stencil.materialize_u32(exec)?;
    values.indexed_with(
        exec,
        indices.column(),
        Some(stencil.column()),
        false,
        output,
    )
}

/// Reverses values into preallocated output storage.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::reverse};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let output = exec.alloc::<u32>(input.len());
///
/// reverse(&exec, input.slice(..), output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![4, 3, 2, 1]);
/// ```
pub fn reverse<R, Values, Output>(
    exec: &Executor<R>,
    values: Values,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Values::Item>,
{
    values.reverse_with(exec, output)
}
