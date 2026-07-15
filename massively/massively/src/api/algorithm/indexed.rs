use cubecl::prelude::Runtime;

use crate::{
    Error, Executor, MIndex, MIter, MIterMut, MStorage, MVec, Materializable, WritableFrom,
};

/// Gathers `values[indices[i]]` into owned device storage.
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
/// let output = gather(&exec, values.slice(..), indices.slice(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![30, 10, 20]);
/// ```
pub fn gather<R, Values, Item, Indices>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Values: MIter<R, Item = Item>,
    Item: Materializable<R>,
    Indices: MIter<R, Item = MIndex>,
{
    let len = indices.len()? as usize;
    let output = exec.alloc_mvec::<Item>(len);
    gather_into(exec, values, indices, output.slice_mut(..))?;
    Ok(output)
}

/// Gathers values into caller-provided storage.
#[doc(hidden)]
pub(crate) fn gather_into<R, Values, Indices, Output>(
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
    Output::Item: WritableFrom<Values::Item>,
{
    crate::indexed::gather_direct(
        exec,
        crate::api::iter::lower_fixed::<R, _>(values),
        crate::api::iter::lower_fixed::<R, _>(indices),
        output.lower_output(),
    )
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
    Output::Item: WritableFrom<Values::Item>,
{
    crate::indexed::gather_where_direct(
        exec,
        crate::api::iter::lower_fixed::<R, _>(values),
        crate::api::iter::lower_fixed::<R, _>(indices),
        crate::api::iter::lower_fixed::<R, _>(stencil),
        output.lower_output(),
    )
}

/// Reverses values into owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::reverse};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let output = reverse(&exec, input.slice(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![4, 3, 2, 1]);
/// ```
pub fn reverse<R, Values, Item>(exec: &Executor<R>, values: Values) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Values: MIter<R, Item = Item>,
    Item: Materializable<R>,
{
    let len = values.len()? as usize;
    let output = exec.alloc_mvec::<Item>(len);
    reverse_into(exec, values, output.slice_mut(..))?;
    Ok(output)
}

/// Reverses values into caller-provided storage.
#[doc(hidden)]
pub(crate) fn reverse_into<R, Values, Output>(
    exec: &Executor<R>,
    values: Values,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
{
    let len = values.len()? as usize;
    crate::indexed::gather_direct(
        exec,
        crate::api::iter::lower_fixed::<R, _>(values),
        crate::ReverseCounting::new(len),
        output.lower_output(),
    )
}
