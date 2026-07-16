use cubecl::prelude::Runtime;

use crate::{Error, Executor, MIter, MIterMut, MStorage, MVec, ToCanonical, WritableFrom};

/// Gathers `values[indices[i]]` into owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op::U32ToUsize, vector::gather};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let indices = lazy::transform(indices.slice(..), U32ToUsize);
/// let output = gather(&exec, values.slice(..), indices).unwrap();
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
    Item: ToCanonical<R>,
    Indices: MIter<R, Item = usize>,
{
    let len = indices.len()? as usize;
    let output = exec.alloc::<Item>(len);
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
    Indices: MIter<R, Item = usize>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
{
    crate::indexed::gather_direct(
        exec,
        crate::api::iter::lower::<R, _>(values),
        crate::api::iter::lower::<R, _>(indices),
        output.lower_output(),
    )
}

/// Gathers from stored `u32` indices after converting them at the read boundary.
pub(crate) fn gather_raw_into<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
{
    crate::indexed::gather_u32(
        exec,
        crate::api::iter::lower::<R, _>(values),
        crate::api::iter::lower::<R, _>(indices),
        output.lower_output(),
    )
}

/// Gathers selected rows while preserving other output rows.
///
/// A false stencil leaves the corresponding output position unchanged.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{
///     Executor, lazy,
///     op::{U32ToBool, U32ToUsize},
///     vector::gather_where,
/// };
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let stencil = exec.to_device(&[1_u32, 0, 1]);
/// let indices = lazy::transform(indices.slice(..), U32ToUsize);
/// let stencil = lazy::transform(stencil.slice(..), U32ToBool);
/// let output = exec.to_device(&[99_u32, 99, 99]);
///
/// gather_where(
///     &exec,
///     values.slice(..),
///     indices,
///     stencil,
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
    Indices: MIter<R, Item = usize>,
    Stencil: MIter<R, Item = bool>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
{
    let indices_len = indices.len()?;
    let stencil_len = stencil.len()?;
    if indices_len != stencil_len {
        return Err(Error::LengthMismatch {
            left: indices_len,
            right: stencil_len,
        });
    }
    let output_len = output.len()?;
    if output_len < indices_len {
        return Err(Error::OutputTooShort {
            input: indices_len,
            output: output_len,
        });
    }

    let control = crate::selection::FlagInput::selected_control(
        crate::api::iter::lower::<R, _>(stencil),
        exec,
    )?;
    if control.count() == 0 {
        return Ok(());
    }

    let scratch = <Output::Item as crate::allocation::ScratchStorage<R>>::alloc_scratch(
        exec,
        control.count() as usize,
    );
    let scratch_write = crate::output::ReassociatedOutput::<
        _,
        Output::Item,
        <<Output::Item as crate::StorageLayout>::StorageLeaves as crate::output::OutputSlotLayout>::Slots,
    >::new(crate::CanonicalStorage::write(&scratch));
    crate::indexed::IndexedCopyInput::indexed_copy_selected(
        crate::api::iter::lower::<R, _>(values),
        exec,
        crate::api::iter::lower::<R, _>(indices),
        Some(control.indices()),
        true,
        scratch_write,
    )?;
    crate::core::scatter::scatter(
        exec,
        crate::CanonicalStorage::read(&scratch),
        crate::Transform::new(control.indices().column(), crate::op::U32ToUsize),
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
    Item: ToCanonical<R>,
{
    let len = values.len()? as usize;
    let output = exec.alloc::<Item>(len);
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
    crate::indexed::gather_u32(
        exec,
        crate::api::iter::lower_fixed::<R, _>(values),
        crate::ReverseCounting::new(len),
        output.lower_output(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn gather_where_does_not_evaluate_rejected_indices() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[10_u32, 20, 30]);
        let encoded_indices = exec.to_device(&[2_u32, u32::MAX, 1]);
        let encoded_stencil = exec.to_device(&[1_u32, 0, 1]);
        let output = exec.to_device(&[99_u32; 3]);

        gather_where(
            &exec,
            values.slice(..),
            crate::lazy::transform(encoded_indices.slice(..), crate::op::U32ToUsize),
            crate::lazy::transform(encoded_stencil.slice(..), crate::op::U32ToBool),
            output.slice_mut(..),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output).unwrap(), vec![30, 99, 20]);
    }
}
