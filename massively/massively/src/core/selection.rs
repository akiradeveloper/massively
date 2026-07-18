//! Stable selection composed from scan, permutation, and materialization.

use cubecl::prelude::*;

use crate::{
    A1, Column, Constant, DeviceSliceMut, DeviceVec, Dispatch, Error, Executor, ReadExpression,
    RowStorage, S1, StorageLayout, Transform, Zip,
    indexed::GatherInput,
    op::UnaryOp,
    output::{LowerOutputExpression, OutputExpression, SliceOutput, StageOutput},
    read::{Env0, Env1, LowerReadExpression},
    reduce::{ReductionOp, StageRead},
    scan::{InclusiveScanDispatch, inclusive_scan, inclusive_scan_u32, last_u32},
    storage::Concat,
    transform::{MaterializeDispatch, materialize},
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked)]
fn selected_indices_kernel(positions: &[u32], len: &[u32], invert: &[u32], indices: &mut [u32]) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        let previous = if index == 0usize {
            0u32
        } else {
            positions[index - 1usize]
        };
        let passes = positions[index] != previous;
        let selected = if invert[0] != 0u32 { !passes } else { passes };
        if selected {
            let rank = if invert[0] != 0u32 {
                index as u32 + 1u32 - positions[index]
            } else {
                positions[index]
            };
            indices[(rank - 1u32) as usize] = index as u32;
        }
    }
}

struct IsTrue;

#[cubecl::cube]
impl UnaryOp<bool> for IsTrue {
    type Output = u32;
    fn apply(input: bool) -> u32 {
        if input { 1u32 } else { 0u32 }
    }
}

struct IsFalse;

#[cubecl::cube]
impl UnaryOp<bool> for IsFalse {
    type Output = u32;
    fn apply(input: bool) -> u32 {
        if input { 0u32 } else { 1u32 }
    }
}

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

/// Recursive fill over output leaves.
#[doc(hidden)]
pub trait FillOutput<R: Runtime>: OutputExpression + Sized {
    fn fill_output(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error>;
}

impl<R, T> FillOutput<R> for DeviceSliceMut<T>
where
    R: Runtime,
    T: crate::MStorageElement + StorageLayout<StorageArity = S1>,
    Constant<T>: ReadExpression<Item = T, ReadArity = A1>
        + LowerReadExpression<Slots = Env1<T>>
        + StageRead<R, Env0>,
    Dispatch<crate::A13, crate::S12>: MaterializeDispatch<
            R,
            Constant<T>,
            DeviceSliceMut<T>,
            crate::read::KernelReadSlots<Env1<T>>,
            crate::output::KernelOutputSlots<Env1<T>>,
        >,
    DeviceSliceMut<T>: OutputExpression<Item = T, StorageArity = S1>
        + LowerOutputExpression<Slots = Env1<T>>
        + StageOutput<R, Env0>,
{
    fn fill_output(self, exec: &Executor<R>, value: T) -> Result<(), Error> {
        let len = self.len();
        materialize(exec, Constant::new(value, len), self)
    }
}

impl<R, Left, Right> FillOutput<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: FillOutput<R>,
    Right: FillOutput<R>,
    Zip<Left, Right>: OutputExpression,
    <Left::Item as StorageLayout>::StorageLeaves: Concat<
            <Right::Item as StorageLayout>::StorageLeaves,
            Output = <<Zip<Left, Right> as OutputExpression>::Item as StorageLayout>::StorageLeaves,
        >,
{
    fn fill_output(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error> {
        let left_len = self.0.logical_len()?;
        let right_len = self.1.logical_len()?;
        if left_len != right_len {
            return Err(Error::LengthMismatch {
                left: left_len,
                right: right_len,
            });
        }
        let (left, right) =
            <Left::Item as StorageLayout>::StorageLeaves::split(value.into_storage_leaves());
        self.0
            .fill_output(exec, Left::Item::from_storage_leaves(left))?;
        self.1
            .fill_output(exec, Right::Item::from_storage_leaves(right))
    }
}

impl<R, Output> FillOutput<R> for crate::output::Slice<R, Output>
where
    R: Runtime,
    Output: FillOutput<R>,
{
    fn fill_output(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error> {
        self.into_inner().fill_output(exec, value)
    }
}

/// Internal capability to consume a logical stencil expression.
#[doc(hidden)]
pub trait FlagInput<R: Runtime>: ReadExpression<Item = bool> + Sized {
    fn flag_len(&self) -> Result<usize, Error>;
    fn selected_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error>;
    fn rejected_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error>;
}

impl<R, Stencil> FlagInput<R> for Stencil
where
    R: Runtime,
    Stencil: ReadExpression<Item = bool> + LowerReadExpression + StageRead<R, Env0>,
    Transform<Stencil, IsTrue>:
        ReadExpression<Item = u32> + LowerReadExpression + StageRead<R, Env0>,
    Transform<Stencil, IsFalse>:
        ReadExpression<Item = u32> + LowerReadExpression + StageRead<R, Env0>,
    Dispatch<crate::A13, crate::S12>: InclusiveScanDispatch<
            R,
            Transform<Stencil, IsTrue>,
            DeviceSliceMut<u32>,
            u32,
            crate::read::KernelReadSlots<
                <Transform<Stencil, IsTrue> as LowerReadExpression>::Slots,
            >,
            crate::output::KernelOutputSlots<Env1<u32>>,
            SumU32,
        >,
    Dispatch<crate::A13, crate::S12>: InclusiveScanDispatch<
            R,
            Transform<Stencil, IsFalse>,
            DeviceSliceMut<u32>,
            u32,
            crate::read::KernelReadSlots<
                <Transform<Stencil, IsFalse> as LowerReadExpression>::Slots,
            >,
            crate::output::KernelOutputSlots<Env1<u32>>,
            SumU32,
        >,
    DeviceSliceMut<u32>: StageOutput<R, Env0>,
{
    fn flag_len(&self) -> Result<usize, Error> {
        self.logical_len()
    }

    fn selected_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error> {
        let len = self.logical_len()?;
        let positions = exec.alloc_row::<u32>(len);
        inclusive_scan(
            exec,
            Transform::new(self, IsTrue),
            SumU32,
            positions.slice_mut(..),
        )?;
        SelectionControl::from_positions(exec, positions, false)
    }

    fn rejected_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error> {
        let len = self.logical_len()?;
        let positions = exec.alloc_row::<u32>(len);
        inclusive_scan(
            exec,
            Transform::new(self, IsFalse),
            SumU32,
            positions.slice_mut(..),
        )?;
        SelectionControl::from_positions(exec, positions, false)
    }
}

/// Stable selected-row control shared by every payload that uses the same
/// flags.  Keeping this separate from the payload prevents by-key algorithms
/// from coupling key arity to value arity.
#[doc(hidden)]
pub struct SelectionControl<R: Runtime> {
    len: usize,
    indices: DeviceVec<R, u32>,
    count: u32,
}

impl<R: Runtime> SelectionControl<R> {
    pub(crate) fn from_indices(len: usize, indices: DeviceVec<R, u32>, count: u32) -> Self {
        Self {
            len,
            indices,
            count,
        }
    }

    pub(crate) fn from_flags(exec: &Executor<R>, flags: DeviceVec<R, u32>) -> Result<Self, Error> {
        let positions = inclusive_scan_u32(exec, &flags)?;
        Self::from_positions(exec, positions, false)
    }

    pub(crate) fn split_from_positions(
        exec: &Executor<R>,
        positions: DeviceVec<R, u32>,
    ) -> Result<(Self, Self), Error> {
        let selected = last_u32(exec, &positions)?;
        let len = u32::try_from(positions.len()).map_err(|_| Error::LengthTooLarge {
            len: positions.len(),
        })?;
        let passing = Self::from_positions_with_count(exec, &positions, selected, false)?;
        let failing = Self::from_positions_with_count(exec, &positions, len - selected, true)?;
        Ok((passing, failing))
    }

    pub(crate) fn from_positions(
        exec: &Executor<R>,
        positions: DeviceVec<R, u32>,
        invert: bool,
    ) -> Result<Self, Error> {
        let selected = last_u32(exec, &positions)?;
        let count = if invert {
            u32::try_from(positions.len()).map_err(|_| Error::LengthTooLarge {
                len: positions.len(),
            })? - selected
        } else {
            selected
        };
        Self::from_positions_with_count(exec, &positions, count, invert)
    }

    pub(crate) fn from_positions_with_count(
        exec: &Executor<R>,
        positions: &DeviceVec<R, u32>,
        count: u32,
        invert: bool,
    ) -> Result<Self, Error> {
        let indices = exec.alloc_row::<u32>(count as usize);
        if count != 0 {
            let len = u32::try_from(positions.len()).map_err(|_| Error::LengthTooLarge {
                len: positions.len(),
            })?;
            let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
            let invert_handle = exec
                .client()
                .create_from_slice(u32::as_bytes(&[u32::from(invert)]));
            unsafe {
                selected_indices_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(positions.len().div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(positions.handle.clone(), positions.len()),
                    BufferArg::from_raw_parts(len_handle, 1),
                    BufferArg::from_raw_parts(invert_handle, 1),
                    BufferArg::from_raw_parts(indices.handle.clone(), indices.len()),
                );
            }
        }
        Ok(Self {
            len: positions.len(),
            indices,
            count,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    pub(crate) fn indices(&self) -> &DeviceVec<R, u32> {
        &self.indices
    }
}

/// Internal capability for copying selected rows from a readable expression.
#[doc(hidden)]
pub trait CopySelected<R: Runtime, Output>: ReadExpression + Sized {
    fn source_len(&self) -> Result<usize, Error>;
    fn copy_selected(
        self,
        exec: &Executor<R>,
        control: &SelectionControl<R>,
        output: Output,
    ) -> Result<u32, Error>;
}

impl<R, Input, Output> CopySelected<R, Output> for Input
where
    R: Runtime,
    Input:
        GatherInput<R, Transform<Column<u32>, crate::op::U32ToUsize>, Output> + StageRead<R, Env0>,
    Output: OutputExpression,
{
    fn source_len(&self) -> Result<usize, Error> {
        self.logical_len()
    }

    fn copy_selected(
        self,
        exec: &Executor<R>,
        control: &SelectionControl<R>,
        output: Output,
    ) -> Result<u32, Error> {
        let source_len = self.logical_len()?;
        if source_len != control.len() {
            return Err(Error::LengthMismatch {
                left: source_len,
                right: control.len(),
            });
        }
        let count = control.count();
        let output_len = output.logical_len()?;
        if output_len < count as usize {
            return Err(Error::OutputTooShort {
                input: count as usize,
                output: output_len,
            });
        }
        if count == 0 {
            return Ok(0);
        }
        self.gather(
            exec,
            Transform::new(control.indices.column(), crate::op::U32ToUsize),
            output,
        )?;
        Ok(count)
    }
}

/// Stably copies values whose stencil is nonzero.
pub(crate) fn copy_where<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<u32, Error>
where
    R: Runtime,
    Input: CopySelected<R, Output>,
    Stencil: FlagInput<R>,
{
    let input_len = input.source_len()?;
    let stencil_len = stencil.flag_len()?;
    if input_len != stencil_len {
        return Err(Error::LengthMismatch {
            left: input_len,
            right: stencil_len,
        });
    }
    let control = stencil.selected_control(exec)?;
    input.copy_selected(exec, &control, output)
}

/// Stably copies values whose stencil is zero.
pub(crate) fn remove_where<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<u32, Error>
where
    R: Runtime,
    Input: CopySelected<R, Output>,
    Stencil: FlagInput<R>,
{
    let input_len = input.source_len()?;
    let stencil_len = stencil.flag_len()?;
    if input_len != stencil_len {
        return Err(Error::LengthMismatch {
            left: input_len,
            right: stencil_len,
        });
    }
    let control = stencil.rejected_control(exec)?;
    input.copy_selected(exec, &control, output)
}

/// Stably partitions passing values before failing values.
pub(crate) fn partition<R, Input, Pred, Output>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
    output: Output,
) -> Result<u32, Error>
where
    R: Runtime,
    Input: Clone + crate::predicate::PredicateInput<R, Pred> + CopySelected<R, Output>,
    Output: SliceOutput,
{
    let len = input.source_len()?;
    let output_len = output.logical_len()?;
    if output_len < len {
        return Err(Error::OutputTooShort {
            input: len,
            output: output_len,
        });
    }
    let positions = input.clone().predicate_positions(exec)?;
    let (passing_control, failing_control) =
        SelectionControl::split_from_positions(exec, positions)?;
    let passing = passing_control.count();
    input.clone().copy_selected(
        exec,
        &passing_control,
        output.slice_output(..passing as usize),
    )?;
    input.copy_selected(
        exec,
        &failing_control,
        output.slice_output(passing as usize..len),
    )?;
    Ok(passing)
}

/// Fills every item in an output tree.
pub(crate) fn fill<R, Output>(
    exec: &Executor<R>,
    value: Output::Item,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: FillOutput<R>,
{
    output.fill_output(exec, value)
}

/// Replaces items whose logical stencil is true.
pub(crate) fn replace_where<R, Stencil, Output>(
    exec: &Executor<R>,
    value: Output::Item,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Stencil: FlagInput<R>,
    Output: OutputExpression,
    Output::Item: crate::allocation::ScratchStorage<R>,
    <Output::Item as crate::allocation::ScratchStorage<R>>::Storage: RowStorage<R>,
    <<Output::Item as crate::allocation::ScratchStorage<R>>::Storage as RowStorage<R>>::Write:
        FillOutput<R>,
    <<Output::Item as crate::allocation::ScratchStorage<R>>::Storage as RowStorage<R>>::Read:
        crate::indexed::IndexedCopyInput<R, Transform<Column<u32>, crate::op::U32ToUsize>, Output>,
{
    let stencil_len = stencil.flag_len()?;
    let output_len = output.logical_len()?;
    if stencil_len != output_len {
        return Err(Error::LengthMismatch {
            left: stencil_len,
            right: output_len,
        });
    }
    let control = stencil.selected_control(exec)?;
    if control.count() == 0 {
        return Ok(());
    }
    let replacements = <Output::Item as crate::allocation::ScratchStorage<R>>::alloc_scratch(
        exec,
        control.count() as usize,
    );
    replacements.write().fill_output(exec, value)?;
    crate::core::scatter::scatter(
        exec,
        replacements.read(),
        Transform::new(control.indices().column(), crate::op::U32ToUsize),
        output,
    )
}

/// Internal capability for a transform selected by a logical stencil.
#[doc(hidden)]
pub trait TransformWhereInput<R: Runtime, Stencil, Output, Op>: ReadExpression + Sized {
    fn transform_where(
        self,
        exec: &Executor<R>,
        op: Op,
        stencil: Stencil,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Input, Stencil, Output, Op> TransformWhereInput<R, Stencil, Output, Op> for Input
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env0>,
    Op: UnaryOp<Input::Item>,
    Transform<Input, Op>: ReadExpression<Item = Op::Output>,
    Transform<Input, Op>: crate::indexed::IndexedCopyInput<
            R,
            Transform<crate::Counting, crate::op::U32ToUsize>,
            Output,
        >,
    Stencil: FlagInput<R>,
    Output: OutputExpression,
{
    fn transform_where(
        self,
        exec: &Executor<R>,
        op: Op,
        stencil: Stencil,
        output: Output,
    ) -> Result<(), Error> {
        let input_len = self.logical_len()?;
        let stencil_len = stencil.flag_len()?;
        let output_len = output.logical_len()?;
        if input_len != stencil_len {
            return Err(Error::LengthMismatch {
                left: input_len,
                right: stencil_len,
            });
        }
        if input_len != output_len {
            return Err(Error::LengthMismatch {
                left: input_len,
                right: output_len,
            });
        }
        let control = stencil.selected_control(exec)?;
        if control.count() == 0 {
            return Ok(());
        }
        crate::indexed::IndexedCopyInput::indexed_copy_selected(
            Transform::new(self, op),
            exec,
            Transform::new(crate::Counting::new(0, input_len), crate::op::U32ToUsize),
            Some(control.indices()),
            false,
            output,
        )
    }
}

/// Applies `op` only where the logical stencil is true.
pub(crate) fn transform_where<R, Input, Stencil, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: TransformWhereInput<R, Stencil, Output, Op>,
{
    input.transform_where(exec, op, stencil, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, Permute, Zip};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn copy_where_preserves_flat_three_column_rows() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[1_u32, 2, 3, 4]);
        let b = exec.to_device(&[10_f32, 20.0, 30.0, 40.0]);
        let c = exec.to_device(&[100_i32, 200, 300, 400]);
        let flags = exec.to_device(&[0_u32, 1, 1, 0]);
        let out_a = exec.to_device(&[0_u32; 4]);
        let out_b = exec.to_device(&[0_f32; 4]);
        let out_c = exec.to_device(&[0_i32; 4]);
        let input = Zip::new(a.column(), Zip::new(b.column(), c.column()));
        let output = Zip::new(
            Zip::new(out_a.slice_mut(..), out_b.slice_mut(..)),
            out_c.slice_mut(..),
        );

        let count = copy_where(
            &exec,
            input,
            Transform::new(flags.column(), crate::op::U32ToBool),
            output,
        )
        .unwrap();
        assert_eq!(count, 2);
        assert_eq!(
            exec.to_host(&out_a.slice(..count as usize)).unwrap(),
            vec![2, 3]
        );
        assert_eq!(
            exec.to_host(&out_b.slice(..count as usize)).unwrap(),
            vec![20.0, 30.0]
        );
        assert_eq!(
            exec.to_host(&out_c.slice(..count as usize)).unwrap(),
            vec![200, 300]
        );
    }

    #[test]
    fn fused_stencil_scan_produces_binary_positions() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let flags = exec.to_device(&[0_u32, 7, 3, 0]);
        let positions = exec.alloc_row::<u32>(4);
        inclusive_scan(
            &exec,
            Transform::new(Transform::new(flags.column(), crate::op::U32ToBool), IsTrue),
            SumU32,
            positions.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&positions).unwrap(), vec![0, 1, 2, 2]);
    }

    #[test]
    fn copy_where_on_seven_leaves_dispatches_eval8_after_permute() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 3])).collect();
        let flags = exec.to_device(&[1_u32, 0, 1]);
        let input = Zip::new(
            inputs[0].column(),
            Zip::new(
                inputs[1].column(),
                Zip::new(
                    inputs[2].column(),
                    Zip::new(
                        inputs[3].column(),
                        Zip::new(
                            inputs[4].column(),
                            Zip::new(inputs[5].column(), inputs[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let output = Zip::new(
            Zip::new(
                Zip::new(
                    Zip::new(
                        Zip::new(
                            Zip::new(outputs[0].slice_mut(..), outputs[1].slice_mut(..)),
                            outputs[2].slice_mut(..),
                        ),
                        outputs[3].slice_mut(..),
                    ),
                    outputs[4].slice_mut(..),
                ),
                outputs[5].slice_mut(..),
            ),
            outputs[6].slice_mut(..),
        );

        assert_eq!(
            copy_where(
                &exec,
                input,
                Transform::new(flags.column(), crate::op::U32ToBool),
                output,
            )
            .unwrap(),
            2
        );
        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(&output.slice(..2)).unwrap(),
                vec![column as u32 * 10 + 1, column as u32 * 10 + 3]
            );
        }
    }

    #[test]
    fn remove_where_inverts_nonzero_stencil() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[10_u32, 20, 30, 40]);
        let flags = exec.to_device(&[0_u32, 1, 0, 1]);
        let output = exec.to_device(&[0_u32; 4]);
        let count = remove_where(
            &exec,
            input.column(),
            Transform::new(flags.column(), crate::op::U32ToBool),
            output.slice_mut(..),
        )
        .unwrap();
        assert_eq!(count, 2);
        assert_eq!(exec.to_host(&output.slice(..2)).unwrap(), vec![10, 30]);
    }

    struct IsEven;

    #[cubecl::cube]
    impl crate::op::PredicateOp<u32> for IsEven {
        fn apply(input: u32) -> bool {
            input % 2u32 == 0u32
        }
    }

    #[test]
    fn partition_is_stable_on_both_sides() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[3_u32, 2, 4, 1, 6, 5]);
        let output = exec.to_device(&[0_u32; 6]);
        let split = partition(&exec, input.column(), IsEven, output.slice_mut(..)).unwrap();
        assert_eq!(split, 3);
        assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4, 6, 3, 1, 5]);
    }

    #[test]
    fn fill_and_replace_where_recurse_over_binary_output_tree() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[0_u32; 4]);
        let b = exec.to_device(&[0_f32; 4]);
        let c = exec.to_device(&[0_i32; 4]);
        let output = || Zip::new(Zip::new(a.slice_mut(..), b.slice_mut(..)), c.slice_mut(..));
        fill(&exec, (7_u32, 2.5_f32, -3_i32), output()).unwrap();
        let flags = exec.to_device(&[0_u32, 1, 0, 1]);
        replace_where(
            &exec,
            (9_u32, 4.5_f32, -8_i32),
            Transform::new(flags.column(), crate::op::U32ToBool),
            output(),
        )
        .unwrap();
        assert_eq!(exec.to_host(&a).unwrap(), vec![7, 9, 7, 9]);
        assert_eq!(exec.to_host(&b).unwrap(), vec![2.5, 4.5, 2.5, 4.5]);
        assert_eq!(exec.to_host(&c).unwrap(), vec![-3, -8, -3, -8]);
    }

    type Seven = (u32, u32, u32, u32, u32, u32, u32);

    struct IncrementSeven;

    #[cubecl::cube]
    impl UnaryOp<Seven> for IncrementSeven {
        type Output = Seven;
        fn apply(input: Seven) -> Seven {
            (
                input.0 + 1u32,
                input.1 + 1u32,
                input.2 + 1u32,
                input.3 + 1u32,
                input.4 + 1u32,
                input.5 + 1u32,
                input.6 + 1u32,
            )
        }
    }

    #[test]
    fn transform_where_normalizes_eval8_before_selected_storage7_copy() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[100_u32; 3])).collect();
        let stencil = exec.to_device(&[1_u32, 0, 1]);
        let seven = Zip::new(
            inputs[0].column(),
            Zip::new(
                inputs[1].column(),
                Zip::new(
                    inputs[2].column(),
                    Zip::new(
                        inputs[3].column(),
                        Zip::new(
                            inputs[4].column(),
                            Zip::new(inputs[5].column(), inputs[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(
            seven,
            Transform::new(Counting::new(0, 3), crate::op::U32ToUsize),
        );
        let output = Zip::new(
            Zip::new(
                Zip::new(
                    Zip::new(
                        Zip::new(
                            Zip::new(outputs[0].slice_mut(..), outputs[1].slice_mut(..)),
                            outputs[2].slice_mut(..),
                        ),
                        outputs[3].slice_mut(..),
                    ),
                    outputs[4].slice_mut(..),
                ),
                outputs[5].slice_mut(..),
            ),
            outputs[6].slice_mut(..),
        );

        transform_where(
            &exec,
            input,
            IncrementSeven,
            Transform::new(stencil.column(), crate::op::U32ToBool),
            output,
        )
        .unwrap();
        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![column as u32 * 10 + 2, 100, column as u32 * 10 + 4]
            );
        }
    }
}
