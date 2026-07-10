//! Indexed algorithms built from the canonical permutation expression.

use cubecl::prelude::Runtime;

use crate::{
    Dispatch, Error, Executor, MAlloc, MStorage, Permute, ReadExpression, ReverseCounting,
    StorageLayout,
    allocation::NormalizeInput,
    masked::MaskedCopyInput,
    output::{LowerOutputExpression, OutputExpression, StageOutput},
    read::{Env0, LowerReadExpression},
    reduce::StageRead,
    transform::{MaterializeDispatch, materialize},
};

/// Internal capability proving the combined value/index arity is supported.
#[doc(hidden)]
pub trait GatherInput<R: Runtime, Indices, Output>: ReadExpression + Sized {
    fn gather(self, exec: &Executor<R>, indices: Indices, output: Output) -> Result<(), Error>;
}

impl<R, Values, Indices, Output> GatherInput<R, Indices, Output> for Values
where
    R: Runtime,
    Values: ReadExpression,
    Values::Item: StorageLayout,
    Indices: ReadExpression<Item = u32>,
    Permute<Values, Indices>:
        ReadExpression<Item = Values::Item> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Item: crate::WriteFrom<Values::Item>,
    Dispatch<<Permute<Values, Indices> as ReadExpression>::ReadArity, Output::StorageArity>:
        MaterializeDispatch<
                R,
                Permute<Values, Indices>,
                Output,
                <Permute<Values, Indices> as LowerReadExpression>::Slots,
                Output::Slots,
            >,
{
    fn gather(self, exec: &Executor<R>, indices: Indices, output: Output) -> Result<(), Error> {
        materialize(exec, Permute::new(self, indices), output)
    }
}

pub(crate) fn gather_direct<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: GatherInput<R, Indices, Output>,
{
    values.gather(exec, indices, output)
}

/// Internal public-API capability that normalizes values and indices
/// independently before applying the permutation.
#[doc(hidden)]
pub trait GatherNormalized<R: Runtime, Indices, Output>: NormalizeInput<R> {
    fn gather_normalized(
        self,
        exec: &Executor<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Values, Indices, Output> GatherNormalized<R, Indices, Output> for Values
where
    R: Runtime,
    Values: NormalizeInput<R>,
    Values::Storage: MStorage<R>,
    Indices: NormalizeInput<R> + ReadExpression<Item = u32>,
    Indices::Storage: MStorage<R>,
    <Indices::Storage as MStorage<R>>::Read: ReadExpression<Item = u32>,
    <Values::Storage as MStorage<R>>::Read:
        GatherInput<R, <Indices::Storage as MStorage<R>>::Read, Output>,
{
    fn gather_normalized(
        self,
        exec: &Executor<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error> {
        let values = self.normalize(exec)?;
        let indices = indices.normalize(exec)?;
        gather_direct(exec, values.read(), indices.read(), output)
    }
}

/// Gathers `values[indices[i]]` into preallocated output storage.
pub(crate) fn gather<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: GatherNormalized<R, Indices, Output>,
{
    values.gather_normalized(exec, indices, output)
}

/// Internal public-API capability for masked gather.
#[doc(hidden)]
pub trait GatherWhereInput<R: Runtime, Indices, Stencil, Output>: NormalizeInput<R> {
    fn gather_where_normalized(
        self,
        exec: &Executor<R>,
        indices: Indices,
        stencil: Stencil,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Values, Indices, Stencil, Output> GatherWhereInput<R, Indices, Stencil, Output> for Values
where
    R: Runtime,
    Values: NormalizeInput<R>,
    Values::Item: MAlloc<R>,
    Values::Storage: MStorage<R>,
    <Values::Item as MAlloc<R>>::Storage: MStorage<R>,
    Indices: NormalizeInput<R> + ReadExpression<Item = u32>,
    Indices::Storage: MStorage<R>,
    <Indices::Storage as MStorage<R>>::Read: ReadExpression<Item = u32>,
    <Values::Storage as MStorage<R>>::Read: GatherInput<
            R,
            <Indices::Storage as MStorage<R>>::Read,
            <<Values::Item as MAlloc<R>>::Storage as MStorage<R>>::Write,
        >,
    <<Values::Item as MAlloc<R>>::Storage as MStorage<R>>::Read: MaskedCopyInput<R, Output>,
    Stencil: crate::selection::FlagInput<R>,
    Output: OutputExpression,
{
    fn gather_where_normalized(
        self,
        exec: &Executor<R>,
        indices: Indices,
        stencil: Stencil,
        output: Output,
    ) -> Result<(), Error> {
        let stencil_len = stencil.flag_len()?;
        let output_len = output.logical_len()?;
        if stencil_len != output_len {
            return Err(Error::LengthMismatch {
                left: stencil_len,
                right: output_len,
            });
        }
        let values = self.normalize(exec)?;
        let indices = indices.normalize(exec)?;
        let gathered = exec.alloc::<Values::Item>(output_len);
        gather_direct(exec, values.read(), indices.read(), gathered.write())?;
        let flags = stencil.materialize_flags(exec)?;
        gathered.read().masked_copy(exec, &flags, output)
    }
}

/// Gathers only rows whose stencil is nonzero, preserving other output rows.
pub(crate) fn gather_where<R, Values, Indices, Stencil, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: GatherWhereInput<R, Indices, Stencil, Output>,
{
    values.gather_where_normalized(exec, indices, stencil, output)
}

/// Internal capability proving reverse permutation has a canonical evaluator.
#[doc(hidden)]
pub trait ReverseInput<R: Runtime, Output>: ReadExpression + Sized {
    fn reverse(self, exec: &Executor<R>, output: Output) -> Result<(), Error>;
}

impl<R, Values, Output> ReverseInput<R, Output> for Values
where
    R: Runtime,
    Values: ReadExpression + StageRead<R, Env0>,
    Values::Item: StorageLayout,
    Permute<Values, ReverseCounting>:
        ReadExpression<Item = Values::Item> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Item: crate::WriteFrom<Values::Item>,
    Dispatch<<Permute<Values, ReverseCounting> as ReadExpression>::ReadArity, Output::StorageArity>:
        MaterializeDispatch<
                R,
                Permute<Values, ReverseCounting>,
                Output,
                <Permute<Values, ReverseCounting> as LowerReadExpression>::Slots,
                Output::Slots,
            >,
{
    fn reverse(self, exec: &Executor<R>, output: Output) -> Result<(), Error> {
        let len = self.logical_len()?;
        materialize(exec, Permute::new(self, ReverseCounting::new(len)), output)
    }
}

/// Reverses values into preallocated output storage.
pub(crate) fn reverse<R, Values, Output>(
    exec: &Executor<R>,
    values: Values,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: ReverseInput<R, Output>,
{
    values.reverse(exec, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, MStorage, Permute, Zip};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn gather_seven_columns_uses_eval8_and_reassociates_output() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 2])).collect();
        let indices = exec.to_device(&[2_u32, 0]);
        let values = Zip::new(
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

        gather(&exec, values, indices.column(), output).unwrap();
        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![column as u32 * 10 + 3, column as u32 * 10 + 1]
            );
        }
    }

    #[test]
    fn reverse_seven_columns_uses_reverse_counting_eval8() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 3])).collect();
        let values = Zip::new(
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

        reverse(&exec, values, output).unwrap();
        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![
                    column as u32 * 10 + 3,
                    column as u32 * 10 + 2,
                    column as u32 * 10 + 1
                ]
            );
        }
    }

    #[test]
    fn gather_normalizes_eval8_values_and_lazy_indices_independently() {
        type Seven = (u32, (u32, (u32, (u32, (u32, (u32, u32))))));
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
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
        let values = Permute::new(seven, Counting::new(0, 3));
        let raw_indices = exec.to_device(&[2_u32, 0]);
        let indices = Permute::new(raw_indices.column(), Counting::new(0, 2));
        let output = exec.alloc::<Seven>(2);

        gather(&exec, values, indices, output.write()).unwrap();
        assert_eq!(exec.to_host(&output.0.0.0.0.0.0).unwrap(), vec![3, 1]);
        assert_eq!(exec.to_host(&output.1).unwrap(), vec![63, 61]);
    }

    #[test]
    fn gather_where_preserves_rows_with_zero_stencil() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[10_u32, 20, 30, 40]);
        let indices = exec.to_device(&[3_u32, 2, 1, 0]);
        let stencil = exec.to_device(&[1_u32, 0, 1, 0]);
        let output = exec.to_device(&[100_u32, 200, 300, 400]);

        gather_where(
            &exec,
            values.column(),
            indices.column(),
            stencil.column(),
            output.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![40, 200, 20, 400]);
    }
}
