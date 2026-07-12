//! Scatter primitives using the fixed 13-read/12-write kernel ABI.

use cubecl::prelude::*;

use crate::{
    A13, Dispatch, Error, Executor, ReadExpression, S12,
    masked::MaskedCopyDispatch,
    output::{
        KernelOutputSlots, LowerOutputExpression, OutputExpression, PaddedOutputSlots, StageOutput,
    },
    read::{Env0, KernelReadSlots, LowerReadExpression},
    reduce::StageRead,
    selection::FlagInput,
};

#[doc(hidden)]
pub trait ScatterInput<R: Runtime, Indices, Output>: ReadExpression + Sized {
    fn scatter_input(
        self,
        exec: &Executor<R>,
        indices: Indices,
        flags: Option<&crate::DeviceVec<R, u32>>,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Values, Indices, Output> ScatterInput<R, Indices, Output> for Values
where
    R: Runtime,
    Values: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots,
    Dispatch<A13, S12>: MaskedCopyDispatch<
            R,
            Values,
            Output,
            KernelReadSlots<Values::Slots>,
            KernelOutputSlots<Output::Slots>,
        >,
    Indices: FlagInput<R>,
{
    fn scatter_input(
        self,
        exec: &Executor<R>,
        indices: Indices,
        flags: Option<&crate::DeviceVec<R, u32>>,
        output: Output,
    ) -> Result<(), Error> {
        let len = self.logical_len()?;
        let indices_len = indices.flag_len()?;
        if indices_len != len {
            return Err(Error::LengthMismatch {
                left: len,
                right: indices_len,
            });
        }
        let indices = indices.materialize_flags(exec)?;
        <Dispatch<A13, S12> as MaskedCopyDispatch<
            R,
            Values,
            Output,
            KernelReadSlots<Values::Slots>,
            KernelOutputSlots<Output::Slots>,
        >>::run(exec, &self, Some(&indices), false, flags, &output)
    }
}

/// Writes each input item to the output position given by its index.
pub(crate) fn scatter<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: ScatterInput<R, Indices, Output>,
{
    values.scatter_input(exec, indices, None, output)
}

/// Scatters rows whose stencil is nonzero, preserving other output rows.
pub(crate) fn scatter_where<R, Values, Indices, Stencil, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: ScatterInput<R, Indices, Output>,
    Stencil: FlagInput<R>,
{
    let flags = stencil.materialize_flags(exec)?;
    values.scatter_input(exec, indices, Some(&flags), output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, Permute, Zip};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn scatter_accepts_eval8_source_and_storage7_output() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 4])).collect();
        let indices = exec.to_device(&[2_u32, 0, 3]);
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

        scatter(&exec, values, indices.column(), output).unwrap();
        for (column, output) in outputs.iter().enumerate() {
            let base = column as u32 * 10;
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![base + 2, 0, base + 1, base + 3]
            );
        }
    }

    #[test]
    fn scatter_where_leaves_unselected_destinations_unchanged() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[10_u32, 20, 30]);
        let indices = exec.to_device(&[2_u32, 0, 1]);
        let flags = exec.to_device(&[1_u32, 0, 1]);
        let output = exec.to_device(&[99_u32; 3]);
        scatter_where(
            &exec,
            values.column(),
            indices.column(),
            flags.column(),
            output.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![99, 30, 10]);
    }
}
