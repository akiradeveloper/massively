use super::memory::{MaterializeOutput, materialize};
use crate::{error::Error, op::GpuOp, policy::CubePolicy};
use cubecl::prelude::*;

/// Reduces read-only device input to a host tuple item.
///
/// This is a borrowing algorithm: pass `&DeviceVec` for one column or [`zip`]
/// for multiple read-only columns. No output device storage is allocated.
///
/// [`zip`]: crate::zip
pub fn reduce<Input, Op>(
    policy: &CubePolicy<<Input as crate::detail::read::KernelReduceInput<Op>>::Runtime>,
    input: Input,
    init: <Input as crate::detail::read::KernelReduceInput<Op>>::Item,
    _op: Op,
) -> Result<<Input as crate::detail::read::KernelReduceInput<Op>>::Item, Error>
where
    Input: crate::detail::read::KernelReduceInput<Op>,
{
    input.reduce_read(policy, init)
}

impl<Keys, Values, KeyEq, Op> crate::detail::read::KernelReduceByKeyCall<Values, KeyEq, Op> for Keys
where
    Keys: crate::detail::read::KernelReduceByKeyKeys<KeyEq>,
    Values: crate::detail::read::KernelReduceByKeyValues<
            Keys::Control,
            KeyEq,
            Op,
            Runtime = Keys::Runtime,
        >,
{
    type Runtime = Keys::Runtime;
    type Init =
        <Values as crate::detail::read::KernelReduceByKeyValues<Keys::Control, KeyEq, Op>>::Init;
    type Output = (
        <Keys as crate::detail::read::KernelReduceByKeyKeys<KeyEq>>::OutputKeys,
        <Values as crate::detail::read::KernelReduceByKeyValues<
            Keys::Control,
            KeyEq,
            Op,
        >>::OutputValues,
    );

    fn reduce_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        _key_eq: GpuOp<KeyEq>,
        init: Self::Init,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let (keys, control) = self.reduce_by_key_control(policy)?;
        let values = values.reduce_by_key_values(policy, &control, init)?;
        Ok((keys, values))
    }
}

/// Reduces contiguous equal-key runs using read-only keys and values.
///
/// This is a borrowing algorithm: values may be a borrowed column or a read-only
/// Zip from [`zip`](crate::zip). The returned keys and values are owned Zip
/// storage.
pub fn reduce_by_key<R, Keys, Values, KeyEq, Op>(
    policy: &CubePolicy<R>,
    keys: Keys,
    values: Values,
    _key_eq: KeyEq,
    init: <Keys as crate::detail::read::KernelReduceByKeyCall<Values, KeyEq, Op>>::Init,
    _op: Op,
) -> Result<
    <<Keys as crate::detail::read::KernelReduceByKeyCall<Values, KeyEq, Op>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    R: Runtime,
    Keys: crate::detail::read::KernelReduceByKeyCall<Values, KeyEq, Op, Runtime = R>,
    <Keys as crate::detail::read::KernelReduceByKeyCall<Values, KeyEq, Op>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        keys.reduce_by_key_call(
            policy,
            values,
            GpuOp::<KeyEq>::new(),
            init,
            GpuOp::<Op>::new(),
        )?,
    )
}
