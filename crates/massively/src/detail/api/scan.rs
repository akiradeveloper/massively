use super::memory::{MaterializeOutput, materialize};
use crate::{error::Error, op::GpuOp, policy::CubePolicy};
use cubecl::prelude::*;

/// Computes an inclusive scan from read-only input into device storage.
pub fn inclusive_scan<InputSource, Op>(
    policy: &CubePolicy<<InputSource as crate::detail::read::KernelInclusiveScanInput<Op>>::Runtime>,
    source: InputSource,
    _op: Op,
) -> Result<
    <<InputSource as crate::detail::read::KernelInclusiveScanInput<Op>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    InputSource: crate::detail::read::KernelInclusiveScanInput<Op>,
    <InputSource as crate::detail::read::KernelInclusiveScanInput<Op>>::Output:
        MaterializeOutput<
            Runtime = <InputSource as crate::detail::read::KernelInclusiveScanInput<Op>>::Runtime,
        >,
{
    materialize(policy, source.inclusive_scan_read(policy)?)
}

/// Computes an exclusive scan from read-only input into device storage.
pub fn exclusive_scan<InputSource, Op>(
    policy: &CubePolicy<<InputSource as crate::detail::read::KernelExclusiveScanInput<Op>>::Runtime>,
    source: InputSource,
    init: <InputSource as crate::detail::read::KernelExclusiveScanInput<Op>>::Init,
    _op: Op,
) -> Result<
    <<InputSource as crate::detail::read::KernelExclusiveScanInput<Op>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    InputSource: crate::detail::read::KernelExclusiveScanInput<Op>,
    <InputSource as crate::detail::read::KernelExclusiveScanInput<Op>>::Output:
        MaterializeOutput<
            Runtime = <InputSource as crate::detail::read::KernelExclusiveScanInput<Op>>::Runtime,
        >,
{
    materialize(policy, source.exclusive_scan_read(policy, init)?)
}

/// Computes adjacent differences into device storage.
pub fn adjacent_difference<Source, Op>(
    policy: &CubePolicy<<Source as crate::detail::read::KernelAdjacentDifferenceInput<Op>>::Runtime>,
    source: Source,
    _op: Op,
) -> Result<
    <<Source as crate::detail::read::KernelAdjacentDifferenceInput<Op>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Source: crate::detail::read::KernelAdjacentDifferenceInput<Op>,
    <Source as crate::detail::read::KernelAdjacentDifferenceInput<Op>>::Output:
        MaterializeOutput<
            Runtime = <Source as crate::detail::read::KernelAdjacentDifferenceInput<Op>>::Runtime,
        >,
{
    materialize(policy, source.adjacent_difference_read(policy)?)
}

impl<Values, Keys, KeyEq, Op> crate::detail::read::KernelInclusiveScanByKeyCall<Values, KeyEq, Op>
    for Keys
where
    Keys: crate::detail::read::KernelScanByKeyKeys<KeyEq>,
    Values: crate::detail::read::KernelInclusiveScanByKeyValues<
            Keys::Control,
            KeyEq,
            Op,
            Runtime = Keys::Runtime,
        >,
{
    type Runtime = Keys::Runtime;
    type Output = <Values as crate::detail::read::KernelInclusiveScanByKeyValues<
        Keys::Control,
        KeyEq,
        Op,
    >>::Output;

    fn inclusive_scan_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let control = self.scan_by_key_control(policy)?;
        values.inclusive_scan_by_key_values(policy, &control)
    }
}

pub fn inclusive_scan_by_key<R, Keys, Values, KeyEq, Op>(
    _policy: &CubePolicy<R>,
    keys: Keys,
    values: Values,
    _key_eq: KeyEq,
    _op: Op,
) -> Result<
    <<Keys as crate::detail::read::KernelInclusiveScanByKeyCall<Values, KeyEq, Op>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    R: Runtime,
    Keys: crate::detail::read::KernelInclusiveScanByKeyCall<Values, KeyEq, Op, Runtime = R>,
    <Keys as crate::detail::read::KernelInclusiveScanByKeyCall<Values, KeyEq, Op>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        _policy,
        keys.inclusive_scan_by_key_call(
            _policy,
            values,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?,
    )
}

impl<Values, Keys, KeyEq, Op> crate::detail::read::KernelExclusiveScanByKeyCall<Values, KeyEq, Op>
    for Keys
where
    Keys: crate::detail::read::KernelScanByKeyKeys<KeyEq>,
    Values: crate::detail::read::KernelExclusiveScanByKeyValues<
            Keys::Control,
            KeyEq,
            Op,
            Runtime = Keys::Runtime,
        >,
{
    type Runtime = Keys::Runtime;
    type Init = <Values as crate::detail::read::KernelExclusiveScanByKeyValues<
        Keys::Control,
        KeyEq,
        Op,
    >>::Init;
    type Output = <Values as crate::detail::read::KernelExclusiveScanByKeyValues<
        Keys::Control,
        KeyEq,
        Op,
    >>::Output;

    fn exclusive_scan_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        init: Self::Init,
        _key_eq: GpuOp<KeyEq>,
        _op: GpuOp<Op>,
    ) -> Result<Self::Output, Error> {
        let control = self.scan_by_key_control(policy)?;
        values.exclusive_scan_by_key_values(policy, &control, init)
    }
}

pub fn exclusive_scan_by_key<R, Keys, Values, KeyEq, Op>(
    _policy: &CubePolicy<R>,
    keys: Keys,
    values: Values,
    _key_eq: KeyEq,
    init: <Keys as crate::detail::read::KernelExclusiveScanByKeyCall<Values, KeyEq, Op>>::Init,
    _op: Op,
) -> Result<
    <<Keys as crate::detail::read::KernelExclusiveScanByKeyCall<Values, KeyEq, Op>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    R: Runtime,
    Keys: crate::detail::read::KernelExclusiveScanByKeyCall<Values, KeyEq, Op, Runtime = R>,
    <Keys as crate::detail::read::KernelExclusiveScanByKeyCall<Values, KeyEq, Op>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        _policy,
        keys.exclusive_scan_by_key_call(
            _policy,
            values,
            init,
            GpuOp::<KeyEq>::new(),
            GpuOp::<Op>::new(),
        )?,
    )
}
