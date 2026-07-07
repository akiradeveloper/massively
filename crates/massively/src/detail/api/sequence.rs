use super::memory::{MaterializeOutput, materialize};
use crate::{error::Error, op::GpuOp, policy::CubePolicy};
use cubecl::prelude::*;

impl<Keys, Values, Eq> crate::detail::read::KernelUniqueByKeyCall<Values, Eq> for Keys
where
    Keys: crate::detail::read::KernelUniqueByKeyKeys<Eq>,
    Values: crate::detail::read::KernelUniqueByKeyValues<Runtime = Keys::Runtime>,
{
    type Runtime = Keys::Runtime;
    type Output = (
        <Keys as crate::detail::read::KernelUniqueByKeyKeys<Eq>>::OutputKeys,
        <Values as crate::detail::read::KernelUniqueByKeyValues>::OutputValues,
    );

    fn unique_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        _eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        let (keys, control) = self.unique_by_key_control(policy)?;
        let values = values.unique_by_key_values(policy, &control)?;
        Ok((keys, values))
    }
}

/// Replaces elements whose staged stencil flag satisfies `Pred`.
#[allow(dead_code)]
pub fn replace_where<R, Input, Stencil, Pred>(
    policy: &CubePolicy<R>,
    input: Input,
    replacement: <Input as crate::detail::read::KernelReplaceWhereInput<Stencil, Pred>>::Item,
    stencil: Stencil,
    _pred: Pred,
) -> Result<
    <<Input as crate::detail::read::KernelReplaceWhereInput<Stencil, Pred>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    R: Runtime,
    Input: crate::detail::read::KernelReplaceWhereInput<Stencil, Pred, Runtime = R>,
    <Input as crate::detail::read::KernelReplaceWhereInput<Stencil, Pred>>::Output:
        MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        input.replace_where_read(policy, replacement, stencil)?,
    )
}

/// Removes consecutive duplicates.
pub fn unique<R, Input, Pred>(
    policy: &CubePolicy<R>,
    input: Input,
    _pred: Pred,
) -> Result<
    <<Input as crate::detail::read::KernelUniqueInput<Pred>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    R: Runtime,
    Input: crate::detail::read::KernelUniqueInput<Pred, Runtime = R>,
    <Input as crate::detail::read::KernelUniqueInput<Pred>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(policy, input.unique_read(policy)?)
}

/// Removes consecutive duplicate keys and carries the first value for each key.
#[allow(dead_code)]
pub fn unique_by_key<R, Keys, Values, Eq>(
    policy: &CubePolicy<R>,
    keys: Keys,
    values: Values,
    _eq: Eq,
) -> Result<<<Keys as crate::detail::read::KernelUniqueByKeyCall<Values, Eq>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Keys: crate::detail::read::KernelUniqueByKeyCall<Values, Eq, Runtime = R>,
    <Keys as crate::detail::read::KernelUniqueByKeyCall<Values, Eq>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        keys.unique_by_key_call(policy, values, GpuOp::<Eq>::new())?,
    )
}
