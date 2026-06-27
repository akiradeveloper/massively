use super::memory::{MaterializeOutput, materialize};
use crate::{error::Error, policy::CubePolicy};
use cubecl::prelude::*;

/// Scatters `values[i]` into a new output at `indices[i]`.
///
/// The output is allocated with `len` elements, initialized with `default`, and
/// then updated by the scatter. For multiple value columns, pass borrowed
/// columns as a tuple, such as `(values.slice(..), tags.slice(..))`, and use
/// the same tuple shape for `default`.
pub fn scatter<Values, Indices>(
    policy: &CubePolicy<<Indices as crate::detail::read::KernelIndexRead>::Runtime>,
    values: Values,
    indices: Indices,
    len: usize,
    default: <Values as crate::detail::read::KernelScatterInput<
        <Indices as crate::detail::read::KernelIndexRead>::Source,
    >>::Default,
) -> Result<
    <<Values as crate::detail::read::KernelScatterInput<
        <Indices as crate::detail::read::KernelIndexRead>::Source,
    >>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Indices: crate::detail::read::KernelIndexRead,
    Values: crate::detail::read::KernelScatterInput<
            <Indices as crate::detail::read::KernelIndexRead>::Source,
            Runtime = <Indices as crate::detail::read::KernelIndexRead>::Runtime,
        >,
    <Values as crate::detail::read::KernelScatterInput<
        <Indices as crate::detail::read::KernelIndexRead>::Source,
    >>::Output:
        MaterializeOutput<Runtime = <Indices as crate::detail::read::KernelIndexRead>::Runtime>,
{
    let indices = indices.index_source()?;
    materialize(policy, values.scatter_read(policy, &indices, len, default)?)
}

/// Scatters selected values into a new output at `indices[i]`.
///
/// The output is allocated with `len` elements, initialized with `default`, and
/// then updated for values satisfying `Pred`.
pub fn scatter_where<Values, Indices, Stencil, Pred>(
    policy: &CubePolicy<<Indices as crate::detail::read::KernelIndexRead>::Runtime>,
    values: Values,
    indices: Indices,
    len: usize,
    default: <Values as crate::detail::read::KernelScatterWhereInput<
        <Indices as crate::detail::read::KernelIndexRead>::Source,
        Stencil,
        Pred,
    >>::Default,
    stencil: Stencil,
    _pred: Pred,
) -> Result<
    <<Values as crate::detail::read::KernelScatterWhereInput<
        <Indices as crate::detail::read::KernelIndexRead>::Source,
        Stencil,
        Pred,
    >>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Indices: crate::detail::read::KernelIndexRead,
    Values: crate::detail::read::KernelScatterWhereInput<
            <Indices as crate::detail::read::KernelIndexRead>::Source,
            Stencil,
            Pred,
            Runtime = <Indices as crate::detail::read::KernelIndexRead>::Runtime,
        >,
    <Values as crate::detail::read::KernelScatterWhereInput<
        <Indices as crate::detail::read::KernelIndexRead>::Source,
        Stencil,
        Pred,
    >>::Output:
        MaterializeOutput<Runtime = <Indices as crate::detail::read::KernelIndexRead>::Runtime>,
{
    let indices = indices.index_source()?;
    materialize(
        policy,
        values.scatter_where_read(policy, &indices, stencil, len, default)?,
    )
}
