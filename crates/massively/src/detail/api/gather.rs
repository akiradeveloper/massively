use super::memory::{MaterializeOutput, materialize};
use crate::{error::Error, policy::CubePolicy};
use cubecl::prelude::*;

/// Gathers `input[indices[i]]` into new owned device storage.
///
/// This is a borrowing algorithm: `input` and `indices` are read-only. For
/// multiple value columns, pass borrowed columns as `Zip2` or `Zip3`.
/// Indices may be passed as `Zip1(indices.slice(..))`.
pub fn gather<Input, Indices>(
    policy: &CubePolicy<<Indices as crate::detail::read::KernelIndexRead>::Runtime>,
    input: Input,
    indices: Indices,
) -> Result<
    <<Input as crate::detail::read::KernelGatherInput<
        <Indices as crate::detail::read::KernelIndexRead>::Source,
    >>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Indices: crate::detail::read::KernelIndexRead,
    Input: crate::detail::read::KernelGatherInput<
            <Indices as crate::detail::read::KernelIndexRead>::Source,
            Runtime = <Indices as crate::detail::read::KernelIndexRead>::Runtime,
        >,
    <Input as crate::detail::read::KernelGatherInput<
        <Indices as crate::detail::read::KernelIndexRead>::Source,
    >>::Output:
        MaterializeOutput<Runtime = <Indices as crate::detail::read::KernelIndexRead>::Runtime>,
{
    let indices = indices.index_source()?;
    materialize(policy, input.gather_read(policy, &indices)?)
}

/// Gathers elements whose staged stencil flag satisfies `Pred`.
///
/// This is a borrowing algorithm: `input` and `indices` are read-only.
pub fn gather_where<Input, Indices, Stencil, Pred>(
    policy: &CubePolicy<<Indices as crate::detail::read::KernelIndexRead>::Runtime>,
    input: Input,
    indices: Indices,
    stencil: Stencil,
    default: <Input as crate::detail::read::KernelGatherWhereInput<
        <Indices as crate::detail::read::KernelIndexRead>::Source,
        Stencil,
        Pred,
    >>::Default,
    _pred: Pred,
) -> Result<
    <<Input as crate::detail::read::KernelGatherWhereInput<
        <Indices as crate::detail::read::KernelIndexRead>::Source,
        Stencil,
        Pred,
    >>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Indices: crate::detail::read::KernelIndexRead,
    Input: crate::detail::read::KernelGatherWhereInput<
            <Indices as crate::detail::read::KernelIndexRead>::Source,
            Stencil,
            Pred,
            Runtime = <Indices as crate::detail::read::KernelIndexRead>::Runtime,
        >,
    <Input as crate::detail::read::KernelGatherWhereInput<
        <Indices as crate::detail::read::KernelIndexRead>::Source,
        Stencil,
        Pred,
    >>::Output:
        MaterializeOutput<Runtime = <Indices as crate::detail::read::KernelIndexRead>::Runtime>,
{
    let indices = indices.index_source()?;
    materialize(
        policy,
        input.gather_where_read(policy, &indices, stencil, default)?,
    )
}
