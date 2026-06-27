use super::memory::{MaterializeOutput, materialize};
use crate::{
    device::{
        KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA, SoA1, SoA2, SoA3, SoAView2, SoAView3,
        StorageKernelColumn,
    },
    error::Error,
    policy::CubePolicy,
};
use cubecl::prelude::*;

#[doc(hidden)]
pub(crate) trait OwnedSelectionInput {}

impl<Source> OwnedSelectionInput for SoA1<Source>
where
    Self: SoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
{
}

impl<Source> OwnedSelectionInput for Source
where
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
{
}

impl<Source> OwnedSelectionInput for (Source,)
where
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
{
}

macro_rules! impl_selection_tuple_input {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+> OwnedSelectionInput for ($( $ty ),+)
        where
            $view<$( $ty ),+>: OwnedSelectionInput,
        {
        }
    };
}

impl_selection_tuple_input!(SoAView2<A, B> { left: 0, right: 1 });
impl_selection_tuple_input!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

macro_rules! impl_owned_selection_tuple {
    ($name:ident < $first:ident, $( $rest:ident ),+ >) => {
        impl<$first, $( $rest ),+> OwnedSelectionInput for $name<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: StorageKernelColumn + KernelColumnAt<S0>,
            $(
                $rest: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
        {
        }
    };
}

impl_owned_selection_tuple!(SoA2<A, B>);
impl_owned_selection_tuple!(SoA3<A, B, C>);

impl<Left, Right> OwnedSelectionInput for SoAView2<Left, Right>
where
    Self: ReadOnlySoA<Scalar = Left::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
{
}

impl<First, Second, Third> OwnedSelectionInput for SoAView3<First, Second, Third>
where
    Self: ReadOnlySoA<Scalar = First::Item>,
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
{
}

/// Keeps values whose staged stencil flag satisfies `Pred`.
///
/// This is a borrowing algorithm. It reads the input and returns newly owned SoA
/// storage containing the selected values.
pub fn copy_where<Source, Stencil, Pred>(
    policy: &CubePolicy<<Source as crate::detail::read::KernelCopyWhereInput<Stencil, Pred>>::Runtime>,
    source: Source,
    stencil: Stencil,
    _pred: Pred,
) -> Result<
    <<Source as crate::detail::read::KernelCopyWhereInput<Stencil, Pred>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Source: crate::detail::read::KernelCopyWhereInput<Stencil, Pred>,
    <Source as crate::detail::read::KernelCopyWhereInput<Stencil, Pred>>::Output:
        MaterializeOutput<
            Runtime = <Source as crate::detail::read::KernelCopyWhereInput<Stencil, Pred>>::Runtime,
        >,
{
    materialize(policy, source.copy_where_read(policy, stencil)?)
}

/// Removes values satisfying `Pred`.
///
/// This is a borrowing algorithm. It reads the input and returns newly owned SoA
/// storage for the remaining values.
pub fn remove_if<Source, Pred>(
    policy: &CubePolicy<<Source as crate::detail::read::KernelSelectInput<Pred>>::Runtime>,
    source: Source,
    _pred: Pred,
) -> Result<
    <<Source as crate::detail::read::KernelSelectInput<Pred>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Source: crate::detail::read::KernelSelectInput<Pred> + OwnedSelectionInput,
    <Source as crate::detail::read::KernelSelectInput<Pred>>::Output: MaterializeOutput<
        Runtime = <Source as crate::detail::read::KernelSelectInput<Pred>>::Runtime,
    >,
{
    materialize(policy, source.select_read(policy, true)?)
}

#[doc(hidden)]
pub trait TuplePair {
    type Left;
    type Right;

    fn into_pair(self) -> (Self::Left, Self::Right);
}

impl<Left, Right> TuplePair for (Left, Right) {
    type Left = Left;
    type Right = Right;

    fn into_pair(self) -> (Self::Left, Self::Right) {
        self
    }
}

/// Counts values satisfying `Pred`.
pub fn count_if<Source, Pred>(
    policy: &CubePolicy<<Source as crate::detail::read::KernelPredicateQueryInput<Pred>>::Runtime>,
    source: Source,
    _pred: Pred,
) -> Result<usize, Error>
where
    Source: crate::detail::read::KernelPredicateQueryInput<Pred>,
{
    source.count_read(policy, false)
}

/// Returns whether all values satisfy `Pred`.
pub fn all_of<Source, Pred>(
    policy: &CubePolicy<<Source as crate::detail::read::KernelPredicateQueryInput<Pred>>::Runtime>,
    source: Source,
    pred: Pred,
) -> Result<bool, Error>
where
    Source: crate::detail::read::KernelPredicateQueryInput<Pred>,
{
    Ok(find_if_not(policy, source, pred)?.is_none())
}

/// Returns whether any value satisfies `Pred`.
pub fn any_of<Source, Pred>(
    policy: &CubePolicy<<Source as crate::detail::read::KernelPredicateQueryInput<Pred>>::Runtime>,
    source: Source,
    pred: Pred,
) -> Result<bool, Error>
where
    Source: crate::detail::read::KernelPredicateQueryInput<Pred>,
{
    Ok(find_if(policy, source, pred)?.is_some())
}

/// Returns whether no values satisfy `Pred`.
pub fn none_of<Source, Pred>(
    policy: &CubePolicy<<Source as crate::detail::read::KernelPredicateQueryInput<Pred>>::Runtime>,
    source: Source,
    pred: Pred,
) -> Result<bool, Error>
where
    Source: crate::detail::read::KernelPredicateQueryInput<Pred>,
{
    Ok(find_if(policy, source, pred)?.is_none())
}

/// Finds the first value satisfying `Pred`.
pub fn find_if<Source, Pred>(
    policy: &CubePolicy<<Source as crate::detail::read::KernelPredicateQueryInput<Pred>>::Runtime>,
    source: Source,
    _pred: Pred,
) -> Result<Option<usize>, Error>
where
    Source: crate::detail::read::KernelPredicateQueryInput<Pred>,
{
    source.find_read(policy, false)
}

fn find_if_not<Source, Pred>(
    policy: &CubePolicy<<Source as crate::detail::read::KernelPredicateQueryInput<Pred>>::Runtime>,
    source: Source,
    _pred: Pred,
) -> Result<Option<usize>, Error>
where
    Source: crate::detail::read::KernelPredicateQueryInput<Pred>,
{
    source.find_read(policy, true)
}

/// Partitions elements by `Pred`, preserving relative order within each side.
pub fn partition<Input, Pred>(
    policy: &CubePolicy<<Input as crate::detail::read::KernelPartitionInput<Pred>>::Runtime>,
    input: Input,
    _pred: Pred,
) -> Result<
    (
        <<<Input as crate::detail::read::KernelPartitionInput<Pred>>::SplitOutput as TuplePair>::Left as MaterializeOutput>::Output,
        <<<Input as crate::detail::read::KernelPartitionInput<Pred>>::SplitOutput as TuplePair>::Right as MaterializeOutput>::Output,
    ),
    Error,
>
where
    Input: crate::detail::read::KernelPartitionInput<Pred>,
    <Input as crate::detail::read::KernelPartitionInput<Pred>>::SplitOutput: TuplePair,
    <<Input as crate::detail::read::KernelPartitionInput<Pred>>::SplitOutput as TuplePair>::Left:
        MaterializeOutput<Runtime = <Input as crate::detail::read::KernelPartitionInput<Pred>>::Runtime>,
    <<Input as crate::detail::read::KernelPartitionInput<Pred>>::SplitOutput as TuplePair>::Right:
        MaterializeOutput<Runtime = <Input as crate::detail::read::KernelPartitionInput<Pred>>::Runtime>,
{
    let (matching, failing) = input.partition_copy_read(policy)?.into_pair();
    Ok((
        materialize(policy, matching)?,
        materialize(policy, failing)?,
    ))
}

/// Returns whether all elements satisfying `Pred` appear before all non-matching elements.
pub fn is_partitioned<Input, Pred>(
    policy: &CubePolicy<<Input as crate::detail::read::KernelPartitionInput<Pred>>::Runtime>,
    input: Input,
    _pred: Pred,
) -> Result<bool, Error>
where
    Input: crate::detail::read::KernelPartitionInput<Pred>,
{
    input.is_partitioned_read(policy)
}
