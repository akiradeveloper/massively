use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA2, SoA3, SoA4, SoA5, SoA6,
        SoA7, SoA8, SoA9, SoA10, SoA11, SoA12, SoAView1, SoAView2, SoAView3, SoAView4, SoAView5,
        SoAView6, SoAView7, SoAView8, SoAView9, SoAView10, SoAView11, SoAView12,
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::{BinaryPredicateOp, GpuOp},
    primitives::{scan, search},
};
use cubecl::prelude::*;

const BLOCK_SEARCH_SIZE: u32 = 256;

fn search_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_SEARCH_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

fn materialize_one<Source>(
    input: SoAView1<Source>,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    SoAView1<Source>: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    ReadOnlySoA::validate(&input)?;
    super::device_expr_collect(&input.source)
}

fn materialize_pair<Left, Right>(
    left: SoAView1<Left>,
    right: SoAView1<Right>,
) -> Result<
    (
        DeviceVec<Left::Runtime, Left::Item>,
        DeviceVec<Left::Runtime, Left::Item>,
    ),
    Error,
>
where
    SoAView1<Left>: ReadOnlySoA<Item = Left::Item, Scalar = Left::Item>,
    SoAView1<Right>: ReadOnlySoA<Item = Right::Item, Scalar = Right::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
{
    let left = materialize_one(left)?;
    let right = materialize_one(right)?;
    Ok((left, right))
}

/// Input accepted by min/max element queries.
#[doc(hidden)]
pub trait MinMaxInput<Less> {
    /// Finds the minimum element index.
    fn min_element_input(self, less: GpuOp<Less>) -> Result<Option<usize>, Error>;
    /// Finds the maximum element index.
    fn max_element_input(self, less: GpuOp<Less>) -> Result<Option<usize>, Error>;
    /// Finds both minimum and maximum element indices.
    fn minmax_element_input(self, less: GpuOp<Less>) -> Result<Option<(usize, usize)>, Error>;
}

impl<Source, Less> MinMaxInput<Less> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    fn min_element_input(self, _less: GpuOp<Less>) -> Result<Option<usize>, Error> {
        ReadOnlySoA::validate(&self)?;
        Ok(super::device_expr_minmax_element::<Source, Less>(&self.source)?.map(|(min, _)| min))
    }

    fn max_element_input(self, _less: GpuOp<Less>) -> Result<Option<usize>, Error> {
        ReadOnlySoA::validate(&self)?;
        Ok(super::device_expr_minmax_element::<Source, Less>(&self.source)?.map(|(_, max)| max))
    }

    fn minmax_element_input(self, _less: GpuOp<Less>) -> Result<Option<(usize, usize)>, Error> {
        ReadOnlySoA::validate(&self)?;
        super::device_expr_minmax_element::<Source, Less>(&self.source)
    }
}

impl<Source, Less> MinMaxInput<Less> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    fn min_element_input(self, less: GpuOp<Less>) -> Result<Option<usize>, Error> {
        <SoAView1<Source> as MinMaxInput<Less>>::min_element_input(SoAView1 { source: self }, less)
    }

    fn max_element_input(self, less: GpuOp<Less>) -> Result<Option<usize>, Error> {
        <SoAView1<Source> as MinMaxInput<Less>>::max_element_input(SoAView1 { source: self }, less)
    }

    fn minmax_element_input(self, less: GpuOp<Less>) -> Result<Option<(usize, usize)>, Error> {
        <SoAView1<Source> as MinMaxInput<Less>>::minmax_element_input(
            SoAView1 { source: self },
            less,
        )
    }
}

/// Input accepted by adjacent-pair search.
#[doc(hidden)]
pub trait AdjacentFindInput<Pred> {
    /// Finds the first adjacent pair that satisfies `Pred`.
    fn adjacent_find_input(self, pred: GpuOp<Pred>) -> Result<Option<usize>, Error>;
}

impl<Source, Pred> AdjacentFindInput<Pred> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<Source::Item>,
{
    fn adjacent_find_input(self, _pred: GpuOp<Pred>) -> Result<Option<usize>, Error> {
        let input = materialize_one(self)?;
        search::adjacent_find(&input, GpuOp::<Pred>::new())
    }
}

impl<Source, Pred> AdjacentFindInput<Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<Source::Item>,
{
    fn adjacent_find_input(self, pred: GpuOp<Pred>) -> Result<Option<usize>, Error> {
        <SoAView1<Source> as AdjacentFindInput<Pred>>::adjacent_find_input(
            SoAView1 { source: self },
            pred,
        )
    }
}

/// Input accepted by sorted single-range search.
#[doc(hidden)]
pub trait SortedSearchInput<Less> {
    /// Element type.
    type Item;

    /// Returns whether `value` is present in this sorted input.
    fn binary_search_input(self, value: Self::Item, less: GpuOp<Less>) -> Result<bool, Error>;
    /// Returns the equal range for `value` in this sorted input.
    fn equal_range_input(
        self,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<(usize, usize), Error>;
    /// Finds the first sorted insertion point for `value`.
    fn lower_bound_input(self, value: Self::Item, less: GpuOp<Less>) -> Result<usize, Error>;
    /// Finds the last sorted insertion point for `value`.
    fn upper_bound_input(self, value: Self::Item, less: GpuOp<Less>) -> Result<usize, Error>;
    /// Returns the first position where sorted order is broken.
    fn is_sorted_until_input(self, less: GpuOp<Less>) -> Result<usize, Error>;
    /// Returns whether this input is sorted.
    fn is_sorted_input(self, less: GpuOp<Less>) -> Result<bool, Error>;
}

impl<Source, Less> SortedSearchInput<Less> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Item = Source::Item;

    fn binary_search_input(self, value: Self::Item, _less: GpuOp<Less>) -> Result<bool, Error> {
        let input = materialize_one(self)?;
        search::binary_search(&input, value, GpuOp::<Less>::new())
    }

    fn equal_range_input(
        self,
        value: Self::Item,
        _less: GpuOp<Less>,
    ) -> Result<(usize, usize), Error> {
        let input = materialize_one(self)?;
        Ok((
            search::lower_bound(&input, value.clone(), GpuOp::<Less>::new())?,
            search::upper_bound(&input, value, GpuOp::<Less>::new())?,
        ))
    }

    fn lower_bound_input(self, value: Self::Item, _less: GpuOp<Less>) -> Result<usize, Error> {
        let input = materialize_one(self)?;
        search::lower_bound(&input, value, GpuOp::<Less>::new())
    }

    fn upper_bound_input(self, value: Self::Item, _less: GpuOp<Less>) -> Result<usize, Error> {
        let input = materialize_one(self)?;
        search::upper_bound(&input, value, GpuOp::<Less>::new())
    }

    fn is_sorted_until_input(self, _less: GpuOp<Less>) -> Result<usize, Error> {
        let input = materialize_one(self)?;
        search::is_sorted_until(&input, GpuOp::<Less>::new())
    }

    fn is_sorted_input(self, less: GpuOp<Less>) -> Result<bool, Error> {
        let len = ReadOnlySoA::len(&self);
        Ok(self.is_sorted_until_input(less)? == len)
    }
}

impl<Source, Less> SortedSearchInput<Less> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Item = Source::Item;

    fn binary_search_input(self, value: Self::Item, less: GpuOp<Less>) -> Result<bool, Error> {
        <SoAView1<Source> as SortedSearchInput<Less>>::binary_search_input(
            SoAView1 { source: self },
            value,
            less,
        )
    }

    fn equal_range_input(
        self,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<(usize, usize), Error> {
        <SoAView1<Source> as SortedSearchInput<Less>>::equal_range_input(
            SoAView1 { source: self },
            value,
            less,
        )
    }

    fn lower_bound_input(self, value: Self::Item, less: GpuOp<Less>) -> Result<usize, Error> {
        <SoAView1<Source> as SortedSearchInput<Less>>::lower_bound_input(
            SoAView1 { source: self },
            value,
            less,
        )
    }

    fn upper_bound_input(self, value: Self::Item, less: GpuOp<Less>) -> Result<usize, Error> {
        <SoAView1<Source> as SortedSearchInput<Less>>::upper_bound_input(
            SoAView1 { source: self },
            value,
            less,
        )
    }

    fn is_sorted_until_input(self, less: GpuOp<Less>) -> Result<usize, Error> {
        <SoAView1<Source> as SortedSearchInput<Less>>::is_sorted_until_input(
            SoAView1 { source: self },
            less,
        )
    }

    fn is_sorted_input(self, less: GpuOp<Less>) -> Result<bool, Error> {
        <SoAView1<Source> as SortedSearchInput<Less>>::is_sorted_input(
            SoAView1 { source: self },
            less,
        )
    }
}

/// Pair of inputs accepted by binary search/comparison algorithms.
#[doc(hidden)]
pub trait PairSearchInput<Other, Op> {
    /// Returns whether two inputs are equal under `Op`.
    fn equal_input(self, other: Other, op: GpuOp<Op>) -> Result<bool, Error>;
    /// Finds the first mismatch between two inputs.
    fn mismatch_input(self, other: Other, op: GpuOp<Op>) -> Result<Option<usize>, Error>;
    /// Finds the first element equal to any value in `other`.
    fn find_first_of_input(self, other: Other, op: GpuOp<Op>) -> Result<Option<usize>, Error>;
    /// Finds the last occurrence of `other` inside this input.
    fn find_end_input(self, other: Other, op: GpuOp<Op>) -> Result<Option<usize>, Error>;
    /// Returns whether sorted `self` includes all values from sorted `other`.
    fn includes_input(self, other: Other, op: GpuOp<Op>) -> Result<bool, Error>;
    /// Lexicographically compares two inputs.
    fn lexicographical_compare_input(self, other: Other, op: GpuOp<Op>) -> Result<bool, Error>;
}

impl<Left, Right, Op> PairSearchInput<SoAView1<Right>, Op> for SoAView1<Left>
where
    Self: ReadOnlySoA<Item = Left::Item, Scalar = Left::Item>,
    SoAView1<Right>: ReadOnlySoA<Item = Right::Item, Scalar = Right::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: BinaryPredicateOp<Left::Item>,
{
    fn equal_input(self, other: SoAView1<Right>, _op: GpuOp<Op>) -> Result<bool, Error> {
        let (left, right) = materialize_pair(self, other)?;
        search::equal(&left, &right, GpuOp::<Op>::new())
    }

    fn mismatch_input(
        self,
        other: SoAView1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<Option<usize>, Error> {
        let (left, right) = materialize_pair(self, other)?;
        search::mismatch(&left, &right, GpuOp::<Op>::new())
    }

    fn find_first_of_input(
        self,
        other: SoAView1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<Option<usize>, Error> {
        let (input, needles) = materialize_pair(self, other)?;
        search::find_first_of(&input, &needles, GpuOp::<Op>::new())
    }

    fn find_end_input(
        self,
        other: SoAView1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<Option<usize>, Error> {
        let (input, pattern) = materialize_pair(self, other)?;
        search::find_end(&input, &pattern, GpuOp::<Op>::new())
    }

    fn includes_input(self, other: SoAView1<Right>, _op: GpuOp<Op>) -> Result<bool, Error> {
        let (left, right) = materialize_pair(self, other)?;
        search::includes(&left, &right, GpuOp::<Op>::new())
    }

    fn lexicographical_compare_input(
        self,
        other: SoAView1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        let (left, right) = materialize_pair(self, other)?;
        search::lexicographical_compare(&left, &right, GpuOp::<Op>::new())
    }
}

impl<Left, Right, Op> PairSearchInput<Right, Op> for Left
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: BinaryPredicateOp<Left::Item>,
{
    fn equal_input(self, other: Right, op: GpuOp<Op>) -> Result<bool, Error> {
        <SoAView1<Left> as PairSearchInput<SoAView1<Right>, Op>>::equal_input(
            SoAView1 { source: self },
            SoAView1 { source: other },
            op,
        )
    }

    fn mismatch_input(self, other: Right, op: GpuOp<Op>) -> Result<Option<usize>, Error> {
        <SoAView1<Left> as PairSearchInput<SoAView1<Right>, Op>>::mismatch_input(
            SoAView1 { source: self },
            SoAView1 { source: other },
            op,
        )
    }

    fn find_first_of_input(self, other: Right, op: GpuOp<Op>) -> Result<Option<usize>, Error> {
        <SoAView1<Left> as PairSearchInput<SoAView1<Right>, Op>>::find_first_of_input(
            SoAView1 { source: self },
            SoAView1 { source: other },
            op,
        )
    }

    fn find_end_input(self, other: Right, op: GpuOp<Op>) -> Result<Option<usize>, Error> {
        <SoAView1<Left> as PairSearchInput<SoAView1<Right>, Op>>::find_end_input(
            SoAView1 { source: self },
            SoAView1 { source: other },
            op,
        )
    }

    fn includes_input(self, other: Right, op: GpuOp<Op>) -> Result<bool, Error> {
        <SoAView1<Left> as PairSearchInput<SoAView1<Right>, Op>>::includes_input(
            SoAView1 { source: self },
            SoAView1 { source: other },
            op,
        )
    }

    fn lexicographical_compare_input(self, other: Right, op: GpuOp<Op>) -> Result<bool, Error> {
        <SoAView1<Left> as PairSearchInput<SoAView1<Right>, Op>>::lexicographical_compare_input(
            SoAView1 { source: self },
            SoAView1 { source: other },
            op,
        )
    }
}

macro_rules! impl_tuple_search {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $name:ident < $first:ident, $( $rest:ident ),+ > {
            $first_field:ident: $first_index:tt,
            $( $field:ident: $index:tt ),+
        },
        $adjacent_kernel:ident,
        $sorted_break_kernel:ident,
        $lower_bound_kernel:ident,
        $upper_bound_kernel:ident,
        $binary_search_at_kernel:ident,
        $minmax_element_kernel:ident,
        $minmax_index_kernel:ident
    ) => {
        impl<$first, $( $rest ),+, Less> MinMaxInput<Less> for $name<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Less: BinaryPredicateOp<(
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            )>,
        {
            fn min_element_input(self, less: GpuOp<Less>) -> Result<Option<usize>, Error> {
                Ok(self.minmax_element_input(less)?.map(|(min, _)| min))
            }

            fn max_element_input(self, less: GpuOp<Less>) -> Result<Option<usize>, Error> {
                Ok(self.minmax_element_input(less)?.map(|(_, max)| max))
            }

            fn minmax_element_input(self, _less: GpuOp<Less>) -> Result<Option<(usize, usize)>, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let len = $first_field.len();
                if len == 0 {
                    return Ok(None);
                }

                let client = $first_field.policy().client();
                let mut current_count = len.div_ceil(BLOCK_SEARCH_SIZE as usize);
                let mut current_count_u32 = u32::try_from(current_count)
                    .map_err(|_| Error::LengthTooLarge { len: current_count })?;
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                let mut current_handle =
                    client.empty(current_count * 2 * std::mem::size_of::<u32>());

                unsafe {
                    $minmax_element_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(current_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(
                            &$first_field.handle,
                            len,
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(
                                &$field.handle,
                                len,
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                        ArrayArg::from_raw_parts::<u32>(
                            &current_handle,
                            current_count * 2,
                            1,
                        ),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }

                while current_count > 1 {
                    let next_count = current_count.div_ceil(BLOCK_SEARCH_SIZE as usize);
                    let next_count_u32 = u32::try_from(next_count)
                        .map_err(|_| Error::LengthTooLarge { len: next_count })?;
                    let candidate_len_handle =
                        client.create_from_slice(u32::as_bytes(&[current_count_u32]));
                    let next_handle =
                        client.empty(next_count * 2 * std::mem::size_of::<u32>());

                    unsafe {
                        $minmax_index_kernel::launch_unchecked::<
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item, )+
                            Less,
                            <$first as KernelColumn>::Runtime,
                        >(
                            client,
                            CubeCount::Static(next_count_u32, 1, 1),
                            CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                            ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(
                                &$first_field.handle,
                                len,
                                1,
                            ),
                            $(
                                ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(
                                    &$field.handle,
                                    len,
                                    1,
                                ),
                            )+
                            ArrayArg::from_raw_parts::<u32>(
                                &current_handle,
                                current_count * 2,
                                1,
                            ),
                            ArrayArg::from_raw_parts::<u32>(&candidate_len_handle, 1, 1),
                            ArrayArg::from_raw_parts::<u32>(&next_handle, next_count * 2, 1),
                        )
                        .map_err(|err| Error::Launch {
                            message: format!("{err:?}"),
                        })?;
                    }

                    current_handle = next_handle;
                    current_count = next_count;
                    current_count_u32 = next_count_u32;
                }

                let bytes = client.read_one(current_handle);
                let indices = u32::from_bytes(&bytes);
                Ok(Some((indices[0] as usize, indices[1] as usize)))
            }
        }

        impl<$first, $( $rest ),+, Pred> AdjacentFindInput<Pred> for $name<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Pred: BinaryPredicateOp<(
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            )>,
        {
            fn adjacent_find_input(self, _pred: GpuOp<Pred>) -> Result<Option<usize>, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let len = $first_field.len();
                if len < 2 {
                    return Ok(None);
                }
                let block_count_u32 = search_block_count(len)?;
                let client = $first_field.policy().client();
                let flag_handle = client.empty(len * std::mem::size_of::<u32>());
                unsafe {
                    $adjacent_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Pred,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&$first_field.handle, len, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.handle, len, 1),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }
                search::first_flag($first_field.policy(), flag_handle, len, len - 1)
            }
        }

        impl<$first, $( $rest ),+, Less> SortedSearchInput<Less> for $name<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Less: BinaryPredicateOp<(
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            )>,
        {
            type Item = (
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            );

            fn binary_search_input(
                self,
                value: Self::Item,
                _less: GpuOp<Less>,
            ) -> Result<bool, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let len = $first_field.len();
                let client = $first_field.policy().client();
                let flag_handle = client.empty(len * std::mem::size_of::<u32>());
                let first_value_handle = client.create_from_slice(
                    <<$first as KernelColumn>::Item as CubeElement>::as_bytes(&[value.$first_index])
                );
                $(
                    let $field = $field;
                    let $field = (
                        $field,
                        client.create_from_slice(
                            <<$rest as KernelColumn>::Item as CubeElement>::as_bytes(&[value.$index])
                        ),
                    );
                )+
                if len != 0 {
                    let block_count_u32 = search_block_count(len)?;
                    unsafe {
                        $lower_bound_kernel::launch_unchecked::<
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item, )+
                            Less,
                            <$first as KernelColumn>::Runtime,
                        >(
                            client,
                            CubeCount::Static(block_count_u32, 1, 1),
                            CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                            ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&$first_field.handle, len, 1),
                            $(
                                ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.0.handle, len, 1),
                            )+
                            ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&first_value_handle, 1, 1),
                            $(
                                ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.1, 1, 1),
                            )+
                            ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
                        )
                        .map_err(|err| Error::Launch {
                            message: format!("{err:?}"),
                        })?;
                    }
                }
                let index = search::first_flag($first_field.policy(), flag_handle, len, len)?
                    .unwrap_or(len);
                if index >= len {
                    return Ok(false);
                }
                let index_handle = client.create_from_slice(u32::as_bytes(&[index as u32]));
                let output_handle = client.empty(std::mem::size_of::<u32>());
                unsafe {
                    $binary_search_at_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::new_single(),
                        CubeDim::new_1d(1),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&$first_field.handle, len, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.0.handle, len, 1),
                        )+
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&first_value_handle, 1, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.1, 1, 1),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&index_handle, 1, 1),
                        ArrayArg::from_raw_parts::<u32>(&output_handle, 1, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }
                Ok(scan::read_u32_scalar::<<$first as KernelColumn>::Runtime>(
                    client,
                    output_handle,
                ) != 0)
            }

            fn equal_range_input(
                self,
                value: Self::Item,
                _less: GpuOp<Less>,
            ) -> Result<(usize, usize), Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let len = $first_field.len();
                if len == 0 {
                    return Ok((0, 0));
                }
                let client = $first_field.policy().client();
                let lower_flag = client.empty(len * std::mem::size_of::<u32>());
                let upper_flag = client.empty(len * std::mem::size_of::<u32>());
                let first_value_handle = client.create_from_slice(
                    <<$first as KernelColumn>::Item as CubeElement>::as_bytes(&[value.$first_index])
                );
                $(
                    let $field = (
                        $field,
                        client.create_from_slice(
                            <<$rest as KernelColumn>::Item as CubeElement>::as_bytes(&[value.$index])
                        ),
                    );
                )+
                let block_count_u32 = search_block_count(len)?;
                unsafe {
                    $lower_bound_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&$first_field.handle, len, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.0.handle, len, 1),
                        )+
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&first_value_handle, 1, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.1, 1, 1),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&lower_flag, len, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                    $upper_bound_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&$first_field.handle, len, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.0.handle, len, 1),
                        )+
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&first_value_handle, 1, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.1, 1, 1),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&upper_flag, len, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }
                Ok((
                    search::first_flag($first_field.policy(), lower_flag, len, len)?.unwrap_or(len),
                    search::first_flag($first_field.policy(), upper_flag, len, len)?.unwrap_or(len),
                ))
            }

            fn lower_bound_input(
                self,
                value: Self::Item,
                _less: GpuOp<Less>,
            ) -> Result<usize, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let len = $first_field.len();
                if len == 0 {
                    return Ok(0);
                }
                let client = $first_field.policy().client();
                let flag_handle = client.empty(len * std::mem::size_of::<u32>());
                let first_value_handle = client.create_from_slice(
                    <<$first as KernelColumn>::Item as CubeElement>::as_bytes(&[value.$first_index])
                );
                $(
                    let $field = (
                        $field,
                        client.create_from_slice(
                            <<$rest as KernelColumn>::Item as CubeElement>::as_bytes(&[value.$index])
                        ),
                    );
                )+
                let block_count_u32 = search_block_count(len)?;
                unsafe {
                    $lower_bound_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&$first_field.handle, len, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.0.handle, len, 1),
                        )+
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&first_value_handle, 1, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.1, 1, 1),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }
                Ok(search::first_flag($first_field.policy(), flag_handle, len, len)?
                    .unwrap_or(len))
            }

            fn upper_bound_input(
                self,
                value: Self::Item,
                _less: GpuOp<Less>,
            ) -> Result<usize, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let len = $first_field.len();
                if len == 0 {
                    return Ok(0);
                }
                let client = $first_field.policy().client();
                let flag_handle = client.empty(len * std::mem::size_of::<u32>());
                let first_value_handle = client.create_from_slice(
                    <<$first as KernelColumn>::Item as CubeElement>::as_bytes(&[value.$first_index])
                );
                $(
                    let $field = (
                        $field,
                        client.create_from_slice(
                            <<$rest as KernelColumn>::Item as CubeElement>::as_bytes(&[value.$index])
                        ),
                    );
                )+
                let block_count_u32 = search_block_count(len)?;
                unsafe {
                    $upper_bound_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&$first_field.handle, len, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.0.handle, len, 1),
                        )+
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&first_value_handle, 1, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.1, 1, 1),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }
                Ok(search::first_flag($first_field.policy(), flag_handle, len, len)?
                    .unwrap_or(len))
            }

            fn is_sorted_until_input(self, _less: GpuOp<Less>) -> Result<usize, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let len = $first_field.len();
                if len <= 1 {
                    return Ok(len);
                }
                let block_count_u32 = search_block_count(len)?;
                let client = $first_field.policy().client();
                let flag_handle = client.empty(len * std::mem::size_of::<u32>());
                unsafe {
                    $sorted_break_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&$first_field.handle, len, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(&$field.handle, len, 1),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }
                Ok(search::first_flag($first_field.policy(), flag_handle, len, len)?
                    .unwrap_or(len))
            }

            fn is_sorted_input(self, less: GpuOp<Less>) -> Result<bool, Error> {
                let len = ReadOnlySoA::len(&self);
                Ok(self.is_sorted_until_input(less)? == len)
            }
        }

    };
}

macro_rules! impl_tuple_pair_search {
    (
        $name:ident <
            $first:ident, $( $rest:ident ),+ ;
            $right_first:ident, $( $right_rest:ident ),+
        > {
            $first_field:ident: $left_first:ident / $right_first_value:ident,
            $( $field:ident: $left_value:ident / $right_value:ident ),+
        },
        $mismatch_kernel:ident,
        $find_first_of_kernel:ident,
        $subrange_match_kernel:ident,
        $lexicographical_diff_kernel:ident,
        $lexicographical_compare_at_kernel:ident,
        $includes_missing_kernel:ident
    ) => {
        impl<$first, $( $rest ),+, $right_first, $( $right_rest ),+, Op>
            PairSearchInput<$name<$right_first, $( $right_rest ),+>, Op>
            for $name<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $name<$right_first, $( $right_rest ),+>: ReadOnlySoA<Scalar = <$right_first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $right_first: KernelColumn<
                    Runtime = <$first as KernelColumn>::Runtime,
                    Item = <$first as KernelColumn>::Item,
                > + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
                $right_rest: KernelColumn<
                        Runtime = <$first as KernelColumn>::Runtime,
                        Item = <$rest as KernelColumn>::Item,
                    > + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            <$right_first as KernelColumn>::Expr: DeviceGpuExpr<<$right_first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
                <$right_rest as KernelColumn>::Expr:
                    DeviceGpuExpr<<$right_rest as KernelColumn>::Item>,
            )+
            Op: BinaryPredicateOp<(
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            )>,
        {
            fn equal_input(
                self,
                other: $name<$right_first, $( $right_rest ),+>,
                op: GpuOp<Op>,
            ) -> Result<bool, Error> {
                if ReadOnlySoA::len(&self) != ReadOnlySoA::len(&other) {
                    return Ok(false);
                }
                Ok(self.mismatch_input(other, op)?.is_none())
            }

            fn mismatch_input(
                self,
                other: $name<$right_first, $( $right_rest ),+>,
                _op: GpuOp<Op>,
            ) -> Result<Option<usize>, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $left_first = super::device_expr_collect(&self.$first_field)?;
                let $right_first_value = super::device_expr_collect(&other.$first_field)?;
                $(
                    let $left_value = super::device_expr_collect(&self.$field)?;
                    let $right_value = super::device_expr_collect(&other.$field)?;
                )+

                let min_len = $left_first.len().min($right_first_value.len());
                if min_len == 0 {
                    return if $left_first.len() == $right_first_value.len() {
                        Ok(None)
                    } else {
                        Ok(Some(0))
                    };
                }

                let block_count_u32 = search_block_count(min_len)?;
                let client = $left_first.policy().client();
                let flag_handle = client.empty(min_len * std::mem::size_of::<u32>());
                unsafe {
                    $mismatch_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Op,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(
                            &$left_first.handle,
                            $left_first.len(),
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(
                                &$left_value.handle,
                                $left_value.len(),
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<<$right_first as KernelColumn>::Item>(
                            &$right_first_value.handle,
                            $right_first_value.len(),
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$right_rest as KernelColumn>::Item>(
                                &$right_value.handle,
                                $right_value.len(),
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&flag_handle, min_len, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }

                if let Some(index) =
                    search::first_flag($left_first.policy(), flag_handle, min_len, min_len)?
                {
                    return Ok(Some(index));
                }
                if $left_first.len() == $right_first_value.len() {
                    Ok(None)
                } else {
                    Ok(Some(min_len))
                }
            }

            fn find_first_of_input(
                self,
                other: $name<$right_first, $( $right_rest ),+>,
                _op: GpuOp<Op>,
            ) -> Result<Option<usize>, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $left_first = super::device_expr_collect(&self.$first_field)?;
                let $right_first_value = super::device_expr_collect(&other.$first_field)?;
                $(
                    let $left_value = super::device_expr_collect(&self.$field)?;
                    let $right_value = super::device_expr_collect(&other.$field)?;
                )+

                if $left_first.len() == 0 || $right_first_value.len() == 0 {
                    return Ok(None);
                }

                let block_count_u32 = search_block_count($left_first.len())?;
                let client = $left_first.policy().client();
                let flag_handle = client.empty($left_first.len() * std::mem::size_of::<u32>());
                unsafe {
                    $find_first_of_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Op,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(
                            &$left_first.handle,
                            $left_first.len(),
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(
                                &$left_value.handle,
                                $left_value.len(),
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<<$right_first as KernelColumn>::Item>(
                            &$right_first_value.handle,
                            $right_first_value.len(),
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$right_rest as KernelColumn>::Item>(
                                &$right_value.handle,
                                $right_value.len(),
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&flag_handle, $left_first.len(), 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }
                search::first_flag(
                    $left_first.policy(),
                    flag_handle,
                    $left_first.len(),
                    $left_first.len(),
                )
            }

            fn find_end_input(
                self,
                other: $name<$right_first, $( $right_rest ),+>,
                _op: GpuOp<Op>,
            ) -> Result<Option<usize>, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $left_first = super::device_expr_collect(&self.$first_field)?;
                let $right_first_value = super::device_expr_collect(&other.$first_field)?;
                $(
                    let $left_value = super::device_expr_collect(&self.$field)?;
                    let $right_value = super::device_expr_collect(&other.$field)?;
                )+

                if $right_first_value.len() == 0 || $right_first_value.len() > $left_first.len() {
                    return Ok(None);
                }

                let candidate_len = $left_first.len() - $right_first_value.len() + 1;
                let block_count_u32 = search_block_count(candidate_len)?;
                let client = $left_first.policy().client();
                let flag_handle = client.empty(candidate_len * std::mem::size_of::<u32>());
                unsafe {
                    $subrange_match_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Op,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(
                            &$left_first.handle,
                            $left_first.len(),
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(
                                &$left_value.handle,
                                $left_value.len(),
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<<$right_first as KernelColumn>::Item>(
                            &$right_first_value.handle,
                            $right_first_value.len(),
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$right_rest as KernelColumn>::Item>(
                                &$right_value.handle,
                                $right_value.len(),
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&flag_handle, candidate_len, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }
                search::last_flag(
                    $left_first.policy(),
                    flag_handle,
                    candidate_len,
                    candidate_len,
                )
            }

            fn includes_input(
                self,
                other: $name<$right_first, $( $right_rest ),+>,
                _op: GpuOp<Op>,
            ) -> Result<bool, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $left_first = super::device_expr_collect(&self.$first_field)?;
                let $right_first_value = super::device_expr_collect(&other.$first_field)?;
                $(
                    let $left_value = super::device_expr_collect(&self.$field)?;
                    let $right_value = super::device_expr_collect(&other.$field)?;
                )+

                if $right_first_value.len() == 0 {
                    return Ok(true);
                }
                if $left_first.len() == 0 {
                    return Ok(false);
                }

                let block_count_u32 = search_block_count($right_first_value.len())?;
                let client = $left_first.policy().client();
                let flag_handle =
                    client.empty($right_first_value.len() * std::mem::size_of::<u32>());
                unsafe {
                    $includes_missing_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Op,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(
                            &$left_first.handle,
                            $left_first.len(),
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(
                                &$left_value.handle,
                                $left_value.len(),
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<<$right_first as KernelColumn>::Item>(
                            &$right_first_value.handle,
                            $right_first_value.len(),
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$right_rest as KernelColumn>::Item>(
                                &$right_value.handle,
                                $right_value.len(),
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<u32>(
                            &flag_handle,
                            $right_first_value.len(),
                            1,
                        ),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }
                Ok(search::first_flag(
                    $left_first.policy(),
                    flag_handle,
                    $right_first_value.len(),
                    $right_first_value.len(),
                )?
                .is_none())
            }

            fn lexicographical_compare_input(
                self,
                other: $name<$right_first, $( $right_rest ),+>,
                _op: GpuOp<Op>,
            ) -> Result<bool, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $left_first = super::device_expr_collect(&self.$first_field)?;
                let $right_first_value = super::device_expr_collect(&other.$first_field)?;
                $(
                    let $left_value = super::device_expr_collect(&self.$field)?;
                    let $right_value = super::device_expr_collect(&other.$field)?;
                )+

                let min_len = $left_first.len().min($right_first_value.len());
                if min_len == 0 {
                    return Ok($left_first.len() < $right_first_value.len());
                }

                let block_count_u32 = search_block_count(min_len)?;
                let client = $left_first.policy().client();
                let flag_handle = client.empty(min_len * std::mem::size_of::<u32>());
                unsafe {
                    $lexicographical_diff_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Op,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(
                            &$left_first.handle,
                            $left_first.len(),
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(
                                &$left_value.handle,
                                $left_value.len(),
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<<$right_first as KernelColumn>::Item>(
                            &$right_first_value.handle,
                            $right_first_value.len(),
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$right_rest as KernelColumn>::Item>(
                                &$right_value.handle,
                                $right_value.len(),
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&flag_handle, min_len, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }

                let Some(index) =
                    search::first_flag($left_first.policy(), flag_handle, min_len, min_len)?
                else {
                    return Ok($left_first.len() < $right_first_value.len());
                };

                let index_handle = client.create_from_slice(u32::as_bytes(&[index as u32]));
                let output_handle = client.empty(std::mem::size_of::<u32>());
                unsafe {
                    $lexicographical_compare_at_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Op,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::new_single(),
                        CubeDim::new_1d(1),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(
                            &$left_first.handle,
                            $left_first.len(),
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(
                                &$left_value.handle,
                                $left_value.len(),
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<<$right_first as KernelColumn>::Item>(
                            &$right_first_value.handle,
                            $right_first_value.len(),
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<<$right_rest as KernelColumn>::Item>(
                                &$right_value.handle,
                                $right_value.len(),
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&index_handle, 1, 1),
                        ArrayArg::from_raw_parts::<u32>(&output_handle, 1, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }
                Ok(scan::read_u32_scalar::<<$first as KernelColumn>::Runtime>(
                    client,
                    output_handle,
                ) != 0)
            }
        }
    };
}

impl_tuple_search!(SoAView2<A, B> { left: 0, right: 1 }, tuple2_adjacent_flags_kernel, tuple2_sorted_break_flags_kernel, tuple2_lower_bound_flags_kernel, tuple2_upper_bound_flags_kernel, tuple2_binary_search_at_kernel, tuple2_minmax_element_partials_kernel, tuple2_minmax_index_partials_kernel);
impl_tuple_search!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 }, tuple3_adjacent_flags_kernel, tuple3_sorted_break_flags_kernel, tuple3_lower_bound_flags_kernel, tuple3_upper_bound_flags_kernel, tuple3_binary_search_at_kernel, tuple3_minmax_element_partials_kernel, tuple3_minmax_index_partials_kernel);
impl_tuple_search!(SoAView4<A, B, C, D> { a: 0, b: 1, c: 2, d: 3 }, tuple4_adjacent_flags_kernel, tuple4_sorted_break_flags_kernel, tuple4_lower_bound_flags_kernel, tuple4_upper_bound_flags_kernel, tuple4_binary_search_at_kernel, tuple4_minmax_element_partials_kernel, tuple4_minmax_index_partials_kernel);
impl_tuple_search!(SoAView5<A, B, C, D, E> { a: 0, b: 1, c: 2, d: 3, e: 4 }, tuple5_adjacent_flags_kernel, tuple5_sorted_break_flags_kernel, tuple5_lower_bound_flags_kernel, tuple5_upper_bound_flags_kernel, tuple5_binary_search_at_kernel, tuple5_minmax_element_partials_kernel, tuple5_minmax_index_partials_kernel);
impl_tuple_search!(SoAView6<A, B, C, D, E, F> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5 }, tuple6_adjacent_flags_kernel, tuple6_sorted_break_flags_kernel, tuple6_lower_bound_flags_kernel, tuple6_upper_bound_flags_kernel, tuple6_binary_search_at_kernel, tuple6_minmax_element_partials_kernel, tuple6_minmax_index_partials_kernel);
impl_tuple_search!(SoAView7<A, B, C, D, E, F, G> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6 }, tuple7_adjacent_flags_kernel, tuple7_sorted_break_flags_kernel, tuple7_lower_bound_flags_kernel, tuple7_upper_bound_flags_kernel, tuple7_binary_search_at_kernel, tuple7_minmax_element_partials_kernel, tuple7_minmax_index_partials_kernel);
impl_tuple_search!(SoAView8<A, B, C, D, E, F, G, H> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7 }, tuple8_adjacent_flags_kernel, tuple8_sorted_break_flags_kernel, tuple8_lower_bound_flags_kernel, tuple8_upper_bound_flags_kernel, tuple8_binary_search_at_kernel, tuple8_minmax_element_partials_kernel, tuple8_minmax_index_partials_kernel);
impl_tuple_search!(SoAView9<A, B, C, D, E, F, G, H, I> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8 }, tuple9_adjacent_flags_kernel, tuple9_sorted_break_flags_kernel, tuple9_lower_bound_flags_kernel, tuple9_upper_bound_flags_kernel, tuple9_binary_search_at_kernel, tuple9_minmax_element_partials_kernel, tuple9_minmax_index_partials_kernel);
impl_tuple_search!(SoAView10<A, B, C, D, E, F, G, H, I, J> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8, j: 9 }, tuple10_adjacent_flags_kernel, tuple10_sorted_break_flags_kernel, tuple10_lower_bound_flags_kernel, tuple10_upper_bound_flags_kernel, tuple10_binary_search_at_kernel, tuple10_minmax_element_partials_kernel, tuple10_minmax_index_partials_kernel);
impl_tuple_search!(SoAView11<A, B, C, D, E, F, G, H, I, J, K> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8, j: 9, k: 10 }, tuple11_adjacent_flags_kernel, tuple11_sorted_break_flags_kernel, tuple11_lower_bound_flags_kernel, tuple11_upper_bound_flags_kernel, tuple11_binary_search_at_kernel, tuple11_minmax_element_partials_kernel, tuple11_minmax_index_partials_kernel);
impl_tuple_search!(SoAView12<A, B, C, D, E, F, G, H, I, J, K, L> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8, j: 9, k: 10, l: 11 }, tuple12_adjacent_flags_kernel, tuple12_sorted_break_flags_kernel, tuple12_lower_bound_flags_kernel, tuple12_upper_bound_flags_kernel, tuple12_binary_search_at_kernel, tuple12_minmax_element_partials_kernel, tuple12_minmax_index_partials_kernel);

impl_tuple_pair_search!(SoAView2<A, B; RA, RB> { left: left_a / right_a, right: left_b / right_b }, tuple2_mismatch_flags_kernel, tuple2_find_first_of_flags_kernel, tuple2_subrange_match_flags_kernel, tuple2_lexicographical_diff_flags_kernel, tuple2_lexicographical_compare_at_kernel, tuple2_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoAView3<A, B, C; RA, RB, RC> { first: left_a / right_a, second: left_b / right_b, third: left_c / right_c }, tuple3_mismatch_flags_kernel, tuple3_find_first_of_flags_kernel, tuple3_subrange_match_flags_kernel, tuple3_lexicographical_diff_flags_kernel, tuple3_lexicographical_compare_at_kernel, tuple3_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoAView4<A, B, C, D; RA, RB, RC, RD> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d }, tuple4_mismatch_flags_kernel, tuple4_find_first_of_flags_kernel, tuple4_subrange_match_flags_kernel, tuple4_lexicographical_diff_flags_kernel, tuple4_lexicographical_compare_at_kernel, tuple4_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoAView5<A, B, C, D, E; RA, RB, RC, RD, RE> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e }, tuple5_mismatch_flags_kernel, tuple5_find_first_of_flags_kernel, tuple5_subrange_match_flags_kernel, tuple5_lexicographical_diff_flags_kernel, tuple5_lexicographical_compare_at_kernel, tuple5_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoAView6<A, B, C, D, E, F; RA, RB, RC, RD, RE, RF> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f }, tuple6_mismatch_flags_kernel, tuple6_find_first_of_flags_kernel, tuple6_subrange_match_flags_kernel, tuple6_lexicographical_diff_flags_kernel, tuple6_lexicographical_compare_at_kernel, tuple6_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoAView7<A, B, C, D, E, F, G; RA, RB, RC, RD, RE, RF, RG> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g }, tuple7_mismatch_flags_kernel, tuple7_find_first_of_flags_kernel, tuple7_subrange_match_flags_kernel, tuple7_lexicographical_diff_flags_kernel, tuple7_lexicographical_compare_at_kernel, tuple7_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoAView8<A, B, C, D, E, F, G, H; RA, RB, RC, RD, RE, RF, RG, RH> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g, h: left_h / right_h }, tuple8_mismatch_flags_kernel, tuple8_find_first_of_flags_kernel, tuple8_subrange_match_flags_kernel, tuple8_lexicographical_diff_flags_kernel, tuple8_lexicographical_compare_at_kernel, tuple8_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoAView9<A, B, C, D, E, F, G, H, I; RA, RB, RC, RD, RE, RF, RG, RH, RI> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g, h: left_h / right_h, i: left_i / right_i }, tuple9_mismatch_flags_kernel, tuple9_find_first_of_flags_kernel, tuple9_subrange_match_flags_kernel, tuple9_lexicographical_diff_flags_kernel, tuple9_lexicographical_compare_at_kernel, tuple9_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoAView10<A, B, C, D, E, F, G, H, I, J; RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g, h: left_h / right_h, i: left_i / right_i, j: left_j / right_j }, tuple10_mismatch_flags_kernel, tuple10_find_first_of_flags_kernel, tuple10_subrange_match_flags_kernel, tuple10_lexicographical_diff_flags_kernel, tuple10_lexicographical_compare_at_kernel, tuple10_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoAView11<A, B, C, D, E, F, G, H, I, J, K; RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ, RK> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g, h: left_h / right_h, i: left_i / right_i, j: left_j / right_j, k: left_k / right_k }, tuple11_mismatch_flags_kernel, tuple11_find_first_of_flags_kernel, tuple11_subrange_match_flags_kernel, tuple11_lexicographical_diff_flags_kernel, tuple11_lexicographical_compare_at_kernel, tuple11_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoAView12<A, B, C, D, E, F, G, H, I, J, K, L; RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ, RK, RL> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g, h: left_h / right_h, i: left_i / right_i, j: left_j / right_j, k: left_k / right_k, l: left_l / right_l }, tuple12_mismatch_flags_kernel, tuple12_find_first_of_flags_kernel, tuple12_subrange_match_flags_kernel, tuple12_lexicographical_diff_flags_kernel, tuple12_lexicographical_compare_at_kernel, tuple12_includes_missing_flags_kernel);
impl_tuple_search!(SoA2<A, B> { left: 0, right: 1 }, tuple2_adjacent_flags_kernel, tuple2_sorted_break_flags_kernel, tuple2_lower_bound_flags_kernel, tuple2_upper_bound_flags_kernel, tuple2_binary_search_at_kernel, tuple2_minmax_element_partials_kernel, tuple2_minmax_index_partials_kernel);
impl_tuple_search!(SoA3<A, B, C> { first: 0, second: 1, third: 2 }, tuple3_adjacent_flags_kernel, tuple3_sorted_break_flags_kernel, tuple3_lower_bound_flags_kernel, tuple3_upper_bound_flags_kernel, tuple3_binary_search_at_kernel, tuple3_minmax_element_partials_kernel, tuple3_minmax_index_partials_kernel);
impl_tuple_search!(SoA4<A, B, C, D> { a: 0, b: 1, c: 2, d: 3 }, tuple4_adjacent_flags_kernel, tuple4_sorted_break_flags_kernel, tuple4_lower_bound_flags_kernel, tuple4_upper_bound_flags_kernel, tuple4_binary_search_at_kernel, tuple4_minmax_element_partials_kernel, tuple4_minmax_index_partials_kernel);
impl_tuple_search!(SoA5<A, B, C, D, E> { a: 0, b: 1, c: 2, d: 3, e: 4 }, tuple5_adjacent_flags_kernel, tuple5_sorted_break_flags_kernel, tuple5_lower_bound_flags_kernel, tuple5_upper_bound_flags_kernel, tuple5_binary_search_at_kernel, tuple5_minmax_element_partials_kernel, tuple5_minmax_index_partials_kernel);
impl_tuple_search!(SoA6<A, B, C, D, E, F> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5 }, tuple6_adjacent_flags_kernel, tuple6_sorted_break_flags_kernel, tuple6_lower_bound_flags_kernel, tuple6_upper_bound_flags_kernel, tuple6_binary_search_at_kernel, tuple6_minmax_element_partials_kernel, tuple6_minmax_index_partials_kernel);
impl_tuple_search!(SoA7<A, B, C, D, E, F, G> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6 }, tuple7_adjacent_flags_kernel, tuple7_sorted_break_flags_kernel, tuple7_lower_bound_flags_kernel, tuple7_upper_bound_flags_kernel, tuple7_binary_search_at_kernel, tuple7_minmax_element_partials_kernel, tuple7_minmax_index_partials_kernel);
impl_tuple_search!(SoA8<A, B, C, D, E, F, G, H> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7 }, tuple8_adjacent_flags_kernel, tuple8_sorted_break_flags_kernel, tuple8_lower_bound_flags_kernel, tuple8_upper_bound_flags_kernel, tuple8_binary_search_at_kernel, tuple8_minmax_element_partials_kernel, tuple8_minmax_index_partials_kernel);
impl_tuple_search!(SoA9<A, B, C, D, E, F, G, H, I> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8 }, tuple9_adjacent_flags_kernel, tuple9_sorted_break_flags_kernel, tuple9_lower_bound_flags_kernel, tuple9_upper_bound_flags_kernel, tuple9_binary_search_at_kernel, tuple9_minmax_element_partials_kernel, tuple9_minmax_index_partials_kernel);
impl_tuple_search!(SoA10<A, B, C, D, E, F, G, H, I, J> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8, j: 9 }, tuple10_adjacent_flags_kernel, tuple10_sorted_break_flags_kernel, tuple10_lower_bound_flags_kernel, tuple10_upper_bound_flags_kernel, tuple10_binary_search_at_kernel, tuple10_minmax_element_partials_kernel, tuple10_minmax_index_partials_kernel);
impl_tuple_search!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8, j: 9, k: 10 }, tuple11_adjacent_flags_kernel, tuple11_sorted_break_flags_kernel, tuple11_lower_bound_flags_kernel, tuple11_upper_bound_flags_kernel, tuple11_binary_search_at_kernel, tuple11_minmax_element_partials_kernel, tuple11_minmax_index_partials_kernel);
impl_tuple_search!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8, j: 9, k: 10, l: 11 }, tuple12_adjacent_flags_kernel, tuple12_sorted_break_flags_kernel, tuple12_lower_bound_flags_kernel, tuple12_upper_bound_flags_kernel, tuple12_binary_search_at_kernel, tuple12_minmax_element_partials_kernel, tuple12_minmax_index_partials_kernel);
impl_tuple_pair_search!(SoA2<A, B; RA, RB> { left: left_a / right_a, right: left_b / right_b }, tuple2_mismatch_flags_kernel, tuple2_find_first_of_flags_kernel, tuple2_subrange_match_flags_kernel, tuple2_lexicographical_diff_flags_kernel, tuple2_lexicographical_compare_at_kernel, tuple2_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoA3<A, B, C; RA, RB, RC> { first: left_a / right_a, second: left_b / right_b, third: left_c / right_c }, tuple3_mismatch_flags_kernel, tuple3_find_first_of_flags_kernel, tuple3_subrange_match_flags_kernel, tuple3_lexicographical_diff_flags_kernel, tuple3_lexicographical_compare_at_kernel, tuple3_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoA4<A, B, C, D; RA, RB, RC, RD> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d }, tuple4_mismatch_flags_kernel, tuple4_find_first_of_flags_kernel, tuple4_subrange_match_flags_kernel, tuple4_lexicographical_diff_flags_kernel, tuple4_lexicographical_compare_at_kernel, tuple4_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoA5<A, B, C, D, E; RA, RB, RC, RD, RE> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e }, tuple5_mismatch_flags_kernel, tuple5_find_first_of_flags_kernel, tuple5_subrange_match_flags_kernel, tuple5_lexicographical_diff_flags_kernel, tuple5_lexicographical_compare_at_kernel, tuple5_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoA6<A, B, C, D, E, F; RA, RB, RC, RD, RE, RF> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f }, tuple6_mismatch_flags_kernel, tuple6_find_first_of_flags_kernel, tuple6_subrange_match_flags_kernel, tuple6_lexicographical_diff_flags_kernel, tuple6_lexicographical_compare_at_kernel, tuple6_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoA7<A, B, C, D, E, F, G; RA, RB, RC, RD, RE, RF, RG> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g }, tuple7_mismatch_flags_kernel, tuple7_find_first_of_flags_kernel, tuple7_subrange_match_flags_kernel, tuple7_lexicographical_diff_flags_kernel, tuple7_lexicographical_compare_at_kernel, tuple7_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoA8<A, B, C, D, E, F, G, H; RA, RB, RC, RD, RE, RF, RG, RH> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g, h: left_h / right_h }, tuple8_mismatch_flags_kernel, tuple8_find_first_of_flags_kernel, tuple8_subrange_match_flags_kernel, tuple8_lexicographical_diff_flags_kernel, tuple8_lexicographical_compare_at_kernel, tuple8_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoA9<A, B, C, D, E, F, G, H, I; RA, RB, RC, RD, RE, RF, RG, RH, RI> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g, h: left_h / right_h, i: left_i / right_i }, tuple9_mismatch_flags_kernel, tuple9_find_first_of_flags_kernel, tuple9_subrange_match_flags_kernel, tuple9_lexicographical_diff_flags_kernel, tuple9_lexicographical_compare_at_kernel, tuple9_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoA10<A, B, C, D, E, F, G, H, I, J; RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g, h: left_h / right_h, i: left_i / right_i, j: left_j / right_j }, tuple10_mismatch_flags_kernel, tuple10_find_first_of_flags_kernel, tuple10_subrange_match_flags_kernel, tuple10_lexicographical_diff_flags_kernel, tuple10_lexicographical_compare_at_kernel, tuple10_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoA11<A, B, C, D, E, F, G, H, I, J, K; RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ, RK> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g, h: left_h / right_h, i: left_i / right_i, j: left_j / right_j, k: left_k / right_k }, tuple11_mismatch_flags_kernel, tuple11_find_first_of_flags_kernel, tuple11_subrange_match_flags_kernel, tuple11_lexicographical_diff_flags_kernel, tuple11_lexicographical_compare_at_kernel, tuple11_includes_missing_flags_kernel);
impl_tuple_pair_search!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L; RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ, RK, RL> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g, h: left_h / right_h, i: left_i / right_i, j: left_j / right_j, k: left_k / right_k, l: left_l / right_l }, tuple12_mismatch_flags_kernel, tuple12_find_first_of_flags_kernel, tuple12_subrange_match_flags_kernel, tuple12_lexicographical_diff_flags_kernel, tuple12_lexicographical_compare_at_kernel, tuple12_includes_missing_flags_kernel);

/// Finds the minimum element index according to `Less`.
pub fn min_element<Input, Less>(input: Input, _less: Less) -> Result<Option<usize>, Error>
where
    Input: MinMaxInput<Less>,
{
    input.min_element_input(GpuOp::<Less>::new())
}

/// Finds the maximum element index according to `Less`.
pub fn max_element<Input, Less>(input: Input, _less: Less) -> Result<Option<usize>, Error>
where
    Input: MinMaxInput<Less>,
{
    input.max_element_input(GpuOp::<Less>::new())
}

/// Finds both minimum and maximum element indices according to `Less`.
pub fn minmax_element<Input, Less>(
    input: Input,
    _less: Less,
) -> Result<Option<(usize, usize)>, Error>
where
    Input: MinMaxInput<Less>,
{
    input.minmax_element_input(GpuOp::<Less>::new())
}

/// Finds the first adjacent pair that satisfies `Pred`.
pub fn adjacent_find<Input, Pred>(input: Input, _pred: Pred) -> Result<Option<usize>, Error>
where
    Input: AdjacentFindInput<Pred>,
{
    input.adjacent_find_input(GpuOp::<Pred>::new())
}

/// Returns whether two inputs are equal under `Eq`.
pub fn equal<Left, Right, Eq>(left: Left, right: Right, _eq: Eq) -> Result<bool, Error>
where
    Left: PairSearchInput<Right, Eq>,
{
    left.equal_input(right, GpuOp::<Eq>::new())
}

/// Finds the first mismatch between two inputs.
pub fn mismatch<Left, Right, Eq>(left: Left, right: Right, _eq: Eq) -> Result<Option<usize>, Error>
where
    Left: PairSearchInput<Right, Eq>,
{
    left.mismatch_input(right, GpuOp::<Eq>::new())
}

/// Finds the first input element equal to any value in `needles`.
pub fn find_first_of<Input, Needles, Eq>(
    input: Input,
    needles: Needles,
    _eq: Eq,
) -> Result<Option<usize>, Error>
where
    Input: PairSearchInput<Needles, Eq>,
{
    input.find_first_of_input(needles, GpuOp::<Eq>::new())
}

/// Returns the equal range for `value` in a sorted input.
pub fn equal_range<Input, Less>(
    input: Input,
    value: <Input as SortedSearchInput<Less>>::Item,
    _less: Less,
) -> Result<(usize, usize), Error>
where
    Input: SortedSearchInput<Less>,
{
    input.equal_range_input(value, GpuOp::<Less>::new())
}

/// Finds the first sorted insertion point for `value`.
pub fn lower_bound<Input, Less>(
    input: Input,
    value: <Input as SortedSearchInput<Less>>::Item,
    _less: Less,
) -> Result<usize, Error>
where
    Input: SortedSearchInput<Less>,
{
    input.lower_bound_input(value, GpuOp::<Less>::new())
}

/// Finds the last sorted insertion point for `value`.
pub fn upper_bound<Input, Less>(
    input: Input,
    value: <Input as SortedSearchInput<Less>>::Item,
    _less: Less,
) -> Result<usize, Error>
where
    Input: SortedSearchInput<Less>,
{
    input.upper_bound_input(value, GpuOp::<Less>::new())
}

/// Returns the first position where the sorted order is broken.
pub fn is_sorted_until<Input, Less>(input: Input, _less: Less) -> Result<usize, Error>
where
    Input: SortedSearchInput<Less>,
{
    input.is_sorted_until_input(GpuOp::<Less>::new())
}

/// Returns whether an input is sorted.
pub fn is_sorted<Input, Less>(input: Input, _less: Less) -> Result<bool, Error>
where
    Input: SortedSearchInput<Less>,
{
    input.is_sorted_input(GpuOp::<Less>::new())
}

/// Lexicographically compares two inputs.
pub fn lexicographical_compare<Left, Right, Less>(
    left: Left,
    right: Right,
    _less: Less,
) -> Result<bool, Error>
where
    Left: PairSearchInput<Right, Less>,
{
    left.lexicographical_compare_input(right, GpuOp::<Less>::new())
}
