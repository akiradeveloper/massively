use crate::{
    detail::op::kernel::PredicateOp2,
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA2, SoA3, SoAView1, SoAView2,
        SoAView3,
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::GpuOp,
    policy::CubePolicy,
    primitives::{scan, search},
};
use cubecl::prelude::*;

const BLOCK_SEARCH_SIZE: u32 = 256;

fn search_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_SEARCH_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

fn materialize_one<Source>(
    policy: &CubePolicy<Source::Runtime>,
    input: SoAView1<Source>,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    SoAView1<Source>: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    ReadOnlySoA::validate(&input)?;
    super::device_expr_collect_with_policy(policy, &input.source)
}

fn materialize_pair<Left, Right>(
    policy: &CubePolicy<Left::Runtime>,
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
    SoAView1<Left>: ReadOnlySoA<Item = (Left::Item,), Scalar = Left::Item>,
    SoAView1<Right>: ReadOnlySoA<Item = (Right::Item,), Scalar = Right::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
{
    let left = materialize_one(policy, left)?;
    let right = materialize_one(policy, right)?;
    Ok((left, right))
}

/// Input accepted by min/max element queries.
#[doc(hidden)]
pub trait MinMaxInput<Less> {
    /// Runtime used by this input.
    type Runtime: Runtime;

    /// Finds the minimum element index.
    fn min_element_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Option<usize>, Error>;
    /// Finds the maximum element index.
    fn max_element_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Option<usize>, Error>;
    /// Finds both minimum and maximum element indices.
    fn minmax_element_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Option<(usize, usize)>, Error>;
}

impl<Source, Less> MinMaxInput<Less> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: PredicateOp2<Source::Item>,
{
    type Runtime = Source::Runtime;

    fn min_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Option<usize>, Error> {
        ReadOnlySoA::validate(&self)?;
        Ok(
            super::device_expr_minmax_element_with_policy::<Source, Less>(policy, &self.source)?
                .map(|(min, _)| min),
        )
    }

    fn max_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Option<usize>, Error> {
        ReadOnlySoA::validate(&self)?;
        Ok(
            super::device_expr_minmax_element_with_policy::<Source, Less>(policy, &self.source)?
                .map(|(_, max)| max),
        )
    }

    fn minmax_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Option<(usize, usize)>, Error> {
        ReadOnlySoA::validate(&self)?;
        super::device_expr_minmax_element_with_policy::<Source, Less>(policy, &self.source)
    }
}

impl<Source, Less> MinMaxInput<Less> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: PredicateOp2<Source::Item>,
{
    type Runtime = Source::Runtime;

    fn min_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Option<usize>, Error> {
        <SoAView1<Source> as MinMaxInput<Less>>::min_element_input(
            SoAView1 { source: self },
            policy,
            less,
        )
    }

    fn max_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Option<usize>, Error> {
        <SoAView1<Source> as MinMaxInput<Less>>::max_element_input(
            SoAView1 { source: self },
            policy,
            less,
        )
    }

    fn minmax_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Option<(usize, usize)>, Error> {
        <SoAView1<Source> as MinMaxInput<Less>>::minmax_element_input(
            SoAView1 { source: self },
            policy,
            less,
        )
    }
}

impl<Source, Less> MinMaxInput<Less> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: PredicateOp2<(Source::Item,)>,
{
    type Runtime = Source::Runtime;

    fn min_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Option<usize>, Error> {
        <Source as MinMaxInput<super::Tuple1Less<Less>>>::min_element_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn max_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Option<usize>, Error> {
        <Source as MinMaxInput<super::Tuple1Less<Less>>>::max_element_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn minmax_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Option<(usize, usize)>, Error> {
        <Source as MinMaxInput<super::Tuple1Less<Less>>>::minmax_element_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }
}

/// Input accepted by adjacent-pair search.
#[doc(hidden)]
pub trait AdjacentFindInput<Pred> {
    /// Runtime used by this input.
    type Runtime: Runtime;

    /// Finds the first adjacent pair that satisfies `Pred`.
    fn adjacent_find_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        pred: GpuOp<Pred>,
    ) -> Result<Option<usize>, Error>;
}

impl<Source, Pred> AdjacentFindInput<Pred> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: PredicateOp2<Source::Item>,
{
    type Runtime = Source::Runtime;

    fn adjacent_find_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _pred: GpuOp<Pred>,
    ) -> Result<Option<usize>, Error> {
        let input = super::device_expr_collect_with_policy(policy, &self.source)?;
        search::adjacent_find(policy, &input, GpuOp::<Pred>::new())
    }
}

impl<Source, Pred> AdjacentFindInput<Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: PredicateOp2<Source::Item>,
{
    type Runtime = Source::Runtime;

    fn adjacent_find_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        pred: GpuOp<Pred>,
    ) -> Result<Option<usize>, Error> {
        <SoAView1<Source> as AdjacentFindInput<Pred>>::adjacent_find_input(
            SoAView1 { source: self },
            policy,
            pred,
        )
    }
}

impl<Source, Pred> AdjacentFindInput<Pred> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: PredicateOp2<(Source::Item,)>,
{
    type Runtime = Source::Runtime;

    fn adjacent_find_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _pred: GpuOp<Pred>,
    ) -> Result<Option<usize>, Error> {
        <Source as AdjacentFindInput<super::Tuple1Less<Pred>>>::adjacent_find_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Pred>>::new(),
        )
    }
}

/// Input accepted by sorted single-range search.
#[doc(hidden)]
pub trait SortedSearchInput<Less> {
    /// Runtime used by this input.
    type Runtime: Runtime;
    /// Element type.
    type Item;

    /// Returns the equal range for `value` in this sorted input.
    fn equal_range_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<(usize, usize), Error>;
    /// Finds the first sorted insertion point for `value`.
    fn lower_bound_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<usize, Error>;
    /// Finds the last sorted insertion point for `value`.
    fn upper_bound_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<usize, Error>;
    /// Returns the first position where sorted order is broken.
    fn is_sorted_until_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<usize, Error>;
    /// Returns whether this input is sorted.
    fn is_sorted_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<bool, Error>;
}

impl<Source, Less> SortedSearchInput<Less> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: PredicateOp2<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Item = Source::Item;

    fn equal_range_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        _less: GpuOp<Less>,
    ) -> Result<(usize, usize), Error> {
        let input = super::device_expr_collect_with_policy(policy, &self.source)?;
        Ok((
            search::lower_bound(policy, &input, value.clone(), GpuOp::<Less>::new())?,
            search::upper_bound(policy, &input, value, GpuOp::<Less>::new())?,
        ))
    }

    fn lower_bound_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        _less: GpuOp<Less>,
    ) -> Result<usize, Error> {
        let input = super::device_expr_collect_with_policy(policy, &self.source)?;
        search::lower_bound(policy, &input, value, GpuOp::<Less>::new())
    }

    fn upper_bound_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        _less: GpuOp<Less>,
    ) -> Result<usize, Error> {
        let input = super::device_expr_collect_with_policy(policy, &self.source)?;
        search::upper_bound(policy, &input, value, GpuOp::<Less>::new())
    }

    fn is_sorted_until_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<usize, Error> {
        let input = super::device_expr_collect_with_policy(policy, &self.source)?;
        search::is_sorted_until(policy, &input, GpuOp::<Less>::new())
    }

    fn is_sorted_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<bool, Error> {
        let len = ReadOnlySoA::len(&self);
        Ok(self.is_sorted_until_input(policy, less)? == len)
    }
}

impl<Source, Less> SortedSearchInput<Less> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: PredicateOp2<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Item = Source::Item;

    fn equal_range_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<(usize, usize), Error> {
        <SoAView1<Source> as SortedSearchInput<Less>>::equal_range_input(
            SoAView1 { source: self },
            policy,
            value,
            less,
        )
    }

    fn lower_bound_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<usize, Error> {
        <SoAView1<Source> as SortedSearchInput<Less>>::lower_bound_input(
            SoAView1 { source: self },
            policy,
            value,
            less,
        )
    }

    fn upper_bound_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<usize, Error> {
        <SoAView1<Source> as SortedSearchInput<Less>>::upper_bound_input(
            SoAView1 { source: self },
            policy,
            value,
            less,
        )
    }

    fn is_sorted_until_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<usize, Error> {
        <SoAView1<Source> as SortedSearchInput<Less>>::is_sorted_until_input(
            SoAView1 { source: self },
            policy,
            less,
        )
    }

    fn is_sorted_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<bool, Error> {
        <SoAView1<Source> as SortedSearchInput<Less>>::is_sorted_input(
            SoAView1 { source: self },
            policy,
            less,
        )
    }
}

impl<Source, Less> SortedSearchInput<Less> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: PredicateOp2<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Item = (Source::Item,);

    fn equal_range_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        _less: GpuOp<Less>,
    ) -> Result<(usize, usize), Error> {
        <Source as SortedSearchInput<super::Tuple1Less<Less>>>::equal_range_input(
            self.0,
            policy,
            value.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn lower_bound_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        _less: GpuOp<Less>,
    ) -> Result<usize, Error> {
        <Source as SortedSearchInput<super::Tuple1Less<Less>>>::lower_bound_input(
            self.0,
            policy,
            value.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn upper_bound_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        _less: GpuOp<Less>,
    ) -> Result<usize, Error> {
        <Source as SortedSearchInput<super::Tuple1Less<Less>>>::upper_bound_input(
            self.0,
            policy,
            value.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn is_sorted_until_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<usize, Error> {
        <Source as SortedSearchInput<super::Tuple1Less<Less>>>::is_sorted_until_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn is_sorted_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<bool, Error> {
        <Source as SortedSearchInput<super::Tuple1Less<Less>>>::is_sorted_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }
}

macro_rules! impl_sorted_search_tuple_input {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Less> SortedSearchInput<Less> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: SortedSearchInput<Less>,
        {
            type Runtime = <$view<$( $ty ),+> as SortedSearchInput<Less>>::Runtime;
            type Item = <$view<$( $ty ),+> as SortedSearchInput<Less>>::Item;

            fn equal_range_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                value: Self::Item,
                less: GpuOp<Less>,
            ) -> Result<(usize, usize), Error> {
                <$view<$( $ty ),+> as SortedSearchInput<Less>>::equal_range_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    value,
                    less,
                )
            }

            fn lower_bound_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                value: Self::Item,
                less: GpuOp<Less>,
            ) -> Result<usize, Error> {
                <$view<$( $ty ),+> as SortedSearchInput<Less>>::lower_bound_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    value,
                    less,
                )
            }

            fn upper_bound_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                value: Self::Item,
                less: GpuOp<Less>,
            ) -> Result<usize, Error> {
                <$view<$( $ty ),+> as SortedSearchInput<Less>>::upper_bound_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    value,
                    less,
                )
            }

            fn is_sorted_until_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                less: GpuOp<Less>,
            ) -> Result<usize, Error> {
                <$view<$( $ty ),+> as SortedSearchInput<Less>>::is_sorted_until_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    less,
                )
            }

            fn is_sorted_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                less: GpuOp<Less>,
            ) -> Result<bool, Error> {
                <$view<$( $ty ),+> as SortedSearchInput<Less>>::is_sorted_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    less,
                )
            }
        }
    };
}

impl_sorted_search_tuple_input!(SoAView2<A, B> { left: 0, right: 1 });
impl_sorted_search_tuple_input!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

/// Pair of inputs accepted by binary search/comparison algorithms.
#[doc(hidden)]
pub trait PairSearchInput<Other, Op> {
    /// Runtime used by both inputs.
    type Runtime: Runtime;

    /// Returns whether two inputs are equal under `Op`.
    fn equal_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        op: GpuOp<Op>,
    ) -> Result<bool, Error>;
    /// Finds the first mismatch between two inputs.
    fn mismatch_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        op: GpuOp<Op>,
    ) -> Result<Option<usize>, Error>;
    /// Finds the first element equal to any value in `other`.
    fn find_first_of_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        op: GpuOp<Op>,
    ) -> Result<Option<usize>, Error>;
    /// Lexicographically compares two inputs.
    fn lexicographical_compare_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        op: GpuOp<Op>,
    ) -> Result<bool, Error>;
}

impl<Left, Right, Op> PairSearchInput<SoAView1<Right>, Op> for SoAView1<Left>
where
    Self: ReadOnlySoA<Item = (Left::Item,), Scalar = Left::Item>,
    SoAView1<Right>: ReadOnlySoA<Item = (Right::Item,), Scalar = Right::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: PredicateOp2<Left::Item>,
{
    type Runtime = Left::Runtime;

    fn equal_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: SoAView1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        let (left, right) = materialize_pair(policy, self, other)?;
        search::equal(policy, &left, &right, GpuOp::<Op>::new())
    }

    fn mismatch_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: SoAView1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<Option<usize>, Error> {
        let (left, right) = materialize_pair(policy, self, other)?;
        search::mismatch(policy, &left, &right, GpuOp::<Op>::new())
    }

    fn find_first_of_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: SoAView1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<Option<usize>, Error> {
        let (input, needles) = materialize_pair(policy, self, other)?;
        search::find_first_of(policy, &input, &needles, GpuOp::<Op>::new())
    }

    fn lexicographical_compare_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: SoAView1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        let (left, right) = materialize_pair(policy, self, other)?;
        search::lexicographical_compare(policy, &left, &right, GpuOp::<Op>::new())
    }
}

impl<Left, Right, Op> PairSearchInput<Right, Op> for Left
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: PredicateOp2<Left::Item>,
{
    type Runtime = Left::Runtime;

    fn equal_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Right,
        op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        <SoAView1<Left> as PairSearchInput<SoAView1<Right>, Op>>::equal_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: other },
            op,
        )
    }

    fn mismatch_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Right,
        op: GpuOp<Op>,
    ) -> Result<Option<usize>, Error> {
        <SoAView1<Left> as PairSearchInput<SoAView1<Right>, Op>>::mismatch_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: other },
            op,
        )
    }

    fn find_first_of_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Right,
        op: GpuOp<Op>,
    ) -> Result<Option<usize>, Error> {
        <SoAView1<Left> as PairSearchInput<SoAView1<Right>, Op>>::find_first_of_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: other },
            op,
        )
    }

    fn lexicographical_compare_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Right,
        op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        <SoAView1<Left> as PairSearchInput<SoAView1<Right>, Op>>::lexicographical_compare_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: other },
            op,
        )
    }
}

impl<Left, Right, Op> PairSearchInput<(Right,), Op> for (Left,)
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: PredicateOp2<(Left::Item,)>,
{
    type Runtime = Left::Runtime;

    fn equal_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: (Right,),
        _op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        <Left as PairSearchInput<Right, super::Tuple1Less<Op>>>::equal_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Op>>::new(),
        )
    }

    fn mismatch_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: (Right,),
        _op: GpuOp<Op>,
    ) -> Result<Option<usize>, Error> {
        <Left as PairSearchInput<Right, super::Tuple1Less<Op>>>::mismatch_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Op>>::new(),
        )
    }

    fn find_first_of_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: (Right,),
        _op: GpuOp<Op>,
    ) -> Result<Option<usize>, Error> {
        <Left as PairSearchInput<Right, super::Tuple1Less<Op>>>::find_first_of_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Op>>::new(),
        )
    }

    fn lexicographical_compare_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: (Right,),
        _op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        <Left as PairSearchInput<Right, super::Tuple1Less<Op>>>::lexicographical_compare_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Op>>::new(),
        )
    }
}

macro_rules! impl_pair_search_tuple_input {
    (
        $view:ident < $( $left_ty:ident ),+ ; $( $right_ty:ident ),+ > {
            $( $field:ident: $left_index:tt / $right_index:tt ),+
        }
    ) => {
        impl<$( $left_ty ),+, $( $right_ty ),+, Op>
            PairSearchInput<($( $right_ty ),+), Op> for ($( $left_ty ),+)
        where
            $view<$( $left_ty ),+>: PairSearchInput<$view<$( $right_ty ),+>, Op>,
        {
            type Runtime =
                <$view<$( $left_ty ),+> as PairSearchInput<$view<$( $right_ty ),+>, Op>>::Runtime;

            fn equal_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: ($( $right_ty ),+),
                op: GpuOp<Op>,
            ) -> Result<bool, Error> {
                <$view<$( $left_ty ),+> as PairSearchInput<$view<$( $right_ty ),+>, Op>>::equal_input(
                    $view { $( $field: self.$left_index ),+ },
                    policy,
                    $view { $( $field: other.$right_index ),+ },
                    op,
                )
            }

            fn mismatch_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: ($( $right_ty ),+),
                op: GpuOp<Op>,
            ) -> Result<Option<usize>, Error> {
                <$view<$( $left_ty ),+> as PairSearchInput<$view<$( $right_ty ),+>, Op>>::mismatch_input(
                    $view { $( $field: self.$left_index ),+ },
                    policy,
                    $view { $( $field: other.$right_index ),+ },
                    op,
                )
            }

            fn find_first_of_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: ($( $right_ty ),+),
                op: GpuOp<Op>,
            ) -> Result<Option<usize>, Error> {
                <$view<$( $left_ty ),+> as PairSearchInput<$view<$( $right_ty ),+>, Op>>::find_first_of_input(
                    $view { $( $field: self.$left_index ),+ },
                    policy,
                    $view { $( $field: other.$right_index ),+ },
                    op,
                )
            }

            fn lexicographical_compare_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: ($( $right_ty ),+),
                op: GpuOp<Op>,
            ) -> Result<bool, Error> {
                <$view<$( $left_ty ),+> as PairSearchInput<$view<$( $right_ty ),+>, Op>>::lexicographical_compare_input(
                    $view { $( $field: self.$left_index ),+ },
                    policy,
                    $view { $( $field: other.$right_index ),+ },
                    op,
                )
            }
        }
    };
}

impl_pair_search_tuple_input!(SoAView2<A, B; RA, RB> { left: 0 / 0, right: 1 / 1 });
impl_pair_search_tuple_input!(SoAView3<A, B, C; RA, RB, RC> { first: 0 / 0, second: 1 / 1, third: 2 / 2 });

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
            Less: PredicateOp2<(
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            fn min_element_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                less: GpuOp<Less>,
            ) -> Result<Option<usize>, Error> {
                Ok(self.minmax_element_input(policy, less)?.map(|(min, _)| min))
            }

            fn max_element_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                less: GpuOp<Less>,
            ) -> Result<Option<usize>, Error> {
                Ok(self.minmax_element_input(policy, less)?.map(|(_, max)| max))
            }

            fn minmax_element_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                _less: GpuOp<Less>,
            ) -> Result<Option<(usize, usize)>, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = super::device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let len = $first_field.len();
                if len == 0 {
                    return Ok(None);
                }

                let client = policy.client();
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
                        unsafe { BufferArg::from_raw_parts($first_field.handle.clone(), len) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.handle.clone(), len) },
                        )+
                        unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count * 2) },
                    );
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
                            unsafe { BufferArg::from_raw_parts($first_field.handle.clone(), len) },
                            $(
                                unsafe { BufferArg::from_raw_parts($field.handle.clone(), len) },
                            )+
                            unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count * 2) },
                            unsafe { BufferArg::from_raw_parts(candidate_len_handle.clone(), 1) },
                            unsafe { BufferArg::from_raw_parts(next_handle.clone(), next_count * 2) },
                        );
                    }

                    current_handle = next_handle;
                    current_count = next_count;
                    current_count_u32 = next_count_u32;
                }

                let bytes = client.read_one(current_handle).map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
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
            Pred: PredicateOp2<(
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            fn adjacent_find_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                _pred: GpuOp<Pred>,
            ) -> Result<Option<usize>, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = super::device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let len = $first_field.len();
                if len < 2 {
                    return Ok(None);
                }
                let block_count_u32 = search_block_count(len)?;
                let client = policy.client();
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
                        unsafe { BufferArg::from_raw_parts($first_field.handle.clone(), len) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.handle.clone(), len) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                    );
                }
                search::first_flag(policy, flag_handle, len, len - 1)
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
            Less: PredicateOp2<(
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            type Item = (
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            );

            fn equal_range_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                value: Self::Item,
                _less: GpuOp<Less>,
            ) -> Result<(usize, usize), Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = super::device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let len = $first_field.len();
                if len == 0 {
                    return Ok((0, 0));
                }
                let client = policy.client();
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
                        unsafe { BufferArg::from_raw_parts($first_field.handle.clone(), len) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.0.handle.clone(), len) },
                        )+
                        unsafe { BufferArg::from_raw_parts(first_value_handle.clone(), 1) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.1.clone(), 1) },
                        )+
                        unsafe { BufferArg::from_raw_parts(lower_flag.clone(), len) },
                    );
                    $upper_bound_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        unsafe { BufferArg::from_raw_parts($first_field.handle.clone(), len) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.0.handle.clone(), len) },
                        )+
                        unsafe { BufferArg::from_raw_parts(first_value_handle.clone(), 1) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.1.clone(), 1) },
                        )+
                        unsafe { BufferArg::from_raw_parts(upper_flag.clone(), len) },
                    );
                }
                Ok((
                    search::first_flag(policy, lower_flag, len, len)?.unwrap_or(len),
                    search::first_flag(policy, upper_flag, len, len)?.unwrap_or(len),
                ))
            }

            fn lower_bound_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                value: Self::Item,
                _less: GpuOp<Less>,
            ) -> Result<usize, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = super::device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let len = $first_field.len();
                if len == 0 {
                    return Ok(0);
                }
                let client = policy.client();
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
                        unsafe { BufferArg::from_raw_parts($first_field.handle.clone(), len) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.0.handle.clone(), len) },
                        )+
                        unsafe { BufferArg::from_raw_parts(first_value_handle.clone(), 1) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.1.clone(), 1) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                    );
                }
                Ok(search::first_flag(policy, flag_handle, len, len)?
                    .unwrap_or(len))
            }

            fn upper_bound_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                value: Self::Item,
                _less: GpuOp<Less>,
            ) -> Result<usize, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = super::device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let len = $first_field.len();
                if len == 0 {
                    return Ok(0);
                }
                let client = policy.client();
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
                        unsafe { BufferArg::from_raw_parts($first_field.handle.clone(), len) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.0.handle.clone(), len) },
                        )+
                        unsafe { BufferArg::from_raw_parts(first_value_handle.clone(), 1) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.1.clone(), 1) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                    );
                }
                Ok(search::first_flag(policy, flag_handle, len, len)?
                    .unwrap_or(len))
            }

            fn is_sorted_until_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                _less: GpuOp<Less>,
            ) -> Result<usize, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = super::device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let len = $first_field.len();
                if len <= 1 {
                    return Ok(len);
                }
                let block_count_u32 = search_block_count(len)?;
                let client = policy.client();
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
                        unsafe { BufferArg::from_raw_parts($first_field.handle.clone(), len) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.handle.clone(), len) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                    );
                }
                Ok(search::first_flag(policy, flag_handle, len, len)?
                    .unwrap_or(len))
            }

            fn is_sorted_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                less: GpuOp<Less>,
            ) -> Result<bool, Error> {
                let len = ReadOnlySoA::len(&self);
                Ok(self.is_sorted_until_input(policy, less)? == len)
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
        $lexicographical_diff_kernel:ident,
        $lexicographical_compare_at_kernel:ident
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
            Op: PredicateOp2<(
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            fn equal_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: $name<$right_first, $( $right_rest ),+>,
                op: GpuOp<Op>,
            ) -> Result<bool, Error> {
                if ReadOnlySoA::len(&self) != ReadOnlySoA::len(&other) {
                    return Ok(false);
                }
                Ok(self.mismatch_input(policy, other, op)?.is_none())
            }

            fn mismatch_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: $name<$right_first, $( $right_rest ),+>,
                _op: GpuOp<Op>,
            ) -> Result<Option<usize>, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $left_first = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                let $right_first_value = super::device_expr_collect_with_policy(policy, &other.$first_field)?;
                $(
                    let $left_value = super::device_expr_collect_with_policy(policy, &self.$field)?;
                    let $right_value = super::device_expr_collect_with_policy(policy, &other.$field)?;
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
                let client = policy.client();
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
                        unsafe { BufferArg::from_raw_parts($left_first.handle.clone(), $left_first.len()) },
                        $(
                            unsafe { BufferArg::from_raw_parts($left_value.handle.clone(), $left_value.len()) },
                        )+
                        unsafe { BufferArg::from_raw_parts($right_first_value.handle.clone(), $right_first_value.len()) },
                        $(
                            unsafe { BufferArg::from_raw_parts($right_value.handle.clone(), $right_value.len()) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), min_len) },
                    );
                }

                if let Some(index) = search::first_flag(policy, flag_handle, min_len, min_len)? {
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
                policy: &CubePolicy<Self::Runtime>,
                other: $name<$right_first, $( $right_rest ),+>,
                _op: GpuOp<Op>,
            ) -> Result<Option<usize>, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $left_first = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                let $right_first_value = super::device_expr_collect_with_policy(policy, &other.$first_field)?;
                $(
                    let $left_value = super::device_expr_collect_with_policy(policy, &self.$field)?;
                    let $right_value = super::device_expr_collect_with_policy(policy, &other.$field)?;
                )+

                if $left_first.len() == 0 || $right_first_value.len() == 0 {
                    return Ok(None);
                }

                let block_count_u32 = search_block_count($left_first.len())?;
                let client = policy.client();
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
                        unsafe { BufferArg::from_raw_parts($left_first.handle.clone(), $left_first.len()) },
                        $(
                            unsafe { BufferArg::from_raw_parts($left_value.handle.clone(), $left_value.len()) },
                        )+
                        unsafe { BufferArg::from_raw_parts($right_first_value.handle.clone(), $right_first_value.len()) },
                        $(
                            unsafe { BufferArg::from_raw_parts($right_value.handle.clone(), $right_value.len()) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), $left_first.len()) },
                    );
                }
                search::first_flag(
                    policy,
                    flag_handle,
                    $left_first.len(),
                    $left_first.len(),
                )
            }

            fn lexicographical_compare_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: $name<$right_first, $( $right_rest ),+>,
                _op: GpuOp<Op>,
            ) -> Result<bool, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $left_first = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                let $right_first_value = super::device_expr_collect_with_policy(policy, &other.$first_field)?;
                $(
                    let $left_value = super::device_expr_collect_with_policy(policy, &self.$field)?;
                    let $right_value = super::device_expr_collect_with_policy(policy, &other.$field)?;
                )+

                let min_len = $left_first.len().min($right_first_value.len());
                if min_len == 0 {
                    return Ok($left_first.len() < $right_first_value.len());
                }

                let block_count_u32 = search_block_count(min_len)?;
                let client = policy.client();
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
                        unsafe { BufferArg::from_raw_parts($left_first.handle.clone(), $left_first.len()) },
                        $(
                            unsafe { BufferArg::from_raw_parts($left_value.handle.clone(), $left_value.len()) },
                        )+
                        unsafe { BufferArg::from_raw_parts($right_first_value.handle.clone(), $right_first_value.len()) },
                        $(
                            unsafe { BufferArg::from_raw_parts($right_value.handle.clone(), $right_value.len()) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), min_len) },
                    );
                }

                let Some(index) = search::first_flag(policy, flag_handle, min_len, min_len)? else {
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
                        unsafe { BufferArg::from_raw_parts($left_first.handle.clone(), $left_first.len()) },
                        $(
                            unsafe { BufferArg::from_raw_parts($left_value.handle.clone(), $left_value.len()) },
                        )+
                        unsafe { BufferArg::from_raw_parts($right_first_value.handle.clone(), $right_first_value.len()) },
                        $(
                            unsafe { BufferArg::from_raw_parts($right_value.handle.clone(), $right_value.len()) },
                        )+
                        unsafe { BufferArg::from_raw_parts(index_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(output_handle.clone(), 1) },
                    );
                }
                Ok(scan::read_u32_scalar::<<$first as KernelColumn>::Runtime>(
                    client,
                    output_handle,
                )? != 0)
            }
        }
    };
}

impl_tuple_search!(SoAView2<A, B> { left: 0, right: 1 }, tuple2_adjacent_flags_kernel, tuple2_sorted_break_flags_kernel, tuple2_lower_bound_flags_kernel, tuple2_upper_bound_flags_kernel, tuple2_minmax_element_partials_kernel, tuple2_minmax_index_partials_kernel);
impl_tuple_search!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 }, tuple3_adjacent_flags_kernel, tuple3_sorted_break_flags_kernel, tuple3_lower_bound_flags_kernel, tuple3_upper_bound_flags_kernel, tuple3_minmax_element_partials_kernel, tuple3_minmax_index_partials_kernel);

impl_tuple_pair_search!(SoAView2<A, B; RA, RB> { left: left_a / right_a, right: left_b / right_b }, tuple2_mismatch_flags_kernel, tuple2_find_first_of_flags_kernel, tuple2_lexicographical_diff_flags_kernel, tuple2_lexicographical_compare_at_kernel);
impl_tuple_pair_search!(SoAView3<A, B, C; RA, RB, RC> { first: left_a / right_a, second: left_b / right_b, third: left_c / right_c }, tuple3_mismatch_flags_kernel, tuple3_find_first_of_flags_kernel, tuple3_lexicographical_diff_flags_kernel, tuple3_lexicographical_compare_at_kernel);
impl_tuple_search!(SoA2<A, B> { left: 0, right: 1 }, tuple2_adjacent_flags_kernel, tuple2_sorted_break_flags_kernel, tuple2_lower_bound_flags_kernel, tuple2_upper_bound_flags_kernel, tuple2_minmax_element_partials_kernel, tuple2_minmax_index_partials_kernel);
impl_tuple_search!(SoA3<A, B, C> { first: 0, second: 1, third: 2 }, tuple3_adjacent_flags_kernel, tuple3_sorted_break_flags_kernel, tuple3_lower_bound_flags_kernel, tuple3_upper_bound_flags_kernel, tuple3_minmax_element_partials_kernel, tuple3_minmax_index_partials_kernel);
impl_tuple_pair_search!(SoA2<A, B; RA, RB> { left: left_a / right_a, right: left_b / right_b }, tuple2_mismatch_flags_kernel, tuple2_find_first_of_flags_kernel, tuple2_lexicographical_diff_flags_kernel, tuple2_lexicographical_compare_at_kernel);
impl_tuple_pair_search!(SoA3<A, B, C; RA, RB, RC> { first: left_a / right_a, second: left_b / right_b, third: left_c / right_c }, tuple3_mismatch_flags_kernel, tuple3_find_first_of_flags_kernel, tuple3_lexicographical_diff_flags_kernel, tuple3_lexicographical_compare_at_kernel);

/// Finds the minimum element index according to `Less`.
pub fn min_element<Input, Less>(
    policy: &CubePolicy<<Input as MinMaxInput<Less>>::Runtime>,
    input: Input,
    _less: Less,
) -> Result<Option<usize>, Error>
where
    Input: MinMaxInput<Less>,
{
    input.min_element_input(policy, GpuOp::<Less>::new())
}

/// Finds the maximum element index according to `Less`.
pub fn max_element<Input, Less>(
    policy: &CubePolicy<<Input as MinMaxInput<Less>>::Runtime>,
    input: Input,
    _less: Less,
) -> Result<Option<usize>, Error>
where
    Input: MinMaxInput<Less>,
{
    input.max_element_input(policy, GpuOp::<Less>::new())
}

/// Finds both minimum and maximum element indices according to `Less`.
pub fn minmax_element<Input, Less>(
    policy: &CubePolicy<<Input as MinMaxInput<Less>>::Runtime>,
    input: Input,
    _less: Less,
) -> Result<Option<(usize, usize)>, Error>
where
    Input: MinMaxInput<Less>,
{
    input.minmax_element_input(policy, GpuOp::<Less>::new())
}

/// Finds the first adjacent pair that satisfies `Pred`.
pub fn adjacent_find<Input, Pred>(
    policy: &CubePolicy<<Input as AdjacentFindInput<Pred>>::Runtime>,
    input: Input,
    _pred: Pred,
) -> Result<Option<usize>, Error>
where
    Input: AdjacentFindInput<Pred>,
{
    input.adjacent_find_input(policy, GpuOp::<Pred>::new())
}

/// Returns whether two inputs are equal under `Eq`.
pub fn equal<Left, Right, Eq>(
    policy: &CubePolicy<<Left as PairSearchInput<Right, Eq>>::Runtime>,
    left: Left,
    right: Right,
    _eq: Eq,
) -> Result<bool, Error>
where
    Left: PairSearchInput<Right, Eq>,
{
    left.equal_input(policy, right, GpuOp::<Eq>::new())
}

/// Finds the first mismatch between two inputs.
pub fn mismatch<Left, Right, Eq>(
    policy: &CubePolicy<<Left as PairSearchInput<Right, Eq>>::Runtime>,
    left: Left,
    right: Right,
    _eq: Eq,
) -> Result<Option<usize>, Error>
where
    Left: PairSearchInput<Right, Eq>,
{
    left.mismatch_input(policy, right, GpuOp::<Eq>::new())
}

/// Finds the first input element equal to any value in `needles`.
pub fn find_first_of<Input, Needles, Eq>(
    policy: &CubePolicy<<Input as PairSearchInput<Needles, Eq>>::Runtime>,
    input: Input,
    needles: Needles,
    _eq: Eq,
) -> Result<Option<usize>, Error>
where
    Input: PairSearchInput<Needles, Eq>,
{
    input.find_first_of_input(policy, needles, GpuOp::<Eq>::new())
}

/// Returns the equal range for `value` in a sorted input.
pub fn equal_range<Input, Less>(
    policy: &CubePolicy<<Input as SortedSearchInput<Less>>::Runtime>,
    input: Input,
    value: <Input as SortedSearchInput<Less>>::Item,
    _less: Less,
) -> Result<(usize, usize), Error>
where
    Input: SortedSearchInput<Less>,
{
    input.equal_range_input(policy, value, GpuOp::<Less>::new())
}

/// Finds the first sorted insertion point for `value`.
pub fn lower_bound<Input, Less>(
    policy: &CubePolicy<<Input as SortedSearchInput<Less>>::Runtime>,
    input: Input,
    value: <Input as SortedSearchInput<Less>>::Item,
    _less: Less,
) -> Result<usize, Error>
where
    Input: SortedSearchInput<Less>,
{
    input.lower_bound_input(policy, value, GpuOp::<Less>::new())
}

/// Finds the last sorted insertion point for `value`.
pub fn upper_bound<Input, Less>(
    policy: &CubePolicy<<Input as SortedSearchInput<Less>>::Runtime>,
    input: Input,
    value: <Input as SortedSearchInput<Less>>::Item,
    _less: Less,
) -> Result<usize, Error>
where
    Input: SortedSearchInput<Less>,
{
    input.upper_bound_input(policy, value, GpuOp::<Less>::new())
}

/// Returns the first position where the sorted order is broken.
pub fn is_sorted_until<Input, Less>(
    policy: &CubePolicy<<Input as SortedSearchInput<Less>>::Runtime>,
    input: Input,
    _less: Less,
) -> Result<usize, Error>
where
    Input: SortedSearchInput<Less>,
{
    input.is_sorted_until_input(policy, GpuOp::<Less>::new())
}

/// Returns whether an input is sorted.
pub fn is_sorted<Input, Less>(
    policy: &CubePolicy<<Input as SortedSearchInput<Less>>::Runtime>,
    input: Input,
    _less: Less,
) -> Result<bool, Error>
where
    Input: SortedSearchInput<Less>,
{
    input.is_sorted_input(policy, GpuOp::<Less>::new())
}

/// Lexicographically compares two inputs.
pub fn lexicographical_compare<Left, Right, Less>(
    policy: &CubePolicy<<Left as PairSearchInput<Right, Less>>::Runtime>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<bool, Error>
where
    Left: PairSearchInput<Right, Less>,
{
    left.lexicographical_compare_input(policy, right, GpuOp::<Less>::new())
}
