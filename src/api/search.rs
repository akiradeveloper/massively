use crate::{
    device::{DeviceVec, KernelColumn, KernelColumnAt, S0, SoVA, SoVA1},
    error::Error,
    expr::DeviceGpuExpr,
    op::{BinaryPredicateOp, GpuOp},
    primitives::search,
};
use cubecl::prelude::*;

fn materialize_one<Source>(
    input: SoVA1<Source>,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    SoVA1<Source>: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    SoVA::validate(&input)?;
    super::device_expr_collect(&input.source)
}

fn materialize_pair<Left, Right>(
    left: SoVA1<Left>,
    right: SoVA1<Right>,
) -> Result<
    (
        DeviceVec<Left::Runtime, Left::Item>,
        DeviceVec<Left::Runtime, Left::Item>,
    ),
    Error,
>
where
    SoVA1<Left>: SoVA<Item = Left::Item, Scalar = Left::Item>,
    SoVA1<Right>: SoVA<Item = Right::Item, Scalar = Right::Item>,
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

impl<Source, Less> MinMaxInput<Less> for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    fn min_element_input(self, _less: GpuOp<Less>) -> Result<Option<usize>, Error> {
        SoVA::validate(&self)?;
        Ok(super::device_expr_minmax_element::<Source, Less>(&self.source)?.map(|(min, _)| min))
    }

    fn max_element_input(self, _less: GpuOp<Less>) -> Result<Option<usize>, Error> {
        SoVA::validate(&self)?;
        Ok(super::device_expr_minmax_element::<Source, Less>(&self.source)?.map(|(_, max)| max))
    }

    fn minmax_element_input(self, _less: GpuOp<Less>) -> Result<Option<(usize, usize)>, Error> {
        SoVA::validate(&self)?;
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
        <SoVA1<Source> as MinMaxInput<Less>>::min_element_input(SoVA1 { source: self }, less)
    }

    fn max_element_input(self, less: GpuOp<Less>) -> Result<Option<usize>, Error> {
        <SoVA1<Source> as MinMaxInput<Less>>::max_element_input(SoVA1 { source: self }, less)
    }

    fn minmax_element_input(self, less: GpuOp<Less>) -> Result<Option<(usize, usize)>, Error> {
        <SoVA1<Source> as MinMaxInput<Less>>::minmax_element_input(SoVA1 { source: self }, less)
    }
}

/// Input accepted by adjacent-pair search.
#[doc(hidden)]
pub trait AdjacentFindInput<Pred> {
    /// Finds the first adjacent pair that satisfies `Pred`.
    fn adjacent_find_input(self, pred: GpuOp<Pred>) -> Result<Option<usize>, Error>;
}

impl<Source, Pred> AdjacentFindInput<Pred> for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
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
        <SoVA1<Source> as AdjacentFindInput<Pred>>::adjacent_find_input(
            SoVA1 { source: self },
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

impl<Source, Less> SortedSearchInput<Less> for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
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
        let len = SoVA::len(&self);
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
        <SoVA1<Source> as SortedSearchInput<Less>>::binary_search_input(
            SoVA1 { source: self },
            value,
            less,
        )
    }

    fn equal_range_input(
        self,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<(usize, usize), Error> {
        <SoVA1<Source> as SortedSearchInput<Less>>::equal_range_input(
            SoVA1 { source: self },
            value,
            less,
        )
    }

    fn lower_bound_input(self, value: Self::Item, less: GpuOp<Less>) -> Result<usize, Error> {
        <SoVA1<Source> as SortedSearchInput<Less>>::lower_bound_input(
            SoVA1 { source: self },
            value,
            less,
        )
    }

    fn upper_bound_input(self, value: Self::Item, less: GpuOp<Less>) -> Result<usize, Error> {
        <SoVA1<Source> as SortedSearchInput<Less>>::upper_bound_input(
            SoVA1 { source: self },
            value,
            less,
        )
    }

    fn is_sorted_until_input(self, less: GpuOp<Less>) -> Result<usize, Error> {
        <SoVA1<Source> as SortedSearchInput<Less>>::is_sorted_until_input(
            SoVA1 { source: self },
            less,
        )
    }

    fn is_sorted_input(self, less: GpuOp<Less>) -> Result<bool, Error> {
        <SoVA1<Source> as SortedSearchInput<Less>>::is_sorted_input(SoVA1 { source: self }, less)
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

impl<Left, Right, Op> PairSearchInput<SoVA1<Right>, Op> for SoVA1<Left>
where
    Self: SoVA<Item = Left::Item, Scalar = Left::Item>,
    SoVA1<Right>: SoVA<Item = Right::Item, Scalar = Right::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: BinaryPredicateOp<Left::Item>,
{
    fn equal_input(self, other: SoVA1<Right>, _op: GpuOp<Op>) -> Result<bool, Error> {
        let (left, right) = materialize_pair(self, other)?;
        search::equal(&left, &right, GpuOp::<Op>::new())
    }

    fn mismatch_input(self, other: SoVA1<Right>, _op: GpuOp<Op>) -> Result<Option<usize>, Error> {
        let (left, right) = materialize_pair(self, other)?;
        search::mismatch(&left, &right, GpuOp::<Op>::new())
    }

    fn find_first_of_input(
        self,
        other: SoVA1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<Option<usize>, Error> {
        let (input, needles) = materialize_pair(self, other)?;
        search::find_first_of(&input, &needles, GpuOp::<Op>::new())
    }

    fn find_end_input(self, other: SoVA1<Right>, _op: GpuOp<Op>) -> Result<Option<usize>, Error> {
        let (input, pattern) = materialize_pair(self, other)?;
        search::find_end(&input, &pattern, GpuOp::<Op>::new())
    }

    fn includes_input(self, other: SoVA1<Right>, _op: GpuOp<Op>) -> Result<bool, Error> {
        let (left, right) = materialize_pair(self, other)?;
        search::includes(&left, &right, GpuOp::<Op>::new())
    }

    fn lexicographical_compare_input(
        self,
        other: SoVA1<Right>,
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
        <SoVA1<Left> as PairSearchInput<SoVA1<Right>, Op>>::equal_input(
            SoVA1 { source: self },
            SoVA1 { source: other },
            op,
        )
    }

    fn mismatch_input(self, other: Right, op: GpuOp<Op>) -> Result<Option<usize>, Error> {
        <SoVA1<Left> as PairSearchInput<SoVA1<Right>, Op>>::mismatch_input(
            SoVA1 { source: self },
            SoVA1 { source: other },
            op,
        )
    }

    fn find_first_of_input(self, other: Right, op: GpuOp<Op>) -> Result<Option<usize>, Error> {
        <SoVA1<Left> as PairSearchInput<SoVA1<Right>, Op>>::find_first_of_input(
            SoVA1 { source: self },
            SoVA1 { source: other },
            op,
        )
    }

    fn find_end_input(self, other: Right, op: GpuOp<Op>) -> Result<Option<usize>, Error> {
        <SoVA1<Left> as PairSearchInput<SoVA1<Right>, Op>>::find_end_input(
            SoVA1 { source: self },
            SoVA1 { source: other },
            op,
        )
    }

    fn includes_input(self, other: Right, op: GpuOp<Op>) -> Result<bool, Error> {
        <SoVA1<Left> as PairSearchInput<SoVA1<Right>, Op>>::includes_input(
            SoVA1 { source: self },
            SoVA1 { source: other },
            op,
        )
    }

    fn lexicographical_compare_input(self, other: Right, op: GpuOp<Op>) -> Result<bool, Error> {
        <SoVA1<Left> as PairSearchInput<SoVA1<Right>, Op>>::lexicographical_compare_input(
            SoVA1 { source: self },
            SoVA1 { source: other },
            op,
        )
    }
}

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

/// Finds the last occurrence of `pattern` inside `input`.
pub fn find_end<Input, Pattern, Eq>(
    input: Input,
    pattern: Pattern,
    _eq: Eq,
) -> Result<Option<usize>, Error>
where
    Input: PairSearchInput<Pattern, Eq>,
{
    input.find_end_input(pattern, GpuOp::<Eq>::new())
}

/// Returns whether `value` is present in a sorted input.
pub fn binary_search<Input, Less>(
    input: Input,
    value: <Input as SortedSearchInput<Less>>::Item,
    _less: Less,
) -> Result<bool, Error>
where
    Input: SortedSearchInput<Less>,
{
    input.binary_search_input(value, GpuOp::<Less>::new())
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

/// Returns whether sorted `left` includes all values from sorted `right`.
pub fn includes<Left, Right, Less>(left: Left, right: Right, _less: Less) -> Result<bool, Error>
where
    Left: PairSearchInput<Right, Less>,
{
    left.includes_input(right, GpuOp::<Less>::new())
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
