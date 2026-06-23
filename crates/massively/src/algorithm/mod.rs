//! Parallel algorithm types, operators, and free-function APIs.

pub(crate) mod api;
pub mod op;

use crate::Backend;
use crate::algorithm::api::sealed;
pub use api::{
    adjacent_difference, adjacent_find, all_of, any_of, copy_if, count_if, equal, equal_range,
    exclusive_scan, exclusive_scan_by_key, find_first_of, find_if, gather, gather_if,
    inclusive_scan, inclusive_scan_by_key, inner_product, is_partitioned, is_sorted,
    is_sorted_until, lexicographical_compare, lower_bound, max_element, merge, merge_by_key,
    min_element, minmax_element, mismatch, none_of, partition, reduce, reduce_by_key, remove_if,
    replace_if, reverse, scatter, scatter_if, set_difference, set_intersection, set_union, sort,
    sort_by_key, stable_sort, stable_sort_by_key, transform, unique, unique_by_key, upper_bound,
};
use cubecl::prelude::CubeType;

/// Single-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA1<A>(pub A);

/// Two-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA2<A, B>(pub A, pub B);

/// Three-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA3<A, B, C>(pub A, pub B, pub C);

impl<A> From<(A,)> for SoA1<A> {
    fn from(value: (A,)) -> Self {
        Self(value.0)
    }
}

impl<A, B> From<(A, B)> for SoA2<A, B> {
    fn from(value: (A, B)) -> Self {
        Self(value.0, value.1)
    }
}

impl<A, B, C> From<(A, B, C)> for SoA3<A, B, C> {
    fn from(value: (A, B, C)) -> Self {
        Self(value.0, value.1, value.2)
    }
}

/// Logical item handled by massively algorithms.
///
/// An `MItem` is one element of an [`MIter`] or [`MVec`]. The current public
/// model represents items as tuples such as `(T,)`, `(T, U)`, and `(T, U, V)`;
/// internally those tuples are stored as SoA device columns for backend `B`.
pub trait MItem<B: Backend>: sealed::MItemDispatch<B> + CubeType + Sized + 'static {
    #[doc(hidden)]
    type Inner;
}

/// Owned massively vector for a logical item.
pub trait MVec<B: Backend>: Sized {
    type Item: MItem<B>;

    #[doc(hidden)]
    fn from_inner(inner: <Self::Item as MItem<B>>::Inner) -> Self;

    /// Returns the logical length.
    fn len(&self) -> usize;

    /// Returns whether this array has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Massively iterator.
pub trait MIter<B: Backend>: sealed::MIterDispatch<B> + Sized {
    type Item: MItem<B>;

    #[doc(hidden)]
    type Inner;

    #[doc(hidden)]
    fn into_inner(self) -> Self::Inner;

    /// Returns the logical length.
    fn len(&self) -> usize;

    /// Returns whether this slice has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
