//! Public API for `massively`.
//!
//! This crate intentionally keeps CubeCL runtime types out of public algorithm
//! signatures. The implementation delegates to the internal detail layer.

use std::any::Any;
use std::marker::PhantomData;

pub use crate::detail::Error;
pub use crate::detail::op;

mod sealed {
    use super::{Error, MIter, MVec, op};

    pub trait Backend {
        type Runtime: cubecl::prelude::Runtime;
    }

    pub trait Value: cubecl::prelude::CubeType {}
    impl<T> Value for T where T: cubecl::prelude::CubeType {}

    pub trait Scalar:
        Value + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement
    {
    }
    impl<T> Scalar for T where T: cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement {}

    pub trait MIterDispatch<B: super::Backend>: Sized {
        fn index_inner(
            &self,
        ) -> Option<(&crate::detail::DeviceVec<<B as Backend>::Runtime, u32>,)> {
            None
        }

        fn column_inner<T: 'static>(
            &self,
        ) -> Option<&crate::detail::DeviceVec<<B as Backend>::Runtime, T>> {
            None
        }

        fn transform_dispatch<Op, Output, Y>(self, op: Op) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::UnaryOp<<Self as MIter<B>>::Item, Output = Y>,
            Y: super::StorageOutput<B>,
            Output: MVec<B, Item = Y>;

        fn reverse_dispatch<Output>(self) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn sort_dispatch<Less, Output>(self, less: Less) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
            self,
            keys: &crate::detail::DeviceVec<<B as Backend>::Runtime, K>,
            _less: Less,
        ) -> Result<(KeyOutput, ValueOutput), Error>
        where
            Self: MIter<B>,
            K: super::Scalar<B> + 'static,
            Less: op::BinaryPredicateOp<(K,)>,
            KeyOutput: MVec<B, Item = (K,)>,
            ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
            self,
            keys: &crate::detail::DeviceVec<<B as Backend>::Runtime, K>,
            _eq: Eq,
        ) -> Result<(KeyOutput, ValueOutput), Error>
        where
            Self: MIter<B>,
            K: super::Scalar<B> + 'static,
            Eq: op::BinaryPredicateOp<(K,)>,
            KeyOutput: MVec<B, Item = (K,)>,
            ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
            self,
            keys: &crate::detail::DeviceVec<<B as Backend>::Runtime, K>,
            key_eq: KeyEq,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            K: super::Scalar<B> + 'static,
            KeyEq: op::BinaryPredicateOp<(K,)>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn exclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
            self,
            keys: &crate::detail::DeviceVec<<B as Backend>::Runtime, K>,
            key_eq: KeyEq,
            _init: <Self as MIter<B>>::Item,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            K: super::Scalar<B> + 'static,
            KeyEq: op::BinaryPredicateOp<(K,)>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn reduce_by_single_key_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
            self,
            keys: &crate::detail::DeviceVec<<B as Backend>::Runtime, K>,
            key_eq: KeyEq,
            _init: <Self as MIter<B>>::Item,
            op: Op,
        ) -> Result<(KeyOutput, ValueOutput), Error>
        where
            Self: MIter<B>,
            K: super::Scalar<B> + 'static,
            KeyEq: op::BinaryPredicateOp<(K,)>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            KeyOutput: MVec<B, Item = (K,)>,
            ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn merge_by_single_key_same_dispatch<K, Less, KeyOutput, ValueOutput>(
            self,
            left_keys: &crate::detail::DeviceVec<<B as Backend>::Runtime, K>,
            right_keys: &crate::detail::DeviceVec<<B as Backend>::Runtime, K>,
            _right_values: Self,
            _less: Less,
        ) -> Result<(KeyOutput, ValueOutput), Error>
        where
            Self: MIter<B>,
            K: super::Scalar<B> + 'static,
            Less: op::BinaryPredicateOp<(K,)>,
            KeyOutput: MVec<B, Item = (K,)>,
            ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
        {
            let _ = (left_keys, right_keys);
            Err(Error::Launch {
                message: "merge_by_key is not supported for this iterator shape".to_string(),
            })
        }

        fn gather_dispatch<Indices, Output>(self, indices: Indices) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Indices: MIter<B, Item = (u32,)>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn gather_if_dispatch<Indices, Stencil, Pred, Output>(
            self,
            _indices: Indices,
            _default: <Self as MIter<B>>::Item,
            _stencil: Stencil,
            _pred: Pred,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Indices: MIter<B, Item = (u32,)>,
            Stencil: MIter<B>,
            Pred: op::PredicateOp<Stencil::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "gather_if is not supported for this iterator shape".to_string(),
            })
        }

        fn scatter_dispatch<Indices, Output>(
            self,
            _indices: Indices,
            _len: usize,
            _default: <Self as MIter<B>>::Item,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Indices: MIter<B, Item = (u32,)>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "scatter is not supported for this iterator shape".to_string(),
            })
        }

        fn scatter_if_dispatch<Indices, Stencil, Pred, Output>(
            self,
            _indices: Indices,
            _len: usize,
            _default: <Self as MIter<B>>::Item,
            _stencil: Stencil,
            _pred: Pred,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Indices: MIter<B, Item = (u32,)>,
            Stencil: MIter<B>,
            Pred: op::PredicateOp<Stencil::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "scatter_if is not supported for this iterator shape".to_string(),
            })
        }

        fn reduce_dispatch<Op>(
            self,
            _init: <Self as MIter<B>>::Item,
            op: Op,
        ) -> Result<<Self as MIter<B>>::Item, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>;

        fn inclusive_scan_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn exclusive_scan_dispatch<Op, Output>(
            self,
            _init: <Self as MIter<B>>::Item,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn adjacent_difference_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn copy_if_dispatch<Stencil, Pred, Output>(
            self,
            _stencil: Stencil,
            _pred: Pred,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Stencil: MIter<B>,
            Pred: op::PredicateOp<Stencil::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn remove_if_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn count_if_dispatch<Pred>(self, pred: Pred) -> Result<usize, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn all_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn any_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn none_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn find_if_dispatch<Pred>(self, pred: Pred) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn partition_dispatch<Pred, Output>(self, pred: Pred) -> Result<(Output, Output), Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn is_partitioned_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn replace_if_dispatch<Stencil, Pred, Output>(
            self,
            replacement: <Self as MIter<B>>::Item,
            _stencil: Stencil,
            _pred: Pred,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Stencil: MIter<B>,
            Pred: op::PredicateOp<Stencil::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        #[doc(hidden)]
        fn selection_stencil_dispatch<Pred>(
            &self,
            _invert: bool,
        ) -> Result<crate::detail::api::PrecomputedSelection<<B as Backend>::Runtime>, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "stencil is not supported for this iterator shape".to_string(),
            })
        }

        fn unique_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn min_element_dispatch<Less>(self, less: Less) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn max_element_dispatch<Less>(self, less: Less) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn minmax_element_dispatch<Less>(self, less: Less) -> Result<Option<(usize, usize)>, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn adjacent_find_dispatch<Pred>(self, pred: Pred) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn equal_dispatch<Right, Eq>(self, _right: Right, _eq: Eq) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "equal is not supported for this iterator shape".to_string(),
            })
        }

        fn mismatch_dispatch<Right, Eq>(
            self,
            _right: Right,
            _eq: Eq,
        ) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "mismatch is not supported for this iterator shape".to_string(),
            })
        }

        fn find_first_of_dispatch<Needles, Eq>(
            self,
            _needles: Needles,
            _eq: Eq,
        ) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Needles: MIter<B, Item = <Self as MIter<B>>::Item>,
            Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "find_first_of is not supported for this iterator shape".to_string(),
            })
        }

        fn lower_bound_dispatch<Less>(
            self,
            value: <Self as MIter<B>>::Item,
            _less: Less,
        ) -> Result<usize, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn upper_bound_dispatch<Less>(
            self,
            value: <Self as MIter<B>>::Item,
            _less: Less,
        ) -> Result<usize, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn equal_range_dispatch<Less>(
            self,
            value: <Self as MIter<B>>::Item,
            _less: Less,
        ) -> Result<(usize, usize), Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn is_sorted_until_dispatch<Less>(self, less: Less) -> Result<usize, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn is_sorted_dispatch<Less>(self, less: Less) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn lexicographical_compare_dispatch<Right, Less>(
            self,
            _right: Right,
            _less: Less,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "lexicographical_compare is not supported for this iterator shape"
                    .to_string(),
            })
        }

        fn merge_dispatch<Right, Output, Less>(
            self,
            _right: Right,
            _less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "merge is not supported for this iterator shape".to_string(),
            })
        }

        fn set_union_dispatch<Right, Output, Less>(
            self,
            _right: Right,
            _less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "set_union is not supported for this iterator shape".to_string(),
            })
        }

        fn set_intersection_dispatch<Right, Output, Less>(
            self,
            _right: Right,
            _less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "set_intersection is not supported for this iterator shape".to_string(),
            })
        }

        fn set_difference_dispatch<Right, Output, Less>(
            self,
            _right: Right,
            _less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "set_difference is not supported for this iterator shape".to_string(),
            })
        }

        fn inner_product_dispatch<Right, TransformOp, ReduceOp>(
            self,
            _right: Right,
            _transform_op: TransformOp,
            _init: <Self as MIter<B>>::Item,
            _reduce_op: ReduceOp,
        ) -> Result<<Self as MIter<B>>::Item, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            TransformOp: op::BinaryOp<<Self as MIter<B>>::Item>,
            ReduceOp: op::BinaryOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "inner_product is not supported for this iterator shape".to_string(),
            })
        }

        fn equal_same_dispatch<Eq>(self, _right: Self, _eq: Eq) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "equal is not supported for this iterator shape".to_string(),
            })
        }

        fn mismatch_same_dispatch<Eq>(self, _right: Self, _eq: Eq) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "mismatch is not supported for this iterator shape".to_string(),
            })
        }

        fn find_first_of_same_dispatch<Eq>(
            self,
            _needles: Self,
            _eq: Eq,
        ) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "find_first_of is not supported for this iterator shape".to_string(),
            })
        }

        fn lexicographical_compare_same_dispatch<Less>(
            self,
            _right: Self,
            _less: Less,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "lexicographical_compare is not supported for this iterator shape"
                    .to_string(),
            })
        }

        fn merge_same_dispatch<Output, Less>(
            self,
            _right: Self,
            _less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "merge is not supported for this iterator shape".to_string(),
            })
        }

        fn set_union_same_dispatch<Output, Less>(
            self,
            _right: Self,
            _less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "set_union is not supported for this iterator shape".to_string(),
            })
        }

        fn set_intersection_same_dispatch<Output, Less>(
            self,
            _right: Self,
            _less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "set_intersection is not supported for this iterator shape".to_string(),
            })
        }

        fn set_difference_same_dispatch<Output, Less>(
            self,
            _right: Self,
            _less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "set_difference is not supported for this iterator shape".to_string(),
            })
        }

        fn inner_product_same_dispatch<TransformOp, ReduceOp>(
            self,
            _right: Self,
            _transform_op: TransformOp,
            _init: <Self as MIter<B>>::Item,
            _reduce_op: ReduceOp,
        ) -> Result<<Self as MIter<B>>::Item, Error>
        where
            Self: MIter<B>,
            TransformOp: op::BinaryOp<<Self as MIter<B>>::Item>,
            ReduceOp: op::BinaryOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "inner_product is not supported for this iterator shape".to_string(),
            })
        }
    }

    pub trait StorageOutputDispatch<B: super::Backend>: Sized {
        fn transform_unary<Input, Op>(
            input: &crate::detail::DeviceVec<<B as Backend>::Runtime, Input>,
            op: Op,
        ) -> Result<<Self as super::StorageOutput<B>>::Inner, Error>
        where
            Self: super::StorageOutput<B>,
            Input: super::Scalar<B>,
            Op: op::UnaryOp<(Input,), Output = Self>;

        fn transform_binary<Left, Right, Op>(
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            left: &crate::detail::DeviceVec<<B as Backend>::Runtime, Left>,
            right: &crate::detail::DeviceVec<<B as Backend>::Runtime, Right>,
            op: Op,
        ) -> Result<<Self as super::StorageOutput<B>>::Inner, Error>
        where
            Self: super::StorageOutput<B>,
            Left: super::Scalar<B>,
            Right: super::Scalar<B>,
            Op: op::UnaryOp<(Left, Right), Output = Self>;

        fn transform_ternary<First, Second, Third, Op>(
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            first: &crate::detail::DeviceVec<<B as Backend>::Runtime, First>,
            second: &crate::detail::DeviceVec<<B as Backend>::Runtime, Second>,
            third: &crate::detail::DeviceVec<<B as Backend>::Runtime, Third>,
            op: Op,
        ) -> Result<<Self as super::StorageOutput<B>>::Inner, Error>
        where
            Self: super::StorageOutput<B>,
            First: super::Scalar<B>,
            Second: super::Scalar<B>,
            Third: super::Scalar<B>,
            Op: op::UnaryOp<(First, Second, Third), Output = Self>;
    }
}

/// Execution backend marker.
///
/// Backend implementations hide the CubeCL runtime type used by the lower
/// implementation layer.
pub trait Backend: sealed::Backend + Copy + Clone + Default + 'static {}

/// Value that can appear as one logical device item.
pub trait Value<B: Backend>: sealed::Value {}
impl<B, T> Value<B> for T
where
    B: Backend,
    T: sealed::Value,
{
}

/// Scalar value that can be stored in one device column.
pub trait Scalar<B: Backend>: Value<B> + sealed::Scalar {}
impl<B, T> Scalar<B> for T
where
    B: Backend,
    T: Value<B> + sealed::Scalar,
{
}

/// WGPU backend marker.
#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, Debug, Default)]
pub struct Wgpu;

#[cfg(feature = "wgpu")]
impl sealed::Backend for Wgpu {
    type Runtime = cubecl::wgpu::WgpuRuntime;
}

#[cfg(feature = "wgpu")]
impl Backend for Wgpu {}

/// Compatibility alias for the default WGPU execution policy.
#[cfg(feature = "wgpu")]
pub type CubeWgpu = Policy<Wgpu>;

/// Execution policy for a facade backend.
#[derive(Debug)]
pub struct Policy<B: Backend> {
    inner: crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> Clone for Policy<B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Policy<B> {
    fn from_inner(inner: crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>) -> Self {
        Self {
            inner,
            _backend: PhantomData,
        }
    }

    /// Copies host data to device-resident storage.
    pub fn to_device<T>(&self, input: &[T]) -> Result<DeviceVec<B, T>, Error>
    where
        T: Scalar<B>,
    {
        Ok(DeviceVec::from_inner(self.inner.to_device(input)?))
    }

    /// Waits until all previously submitted work for this policy is complete.
    pub fn sync(&self) -> Result<(), Error> {
        futures_lite::future::block_on(self.inner.client().sync()).map_err(|err| Error::Launch {
            message: err.to_string(),
        })
    }
}

#[cfg(feature = "wgpu")]
impl Policy<Wgpu> {
    /// Creates a WGPU policy backed by the default device.
    pub fn new() -> Self {
        Self::from_inner(crate::detail::CubeWgpu::new())
    }

    /// Creates a WGPU policy backed by the CPU adapter.
    pub fn cpu() -> Self {
        Self::from_inner(crate::detail::CubeWgpu::cpu())
    }
}

#[cfg(feature = "wgpu")]
impl Default for Policy<Wgpu> {
    fn default() -> Self {
        Self::new()
    }
}

/// Owned device column.
#[derive(Debug)]
pub struct DeviceVec<B: Backend, T> {
    inner: crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>,
    _backend: PhantomData<fn() -> B>,
}

impl<B, T> DeviceVec<B, T>
where
    B: Backend,
{
    fn from_inner(inner: crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>) -> Self {
        Self {
            inner,
            _backend: PhantomData,
        }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether this column is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Copies this device column to host memory.
    pub fn to_vec(&self) -> Result<Vec<T>, Error>
    where
        T: Scalar<B>,
    {
        self.inner.to_vec()
    }
}

/// Storage materialization for owned outputs.
pub trait StorageOutput<B: Backend>: sealed::StorageOutputDispatch<B> + Value<B> + Sized {
    #[doc(hidden)]
    type Inner;
}

/// Owned massively vector for a logical item.
pub trait MVec<B: Backend>: Sized {
    type Item: StorageOutput<B>;

    #[doc(hidden)]
    fn from_inner(inner: <Self::Item as StorageOutput<B>>::Inner) -> Self;

    /// Returns the logical length.
    fn len(&self) -> usize;

    /// Returns whether this array has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn array_from_inner<B, Item, Output>(inner: <Item as StorageOutput<B>>::Inner) -> Output
where
    B: Backend,
    Item: StorageOutput<B>,
    Output: MVec<B, Item = Item>,
{
    <Output as MVec<B>>::from_inner(inner)
}

fn gather_index_inner<B, Indices>(
    indices: &Indices,
) -> Result<(&crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, u32>,), Error>
where
    B: Backend,
    Indices: MIter<B, Item = (u32,)>,
{
    <Indices as sealed::MIterDispatch<B>>::index_inner(indices).ok_or_else(|| Error::Launch {
        message: "gather indices must be backed by one u32 DeviceVec".to_string(),
    })
}

#[allow(dead_code)]
trait MergeBySingleKeyValues<B, RightValues, K, Less>: MIter<B>
where
    B: Backend,
    Self::Item: StorageOutput<B>,
{
    fn merge_by_single_key_values(
        self,
        left_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        right_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        right_values: RightValues,
    ) -> Result<
        (
            (crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,),
            <Self::Item as StorageOutput<B>>::Inner,
        ),
        Error,
    >;
}

trait PairSearchValues<B, Right, Op>: MIter<B>
where
    B: Backend,
{
    fn equal_values(self, right: Right, op: Op) -> Result<bool, Error>;
    fn mismatch_values(self, right: Right, op: Op) -> Result<Option<usize>, Error>;
    fn find_first_of_values(self, needles: Right, op: Op) -> Result<Option<usize>, Error>;
    fn lexicographical_compare_values(self, right: Right, op: Op) -> Result<bool, Error>;
}

trait PairOrderingValues<B, Right, Less>: MIter<B>
where
    B: Backend,
    Self::Item: StorageOutput<B>,
{
    fn merge_values(
        self,
        right: Right,
        less: Less,
    ) -> Result<<Self::Item as StorageOutput<B>>::Inner, Error>;
    fn set_union_values(
        self,
        right: Right,
        less: Less,
    ) -> Result<<Self::Item as StorageOutput<B>>::Inner, Error>;
    fn set_intersection_values(
        self,
        right: Right,
        less: Less,
    ) -> Result<<Self::Item as StorageOutput<B>>::Inner, Error>;
    fn set_difference_values(
        self,
        right: Right,
        less: Less,
    ) -> Result<<Self::Item as StorageOutput<B>>::Inner, Error>;
}

#[doc(hidden)]
pub struct NoPredicate;

#[cubecl::cube]
impl op::PredicateOp<(u32,)> for NoPredicate {
    fn apply(_input: (u32,)) -> bool {
        true
    }
}

#[allow(dead_code)]
trait InnerProductValues<B, Right, TransformOp, ReduceOp>: MIter<B>
where
    B: Backend,
{
    fn inner_product_values(
        self,
        right: Right,
        transform_op: TransformOp,
        init: Self::Item,
        reduce_op: ReduceOp,
    ) -> Result<Self::Item, Error>;
}

macro_rules! impl_storage_output_tuple {
    ($( $ty:ident : $var:ident ),+) => {
        impl<B, $( $ty ),+> StorageOutput<B> for ($( $ty, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
        {
            type Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+);
        }

        impl<B, $( $ty ),+> MVec<B> for ($( DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
        {
            type Item = ($( $ty, )+);

            fn from_inner(inner: <Self::Item as StorageOutput<B>>::Inner) -> Self {
                let ($( $var, )+) = inner;
                ($( DeviceVec::from_inner($var), )+)
            }

            fn len(&self) -> usize {
                self.0.len()
            }
        }

        impl<B, $( $ty ),+> sealed::StorageOutputDispatch<B> for ($( $ty, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
        {
            fn transform_unary<Input, Op>(
                input: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, Input>,
                op: Op,
            ) -> Result<<Self as StorageOutput<B>>::Inner, Error>
            where
                Input: Scalar<B>,
                Op: op::UnaryOp<(Input,), Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    <B as sealed::Backend>::Runtime,
                    Input,
                    Op,
                >,
                <Self as crate::detail::StorageOutput<
                    <B as sealed::Backend>::Runtime,
                >>::Storage: crate::detail::MaterializeOutput<
                    Output = ($(
                        crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformUnaryOutput<
                        <B as sealed::Backend>::Runtime,
                        Input,
                        Op,
                    >>::run(input)?;
                crate::detail::MaterializeOutput::materialize_output(storage)
            }

            fn transform_binary<Left, Right, Op>(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                left: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, Left>,
                right: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, Right>,
                op: Op,
            ) -> Result<<Self as StorageOutput<B>>::Inner, Error>
            where
                Left: Scalar<B>,
                Right: Scalar<B>,
                Op: op::UnaryOp<(Left, Right), Output = Self>,
                Self: crate::detail::TransformSoA2Output<
                    <B as sealed::Backend>::Runtime,
                    Left,
                    Right,
                    Op,
                >,
                <Self as crate::detail::StorageOutput<
                    <B as sealed::Backend>::Runtime,
                >>::Storage: crate::detail::MaterializeOutput<
                    Output = ($(
                        crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformSoA2Output<
                        <B as sealed::Backend>::Runtime,
                        Left,
                        Right,
                        Op,
                    >>::run(policy, left, right)?;
                crate::detail::MaterializeOutput::materialize_output(storage)
            }

            fn transform_ternary<First, Second, Third, Op>(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                first: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, First>,
                second: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, Second>,
                third: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, Third>,
                op: Op,
            ) -> Result<<Self as StorageOutput<B>>::Inner, Error>
            where
                First: Scalar<B>,
                Second: Scalar<B>,
                Third: Scalar<B>,
                Op: op::UnaryOp<(First, Second, Third), Output = Self>,
                Self: crate::detail::TransformSoA3Output<
                    <B as sealed::Backend>::Runtime,
                    First,
                    Second,
                    Third,
                    Op,
                >,
                <Self as crate::detail::StorageOutput<
                    <B as sealed::Backend>::Runtime,
                >>::Storage: crate::detail::MaterializeOutput<
                    Output = ($(
                        crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformSoA3Output<
                        <B as sealed::Backend>::Runtime,
                        First,
                        Second,
                        Third,
                        Op,
                    >>::run(
                        policy,
                        first,
                        second,
                        third,
                    )?;
                crate::detail::MaterializeOutput::materialize_output(storage)
            }
        }
    };
}

impl_storage_output_tuple!(A: a);
impl_storage_output_tuple!(A: a, B0: b);
impl_storage_output_tuple!(A: a, B0: b, C: c);

/// Massively iterator.
pub trait MIter<B: Backend>: sealed::MIterDispatch<B> + Sized {
    type Item: StorageOutput<B>;

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

impl<'a, B, T> MIter<B> for (&'a DeviceVec<B, T>,)
where
    B: Backend,
    T: Scalar<B> + 'static,
    (T,): StorageOutput<B, Inner = (crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>,)>,
{
    type Item = (T,);
    type Inner = (&'a crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>,);

    fn len(&self) -> usize {
        self.0.inner.len()
    }

    fn into_inner(self) -> Self::Inner {
        (&self.0.inner,)
    }
}

impl<'a, B, T> sealed::MIterDispatch<B> for (&'a DeviceVec<B, T>,)
where
    B: Backend,
    T: Scalar<B> + 'static,
    (T,): StorageOutput<B, Inner = (crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>,)>,
{
    fn index_inner(
        &self,
    ) -> Option<(&crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, u32>,)> {
        let column = self.0 as &dyn Any;
        let column = column.downcast_ref::<DeviceVec<B, u32>>()?;
        Some((&column.inner,))
    }

    fn column_inner<U: 'static>(
        &self,
    ) -> Option<&crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, U>> {
        let column = self.0 as &dyn Any;
        let column = column.downcast_ref::<DeviceVec<B, U>>()?;
        Some(&column.inner)
    }

    fn selection_stencil_dispatch<Pred>(
        &self,
        invert: bool,
    ) -> Result<crate::detail::api::PrecomputedSelection<<B as sealed::Backend>::Runtime>, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        let stencil = self.into_inner();
        crate::detail::api::PrecomputedSelection::from_stencil::<_, Pred>(&stencil, invert)
    }

    fn transform_dispatch<Op, Output, Y>(self, op: Op) -> Result<Output, Error>
    where
        Op: op::UnaryOp<<Self as MIter<B>>::Item, Output = Y>,
        Y: StorageOutput<B>,
        Output: MVec<B, Item = Y>,
    {
        let input = self.into_inner();
        let inner = <Y as sealed::StorageOutputDispatch<B>>::transform_unary(input.0, op)?;
        Ok(array_from_inner::<B, Y, Output>(inner))
    }

    fn reverse_dispatch<Output>(self) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::reverse(self.into_inner())?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn sort_dispatch<Less, Output>(self, _less: Less) -> Result<Output, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::sort(self.into_inner(), _less)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
        self,
        keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K: Scalar<B> + 'static,
        Less: op::BinaryPredicateOp<(K,)>,
        KeyOutput: MVec<B, Item = (K,)>,
        ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let (key_inner, value_inner) =
            crate::detail::sort_by_key((keys,), self.into_inner(), less)?;
        Ok((
            array_from_inner::<B, (K,), KeyOutput>(key_inner),
            array_from_inner::<B, (T,), ValueOutput>(value_inner),
        ))
    }

    fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
        self,
        keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K: Scalar<B> + 'static,
        Eq: op::BinaryPredicateOp<(K,)>,
        KeyOutput: MVec<B, Item = (K,)>,
        ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let (key_inner, value_inner) =
            crate::detail::unique_by_key((keys,), self.into_inner(), eq)?;
        Ok((
            array_from_inner::<B, (K,), KeyOutput>(key_inner),
            array_from_inner::<B, (T,), ValueOutput>(value_inner),
        ))
    }

    fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
        self,
        keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        key_eq: KeyEq,
        op: Op,
    ) -> Result<Output, Error>
    where
        K: Scalar<B> + 'static,
        KeyEq: op::BinaryPredicateOp<(K,)>,
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::inclusive_scan_by_key((keys,), self.into_inner(), key_eq, op)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn exclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
        self,
        keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        key_eq: KeyEq,
        init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        K: Scalar<B> + 'static,
        KeyEq: op::BinaryPredicateOp<(K,)>,
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner =
            crate::detail::exclusive_scan_by_key((keys,), self.into_inner(), key_eq, init, op)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn reduce_by_single_key_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        key_eq: KeyEq,
        init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K: Scalar<B> + 'static,
        KeyEq: op::BinaryPredicateOp<(K,)>,
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        KeyOutput: MVec<B, Item = (K,)>,
        ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let (key_inner, value_inner) =
            crate::detail::reduce_by_key((keys,), self.into_inner(), key_eq, init, op)?;
        Ok((
            array_from_inner::<B, (K,), KeyOutput>(key_inner),
            array_from_inner::<B, (T,), ValueOutput>(value_inner),
        ))
    }

    fn merge_by_single_key_same_dispatch<K, Less, KeyOutput, ValueOutput>(
        self,
        left_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        right_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        right_values: Self,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K: Scalar<B> + 'static,
        Less: op::BinaryPredicateOp<(K,)>,
        KeyOutput: MVec<B, Item = (K,)>,
        ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::merge_by_key(
            left_keys,
            &self.0.inner,
            right_keys,
            &right_values.0.inner,
            crate::detail::api::Tuple1Less::<Less>::default(),
        )?;
        Ok((
            array_from_inner::<B, (K,), KeyOutput>(key_inner),
            array_from_inner::<B, (T,), ValueOutput>(value_inner),
        ))
    }

    fn gather_dispatch<Indices, Output>(self, indices: Indices) -> Result<Output, Error>
    where
        Indices: MIter<B, Item = (u32,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let indices = gather_index_inner::<B, Indices>(&indices)?;
        let inner = crate::detail::gather(self.into_inner(), indices)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn gather_if_dispatch<Indices, Stencil, Pred, Output>(
        self,
        indices: Indices,
        default: <Self as MIter<B>>::Item,
        stencil: Stencil,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Indices: MIter<B, Item = (u32,)>,
        Stencil: MIter<B>,
        Pred: op::PredicateOp<Stencil::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let indices = gather_index_inner::<B, Indices>(&indices)?;
        let stencil = <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<Pred>(
            &stencil, false,
        )?;
        let inner = crate::detail::gather_if(self.into_inner(), indices, stencil, default, pred)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn scatter_dispatch<Indices, Output>(
        self,
        indices: Indices,
        len: usize,
        default: <Self as MIter<B>>::Item,
    ) -> Result<Output, Error>
    where
        Indices: MIter<B, Item = (u32,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let indices = gather_index_inner::<B, Indices>(&indices)?;
        let inner = crate::detail::scatter(self.into_inner(), indices, len, default.0)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn scatter_if_dispatch<Indices, Stencil, Pred, Output>(
        self,
        indices: Indices,
        len: usize,
        default: <Self as MIter<B>>::Item,
        stencil: Stencil,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Indices: MIter<B, Item = (u32,)>,
        Stencil: MIter<B>,
        Pred: op::PredicateOp<Stencil::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let indices = gather_index_inner::<B, Indices>(&indices)?;
        let stencil = <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<Pred>(
            &stencil, false,
        )?;
        let inner =
            crate::detail::scatter_if(self.into_inner(), indices, len, default.0, stencil, pred)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn reduce_dispatch<Op>(
        self,
        init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<<Self as MIter<B>>::Item, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::reduce(self.into_inner(), init, op)
    }

    fn inclusive_scan_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::inclusive_scan(self.into_inner(), op)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn exclusive_scan_dispatch<Op, Output>(
        self,
        init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::exclusive_scan(self.into_inner(), init, op)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn adjacent_difference_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::adjacent_difference(self.into_inner(), op)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn copy_if_dispatch<Stencil, Pred, Output>(
        self,
        stencil: Stencil,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Stencil: MIter<B>,
        Pred: op::PredicateOp<Stencil::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let stencil = <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<Pred>(
            &stencil, false,
        )?;
        let inner = crate::detail::copy_if(self.into_inner(), stencil, pred)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn remove_if_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::remove_if(self.into_inner(), pred)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn count_if_dispatch<Pred>(self, pred: Pred) -> Result<usize, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::count_if(self.into_inner(), pred)
    }

    fn all_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::all_of(self.into_inner(), pred)
    }

    fn any_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::any_of(self.into_inner(), pred)
    }

    fn none_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::none_of(self.into_inner(), pred)
    }

    fn find_if_dispatch<Pred>(self, pred: Pred) -> Result<Option<usize>, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::find_if(self.into_inner(), pred)
    }

    fn partition_dispatch<Pred, Output>(self, pred: Pred) -> Result<(Output, Output), Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let (matching, failing) = crate::detail::partition(self.into_inner(), pred)?;
        Ok((
            array_from_inner::<B, (T,), Output>(matching),
            array_from_inner::<B, (T,), Output>(failing),
        ))
    }

    fn is_partitioned_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::is_partitioned(self.into_inner(), pred)
    }

    fn replace_if_dispatch<Stencil, Pred, Output>(
        self,
        replacement: <Self as MIter<B>>::Item,
        stencil: Stencil,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Stencil: MIter<B>,
        Pred: op::PredicateOp<Stencil::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let stencil = <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<Pred>(
            &stencil, false,
        )?;
        let inner = crate::detail::replace_if(self.into_inner(), replacement, stencil, pred)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn unique_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
    where
        Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::unique(self.into_inner(), pred)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn min_element_dispatch<Less>(self, less: Less) -> Result<Option<usize>, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::min_element(self.into_inner(), less)
    }

    fn max_element_dispatch<Less>(self, less: Less) -> Result<Option<usize>, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::max_element(self.into_inner(), less)
    }

    fn minmax_element_dispatch<Less>(self, less: Less) -> Result<Option<(usize, usize)>, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::minmax_element(self.into_inner(), less)
    }

    fn adjacent_find_dispatch<Pred>(self, pred: Pred) -> Result<Option<usize>, Error>
    where
        Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::adjacent_find(self.into_inner(), pred)
    }

    fn equal_dispatch<Right, Eq>(self, right: Right, eq: Eq) -> Result<bool, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        let right =
            <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
                Error::Launch {
                    message: "equal right input must be backed by one DeviceVec".to_string(),
                }
            })?;
        crate::detail::equal(self.into_inner(), (right,), eq)
    }

    fn mismatch_dispatch<Right, Eq>(self, right: Right, eq: Eq) -> Result<Option<usize>, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        let right =
            <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
                Error::Launch {
                    message: "mismatch right input must be backed by one DeviceVec".to_string(),
                }
            })?;
        crate::detail::mismatch(self.into_inner(), (right,), eq)
    }

    fn find_first_of_dispatch<Needles, Eq>(
        self,
        needles: Needles,
        eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Needles: MIter<B, Item = <Self as MIter<B>>::Item>,
        Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        let needles = <Needles as sealed::MIterDispatch<B>>::column_inner::<T>(&needles)
            .ok_or_else(|| Error::Launch {
                message: "find_first_of needles must be backed by one DeviceVec".to_string(),
            })?;
        crate::detail::find_first_of(self.into_inner(), (needles,), eq)
    }

    fn lower_bound_dispatch<Less>(
        self,
        value: <Self as MIter<B>>::Item,
        less: Less,
    ) -> Result<usize, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::lower_bound(self.into_inner(), value, less)
    }

    fn upper_bound_dispatch<Less>(
        self,
        value: <Self as MIter<B>>::Item,
        less: Less,
    ) -> Result<usize, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::upper_bound(self.into_inner(), value, less)
    }

    fn equal_range_dispatch<Less>(
        self,
        value: <Self as MIter<B>>::Item,
        less: Less,
    ) -> Result<(usize, usize), Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::equal_range(self.into_inner(), value, less)
    }

    fn is_sorted_until_dispatch<Less>(self, less: Less) -> Result<usize, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::is_sorted_until(self.into_inner(), less)
    }

    fn is_sorted_dispatch<Less>(self, less: Less) -> Result<bool, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::is_sorted(self.into_inner(), less)
    }

    fn lexicographical_compare_dispatch<Right, Less>(
        self,
        right: Right,
        less: Less,
    ) -> Result<bool, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        let right =
            <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
                Error::Launch {
                    message: "lexicographical_compare right input must be backed by one DeviceVec"
                        .to_string(),
                }
            })?;
        crate::detail::lexicographical_compare(self.into_inner(), (right,), less)
    }

    fn merge_dispatch<Right, Output, Less>(self, right: Right, less: Less) -> Result<Output, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        let right =
            <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
                Error::Launch {
                    message: "merge right input must be backed by one DeviceVec".to_string(),
                }
            })?;
        let inner = crate::detail::merge(self.into_inner(), (right,), less)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn set_union_dispatch<Right, Output, Less>(
        self,
        right: Right,
        less: Less,
    ) -> Result<Output, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        let right =
            <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
                Error::Launch {
                    message: "set_union right input must be backed by one DeviceVec".to_string(),
                }
            })?;
        let inner = crate::detail::set_union(self.into_inner(), (right,), less)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn set_intersection_dispatch<Right, Output, Less>(
        self,
        right: Right,
        less: Less,
    ) -> Result<Output, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        let right =
            <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
                Error::Launch {
                    message: "set_intersection right input must be backed by one DeviceVec"
                        .to_string(),
                }
            })?;
        let inner = crate::detail::set_intersection(self.into_inner(), (right,), less)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn set_difference_dispatch<Right, Output, Less>(
        self,
        right: Right,
        less: Less,
    ) -> Result<Output, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        let right =
            <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
                Error::Launch {
                    message: "set_difference right input must be backed by one DeviceVec"
                        .to_string(),
                }
            })?;
        let inner = crate::detail::set_difference(self.into_inner(), (right,), less)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn inner_product_dispatch<Right, TransformOp, ReduceOp>(
        self,
        right: Right,
        _transform_op: TransformOp,
        init: <Self as MIter<B>>::Item,
        _reduce_op: ReduceOp,
    ) -> Result<<Self as MIter<B>>::Item, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        TransformOp: op::BinaryOp<<Self as MIter<B>>::Item>,
        ReduceOp: op::BinaryOp<<Self as MIter<B>>::Item>,
    {
        let right =
            <Right as sealed::MIterDispatch<B>>::column_inner::<T>(&right).ok_or_else(|| {
                Error::Launch {
                    message: "inner_product right input must be backed by one DeviceVec"
                        .to_string(),
                }
            })?;
        let value = crate::detail::inner_product(
            &self.0.inner,
            right,
            crate::detail::api::Tuple1BinaryOp::<TransformOp>::default(),
            init.0,
            crate::detail::api::Tuple1BinaryOp::<ReduceOp>::default(),
        )?;
        Ok((value,))
    }

    fn equal_same_dispatch<Eq>(self, right: Self, eq: Eq) -> Result<bool, Error>
    where
        Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        self.equal_dispatch(right, eq)
    }

    fn mismatch_same_dispatch<Eq>(self, right: Self, eq: Eq) -> Result<Option<usize>, Error>
    where
        Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        self.mismatch_dispatch(right, eq)
    }

    fn find_first_of_same_dispatch<Eq>(self, needles: Self, eq: Eq) -> Result<Option<usize>, Error>
    where
        Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        self.find_first_of_dispatch(needles, eq)
    }

    fn lexicographical_compare_same_dispatch<Less>(
        self,
        right: Self,
        less: Less,
    ) -> Result<bool, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        self.lexicographical_compare_dispatch(right, less)
    }

    fn merge_same_dispatch<Output, Less>(self, right: Self, less: Less) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        self.merge_dispatch(right, less)
    }

    fn set_union_same_dispatch<Output, Less>(self, right: Self, less: Less) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        self.set_union_dispatch(right, less)
    }

    fn set_intersection_same_dispatch<Output, Less>(
        self,
        right: Self,
        less: Less,
    ) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        self.set_intersection_dispatch(right, less)
    }

    fn set_difference_same_dispatch<Output, Less>(
        self,
        right: Self,
        less: Less,
    ) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        self.set_difference_dispatch(right, less)
    }

    fn inner_product_same_dispatch<TransformOp, ReduceOp>(
        self,
        right: Self,
        transform_op: TransformOp,
        init: <Self as MIter<B>>::Item,
        reduce_op: ReduceOp,
    ) -> Result<<Self as MIter<B>>::Item, Error>
    where
        TransformOp: op::BinaryOp<<Self as MIter<B>>::Item>,
        ReduceOp: op::BinaryOp<<Self as MIter<B>>::Item>,
    {
        self.inner_product_dispatch(right, transform_op, init, reduce_op)
    }
}

macro_rules! impl_miter_tuple {
    ($( $ty:ident : $idx:tt ),+ => $transform:ident) => {
        impl<'a, B, $( $ty ),+> MIter<B> for ($( &'a DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
            ($( $ty, )+): StorageOutput<
                B,
                Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( &'a crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+);

            fn len(&self) -> usize {
                self.0.inner.len()
            }

            fn into_inner(self) -> Self::Inner {
                ($( &self.$idx.inner, )+)
            }
        }

        impl<'a, B, $( $ty ),+> sealed::MIterDispatch<B> for ($( &'a DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
            ($( $ty, )+): StorageOutput<
                B,
                Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+),
            >,
        {
            fn transform_dispatch<Op, Output, Y>(self, op: Op) -> Result<Output, Error>
            where
                Op: op::UnaryOp<<Self as MIter<B>>::Item, Output = Y>,
                Y: StorageOutput<B>,
                Output: MVec<B, Item = Y>,
            {
                let inner_input = self.into_inner();
                let inner = <Y as sealed::StorageOutputDispatch<B>>::$transform(
                    inner_input.0.policy(),
                    $( inner_input.$idx, )+
                    op,
                )?;
                Ok(array_from_inner::<B, Y, Output>(inner))
            }

            fn reverse_dispatch<Output>(self) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::reverse(self.into_inner())?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn sort_dispatch<Less, Output>(self, less: Less) -> Result<Output, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::sort(self.into_inner(), less)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
                self,
                keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar<B> + 'static,
                Less: op::BinaryPredicateOp<(K,)>,
                KeyOutput: MVec<B, Item = (K,)>,
                ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let values = self.into_inner();
                let values = impl_miter_tuple!(@view values; $( $idx ),+);
                let (key_inner, value_inner) = crate::detail::sort_by_key((keys,), (values,), less)?;
                Ok((
                    array_from_inner::<B, (K,), KeyOutput>(key_inner),
                    array_from_inner::<B, ($( $ty, )+), ValueOutput>(value_inner),
                ))
            }

            fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
                self,
                keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                eq: Eq,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar<B> + 'static,
                Eq: op::BinaryPredicateOp<(K,)>,
                KeyOutput: MVec<B, Item = (K,)>,
                ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let values = self.into_inner();
                let values = impl_miter_tuple!(@view values; $( $idx ),+);
                let (key_inner, value_inner) =
                    crate::detail::unique_by_key((keys,), (values,), eq)?;
                Ok((
                    array_from_inner::<B, (K,), KeyOutput>(key_inner),
                    array_from_inner::<B, ($( $ty, )+), ValueOutput>(value_inner),
                ))
            }

            fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
                self,
                keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                _key_eq: KeyEq,
                op: Op,
            ) -> Result<Output, Error>
            where
                K: Scalar<B> + 'static,
                KeyEq: op::BinaryPredicateOp<(K,)>,
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let values = self.into_inner();
                let values = impl_miter_tuple!(@view values; $( $idx ),+);
                let inner = crate::detail::inclusive_scan_by_key(
                    keys,
                    values,
                    crate::detail::api::Tuple1Less::<KeyEq>::default(),
                    op,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn exclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
                self,
                keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<B>>::Item,
                op: Op,
            ) -> Result<Output, Error>
            where
                K: Scalar<B> + 'static,
                KeyEq: op::BinaryPredicateOp<(K,)>,
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let values = self.into_inner();
                let values = impl_miter_tuple!(@view values; $( $idx ),+);
                let inner = crate::detail::exclusive_scan_by_key(
                    keys,
                    values,
                    crate::detail::api::Tuple1Less::<KeyEq>::default(),
                    init,
                    op,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn reduce_by_single_key_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<B>>::Item,
                op: Op,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar<B> + 'static,
                KeyEq: op::BinaryPredicateOp<(K,)>,
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
                KeyOutput: MVec<B, Item = (K,)>,
                ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let values = self.into_inner();
                let values = impl_miter_tuple!(@view values; $( $idx ),+);
                let (key_inner, value_inner) = crate::detail::reduce_by_key(
                    keys,
                    values,
                    crate::detail::api::Tuple1Less::<KeyEq>::default(),
                    init,
                    op,
                )?;
                Ok((
                    array_from_inner::<B, (K,), KeyOutput>(key_inner),
                    array_from_inner::<B, ($( $ty, )+), ValueOutput>(value_inner),
                ))
            }

            fn merge_by_single_key_same_dispatch<K, Less, KeyOutput, ValueOutput>(
                self,
                left_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                right_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                right_values: Self,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar<B> + 'static,
                Less: op::BinaryPredicateOp<(K,)>,
                KeyOutput: MVec<B, Item = (K,)>,
                ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let left_values = self.into_inner();
                let right_values = right_values.into_inner();
                let left_values = impl_miter_tuple!(@view left_values; $( $idx ),+);
                let right_values = impl_miter_tuple!(@view right_values; $( $idx ),+);
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    left_keys,
                    left_values,
                    right_keys,
                    right_values,
                    crate::detail::api::Tuple1Less::<Less>::default(),
                )?;
                Ok((
                    array_from_inner::<B, (K,), KeyOutput>(key_inner),
                    array_from_inner::<B, ($( $ty, )+), ValueOutput>(value_inner),
                ))
            }

            fn gather_dispatch<Indices, Output>(self, indices: Indices) -> Result<Output, Error>
            where
                Indices: MIter<B, Item = (u32,)>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let indices = gather_index_inner::<B, Indices>(&indices)?;
                let inner = crate::detail::gather(self.into_inner(), indices)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn selection_stencil_dispatch<Pred>(
                &self,
                invert: bool,
            ) -> Result<crate::detail::api::PrecomputedSelection<<B as sealed::Backend>::Runtime>, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                let stencil = (*self).into_inner();
                crate::detail::api::PrecomputedSelection::from_stencil::<_, Pred>(&stencil, invert)
            }

            fn reduce_dispatch<Op>(
                self,
                init: <Self as MIter<B>>::Item,
                op: Op,
            ) -> Result<<Self as MIter<B>>::Item, Error>
            where
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::reduce(self.into_inner(), init, op)
            }

            fn inclusive_scan_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
            where
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::inclusive_scan(self.into_inner(), op)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn exclusive_scan_dispatch<Op, Output>(
                self,
                init: <Self as MIter<B>>::Item,
                op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::exclusive_scan(self.into_inner(), init, op)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn adjacent_difference_dispatch<Op, Output>(self, op: Op) -> Result<Output, Error>
            where
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::adjacent_difference(self.into_inner(), op)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn copy_if_dispatch<Stencil, Pred, Output>(
                self,
                stencil: Stencil,
                pred: Pred,
            ) -> Result<Output, Error>
            where
        Stencil: MIter<B>,
        Pred: op::PredicateOp<Stencil::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let stencil = <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<Pred>(&stencil, false)?;
                let inner = crate::detail::copy_if(self.into_inner(), stencil, pred)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn remove_if_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::remove_if(self.into_inner(), pred)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn count_if_dispatch<Pred>(self, pred: Pred) -> Result<usize, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::count_if(self.into_inner(), pred)
            }

            fn all_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::all_of(self.into_inner(), pred)
            }

            fn any_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::any_of(self.into_inner(), pred)
            }

            fn none_of_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::none_of(self.into_inner(), pred)
            }

            fn find_if_dispatch<Pred>(self, pred: Pred) -> Result<Option<usize>, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::find_if(self.into_inner(), pred)
            }

            fn partition_dispatch<Pred, Output>(self, pred: Pred) -> Result<(Output, Output), Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let (matching, failing) = crate::detail::partition(self.into_inner(), pred)?;
                Ok((
                    array_from_inner::<B, ($( $ty, )+), Output>(matching),
                    array_from_inner::<B, ($( $ty, )+), Output>(failing),
                ))
            }

            fn is_partitioned_dispatch<Pred>(self, pred: Pred) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                crate::detail::is_partitioned(self.into_inner(), pred)
            }

            fn replace_if_dispatch<Stencil, Pred, Output>(
                self,
                replacement: <Self as MIter<B>>::Item,
                stencil: Stencil,
                pred: Pred,
            ) -> Result<Output, Error>
            where
        Stencil: MIter<B>,
        Pred: op::PredicateOp<Stencil::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let stencil = <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<Pred>(&stencil, false)?;
                let inner = crate::detail::replace_if(self.into_inner(), replacement, stencil, pred)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn unique_dispatch<Pred, Output>(self, pred: Pred) -> Result<Output, Error>
            where
                Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let inner = crate::detail::unique(self.into_inner(), pred)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn min_element_dispatch<Less>(self, less: Less) -> Result<Option<usize>, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let input = impl_miter_tuple!(@view input; $( $idx ),+);
                crate::detail::min_element(input, less)
            }

            fn max_element_dispatch<Less>(self, less: Less) -> Result<Option<usize>, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let input = impl_miter_tuple!(@view input; $( $idx ),+);
                crate::detail::max_element(input, less)
            }

            fn minmax_element_dispatch<Less>(
                self,
                less: Less,
            ) -> Result<Option<(usize, usize)>, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let input = impl_miter_tuple!(@view input; $( $idx ),+);
                crate::detail::minmax_element(input, less)
            }

            fn adjacent_find_dispatch<Pred>(self, pred: Pred) -> Result<Option<usize>, Error>
            where
                Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let input = impl_miter_tuple!(@view input; $( $idx ),+);
                crate::detail::adjacent_find(input, pred)
            }

            fn lower_bound_dispatch<Less>(
                self,
                value: <Self as MIter<B>>::Item,
                less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let input = impl_miter_tuple!(@view input; $( $idx ),+);
                crate::detail::lower_bound(input, value, less)
            }

            fn upper_bound_dispatch<Less>(
                self,
                value: <Self as MIter<B>>::Item,
                less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let input = impl_miter_tuple!(@view input; $( $idx ),+);
                crate::detail::upper_bound(input, value, less)
            }

            fn equal_range_dispatch<Less>(
                self,
                value: <Self as MIter<B>>::Item,
                less: Less,
            ) -> Result<(usize, usize), Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let input = impl_miter_tuple!(@view input; $( $idx ),+);
                crate::detail::equal_range(input, value, less)
            }

            fn is_sorted_until_dispatch<Less>(self, less: Less) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let input = impl_miter_tuple!(@view input; $( $idx ),+);
                crate::detail::is_sorted_until(input, less)
            }

            fn is_sorted_dispatch<Less>(self, less: Less) -> Result<bool, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let input = impl_miter_tuple!(@view input; $( $idx ),+);
                crate::detail::is_sorted(input, less)
            }

            fn gather_if_dispatch<Indices, Stencil, Pred, Output>(
                self,
                indices: Indices,
                default: <Self as MIter<B>>::Item,
                stencil: Stencil,
                pred: Pred,
            ) -> Result<Output, Error>
            where
                Indices: MIter<B, Item = (u32,)>,
                Stencil: MIter<B>,
                Pred: op::PredicateOp<Stencil::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<B>>::column_inner::<u32>(&indices).ok_or_else(|| {
                        Error::Launch {
                            message: "gather_if indices must be backed by one u32 DeviceVec".to_string(),
                        }
                    })?;
                let stencil = <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<Pred>(&stencil, false)?;
                let input = self.into_inner();
                let inner = crate::detail::gather_if(
                    impl_miter_tuple!(@view input; $( $idx ),+),
                    indices,
                    stencil,
                    default,
                    pred,
                )?;
                Ok(array_from_inner::<B, <Self as MIter<B>>::Item, Output>(inner))
            }

            fn scatter_dispatch<Indices, Output>(
                self,
                indices: Indices,
                len: usize,
                default: <Self as MIter<B>>::Item,
            ) -> Result<Output, Error>
            where
                Indices: MIter<B, Item = (u32,)>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<B>>::column_inner::<u32>(&indices).ok_or_else(|| {
                        Error::Launch {
                            message: "scatter indices must be backed by one u32 DeviceVec".to_string(),
                        }
                    })?;
                let input = self.into_inner();
                let inner = crate::detail::scatter(
                    impl_miter_tuple!(@view input; $( $idx ),+),
                    indices,
                    len,
                    default,
                )?;
                Ok(array_from_inner::<B, <Self as MIter<B>>::Item, Output>(inner))
            }

            fn scatter_if_dispatch<Indices, Stencil, Pred, Output>(
                self,
                indices: Indices,
                len: usize,
                default: <Self as MIter<B>>::Item,
                stencil: Stencil,
                pred: Pred,
            ) -> Result<Output, Error>
            where
                Indices: MIter<B, Item = (u32,)>,
        Stencil: MIter<B>,
        Pred: op::PredicateOp<Stencil::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<B>>::column_inner::<u32>(&indices).ok_or_else(|| {
                        Error::Launch {
                            message: "scatter_if indices must be backed by one u32 DeviceVec".to_string(),
                        }
                    })?;
                let stencil = <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<Pred>(&stencil, false)?;
                let input = self.into_inner();
                let inner = crate::detail::scatter_if(
                    impl_miter_tuple!(@view input; $( $idx ),+),
                    indices,
                    len,
                    default,
                    stencil,
                    pred,
                )?;
                Ok(array_from_inner::<B, <Self as MIter<B>>::Item, Output>(inner))
            }

            fn equal_same_dispatch<Eq>(self, right: Self, eq: Eq) -> Result<bool, Error>
            where
                Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                self.equal_values(right, eq)
            }

            fn mismatch_same_dispatch<Eq>(self, right: Self, eq: Eq) -> Result<Option<usize>, Error>
            where
                Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                self.mismatch_values(right, eq)
            }

            fn find_first_of_same_dispatch<Eq>(
                self,
                needles: Self,
                eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                self.find_first_of_values(needles, eq)
            }

            fn lexicographical_compare_same_dispatch<Less>(
                self,
                right: Self,
                less: Less,
            ) -> Result<bool, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                self.lexicographical_compare_values(right, less)
            }

            fn merge_same_dispatch<Output, Less>(
                self,
                right: Self,
                less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let inner = self.merge_values(right, less)?;
                Ok(array_from_inner::<B, <Self as MIter<B>>::Item, Output>(inner))
            }

            fn set_union_same_dispatch<Output, Less>(
                self,
                right: Self,
                less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let inner = self.set_union_values(right, less)?;
                Ok(array_from_inner::<B, <Self as MIter<B>>::Item, Output>(inner))
            }

            fn set_intersection_same_dispatch<Output, Less>(
                self,
                right: Self,
                less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let inner = self.set_intersection_values(right, less)?;
                Ok(array_from_inner::<B, <Self as MIter<B>>::Item, Output>(inner))
            }

            fn set_difference_same_dispatch<Output, Less>(
                self,
                right: Self,
                less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let inner = self.set_difference_values(right, less)?;
                Ok(array_from_inner::<B, <Self as MIter<B>>::Item, Output>(inner))
            }

            fn inner_product_same_dispatch<TransformOp, ReduceOp>(
                self,
                right: Self,
                transform_op: TransformOp,
                init: <Self as MIter<B>>::Item,
                reduce_op: ReduceOp,
            ) -> Result<<Self as MIter<B>>::Item, Error>
            where
                TransformOp: op::BinaryOp<<Self as MIter<B>>::Item>,
                ReduceOp: op::BinaryOp<<Self as MIter<B>>::Item>,
            {
                let _ = (right, transform_op, init, reduce_op);
                Err(Error::Launch {
                    message: "inner_product for tuple inputs requires tuple-valued detail support"
                        .to_string(),
                })
            }
        }

    };

    (@view $input:ident; 0, 1) => {
        crate::detail::device::SoAView2 {
            left: $input.0,
            right: $input.1,
        }
    };

    (@view $input:ident; 0, 1, 2) => {
        crate::detail::device::SoAView3 {
            first: $input.0,
            second: $input.1,
            third: $input.2,
        }
    };

}

impl_miter_tuple!(A: 0, C: 1 => transform_binary);
impl_miter_tuple!(A: 0, C: 1, D: 2 => transform_ternary);

impl<'a, 'b, B, T, Op> PairSearchValues<B, (&'b DeviceVec<B, T>,), Op> for (&'a DeviceVec<B, T>,)
where
    B: Backend,
    T: Scalar<B> + 'static,
    Op: op::BinaryPredicateOp<(T,)>,
{
    fn equal_values(self, right: (&'b DeviceVec<B, T>,), op: Op) -> Result<bool, Error> {
        crate::detail::equal((&self.0.inner,), (&right.0.inner,), op)
    }

    fn mismatch_values(
        self,
        right: (&'b DeviceVec<B, T>,),
        op: Op,
    ) -> Result<Option<usize>, Error> {
        crate::detail::mismatch((&self.0.inner,), (&right.0.inner,), op)
    }

    fn find_first_of_values(
        self,
        needles: (&'b DeviceVec<B, T>,),
        op: Op,
    ) -> Result<Option<usize>, Error> {
        crate::detail::find_first_of((&self.0.inner,), (&needles.0.inner,), op)
    }

    fn lexicographical_compare_values(
        self,
        right: (&'b DeviceVec<B, T>,),
        op: Op,
    ) -> Result<bool, Error> {
        crate::detail::lexicographical_compare((&self.0.inner,), (&right.0.inner,), op)
    }
}

impl<'a, 'b, B, T, Less> PairOrderingValues<B, (&'b DeviceVec<B, T>,), Less>
    for (&'a DeviceVec<B, T>,)
where
    B: Backend,
    T: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(T,)>,
{
    fn merge_values(
        self,
        right: (&'b DeviceVec<B, T>,),
        less: Less,
    ) -> Result<<Self::Item as StorageOutput<B>>::Inner, Error> {
        crate::detail::merge((&self.0.inner,), (&right.0.inner,), less)
    }

    fn set_union_values(
        self,
        right: (&'b DeviceVec<B, T>,),
        less: Less,
    ) -> Result<<Self::Item as StorageOutput<B>>::Inner, Error> {
        crate::detail::set_union((&self.0.inner,), (&right.0.inner,), less)
    }

    fn set_intersection_values(
        self,
        right: (&'b DeviceVec<B, T>,),
        less: Less,
    ) -> Result<<Self::Item as StorageOutput<B>>::Inner, Error> {
        crate::detail::set_intersection((&self.0.inner,), (&right.0.inner,), less)
    }

    fn set_difference_values(
        self,
        right: (&'b DeviceVec<B, T>,),
        less: Less,
    ) -> Result<<Self::Item as StorageOutput<B>>::Inner, Error> {
        crate::detail::set_difference((&self.0.inner,), (&right.0.inner,), less)
    }
}

impl<'a, 'b, B, T, TransformOp, ReduceOp>
    InnerProductValues<B, (&'b DeviceVec<B, T>,), TransformOp, ReduceOp> for (&'a DeviceVec<B, T>,)
where
    B: Backend,
    T: Scalar<B> + 'static,
    TransformOp: op::BinaryOp<(T,)>,
    ReduceOp: op::BinaryOp<(T,)>,
{
    fn inner_product_values(
        self,
        right: (&'b DeviceVec<B, T>,),
        _transform_op: TransformOp,
        init: Self::Item,
        _reduce_op: ReduceOp,
    ) -> Result<Self::Item, Error> {
        let value = crate::detail::inner_product(
            &self.0.inner,
            &right.0.inner,
            crate::detail::api::Tuple1BinaryOp::<TransformOp>::default(),
            init.0,
            crate::detail::api::Tuple1BinaryOp::<ReduceOp>::default(),
        )?;
        Ok((value,))
    }
}

macro_rules! impl_pair_values_tuple {
    ($( $ty:ident : $idx:tt ),+) => {
        impl<'a, 'b, B, Op, $( $ty ),+>
            PairSearchValues<B, ($( &'b DeviceVec<B, $ty>, )+), Op>
            for ($( &'a DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
            Op: op::BinaryPredicateOp<($( $ty, )+)>,
        {
            fn equal_values(
                self,
                right: ($( &'b DeviceVec<B, $ty>, )+),
                op: Op,
            ) -> Result<bool, Error> {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::equal(
                    impl_miter_tuple!(@view left; $( $idx ),+),
                    impl_miter_tuple!(@view right; $( $idx ),+),
                    op,
                )
            }

            fn mismatch_values(
                self,
                right: ($( &'b DeviceVec<B, $ty>, )+),
                op: Op,
            ) -> Result<Option<usize>, Error> {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::mismatch(
                    impl_miter_tuple!(@view left; $( $idx ),+),
                    impl_miter_tuple!(@view right; $( $idx ),+),
                    op,
                )
            }

            fn find_first_of_values(
                self,
                needles: ($( &'b DeviceVec<B, $ty>, )+),
                op: Op,
            ) -> Result<Option<usize>, Error> {
                let input = self.into_inner();
                let needles = needles.into_inner();
                crate::detail::find_first_of(
                    impl_miter_tuple!(@view input; $( $idx ),+),
                    impl_miter_tuple!(@view needles; $( $idx ),+),
                    op,
                )
            }

            fn lexicographical_compare_values(
                self,
                right: ($( &'b DeviceVec<B, $ty>, )+),
                op: Op,
            ) -> Result<bool, Error> {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::lexicographical_compare(
                    impl_miter_tuple!(@view left; $( $idx ),+),
                    impl_miter_tuple!(@view right; $( $idx ),+),
                    op,
                )
            }
        }

        impl<'a, 'b, B, Less, $( $ty ),+>
            PairOrderingValues<B, ($( &'b DeviceVec<B, $ty>, )+), Less>
            for ($( &'a DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
            Less: op::BinaryPredicateOp<($( $ty, )+)>,
            ($( $ty, )+): StorageOutput<
                B,
                Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+),
            >,
        {
            fn merge_values(
                self,
                right: ($( &'b DeviceVec<B, $ty>, )+),
                less: Less,
            ) -> Result<<Self::Item as StorageOutput<B>>::Inner, Error> {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::merge(
                    impl_miter_tuple!(@view left; $( $idx ),+),
                    impl_miter_tuple!(@view right; $( $idx ),+),
                    less,
                )
            }

            fn set_union_values(
                self,
                right: ($( &'b DeviceVec<B, $ty>, )+),
                less: Less,
            ) -> Result<<Self::Item as StorageOutput<B>>::Inner, Error> {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::set_union(
                    impl_miter_tuple!(@view left; $( $idx ),+),
                    impl_miter_tuple!(@view right; $( $idx ),+),
                    less,
                )
            }

            fn set_intersection_values(
                self,
                right: ($( &'b DeviceVec<B, $ty>, )+),
                less: Less,
            ) -> Result<<Self::Item as StorageOutput<B>>::Inner, Error> {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::set_intersection(
                    impl_miter_tuple!(@view left; $( $idx ),+),
                    impl_miter_tuple!(@view right; $( $idx ),+),
                    less,
                )
            }

            fn set_difference_values(
                self,
                right: ($( &'b DeviceVec<B, $ty>, )+),
                less: Less,
            ) -> Result<<Self::Item as StorageOutput<B>>::Inner, Error> {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::set_difference(
                    impl_miter_tuple!(@view left; $( $idx ),+),
                    impl_miter_tuple!(@view right; $( $idx ),+),
                    less,
                )
            }
        }

        impl<'a, 'b, B, TransformOp, ReduceOp, $( $ty ),+>
            InnerProductValues<B, ($( &'b DeviceVec<B, $ty>, )+), TransformOp, ReduceOp>
            for ($( &'a DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar<B>, )+
            TransformOp: $( op::BinaryOp<$ty> + )+,
            ReduceOp: $( op::BinaryOp<$ty> + )+,
        {
            fn inner_product_values(
                self,
                right: ($( &'b DeviceVec<B, $ty>, )+),
                transform_op: TransformOp,
                init: Self::Item,
                reduce_op: ReduceOp,
            ) -> Result<Self::Item, Error> {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::inner_product(
                    impl_miter_tuple!(@view left; $( $idx ),+),
                    impl_miter_tuple!(@view right; $( $idx ),+),
                    transform_op,
                    init,
                    reduce_op,
                )
            }
        }
    };
}

impl_pair_values_tuple!(A: 0, C: 1);
impl_pair_values_tuple!(A: 0, C: 1, D: 2);

impl<'a, B, T, K, Less> MergeBySingleKeyValues<B, (&'a DeviceVec<B, T>,), K, Less>
    for (&'a DeviceVec<B, T>,)
where
    B: Backend,
    T: Scalar<B> + 'static,
    K: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(K,)>,
{
    fn merge_by_single_key_values(
        self,
        left_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        right_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        right_values: (&'a DeviceVec<B, T>,),
    ) -> Result<
        (
            (crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,),
            <Self::Item as StorageOutput<B>>::Inner,
        ),
        Error,
    > {
        crate::detail::merge_by_key(
            left_keys,
            &self.0.inner,
            right_keys,
            &right_values.0.inner,
            crate::detail::api::Tuple1Less::<Less>::default(),
        )
    }
}

macro_rules! impl_merge_by_single_key_values_tuple {
    ($( $ty:ident : $idx:tt ),+) => {
        impl<'a, B, K, Less, $( $ty ),+>
            MergeBySingleKeyValues<B, ($( &'a DeviceVec<B, $ty>, )+), K, Less>
            for ($( &'a DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            K: Scalar<B> + 'static,
            Less: op::BinaryPredicateOp<(K,)>,
            $( $ty: Scalar<B>, )+
            ($( $ty, )+): StorageOutput<
                B,
                Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+),
            >,
        {
            fn merge_by_single_key_values(
                self,
                left_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                right_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                right_values: ($( &'a DeviceVec<B, $ty>, )+),
            ) -> Result<
                (
                    (crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,),
                    <Self::Item as StorageOutput<B>>::Inner,
                ),
                Error,
            > {
                let left_values = self.into_inner();
                let right_values = right_values.into_inner();
                let left_values = impl_miter_tuple!(@view left_values; $( $idx ),+);
                let right_values = impl_miter_tuple!(@view right_values; $( $idx ),+);
                crate::detail::merge_by_key(
                    left_keys,
                    left_values,
                    right_keys,
                    right_values,
                    crate::detail::api::Tuple1Less::<Less>::default(),
                )
            }
        }
    };
}

impl_merge_by_single_key_values_tuple!(A: 0, C: 1);
impl_merge_by_single_key_values_tuple!(A: 0, C: 1, D: 2);

/// Applies a unary transform to a massively iterator.
pub fn transform<B, Input, Output, Op>(source: Input, op: Op) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B>,
    Op: op::UnaryOp<Input::Item, Output = Output::Item>,
{
    <Input as sealed::MIterDispatch<B>>::transform_dispatch(source, op)
}

/// Reverses a massively iterator into an owned vector.
pub fn reverse<B, Input, Output>(source: Input) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::reverse_dispatch(source)
}

/// Sorts a massively iterator into an owned vector.
pub fn sort<B, Input, Output, Less>(source: Input, less: Less) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::sort_dispatch(source, less)
}

/// Gathers a massively iterator at index positions into an owned vector.
pub fn gather<B, Input, Indices, Output>(source: Input, indices: Indices) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Indices: MIter<B, Item = (u32,)>,
    Output: MVec<B, Item = Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::gather_dispatch(source, indices)
}

/// Reduces a massively iterator to one host item.
pub fn reduce<B, Input, Op>(source: Input, init: Input::Item, op: Op) -> Result<Input::Item, Error>
where
    B: Backend,
    Input: MIter<B>,
    Op: op::BinaryOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::reduce_dispatch(source, init, op)
}

/// Computes an inclusive scan.
pub fn inclusive_scan<B, Input, Output, Op>(source: Input, op: Op) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::BinaryOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::inclusive_scan_dispatch(source, op)
}

/// Computes an exclusive scan.
pub fn exclusive_scan<B, Input, Output, Op>(
    source: Input,
    init: Input::Item,
    op: Op,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::BinaryOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::exclusive_scan_dispatch(source, init, op)
}

/// Computes adjacent differences.
pub fn adjacent_difference<B, Input, Output, Op>(source: Input, op: Op) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::BinaryOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::adjacent_difference_dispatch(source, op)
}

/// Copies elements whose stencil value satisfies `pred`.
pub fn copy_if<B, Input, Stencil, Output, Pred>(
    source: Input,
    stencil: Stencil,
    pred: Pred,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Stencil: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<Stencil::Item>,
{
    <Input as sealed::MIterDispatch<B>>::copy_if_dispatch(source, stencil, pred)
}

/// Removes elements satisfying `pred`.
pub fn remove_if<B, Input, Output, Pred>(source: Input, pred: Pred) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::remove_if_dispatch(source, pred)
}

/// Counts elements satisfying `pred`.
pub fn count_if<B, Input, Pred>(source: Input, pred: Pred) -> Result<usize, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::count_if_dispatch(source, pred)
}

/// Returns whether all elements satisfy `pred`.
pub fn all_of<B, Input, Pred>(source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::all_of_dispatch(source, pred)
}

/// Returns whether any element satisfies `pred`.
pub fn any_of<B, Input, Pred>(source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::any_of_dispatch(source, pred)
}

/// Returns whether no elements satisfy `pred`.
pub fn none_of<B, Input, Pred>(source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::none_of_dispatch(source, pred)
}

/// Finds the first element satisfying `pred`.
pub fn find_if<B, Input, Pred>(source: Input, pred: Pred) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::find_if_dispatch(source, pred)
}

/// Partitions elements by `pred`.
pub fn partition<B, Input, Output, Pred>(
    source: Input,
    pred: Pred,
) -> Result<(Output, Output), Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::partition_dispatch(source, pred)
}

/// Returns whether input is partitioned by `pred`.
pub fn is_partitioned<B, Input, Pred>(source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::is_partitioned_dispatch(source, pred)
}

/// Replaces elements whose stencil value satisfies `pred`.
pub fn replace_if<B, Input, Stencil, Output, Pred>(
    source: Input,
    replacement: Input::Item,
    stencil: Stencil,
    pred: Pred,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Stencil: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<Stencil::Item>,
{
    <Input as sealed::MIterDispatch<B>>::replace_if_dispatch(source, replacement, stencil, pred)
}

/// Removes consecutive duplicates under `pred`.
pub fn unique<B, Input, Output, Pred>(source: Input, pred: Pred) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::unique_dispatch(source, pred)
}

/// Stable sort. The current lower implementation is stable.
pub fn stable_sort<B, Input, Output, Less>(source: Input, less: Less) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    sort(source, less)
}

/// Gathers selected elements.
pub fn gather_if<B, Input, Indices, Stencil, Output, Pred>(
    source: Input,
    indices: Indices,
    default: Input::Item,
    stencil: Stencil,
    pred: Pred,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Indices: MIter<B, Item = (u32,)>,
    Stencil: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<Stencil::Item>,
{
    <Input as sealed::MIterDispatch<B>>::gather_if_dispatch(source, indices, default, stencil, pred)
}

/// Scatters values into a newly allocated output.
pub fn scatter<B, Input, Indices, Output>(
    source: Input,
    indices: Indices,
    len: usize,
    default: Input::Item,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Indices: MIter<B, Item = (u32,)>,
    Output: MVec<B, Item = Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::scatter_dispatch(source, indices, len, default)
}

/// Scatters selected values into a newly allocated output.
pub fn scatter_if<B, Input, Indices, Stencil, Output, Pred>(
    source: Input,
    indices: Indices,
    len: usize,
    default: Input::Item,
    stencil: Stencil,
    pred: Pred,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Indices: MIter<B, Item = (u32,)>,
    Stencil: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<Stencil::Item>,
{
    <Input as sealed::MIterDispatch<B>>::scatter_if_dispatch(
        source, indices, len, default, stencil, pred,
    )
}

/// Finds the first adjacent pair satisfying `pred`.
pub fn adjacent_find<B, Input, Pred>(source: Input, pred: Pred) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::adjacent_find_dispatch(source, pred)
}

/// Returns whether two inputs are equal under `eq`.
pub fn equal<B, Input, Eq>(left: Input, right: Input, eq: Eq) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Eq: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::equal_same_dispatch(left, right, eq)
}

/// Finds the first mismatch between two inputs.
pub fn mismatch<B, Input, Eq>(left: Input, right: Input, eq: Eq) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Eq: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::mismatch_same_dispatch(left, right, eq)
}

/// Finds the first input element equal to any needle.
pub fn find_first_of<B, Input, Eq>(
    source: Input,
    needles: Input,
    eq: Eq,
) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Eq: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::find_first_of_same_dispatch(source, needles, eq)
}

/// Finds the minimum element index.
pub fn min_element<B, Input, Less>(source: Input, less: Less) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::min_element_dispatch(source, less)
}

/// Finds the maximum element index.
pub fn max_element<B, Input, Less>(source: Input, less: Less) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::max_element_dispatch(source, less)
}

/// Finds both minimum and maximum element indices.
pub fn minmax_element<B, Input, Less>(
    source: Input,
    less: Less,
) -> Result<Option<(usize, usize)>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::minmax_element_dispatch(source, less)
}

/// Finds the lower bound of `value` in a sorted input.
pub fn lower_bound<B, Input, Less>(
    source: Input,
    value: Input::Item,
    less: Less,
) -> Result<usize, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::lower_bound_dispatch(source, value, less)
}

/// Finds the upper bound of `value` in a sorted input.
pub fn upper_bound<B, Input, Less>(
    source: Input,
    value: Input::Item,
    less: Less,
) -> Result<usize, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::upper_bound_dispatch(source, value, less)
}

/// Finds the equal range of `value` in a sorted input.
pub fn equal_range<B, Input, Less>(
    source: Input,
    value: Input::Item,
    less: Less,
) -> Result<(usize, usize), Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::equal_range_dispatch(source, value, less)
}

/// Returns the first position where sorted order is broken.
pub fn is_sorted_until<B, Input, Less>(source: Input, less: Less) -> Result<usize, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::is_sorted_until_dispatch(source, less)
}

/// Returns whether input is sorted.
pub fn is_sorted<B, Input, Less>(source: Input, less: Less) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::is_sorted_dispatch(source, less)
}

/// Lexicographically compares two inputs.
pub fn lexicographical_compare<B, Input, Less>(
    left: Input,
    right: Input,
    less: Less,
) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::lexicographical_compare_same_dispatch(left, right, less)
}

/// Merges two sorted inputs.
pub fn merge<B, Input, Output, Less>(left: Input, right: Input, less: Less) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::merge_same_dispatch(left, right, less)
}

/// Computes the sorted set union of two sorted inputs.
pub fn set_union<B, Input, Output, Less>(
    left: Input,
    right: Input,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::set_union_same_dispatch(left, right, less)
}

/// Computes the sorted set intersection of two sorted inputs.
pub fn set_intersection<B, Input, Output, Less>(
    left: Input,
    right: Input,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::set_intersection_same_dispatch(left, right, less)
}

/// Computes the sorted set difference of two sorted inputs.
pub fn set_difference<B, Input, Output, Less>(
    left: Input,
    right: Input,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::set_difference_same_dispatch(left, right, less)
}

/// Applies a binary transform over two inputs and reduces the result.
pub fn inner_product<B, Input, TransformOp, ReduceOp>(
    left: Input,
    right: Input,
    transform_op: TransformOp,
    init: Input::Item,
    reduce_op: ReduceOp,
) -> Result<Input::Item, Error>
where
    B: Backend,
    Input: MIter<B>,
    TransformOp: op::BinaryOp<Input::Item>,
    ReduceOp: op::BinaryOp<Input::Item>,
{
    <Input as sealed::MIterDispatch<B>>::inner_product_same_dispatch(
        left,
        right,
        transform_op,
        init,
        reduce_op,
    )
}

/// Inclusive scan by key.
pub fn inclusive_scan_by_key<B, Keys, Values, K, KeyEq, Op, Output>(
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    op: Op,
) -> Result<Output, Error>
where
    B: Backend,
    Keys: MIter<B, Item = (K,)>,
    Values: MIter<B>,
    K: Scalar<B> + 'static,
    KeyEq: op::BinaryPredicateOp<(K,)>,
    Op: op::BinaryOp<Values::Item>,
    Output: MVec<B, Item = Values::Item>,
{
    let keys = <Keys as sealed::MIterDispatch<B>>::column_inner::<K>(&keys).ok_or_else(|| {
        Error::Launch {
            message: "inclusive_scan_by_key keys must be backed by one DeviceVec".to_string(),
        }
    })?;
    <Values as sealed::MIterDispatch<B>>::inclusive_scan_by_single_key_dispatch(
        values, keys, key_eq, op,
    )
}

/// Exclusive scan by key.
pub fn exclusive_scan_by_key<B, Keys, Values, K, KeyEq, Op, Output>(
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Values::Item,
    op: Op,
) -> Result<Output, Error>
where
    B: Backend,
    Keys: MIter<B, Item = (K,)>,
    Values: MIter<B>,
    K: Scalar<B> + 'static,
    KeyEq: op::BinaryPredicateOp<(K,)>,
    Op: op::BinaryOp<Values::Item>,
    Output: MVec<B, Item = Values::Item>,
{
    let keys = <Keys as sealed::MIterDispatch<B>>::column_inner::<K>(&keys).ok_or_else(|| {
        Error::Launch {
            message: "exclusive_scan_by_key keys must be backed by one DeviceVec".to_string(),
        }
    })?;
    <Values as sealed::MIterDispatch<B>>::exclusive_scan_by_single_key_dispatch(
        values, keys, key_eq, init, op,
    )
}

/// Reduces consecutive values with equal keys.
pub fn reduce_by_key<B, Keys, Values, K, KeyEq, Op, KeyOutput, ValueOutput>(
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Values::Item,
    op: Op,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    Keys: MIter<B, Item = (K,)>,
    Values: MIter<B>,
    K: Scalar<B> + 'static,
    KeyEq: op::BinaryPredicateOp<(K,)>,
    Op: op::BinaryOp<Values::Item>,
    KeyOutput: MVec<B, Item = (K,)>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    let keys = <Keys as sealed::MIterDispatch<B>>::column_inner::<K>(&keys).ok_or_else(|| {
        Error::Launch {
            message: "reduce_by_key keys must be backed by one DeviceVec".to_string(),
        }
    })?;
    <Values as sealed::MIterDispatch<B>>::reduce_by_single_key_dispatch(
        values, keys, key_eq, init, op,
    )
}

/// Removes consecutive duplicate keys and keeps their values.
pub fn unique_by_key<B, Keys, Values, K, Eq, KeyOutput, ValueOutput>(
    keys: Keys,
    values: Values,
    eq: Eq,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    Keys: MIter<B, Item = (K,)>,
    Values: MIter<B>,
    K: Scalar<B> + 'static,
    Eq: op::BinaryPredicateOp<(K,)>,
    KeyOutput: MVec<B, Item = (K,)>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    let keys = <Keys as sealed::MIterDispatch<B>>::column_inner::<K>(&keys).ok_or_else(|| {
        Error::Launch {
            message: "unique_by_key keys must be backed by one DeviceVec".to_string(),
        }
    })?;
    <Values as sealed::MIterDispatch<B>>::unique_by_single_key_dispatch(values, keys, eq)
}

/// Sorts key-value pairs by key.
pub fn sort_by_key<B, Keys, Values, K, Less, KeyOutput, ValueOutput>(
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    Keys: MIter<B, Item = (K,)>,
    Values: MIter<B>,
    K: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(K,)>,
    KeyOutput: MVec<B, Item = (K,)>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    let keys = <Keys as sealed::MIterDispatch<B>>::column_inner::<K>(&keys).ok_or_else(|| {
        Error::Launch {
            message: "sort_by_key keys must be backed by one DeviceVec".to_string(),
        }
    })?;
    <Values as sealed::MIterDispatch<B>>::sort_by_single_key_dispatch(values, keys, less)
}

/// Stable key-value sort. The current lower implementation is stable.
pub fn stable_sort_by_key<B, Keys, Values, K, Less, KeyOutput, ValueOutput>(
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    Keys: MIter<B, Item = (K,)>,
    Values: MIter<B>,
    K: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(K,)>,
    KeyOutput: MVec<B, Item = (K,)>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    sort_by_key(keys, values, less)
}

/// Merges two sorted key-value ranges by key.
pub fn merge_by_key<B, LeftKeys, Values, RightKeys, K, Less, KeyOutput, ValueOutput>(
    left_keys: LeftKeys,
    left_values: Values,
    right_keys: RightKeys,
    right_values: Values,
    less: Less,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    LeftKeys: MIter<B, Item = (K,)>,
    RightKeys: MIter<B, Item = (K,)>,
    Values: MIter<B>,
    K: Scalar<B> + 'static,
    Less: op::BinaryPredicateOp<(K,)>,
    KeyOutput: MVec<B, Item = (K,)>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    let left_keys = <LeftKeys as sealed::MIterDispatch<B>>::column_inner::<K>(&left_keys)
        .ok_or_else(|| Error::Launch {
            message: "merge_by_key left keys must be backed by one DeviceVec".to_string(),
        })?;
    let right_keys = <RightKeys as sealed::MIterDispatch<B>>::column_inner::<K>(&right_keys)
        .ok_or_else(|| Error::Launch {
            message: "merge_by_key right keys must be backed by one DeviceVec".to_string(),
        })?;
    <Values as sealed::MIterDispatch<B>>::merge_by_single_key_same_dispatch(
        left_values,
        left_keys,
        right_keys,
        right_values,
        less,
    )
}
