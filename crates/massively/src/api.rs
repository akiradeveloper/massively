//! Public API for `massively`.
//!
//! This crate intentionally keeps CubeCL runtime types out of public algorithm
//! signatures. The implementation delegates to the internal detail layer.

use std::any::Any;
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};

use cubecl::frontend::PartialOrdExpand;

pub use crate::detail::Error;
pub use crate::detail::op;

mod sealed {
    use super::{Error, Executor, MIter, MVec, op};

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

    pub trait ToHostDispatch<B: super::Backend> {
        type Output;

        fn to_host_with(&self, exec: &Executor<B>) -> Result<Self::Output, Error>;
    }

    pub trait MIterDispatch<B: super::Backend>: Sized {
        fn validate_executor(&self, _exec: &Executor<B>) -> Result<(), Error> {
            Ok(())
        }

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

        fn column_vec_inner<T: 'static>(
            &self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
        ) -> Result<Option<crate::detail::DeviceVec<<B as Backend>::Runtime, T>>, Error>
        where
            T: super::Scalar<B>,
        {
            Ok(self
                .column_view_inner::<T>()?
                .map(|view| view.materialize(policy))
                .transpose()?)
        }

        fn column_view_inner<T: 'static>(
            &self,
        ) -> Result<
            Option<crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, T>>,
            Error,
        >
        where
            T: super::Scalar<B>,
        {
            Ok(self
                .column_inner::<T>()
                .map(crate::detail::device::DeviceColumnView::from_column))
        }

        fn transform_dispatch<Op, Output, Y>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::UnaryOp<<Self as MIter<B>>::Item, Output = Y>,
            Y: super::StorageOutput<B>,
            Output: MVec<B, Item = Y>;

        fn reverse_dispatch<Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn sort_dispatch<Less, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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

        fn gather_dispatch<Indices, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            indices: Indices,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Indices: MIter<B, Item = (u32,)>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn gather_if_dispatch<Indices, Stencil, Output>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _indices: Indices,
            _default: <Self as MIter<B>>::Item,
            _stencil: Stencil,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Indices: MIter<B, Item = (u32,)>,
            Stencil: MIter<B, Item = (u32,)>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "gather_if is not supported for this iterator shape".to_string(),
            })
        }

        fn scatter_dispatch<Indices, Output>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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

        fn scatter_if_dispatch<Indices, Stencil, Output>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _indices: Indices,
            _len: usize,
            _default: <Self as MIter<B>>::Item,
            _stencil: Stencil,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Indices: MIter<B, Item = (u32,)>,
            Stencil: MIter<B, Item = (u32,)>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "scatter_if is not supported for this iterator shape".to_string(),
            })
        }

        fn reduce_dispatch<Op>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _init: <Self as MIter<B>>::Item,
            op: Op,
        ) -> Result<<Self as MIter<B>>::Item, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>;

        fn inclusive_scan_dispatch<Op, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn exclusive_scan_dispatch<Op, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _init: <Self as MIter<B>>::Item,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn adjacent_difference_dispatch<Op, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn copy_if_dispatch<Stencil, Output>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _stencil: Stencil,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Stencil: MIter<B, Item = (u32,)>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn remove_if_dispatch<Pred, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn count_if_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<usize, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn all_of_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn any_of_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn none_of_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn find_if_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn partition_dispatch<Pred, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<(Output, Output), Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn is_partitioned_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp<<Self as MIter<B>>::Item>;

        fn replace_if_dispatch<Stencil, Output>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            replacement: <Self as MIter<B>>::Item,
            _stencil: Stencil,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Stencil: MIter<B, Item = (u32,)>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        #[doc(hidden)]
        fn selection_stencil_dispatch<Pred>(
            &self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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

        fn unique_dispatch<Pred, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn min_element_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            less: Less,
        ) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn max_element_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            less: Less,
        ) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn minmax_element_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            less: Less,
        ) -> Result<Option<(usize, usize)>, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn adjacent_find_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<Option<usize>, Error>
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
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            value: <Self as MIter<B>>::Item,
            _less: Less,
        ) -> Result<usize, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn upper_bound_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            value: <Self as MIter<B>>::Item,
            _less: Less,
        ) -> Result<usize, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn equal_range_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            value: <Self as MIter<B>>::Item,
            _less: Less,
        ) -> Result<(usize, usize), Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn is_sorted_until_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            less: Less,
        ) -> Result<usize, Error>
        where
            Self: MIter<B>,
            Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>;

        fn is_sorted_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            less: Less,
        ) -> Result<bool, Error>
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

        fn equal_same_dispatch<Eq>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _right: Self,
            _eq: Eq,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "equal is not supported for this iterator shape".to_string(),
            })
        }

        fn mismatch_same_dispatch<Eq>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _right: Self,
            _eq: Eq,
        ) -> Result<Option<usize>, Error>
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
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
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
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            input: crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, Input>,
            op: Op,
        ) -> Result<<Self as super::StorageOutput<B>>::Inner, Error>
        where
            Self: super::StorageOutput<B>,
            Input: super::Scalar<B>,
            Op: op::UnaryOp<(Input,), Output = Self>;

        fn transform_binary<Left, Right, Op>(
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            left: crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, Left>,
            right: crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, Right>,
            op: Op,
        ) -> Result<<Self as super::StorageOutput<B>>::Inner, Error>
        where
            Self: super::StorageOutput<B>,
            Left: super::Scalar<B>,
            Right: super::Scalar<B>,
            Op: op::UnaryOp<(Left, Right), Output = Self>;

        fn transform_ternary<First, Second, Third, Op>(
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            first: crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, First>,
            second: crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, Second>,
            third: crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, Third>,
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

/// Device-resident data that can be copied back to host memory by an executor.
pub trait ToHost<B: Backend>:
    sealed::ToHostDispatch<B, Output = <Self as ToHost<B>>::Output>
{
    type Output;
}

impl<B, T> ToHost<B> for T
where
    B: Backend,
    T: sealed::ToHostDispatch<B>,
{
    type Output = <T as sealed::ToHostDispatch<B>>::Output;
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

/// Execution context for a facade backend.
#[derive(Debug)]
pub struct Executor<B: Backend> {
    inner: crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> Clone for Executor<B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Executor<B> {
    fn from_inner(inner: crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>) -> Self {
        Self {
            inner,
            _backend: PhantomData,
        }
    }

    fn ensure_policy_id(&self, id: crate::detail::policy::CubePolicyId) -> Result<(), Error> {
        if self.inner.id() == id {
            Ok(())
        } else {
            Err(Error::Launch {
                message: "executor does not own this device data".to_string(),
            })
        }
    }

    fn policy(&self) -> &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime> {
        &self.inner
    }

    /// Copies host data to device-resident storage.
    pub fn to_device<T>(&self, input: &[T]) -> Result<DeviceVec<B, T>, Error>
    where
        T: Scalar<B>,
    {
        Ok(DeviceVec::from_inner(self.inner.to_device(input)?))
    }

    /// Allocates device-resident storage and fills it with `value`.
    pub fn filled<T>(&self, len: usize, value: T) -> Result<DeviceVec<B, T>, Error>
    where
        T: Scalar<B>,
    {
        Ok(DeviceVec::from_inner(self.inner.device_filled(len, value)?))
    }

    /// Copies device-resident storage back to host memory.
    pub fn to_host<Input>(&self, input: &Input) -> Result<<Input as ToHost<B>>::Output, Error>
    where
        Input: ToHost<B>,
    {
        input.to_host_with(self)
    }

    /// Waits until all previously submitted work for this executor is complete.
    pub fn sync(&self) -> Result<(), Error> {
        futures_lite::future::block_on(self.inner.client().sync()).map_err(|err| Error::Launch {
            message: err.to_string(),
        })
    }
}

#[cfg(feature = "wgpu")]
impl Executor<Wgpu> {
    /// Creates a WGPU executor backed by the default device.
    pub fn new() -> Self {
        Self::from_inner(crate::detail::CubeWgpu::new())
    }

    /// Creates a WGPU executor backed by the CPU adapter.
    pub fn cpu() -> Self {
        Self::from_inner(crate::detail::CubeWgpu::cpu())
    }
}

#[cfg(feature = "wgpu")]
impl Default for Executor<Wgpu> {
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

    /// Returns a read-only device slice for the given range.
    ///
    /// The range is checked like a Rust slice range and panics if it is out of
    /// bounds or if the start is greater than the end.
    pub fn slice<R>(&self, range: R) -> DeviceSlice<'_, B, T>
    where
        R: RangeBounds<usize>,
    {
        let (offset, len) = resolve_slice_range(self.len(), range);
        DeviceSlice {
            source: self,
            offset,
            len,
        }
    }
}

impl<B, T> sealed::ToHostDispatch<B> for DeviceVec<B, T>
where
    B: Backend,
    T: Scalar<B>,
{
    type Output = Vec<T>;

    fn to_host_with(&self, exec: &Executor<B>) -> Result<Self::Output, Error> {
        exec.ensure_policy_id(self.inner.policy_id())?;
        self.inner.read_to_host(exec.policy())
    }
}

/// Read-only view into a contiguous range of a [`DeviceVec`].
#[derive(Debug)]
pub struct DeviceSlice<'a, B: Backend, T> {
    source: &'a DeviceVec<B, T>,
    offset: usize,
    len: usize,
}

impl<'a, B, T> Copy for DeviceSlice<'a, B, T> where B: Backend {}

impl<'a, B, T> Clone for DeviceSlice<'a, B, T>
where
    B: Backend,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, B, T> DeviceSlice<'a, B, T>
where
    B: Backend,
{
    /// Returns the number of elements in this slice.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether this slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn materialize_with(&self, exec: &Executor<B>) -> Result<DeviceVec<B, T>, Error>
    where
        T: Scalar<B>,
    {
        exec.ensure_policy_id(self.source.inner.policy_id())?;
        Ok(DeviceVec::from_inner(
            crate::detail::primitives::range::copy_slice_with_policy(
                exec.policy(),
                &self.source.inner,
                self.offset,
                self.len,
            )?,
        ))
    }
}

impl<'a, B, T> sealed::ToHostDispatch<B> for DeviceSlice<'a, B, T>
where
    B: Backend,
    T: Scalar<B>,
{
    type Output = Vec<T>;

    fn to_host_with(&self, exec: &Executor<B>) -> Result<Self::Output, Error> {
        self.materialize_with(exec)?
            .inner
            .read_to_host(exec.policy())
    }
}

fn resolve_slice_range<R>(len: usize, range: R) -> (usize, usize)
where
    R: RangeBounds<usize>,
{
    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start.checked_add(1).expect("slice start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&end) => end.checked_add(1).expect("slice end overflow"),
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };
    assert!(
        start <= end,
        "slice start ({start}) is greater than slice end ({end})"
    );
    assert!(
        end <= len,
        "slice end ({end}) is out of bounds for DeviceVec of length {len}"
    );
    (start, end - start)
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
    policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
    indices: &Indices,
) -> Result<crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, u32>, Error>
where
    B: Backend,
    Indices: MIter<B, Item = (u32,)>,
{
    <Indices as sealed::MIterDispatch<B>>::column_vec_inner::<u32>(indices, policy)?.ok_or_else(
        || Error::Launch {
            message: "gather indices must be backed by one u32 DeviceVec or DeviceSlice"
                .to_string(),
        },
    )
}

fn single_column_inner<B, Input, T>(
    policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
    input: &Input,
    message: &'static str,
) -> Result<crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>, Error>
where
    B: Backend,
    Input: MIter<B, Item = (T,)>,
    T: Scalar<B> + 'static,
{
    <Input as sealed::MIterDispatch<B>>::column_vec_inner::<T>(input, policy)?.ok_or_else(|| {
        Error::Launch {
            message: message.to_string(),
        }
    })
}

fn validate_input<B, Input>(exec: &Executor<B>, input: &Input) -> Result<(), Error>
where
    B: Backend,
    Input: MIter<B>,
{
    <Input as sealed::MIterDispatch<B>>::validate_executor(input, exec)
}

#[doc(hidden)]
pub struct StencilFlag;

#[cubecl::cube]
impl op::PredicateOp<(u32,)> for StencilFlag {
    fn apply(input: (u32,)) -> bool {
        input.0 > 0
    }
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
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                input: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    Input,
                >,
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
                    Runtime = <B as sealed::Backend>::Runtime,
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
                    >>::run(policy, input)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_binary<Left, Right, Op>(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                left: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    Left,
                >,
                right: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    Right,
                >,
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
                    Runtime = <B as sealed::Backend>::Runtime,
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
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_ternary<First, Second, Third, Op>(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                first: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    First,
                >,
                second: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    Second,
                >,
                third: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    Third,
                >,
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
                    Runtime = <B as sealed::Backend>::Runtime,
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
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
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

impl<'a, B, T> MIter<B> for (DeviceSlice<'a, B, T>,)
where
    B: Backend,
    T: Scalar<B> + 'static,
    (T,): StorageOutput<B, Inner = (crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>,)>,
{
    type Item = (T,);
    type Inner = (crate::detail::device::DeviceColumnView<<B as sealed::Backend>::Runtime, T>,);

    fn len(&self) -> usize {
        self.0.len()
    }

    fn into_inner(self) -> Self::Inner {
        (crate::detail::device::DeviceColumnView::from_slice(
            &self.0.source.inner,
            self.0.offset,
            self.0.len,
        ),)
    }
}

impl<'a, B, T> sealed::MIterDispatch<B> for (DeviceSlice<'a, B, T>,)
where
    B: Backend,
    T: Scalar<B> + 'static,
    (T,): StorageOutput<B, Inner = (crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>,)>,
{
    fn validate_executor(&self, exec: &Executor<B>) -> Result<(), Error> {
        exec.ensure_policy_id(self.0.source.inner.policy_id())
    }

    fn column_view_inner<U: 'static>(
        &self,
    ) -> Result<
        Option<crate::detail::device::DeviceColumnView<<B as sealed::Backend>::Runtime, U>>,
        Error,
    >
    where
        U: Scalar<B>,
    {
        let source = self.0.source as &dyn Any;
        let source = match source.downcast_ref::<DeviceVec<B, U>>() {
            Some(source) => source,
            None => return Ok(None),
        };
        Ok(Some(crate::detail::device::DeviceColumnView::from_slice(
            &source.inner,
            self.0.offset,
            self.0.len,
        )))
    }

    fn selection_stencil_dispatch<Pred>(
        &self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        invert: bool,
    ) -> Result<crate::detail::api::PrecomputedSelection<<B as sealed::Backend>::Runtime>, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        let stencil = self.into_inner();
        crate::detail::api::PrecomputedSelection::from_stencil_with_policy::<_, Pred>(
            policy, &stencil, invert,
        )
    }

    fn transform_dispatch<Op, Output, Y>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        op: Op,
    ) -> Result<Output, Error>
    where
        Op: op::UnaryOp<<Self as MIter<B>>::Item, Output = Y>,
        Y: StorageOutput<B>,
        Output: MVec<B, Item = Y>,
    {
        let input = self.into_inner().0;
        let inner = <Y as sealed::StorageOutputDispatch<B>>::transform_unary(policy, input, op)?;
        Ok(array_from_inner::<B, Y, Output>(inner))
    }

    fn reverse_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
    ) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::reverse(policy, self.into_inner())?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn sort_dispatch<Less, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        less: Less,
    ) -> Result<Output, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::sort(policy, self.into_inner(), less)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
            crate::detail::sort_by_key(policy, (keys,), self.into_inner(), less)?;
        Ok((
            array_from_inner::<B, (K,), KeyOutput>(key_inner),
            array_from_inner::<B, (T,), ValueOutput>(value_inner),
        ))
    }

    fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
            crate::detail::unique_by_key(policy, (keys,), self.into_inner(), eq)?;
        Ok((
            array_from_inner::<B, (K,), KeyOutput>(key_inner),
            array_from_inner::<B, (T,), ValueOutput>(value_inner),
        ))
    }

    fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
        let inner =
            crate::detail::inclusive_scan_by_key(policy, (keys,), self.into_inner(), key_eq, op)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn exclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
        let inner = crate::detail::exclusive_scan_by_key(
            policy,
            (keys,),
            self.into_inner(),
            key_eq,
            init,
            op,
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn reduce_by_single_key_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
            crate::detail::reduce_by_key(policy, (keys,), self.into_inner(), key_eq, init, op)?;
        Ok((
            array_from_inner::<B, (K,), KeyOutput>(key_inner),
            array_from_inner::<B, (T,), ValueOutput>(value_inner),
        ))
    }

    fn merge_by_single_key_same_dispatch<K, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
        let left_value = self.into_inner().0;
        let right_value = right_values.into_inner().0;
        let (key_inner, value_inner) = crate::detail::merge_by_key(
            policy,
            crate::detail::device::SoAView1 { source: left_keys },
            crate::detail::device::SoAView1 { source: left_value },
            crate::detail::device::SoAView1 { source: right_keys },
            crate::detail::device::SoAView1 {
                source: right_value,
            },
            crate::detail::api::Tuple1Less::<Less>::default(),
        )?;
        Ok((
            array_from_inner::<B, (K,), KeyOutput>(key_inner),
            array_from_inner::<B, (T,), ValueOutput>(value_inner),
        ))
    }

    fn gather_dispatch<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        indices: Indices,
    ) -> Result<Output, Error>
    where
        Indices: MIter<B, Item = (u32,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let indices = gather_index_inner::<B, Indices>(policy, &indices)?;
        let inner = crate::detail::gather(policy, self.into_inner(), (&indices,))?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn reduce_dispatch<Op>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<<Self as MIter<B>>::Item, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::reduce(policy, self.into_inner(), init, op)
    }

    fn inclusive_scan_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        op: Op,
    ) -> Result<Output, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::inclusive_scan(policy, self.into_inner(), op)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn exclusive_scan_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::exclusive_scan(policy, self.into_inner(), init, op)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn adjacent_difference_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        op: Op,
    ) -> Result<Output, Error>
    where
        Op: op::BinaryOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::adjacent_difference(policy, self.into_inner(), op)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn copy_if_dispatch<Stencil, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        stencil: Stencil,
    ) -> Result<Output, Error>
    where
        Stencil: MIter<B, Item = (u32,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let stencil = <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<
            StencilFlag,
        >(&stencil, policy, false)?;
        let inner = crate::detail::copy_if(policy, self.into_inner(), stencil, StencilFlag)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn remove_if_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::remove_if(policy, self.into_inner(), pred)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn count_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        pred: Pred,
    ) -> Result<usize, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::count_if(policy, self.into_inner(), pred)
    }

    fn all_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::all_of(policy, self.into_inner(), pred)
    }

    fn any_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::any_of(policy, self.into_inner(), pred)
    }

    fn none_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::none_of(policy, self.into_inner(), pred)
    }

    fn find_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        pred: Pred,
    ) -> Result<Option<usize>, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::find_if(policy, self.into_inner(), pred)
    }

    fn partition_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        pred: Pred,
    ) -> Result<(Output, Output), Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let (matching, failing) = crate::detail::partition(policy, self.into_inner(), pred)?;
        Ok((
            array_from_inner::<B, (T,), Output>(matching),
            array_from_inner::<B, (T,), Output>(failing),
        ))
    }

    fn is_partitioned_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::is_partitioned(policy, self.into_inner(), pred)
    }

    fn replace_if_dispatch<Stencil, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        replacement: <Self as MIter<B>>::Item,
        stencil: Stencil,
    ) -> Result<Output, Error>
    where
        Stencil: MIter<B, Item = (u32,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let stencil = <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<
            StencilFlag,
        >(&stencil, policy, false)?;
        let inner = crate::detail::replace_if(
            policy,
            self.into_inner(),
            replacement,
            stencil,
            StencilFlag,
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn unique_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::unique(policy, self.into_inner(), pred)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn min_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        less: Less,
    ) -> Result<Option<usize>, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::min_element(policy, self.into_inner(), less)
    }

    fn max_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        less: Less,
    ) -> Result<Option<usize>, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::max_element(policy, self.into_inner(), less)
    }

    fn minmax_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        less: Less,
    ) -> Result<Option<(usize, usize)>, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::minmax_element(policy, self.into_inner(), less)
    }

    fn adjacent_find_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        pred: Pred,
    ) -> Result<Option<usize>, Error>
    where
        Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::adjacent_find(policy, self.into_inner(), pred)
    }

    fn lower_bound_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        value: <Self as MIter<B>>::Item,
        less: Less,
    ) -> Result<usize, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::lower_bound(policy, self.into_inner(), value, less)
    }

    fn upper_bound_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        value: <Self as MIter<B>>::Item,
        less: Less,
    ) -> Result<usize, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::upper_bound(policy, self.into_inner(), value, less)
    }

    fn equal_range_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        value: <Self as MIter<B>>::Item,
        less: Less,
    ) -> Result<(usize, usize), Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::equal_range(policy, self.into_inner(), value, less)
    }

    fn is_sorted_until_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        less: Less,
    ) -> Result<usize, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::is_sorted_until(policy, self.into_inner(), less)
    }

    fn is_sorted_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        less: Less,
    ) -> Result<bool, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::is_sorted(policy, self.into_inner(), less)
    }

    fn gather_if_dispatch<Indices, Stencil, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        indices: Indices,
        default: <Self as MIter<B>>::Item,
        stencil: Stencil,
    ) -> Result<Output, Error>
    where
        Indices: MIter<B, Item = (u32,)>,
        Stencil: MIter<B, Item = (u32,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let indices = gather_index_inner::<B, Indices>(policy, &indices)?;
        let stencil = <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<
            StencilFlag,
        >(&stencil, policy, false)?;
        let inner = crate::detail::gather_if(
            policy,
            self.into_inner(),
            (&indices,),
            stencil,
            default,
            StencilFlag,
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn scatter_dispatch<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        indices: Indices,
        len: usize,
        default: <Self as MIter<B>>::Item,
    ) -> Result<Output, Error>
    where
        Indices: MIter<B, Item = (u32,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let indices = gather_index_inner::<B, Indices>(policy, &indices)?;
        let inner = crate::detail::scatter(policy, self.into_inner(), (&indices,), len, default.0)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn scatter_if_dispatch<Indices, Stencil, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        indices: Indices,
        len: usize,
        default: <Self as MIter<B>>::Item,
        stencil: Stencil,
    ) -> Result<Output, Error>
    where
        Indices: MIter<B, Item = (u32,)>,
        Stencil: MIter<B, Item = (u32,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let indices = gather_index_inner::<B, Indices>(policy, &indices)?;
        let stencil = <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<
            StencilFlag,
        >(&stencil, policy, false)?;
        let inner = crate::detail::scatter_if(
            policy,
            self.into_inner(),
            (&indices,),
            len,
            default.0,
            stencil,
            StencilFlag,
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn equal_same_dispatch<Eq>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        eq: Eq,
    ) -> Result<bool, Error>
    where
        Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::equal(policy, self.into_inner(), right.into_inner(), eq)
    }

    fn mismatch_same_dispatch<Eq>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::mismatch(policy, self.into_inner(), right.into_inner(), eq)
    }

    fn find_first_of_same_dispatch<Eq>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        needles: Self,
        eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::find_first_of(policy, self.into_inner(), needles.into_inner(), eq)
    }

    fn lexicographical_compare_same_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        less: Less,
    ) -> Result<bool, Error>
    where
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        crate::detail::lexicographical_compare(policy, self.into_inner(), right.into_inner(), less)
    }

    fn merge_same_dispatch<Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        less: Less,
    ) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::merge(policy, self.into_inner(), right.into_inner(), less)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn set_union_same_dispatch<Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        less: Less,
    ) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::set_union(policy, self.into_inner(), right.into_inner(), less)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn set_intersection_same_dispatch<Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        less: Less,
    ) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        let inner =
            crate::detail::set_intersection(policy, self.into_inner(), right.into_inner(), less)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn set_difference_same_dispatch<Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        less: Less,
    ) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
    {
        let inner =
            crate::detail::set_difference(policy, self.into_inner(), right.into_inner(), less)?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn inner_product_same_dispatch<TransformOp, ReduceOp>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        _transform_op: TransformOp,
        init: <Self as MIter<B>>::Item,
        _reduce_op: ReduceOp,
    ) -> Result<<Self as MIter<B>>::Item, Error>
    where
        TransformOp: op::BinaryOp<<Self as MIter<B>>::Item>,
        ReduceOp: op::BinaryOp<<Self as MIter<B>>::Item>,
    {
        let (left,) = self.into_inner();
        let (right,) = right.into_inner();
        let value = crate::detail::inner_product(
            policy,
            left,
            right,
            crate::detail::api::Tuple1BinaryOp::<TransformOp>::default(),
            init.0,
            crate::detail::api::Tuple1BinaryOp::<ReduceOp>::default(),
        )?;
        Ok((value,))
    }
}

macro_rules! impl_miter_view {
    ($input:ident; 0, 1) => {
        crate::detail::device::SoAView2 {
            left: $input.0,
            right: $input.1,
        }
    };

    ($input:ident; 0, 1, 2) => {
        crate::detail::device::SoAView3 {
            first: $input.0,
            second: $input.1,
            third: $input.2,
        }
    };
}

macro_rules! impl_miter_slice_tuple {
    ($( $ty:ident : $idx:tt : $tmp:ident ),+ => $transform:ident) => {
        impl<'a, B, $( $ty ),+> MIter<B> for ($( DeviceSlice<'a, B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar<B> + 'static, )+
            ($( $ty, )+): StorageOutput<
                B,
                Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( crate::detail::device::DeviceColumnView<<B as sealed::Backend>::Runtime, $ty>, )+);

            fn len(&self) -> usize {
                self.0.len()
            }

            fn into_inner(self) -> Self::Inner {
                ($(
                    crate::detail::device::DeviceColumnView::from_slice(
                        &self.$idx.source.inner,
                        self.$idx.offset,
                        self.$idx.len,
                    ),
                )+)
            }
        }

        impl<'a, B, $( $ty ),+> sealed::MIterDispatch<B> for ($( DeviceSlice<'a, B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar<B> + 'static, )+
            ($( $ty, )+): StorageOutput<
                B,
                Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+),
            >,
        {
            fn validate_executor(&self, exec: &Executor<B>) -> Result<(), Error> {
                $(
                    exec.ensure_policy_id(self.$idx.source.inner.policy_id())?;
                )+
                Ok(())
            }

            fn selection_stencil_dispatch<Pred>(
                &self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                invert: bool,
            ) -> Result<crate::detail::api::PrecomputedSelection<<B as sealed::Backend>::Runtime>, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                let stencil = self.into_inner();
                let stencil = impl_miter_view!(stencil; $( $idx ),+);
                crate::detail::api::PrecomputedSelection::from_stencil_with_policy::<_, Pred>(
                    policy,
                    &stencil,
                    invert,
                )
            }

            fn transform_dispatch<Op, Output, Y>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::UnaryOp<<Self as MIter<B>>::Item, Output = Y>,
                Y: StorageOutput<B>,
                Output: MVec<B, Item = Y>,
            {
                let input = self.into_inner();
                let inner = <Y as sealed::StorageOutputDispatch<B>>::$transform(
                    policy,
                    $( input.$idx, )+
                    op,
                )?;
                Ok(array_from_inner::<B, Y, Output>(inner))
            }

            fn reverse_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::reverse(policy, impl_miter_view!(input; $( $idx ),+))?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn sort_dispatch<Less, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                less: Less,
            ) -> Result<Output, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::sort(policy, impl_miter_view!(input; $( $idx ),+), less)?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) = crate::detail::sort_by_key(policy, (keys,), (values,), less)?;
                Ok((
                    array_from_inner::<B, (K,), KeyOutput>(key_inner),
                    array_from_inner::<B, ($( $ty, )+), ValueOutput>(value_inner),
                ))
            }

            fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) =
                    crate::detail::unique_by_key(policy, (keys,), (values,), eq)?;
                Ok((
                    array_from_inner::<B, (K,), KeyOutput>(key_inner),
                    array_from_inner::<B, ($( $ty, )+), ValueOutput>(value_inner),
                ))
            }

            fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
                let values = impl_miter_view!(values; $( $idx ),+);
                let inner = crate::detail::inclusive_scan_by_key(
                    policy,
                    keys,
                    values,
                    crate::detail::api::Tuple1Less::<KeyEq>::default(),
                    op,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn exclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
                let values = impl_miter_view!(values; $( $idx ),+);
                let inner = crate::detail::exclusive_scan_by_key(
                    policy,
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
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) = crate::detail::reduce_by_key(
                    policy,
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
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    crate::detail::device::SoAView1 { source: left_keys },
                    impl_miter_view!(left_values; $( $idx ),+),
                    crate::detail::device::SoAView1 { source: right_keys },
                    impl_miter_view!(right_values; $( $idx ),+),
                    crate::detail::api::Tuple1Less::<Less>::default(),
                )?;
                Ok((
                    array_from_inner::<B, (K,), KeyOutput>(key_inner),
                    array_from_inner::<B, ($( $ty, )+), ValueOutput>(value_inner),
                ))
            }

            fn gather_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                indices: Indices,
            ) -> Result<Output, Error>
            where
                Indices: MIter<B, Item = (u32,)>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let indices = gather_index_inner::<B, Indices>(policy, &indices)?;
                let input = self.into_inner();
                let inner = crate::detail::gather(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    &indices,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn reduce_dispatch<Op>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                init: <Self as MIter<B>>::Item,
                op: Op,
            ) -> Result<<Self as MIter<B>>::Item, Error>
            where
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::reduce(policy, impl_miter_view!(input; $( $idx ),+), init, op)
            }

            fn inclusive_scan_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::inclusive_scan(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    op,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn exclusive_scan_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                init: <Self as MIter<B>>::Item,
                op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::exclusive_scan(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    init,
                    op,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn adjacent_difference_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::BinaryOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::adjacent_difference(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    op,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn copy_if_dispatch<Stencil, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                stencil: Stencil,
            ) -> Result<Output, Error>
            where
                Stencil: MIter<B, Item = (u32,)>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let stencil =
                    <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<StencilFlag>(
                        &stencil, policy, false,
                    )?;
                let input = self.into_inner();
                let inner = crate::detail::copy_if(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    stencil,
                    StencilFlag,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn remove_if_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                pred: Pred,
            ) -> Result<Output, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::remove_if(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    pred,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn count_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                pred: Pred,
            ) -> Result<usize, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::count_if(policy, impl_miter_view!(input; $( $idx ),+), pred)
            }

            fn all_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::all_of(policy, impl_miter_view!(input; $( $idx ),+), pred)
            }

            fn any_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::any_of(policy, impl_miter_view!(input; $( $idx ),+), pred)
            }

            fn none_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::none_of(policy, impl_miter_view!(input; $( $idx ),+), pred)
            }

            fn find_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                pred: Pred,
            ) -> Result<Option<usize>, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::find_if(policy, impl_miter_view!(input; $( $idx ),+), pred)
            }

            fn partition_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                pred: Pred,
            ) -> Result<(Output, Output), Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let (matching, failing) = crate::detail::partition(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    pred,
                )?;
                Ok((
                    array_from_inner::<B, ($( $ty, )+), Output>(matching),
                    array_from_inner::<B, ($( $ty, )+), Output>(failing),
                ))
            }

            fn is_partitioned_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::is_partitioned(policy, impl_miter_view!(input; $( $idx ),+), pred)
            }

            fn replace_if_dispatch<Stencil, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                replacement: <Self as MIter<B>>::Item,
                stencil: Stencil,
            ) -> Result<Output, Error>
            where
                Stencil: MIter<B, Item = (u32,)>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let stencil =
                    <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<StencilFlag>(
                        &stencil, policy, false,
                    )?;
                let input = self.into_inner();
                let inner = crate::detail::replace_if(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    replacement,
                    stencil,
                    StencilFlag,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn unique_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                pred: Pred,
            ) -> Result<Output, Error>
            where
                Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::unique(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    pred,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn min_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                less: Less,
            ) -> Result<Option<usize>, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::min_element(policy, impl_miter_view!(input; $( $idx ),+), less)
            }

            fn max_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                less: Less,
            ) -> Result<Option<usize>, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::max_element(policy, impl_miter_view!(input; $( $idx ),+), less)
            }

            fn minmax_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                less: Less,
            ) -> Result<Option<(usize, usize)>, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::minmax_element(policy, impl_miter_view!(input; $( $idx ),+), less)
            }

            fn adjacent_find_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                pred: Pred,
            ) -> Result<Option<usize>, Error>
            where
                Pred: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::adjacent_find(policy, impl_miter_view!(input; $( $idx ),+), pred)
            }

            fn lower_bound_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                value: <Self as MIter<B>>::Item,
                less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::lower_bound(policy, impl_miter_view!(input; $( $idx ),+), value, less)
            }

            fn upper_bound_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                value: <Self as MIter<B>>::Item,
                less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::upper_bound(policy, impl_miter_view!(input; $( $idx ),+), value, less)
            }

            fn equal_range_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                value: <Self as MIter<B>>::Item,
                less: Less,
            ) -> Result<(usize, usize), Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::equal_range(policy, impl_miter_view!(input; $( $idx ),+), value, less)
            }

            fn is_sorted_until_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::is_sorted_until(policy, impl_miter_view!(input; $( $idx ),+), less)
            }

            fn is_sorted_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                less: Less,
            ) -> Result<bool, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::is_sorted(policy, impl_miter_view!(input; $( $idx ),+), less)
            }

            fn gather_if_dispatch<Indices, Stencil, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                indices: Indices,
                default: <Self as MIter<B>>::Item,
                stencil: Stencil,
            ) -> Result<Output, Error>
            where
                Indices: MIter<B, Item = (u32,)>,
                Stencil: MIter<B, Item = (u32,)>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let indices = gather_index_inner::<B, Indices>(policy, &indices)?;
                let stencil =
                    <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<StencilFlag>(
                        &stencil, policy, false,
                    )?;
                let input = self.into_inner();
                let inner = crate::detail::gather_if(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    &indices,
                    stencil,
                    default,
                    StencilFlag,
                )?;
                Ok(array_from_inner::<B, <Self as MIter<B>>::Item, Output>(inner))
            }

            fn scatter_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                indices: Indices,
                len: usize,
                default: <Self as MIter<B>>::Item,
            ) -> Result<Output, Error>
            where
                Indices: MIter<B, Item = (u32,)>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let indices = gather_index_inner::<B, Indices>(policy, &indices)?;
                let input = self.into_inner();
                let inner = crate::detail::scatter(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    &indices,
                    len,
                    default,
                )?;
                Ok(array_from_inner::<B, <Self as MIter<B>>::Item, Output>(inner))
            }

            fn scatter_if_dispatch<Indices, Stencil, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                indices: Indices,
                len: usize,
                default: <Self as MIter<B>>::Item,
                stencil: Stencil,
            ) -> Result<Output, Error>
            where
                Indices: MIter<B, Item = (u32,)>,
                Stencil: MIter<B, Item = (u32,)>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let indices = gather_index_inner::<B, Indices>(policy, &indices)?;
                let stencil =
                    <Stencil as sealed::MIterDispatch<B>>::selection_stencil_dispatch::<StencilFlag>(
                        &stencil, policy, false,
                    )?;
                let input = self.into_inner();
                let inner = crate::detail::scatter_if(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    &indices,
                    len,
                    default,
                    stencil,
                    StencilFlag,
                )?;
                Ok(array_from_inner::<B, <Self as MIter<B>>::Item, Output>(inner))
            }

            fn equal_same_dispatch<Eq>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                eq: Eq,
            ) -> Result<bool, Error>
            where
                Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::equal(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    eq,
                )
            }

            fn mismatch_same_dispatch<Eq>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::mismatch(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    eq,
                )
            }

            fn find_first_of_same_dispatch<Eq>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                needles: Self,
                eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Eq: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let needles = needles.into_inner();
                crate::detail::find_first_of(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    impl_miter_view!(needles; $( $idx ),+),
                    eq,
                )
            }

            fn lexicographical_compare_same_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                less: Less,
            ) -> Result<bool, Error>
            where
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::lexicographical_compare(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    less,
                )
            }

            fn merge_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                let inner = crate::detail::merge(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    less,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn set_union_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                let inner = crate::detail::set_union(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    less,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn set_intersection_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                let inner = crate::detail::set_intersection(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    less,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn set_difference_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<<Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                let inner = crate::detail::set_difference(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    less,
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn inner_product_same_dispatch<TransformOp, ReduceOp>(
                self,
                _policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
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
}

impl_miter_slice_tuple!(A: 0: a, C: 1: c => transform_binary);
impl_miter_slice_tuple!(A: 0: a, C: 1: c, D: 2: d => transform_ternary);

/// Applies a unary transform to a massively iterator.
pub fn transform<B, Input, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    op: Op,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B>,
    Op: op::UnaryOp<Input::Item, Output = Output::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::transform_dispatch(source, exec.policy(), op)
}

/// Reverses a massively iterator into an owned vector.
pub fn reverse<B, Input, Output>(exec: &Executor<B>, source: Input) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::reverse_dispatch(source, exec.policy())
}

/// Sorts a massively iterator into an owned vector.
pub fn sort<B, Input, Output, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::sort_dispatch(source, exec.policy(), less)
}

/// Gathers a massively iterator at index positions into an owned vector.
pub fn gather<B, Input, Indices, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: Indices,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Indices: MIter<B, Item = (u32,)>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &indices)?;
    <Input as sealed::MIterDispatch<B>>::gather_dispatch(source, exec.policy(), indices)
}

/// Reduces a massively iterator to one host item.
pub fn reduce<B, Input, Op>(
    exec: &Executor<B>,
    source: Input,
    init: Input::Item,
    op: Op,
) -> Result<Input::Item, Error>
where
    B: Backend,
    Input: MIter<B>,
    Op: op::BinaryOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::reduce_dispatch(source, exec.policy(), init, op)
}

/// Computes an inclusive scan.
pub fn inclusive_scan<B, Input, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    op: Op,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::BinaryOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::inclusive_scan_dispatch(source, exec.policy(), op)
}

/// Computes an exclusive scan.
pub fn exclusive_scan<B, Input, Output, Op>(
    exec: &Executor<B>,
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
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::exclusive_scan_dispatch(source, exec.policy(), init, op)
}

/// Computes adjacent differences.
pub fn adjacent_difference<B, Input, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    op: Op,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::BinaryOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::adjacent_difference_dispatch(source, exec.policy(), op)
}

/// Copies elements whose one-column `u32` stencil flag is non-zero.
///
/// The stencil must be an `MIter` whose item is `(u32,)`. For predicate-based
/// filtering over the input values themselves, use [`remove_if`] or
/// [`partition`].
pub fn copy_if<B, Input, Stencil, Output>(
    exec: &Executor<B>,
    source: Input,
    stencil: Stencil,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Stencil: MIter<B, Item = (u32,)>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &stencil)?;
    <Input as sealed::MIterDispatch<B>>::copy_if_dispatch(source, exec.policy(), stencil)
}

/// Removes elements satisfying `pred`.
pub fn remove_if<B, Input, Output, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::remove_if_dispatch(source, exec.policy(), pred)
}

/// Counts elements satisfying `pred`.
pub fn count_if<B, Input, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<usize, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::count_if_dispatch(source, exec.policy(), pred)
}

/// Returns whether all elements satisfy `pred`.
pub fn all_of<B, Input, Pred>(exec: &Executor<B>, source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::all_of_dispatch(source, exec.policy(), pred)
}

/// Returns whether any element satisfies `pred`.
pub fn any_of<B, Input, Pred>(exec: &Executor<B>, source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::any_of_dispatch(source, exec.policy(), pred)
}

/// Returns whether no elements satisfy `pred`.
pub fn none_of<B, Input, Pred>(exec: &Executor<B>, source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::none_of_dispatch(source, exec.policy(), pred)
}

/// Finds the first element satisfying `pred`.
pub fn find_if<B, Input, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::find_if_dispatch(source, exec.policy(), pred)
}

/// Partitions elements by `pred`.
pub fn partition<B, Input, Output, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<(Output, Output), Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::partition_dispatch(source, exec.policy(), pred)
}

/// Returns whether input is partitioned by `pred`.
pub fn is_partitioned<B, Input, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::is_partitioned_dispatch(source, exec.policy(), pred)
}

/// Replaces elements whose one-column `u32` stencil flag is non-zero.
pub fn replace_if<B, Input, Stencil, Output>(
    exec: &Executor<B>,
    source: Input,
    replacement: Input::Item,
    stencil: Stencil,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Stencil: MIter<B, Item = (u32,)>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &stencil)?;
    <Input as sealed::MIterDispatch<B>>::replace_if_dispatch(
        source,
        exec.policy(),
        replacement,
        stencil,
    )
}

/// Removes consecutive duplicates under `pred`.
pub fn unique<B, Input, Output, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::unique_dispatch(source, exec.policy(), pred)
}

/// Stable sort. The current lower implementation is stable.
pub fn stable_sort<B, Input, Output, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    sort(exec, source, less)
}

/// Gathers elements whose one-column `u32` stencil flag is non-zero.
pub fn gather_if<B, Input, Indices, Stencil, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: Indices,
    default: Input::Item,
    stencil: Stencil,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Indices: MIter<B, Item = (u32,)>,
    Stencil: MIter<B, Item = (u32,)>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &indices)?;
    validate_input(exec, &stencil)?;
    <Input as sealed::MIterDispatch<B>>::gather_if_dispatch(
        source,
        exec.policy(),
        indices,
        default,
        stencil,
    )
}

/// Scatters values into a newly allocated output.
pub fn scatter<B, Input, Indices, Output>(
    exec: &Executor<B>,
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
    validate_input(exec, &source)?;
    validate_input(exec, &indices)?;
    <Input as sealed::MIterDispatch<B>>::scatter_dispatch(
        source,
        exec.policy(),
        indices,
        len,
        default,
    )
}

/// Scatters values whose one-column `u32` stencil flag is non-zero into a newly allocated output.
pub fn scatter_if<B, Input, Indices, Stencil, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: Indices,
    len: usize,
    default: Input::Item,
    stencil: Stencil,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Indices: MIter<B, Item = (u32,)>,
    Stencil: MIter<B, Item = (u32,)>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &indices)?;
    validate_input(exec, &stencil)?;
    <Input as sealed::MIterDispatch<B>>::scatter_if_dispatch(
        source,
        exec.policy(),
        indices,
        len,
        default,
        stencil,
    )
}

/// Finds the first adjacent pair satisfying `pred`.
pub fn adjacent_find<B, Input, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::adjacent_find_dispatch(source, exec.policy(), pred)
}

/// Returns whether two inputs are equal under `eq`.
pub fn equal<B, Input, Eq>(
    exec: &Executor<B>,
    left: Input,
    right: Input,
    eq: Eq,
) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Eq: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Input as sealed::MIterDispatch<B>>::equal_same_dispatch(left, exec.policy(), right, eq)
}

/// Finds the first mismatch between two inputs.
pub fn mismatch<B, Input, Eq>(
    exec: &Executor<B>,
    left: Input,
    right: Input,
    eq: Eq,
) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Eq: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Input as sealed::MIterDispatch<B>>::mismatch_same_dispatch(left, exec.policy(), right, eq)
}

/// Finds the first input element equal to any needle.
pub fn find_first_of<B, Input, Eq>(
    exec: &Executor<B>,
    source: Input,
    needles: Input,
    eq: Eq,
) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Eq: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &needles)?;
    <Input as sealed::MIterDispatch<B>>::find_first_of_same_dispatch(
        source,
        exec.policy(),
        needles,
        eq,
    )
}

/// Finds the minimum element index.
pub fn min_element<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::min_element_dispatch(source, exec.policy(), less)
}

/// Finds the maximum element index.
pub fn max_element<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::max_element_dispatch(source, exec.policy(), less)
}

/// Finds both minimum and maximum element indices.
pub fn minmax_element<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<Option<(usize, usize)>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::minmax_element_dispatch(source, exec.policy(), less)
}

/// Finds the lower bound of `value` in a sorted input.
pub fn lower_bound<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    value: Input::Item,
    less: Less,
) -> Result<usize, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::lower_bound_dispatch(source, exec.policy(), value, less)
}

/// Finds the upper bound of `value` in a sorted input.
pub fn upper_bound<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    value: Input::Item,
    less: Less,
) -> Result<usize, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::upper_bound_dispatch(source, exec.policy(), value, less)
}

/// Finds the equal range of `value` in a sorted input.
pub fn equal_range<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    value: Input::Item,
    less: Less,
) -> Result<(usize, usize), Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::equal_range_dispatch(source, exec.policy(), value, less)
}

/// Returns the first position where sorted order is broken.
pub fn is_sorted_until<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<usize, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::is_sorted_until_dispatch(source, exec.policy(), less)
}

/// Returns whether input is sorted.
pub fn is_sorted<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::is_sorted_dispatch(source, exec.policy(), less)
}

/// Lexicographically compares two inputs.
pub fn lexicographical_compare<B, Input, Less>(
    exec: &Executor<B>,
    left: Input,
    right: Input,
    less: Less,
) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<Input::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Input as sealed::MIterDispatch<B>>::lexicographical_compare_same_dispatch(
        left,
        exec.policy(),
        right,
        less,
    )
}

/// Merges two sorted inputs.
pub fn merge<B, Input, Output, Less>(
    exec: &Executor<B>,
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
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Input as sealed::MIterDispatch<B>>::merge_same_dispatch(left, exec.policy(), right, less)
}

/// Computes the sorted set union of two sorted inputs.
pub fn set_union<B, Input, Output, Less>(
    exec: &Executor<B>,
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
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Input as sealed::MIterDispatch<B>>::set_union_same_dispatch(left, exec.policy(), right, less)
}

/// Computes the sorted set intersection of two sorted inputs.
pub fn set_intersection<B, Input, Output, Less>(
    exec: &Executor<B>,
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
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Input as sealed::MIterDispatch<B>>::set_intersection_same_dispatch(
        left,
        exec.policy(),
        right,
        less,
    )
}

/// Computes the sorted set difference of two sorted inputs.
pub fn set_difference<B, Input, Output, Less>(
    exec: &Executor<B>,
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
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Input as sealed::MIterDispatch<B>>::set_difference_same_dispatch(
        left,
        exec.policy(),
        right,
        less,
    )
}

/// Applies a binary transform over two inputs and reduces the result.
pub fn inner_product<B, Input, TransformOp, ReduceOp>(
    exec: &Executor<B>,
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
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Input as sealed::MIterDispatch<B>>::inner_product_same_dispatch(
        left,
        exec.policy(),
        right,
        transform_op,
        init,
        reduce_op,
    )
}

/// Inclusive scan by key.
pub fn inclusive_scan_by_key<B, Keys, Values, K, KeyEq, Op, Output>(
    exec: &Executor<B>,
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
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    let keys = single_column_inner::<B, Keys, K>(
        exec.policy(),
        &keys,
        "inclusive_scan_by_key keys must be backed by one DeviceVec or DeviceSlice",
    )?;
    <Values as sealed::MIterDispatch<B>>::inclusive_scan_by_single_key_dispatch(
        values,
        exec.policy(),
        &keys,
        key_eq,
        op,
    )
}

/// Exclusive scan by key.
pub fn exclusive_scan_by_key<B, Keys, Values, K, KeyEq, Op, Output>(
    exec: &Executor<B>,
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
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    let keys = single_column_inner::<B, Keys, K>(
        exec.policy(),
        &keys,
        "exclusive_scan_by_key keys must be backed by one DeviceVec or DeviceSlice",
    )?;
    <Values as sealed::MIterDispatch<B>>::exclusive_scan_by_single_key_dispatch(
        values,
        exec.policy(),
        &keys,
        key_eq,
        init,
        op,
    )
}

/// Reduces consecutive values with equal keys.
pub fn reduce_by_key<B, Keys, Values, K, KeyEq, Op, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
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
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    let keys = single_column_inner::<B, Keys, K>(
        exec.policy(),
        &keys,
        "reduce_by_key keys must be backed by one DeviceVec or DeviceSlice",
    )?;
    <Values as sealed::MIterDispatch<B>>::reduce_by_single_key_dispatch(
        values,
        exec.policy(),
        &keys,
        key_eq,
        init,
        op,
    )
}

/// Removes consecutive duplicate keys and keeps their values.
pub fn unique_by_key<B, Keys, Values, K, Eq, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
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
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    let keys = single_column_inner::<B, Keys, K>(
        exec.policy(),
        &keys,
        "unique_by_key keys must be backed by one DeviceVec or DeviceSlice",
    )?;
    <Values as sealed::MIterDispatch<B>>::unique_by_single_key_dispatch(
        values,
        exec.policy(),
        &keys,
        eq,
    )
}

/// Sorts key-value pairs by key.
pub fn sort_by_key<B, Keys, Values, K, Less, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
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
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    let keys = single_column_inner::<B, Keys, K>(
        exec.policy(),
        &keys,
        "sort_by_key keys must be backed by one DeviceVec or DeviceSlice",
    )?;
    <Values as sealed::MIterDispatch<B>>::sort_by_single_key_dispatch(
        values,
        exec.policy(),
        &keys,
        less,
    )
}

/// Stable key-value sort. The current lower implementation is stable.
pub fn stable_sort_by_key<B, Keys, Values, K, Less, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
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
    sort_by_key(exec, keys, values, less)
}

/// Merges two sorted key-value ranges by key.
pub fn merge_by_key<B, LeftKeys, Values, RightKeys, K, Less, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
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
    validate_input(exec, &left_keys)?;
    validate_input(exec, &left_values)?;
    validate_input(exec, &right_keys)?;
    validate_input(exec, &right_values)?;
    let left_keys = single_column_inner::<B, LeftKeys, K>(
        exec.policy(),
        &left_keys,
        "merge_by_key left keys must be backed by one DeviceVec or DeviceSlice",
    )?;
    let right_keys = single_column_inner::<B, RightKeys, K>(
        exec.policy(),
        &right_keys,
        "merge_by_key right keys must be backed by one DeviceVec or DeviceSlice",
    )?;
    <Values as sealed::MIterDispatch<B>>::merge_by_single_key_same_dispatch(
        left_values,
        exec.policy(),
        &left_keys,
        &right_keys,
        right_values,
        less,
    )
}
