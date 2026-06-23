//! Public algorithm API implementation for `massively`.
//!
//! This crate intentionally keeps CubeCL runtime types out of public algorithm
//! signatures. The implementation delegates to the internal detail layer.

use std::any::Any;
use std::marker::PhantomData;

use cubecl::frontend::PartialOrdExpand;

use crate::algorithm::op;
use crate::algorithm::{MItem, MIter, MVec, SoA1, SoA2, SoA3};
use crate::runtime::{Backend, DeviceSlice, DeviceVec, Executor, Scalar};

pub use crate::Error;

pub(crate) mod sealed {
    use super::{Error, Executor, MIter, MVec, op};

    pub trait Backend {
        type Runtime: cubecl::prelude::Runtime;
    }

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
            T: super::Scalar,
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
            T: super::Scalar,
        {
            Ok(self
                .column_inner::<T>()
                .map(crate::detail::device::DeviceColumnView::from_column))
        }

        fn column_view_by_index_inner<T: 'static>(
            &self,
            index: usize,
        ) -> Result<
            Option<crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, T>>,
            Error,
        >
        where
            T: super::Scalar,
        {
            if index == 0 {
                self.column_view_inner::<T>()
            } else {
                Ok(None)
            }
        }

        fn transform_dispatch<Op, Output, Y>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::UnaryOp<B, <Self as MIter<B>>::Item, Output = Y>,
            Y: super::MItem<B>,
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
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            keys: &crate::detail::DeviceVec<<B as Backend>::Runtime, K>,
            _less: Less,
        ) -> Result<(KeyOutput, ValueOutput), Error>
        where
            Self: MIter<B>,
            K: super::Scalar + 'static,
            Less: op::PredicateOp2<B, (K,)>,
            KeyOutput: MVec<B, Item = (K,)>,
            ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn sort_by_key_dispatch<Values, Less, KeyOutput, ValueOutput>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _values: Values,
            _less: Less,
        ) -> Result<(KeyOutput, ValueOutput), Error>
        where
            Self: MIter<B>,
            Values: MIter<B>,
            <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            KeyOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            ValueOutput: MVec<B, Item = <Values as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "sort_by_key is not supported for this key iterator shape".to_string(),
            })
        }

        fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            keys: &crate::detail::DeviceVec<<B as Backend>::Runtime, K>,
            _eq: Eq,
        ) -> Result<(KeyOutput, ValueOutput), Error>
        where
            Self: MIter<B>,
            K: super::Scalar + 'static,
            Eq: op::PredicateOp2<B, (K,)>,
            KeyOutput: MVec<B, Item = (K,)>,
            ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn unique_by_key_dispatch<Values, Eq, KeyOutput, ValueOutput>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _values: Values,
            _eq: Eq,
        ) -> Result<(KeyOutput, ValueOutput), Error>
        where
            Self: MIter<B>,
            Values: MIter<B>,
            <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
            Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            KeyOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            ValueOutput: MVec<B, Item = <Values as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "unique_by_key is not supported for this key iterator shape".to_string(),
            })
        }

        fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            keys: &crate::detail::DeviceVec<<B as Backend>::Runtime, K>,
            key_eq: KeyEq,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            K: super::Scalar + 'static,
            KeyEq: op::PredicateOp2<B, (K,)>,
            Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn inclusive_scan_by_key_dispatch<Values, KeyEq, Op, Output>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _values: Values,
            _key_eq: KeyEq,
            _op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Values: MIter<B>,
            <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
            KeyEq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            Op: op::BinaryOp1<B, <Values as MIter<B>>::Item>,
            Output: MVec<B, Item = <Values as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "inclusive_scan_by_key is not supported for this key iterator shape"
                    .to_string(),
            })
        }

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
            K: super::Scalar + 'static,
            KeyEq: op::PredicateOp2<B, (K,)>,
            Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn exclusive_scan_by_key_dispatch<Values, KeyEq, Op, Output>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _values: Values,
            _key_eq: KeyEq,
            _init: <Values as MIter<B>>::Item,
            _op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Values: MIter<B>,
            <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
            KeyEq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            Op: op::BinaryOp1<B, <Values as MIter<B>>::Item>,
            Output: MVec<B, Item = <Values as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "exclusive_scan_by_key is not supported for this key iterator shape"
                    .to_string(),
            })
        }

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
            K: super::Scalar + 'static,
            KeyEq: op::PredicateOp2<B, (K,)>,
            Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
            KeyOutput: MVec<B, Item = (K,)>,
            ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn reduce_by_key_dispatch<Values, KeyEq, Op, KeyOutput, ValueOutput>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _values: Values,
            _key_eq: KeyEq,
            _init: <Values as MIter<B>>::Item,
            _op: Op,
        ) -> Result<(KeyOutput, ValueOutput), Error>
        where
            Self: MIter<B>,
            Values: MIter<B>,
            <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
            KeyEq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            Op: op::BinaryOp1<B, <Values as MIter<B>>::Item>,
            KeyOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            ValueOutput: MVec<B, Item = <Values as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "reduce_by_key is not supported for this key iterator shape".to_string(),
            })
        }

        fn merge_by_single_key_same_dispatch<K, RightValues, Less, KeyOutput, ValueOutput>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            left_keys: &crate::detail::DeviceVec<<B as Backend>::Runtime, K>,
            right_keys: &crate::detail::DeviceVec<<B as Backend>::Runtime, K>,
            _right_values: RightValues,
            _less: Less,
        ) -> Result<(KeyOutput, ValueOutput), Error>
        where
            Self: MIter<B>,
            RightValues: MIter<B, Item = <Self as MIter<B>>::Item>,
            K: super::Scalar + 'static,
            Less: op::PredicateOp2<B, (K,)>,
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
            Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>;

        fn inclusive_scan_dispatch<Op, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn exclusive_scan_dispatch<Op, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _init: <Self as MIter<B>>::Item,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn adjacent_difference_dispatch<Op, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            op: Op,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
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
            Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn count_if_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<usize, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>;

        fn all_of_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>;

        fn any_of_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>;

        fn none_of_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>;

        fn find_if_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>;

        fn partition_dispatch<Pred, Output>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<(Output, Output), Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn is_partitioned_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>;

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
            Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
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
            Pred: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>;

        fn min_element_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            less: Less,
        ) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>;

        fn max_element_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            less: Less,
        ) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>;

        fn minmax_element_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            less: Less,
        ) -> Result<Option<(usize, usize)>, Error>
        where
            Self: MIter<B>,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>;

        fn adjacent_find_dispatch<Pred>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            pred: Pred,
        ) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Pred: op::PredicateOp2<B, <Self as MIter<B>>::Item>;

        fn equal_dispatch<Right, Eq>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _right: Right,
            _eq: Eq,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "equal is not supported for this iterator shape".to_string(),
            })
        }

        fn mismatch_dispatch<Right, Eq>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _right: Right,
            _eq: Eq,
        ) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "mismatch is not supported for this iterator shape".to_string(),
            })
        }

        fn find_first_of_dispatch<Needles, Eq>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _needles: Needles,
            _eq: Eq,
        ) -> Result<Option<usize>, Error>
        where
            Self: MIter<B>,
            Needles: MIter<B, Item = <Self as MIter<B>>::Item>,
            Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
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
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>;

        fn upper_bound_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            value: <Self as MIter<B>>::Item,
            _less: Less,
        ) -> Result<usize, Error>
        where
            Self: MIter<B>,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>;

        fn equal_range_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            value: <Self as MIter<B>>::Item,
            _less: Less,
        ) -> Result<(usize, usize), Error>
        where
            Self: MIter<B>,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>;

        fn is_sorted_until_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            less: Less,
        ) -> Result<usize, Error>
        where
            Self: MIter<B>,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>;

        fn is_sorted_dispatch<Less>(
            self,
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            less: Less,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>;

        fn lexicographical_compare_dispatch<Right, Less>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _right: Right,
            _less: Less,
        ) -> Result<bool, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "lexicographical_compare is not supported for this iterator shape"
                    .to_string(),
            })
        }

        fn merge_dispatch<Right, Output, Less>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _right: Right,
            _less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "merge is not supported for this iterator shape".to_string(),
            })
        }

        fn set_union_dispatch<Right, Output, Less>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _right: Right,
            _less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "set_union is not supported for this iterator shape".to_string(),
            })
        }

        fn set_intersection_dispatch<Right, Output, Less>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _right: Right,
            _less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "set_intersection is not supported for this iterator shape".to_string(),
            })
        }

        fn set_difference_dispatch<Right, Output, Less>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _right: Right,
            _less: Less,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Right: MIter<B, Item = <Self as MIter<B>>::Item>,
            Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "set_difference is not supported for this iterator shape".to_string(),
            })
        }

        fn inner_product_dispatch<Right, TransformOp, ReduceOp, Output>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _right: Right,
            _transform_op: TransformOp,
            _init: Output,
            _reduce_op: ReduceOp,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            Right: MIter<B>,
            TransformOp: op::BinaryOp2<
                    B,
                    <Self as MIter<B>>::Item,
                    <Right as MIter<B>>::Item,
                    Output = Output,
                >,
            Output: super::MItem<B>,
            ReduceOp: op::BinaryOp1<B, Output>,
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
            Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
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
            Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
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
            Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
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
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
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
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
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
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
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
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
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
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "set_difference is not supported for this iterator shape".to_string(),
            })
        }

        fn inner_product_same_dispatch<TransformOp, ReduceOp, Output>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _right: Self,
            _transform_op: TransformOp,
            _init: Output,
            _reduce_op: ReduceOp,
        ) -> Result<Output, Error>
        where
            Self: MIter<B>,
            TransformOp: op::BinaryOp2<
                    B,
                    <Self as MIter<B>>::Item,
                    <Self as MIter<B>>::Item,
                    Output = Output,
                >,
            Output: super::MItem<B>,
            ReduceOp: op::BinaryOp1<B, Output>,
        {
            Err(Error::Launch {
                message: "inner_product is not supported for this iterator shape".to_string(),
            })
        }

        fn merge_by_key_dispatch<RightKeys, LeftValues, RightValues, Less, KeyOutput, ValueOutput>(
            self,
            _policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            _right_keys: RightKeys,
            _left_values: LeftValues,
            _right_values: RightValues,
            _less: Less,
        ) -> Result<(KeyOutput, ValueOutput), Error>
        where
            Self: MIter<B>,
            RightKeys: MIter<B, Item = <Self as MIter<B>>::Item>,
            LeftValues: MIter<B>,
            RightValues: MIter<B, Item = <LeftValues as MIter<B>>::Item>,
            <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
            Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            KeyOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            ValueOutput: MVec<B, Item = <LeftValues as MIter<B>>::Item>,
        {
            Err(Error::Launch {
                message: "merge_by_key is not supported for this key iterator shape".to_string(),
            })
        }
    }

    pub trait MItemDispatch<B: super::Backend>: Sized {
        fn transform_unary<Input, Op>(
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            input: crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, Input>,
            op: Op,
        ) -> Result<<Self as super::MItem<B>>::Inner, Error>
        where
            Self: super::MItem<B>,
            Input: super::Scalar,
            Op: op::UnaryOp<B, (Input,), Output = Self>,
        {
            let _ = (policy, input, op);
            Err(Error::Launch {
                message: "transform is not supported for this output item shape".to_string(),
            })
        }

        fn transform_binary<Left, Right, Op>(
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            left: crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, Left>,
            right: crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, Right>,
            op: Op,
        ) -> Result<<Self as super::MItem<B>>::Inner, Error>
        where
            Self: super::MItem<B>,
            Left: super::Scalar,
            Right: super::Scalar,
            Op: op::UnaryOp<B, (Left, Right), Output = Self>,
        {
            let _ = (policy, left, right, op);
            Err(Error::Launch {
                message: "transform is not supported for this output item shape".to_string(),
            })
        }

        fn transform_ternary<First, Second, Third, Op>(
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            first: crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, First>,
            second: crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, Second>,
            third: crate::detail::device::DeviceColumnView<<B as Backend>::Runtime, Third>,
            op: Op,
        ) -> Result<<Self as super::MItem<B>>::Inner, Error>
        where
            Self: super::MItem<B>,
            First: super::Scalar,
            Second: super::Scalar,
            Third: super::Scalar,
            Op: op::UnaryOp<B, (First, Second, Third), Output = Self>,
        {
            let _ = (policy, first, second, third, op);
            Err(Error::Launch {
                message: "transform is not supported for this output item shape".to_string(),
            })
        }

        fn reduce_inner<Op>(
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            input: <Self as super::MItem<B>>::Inner,
            init: Self,
            op: Op,
        ) -> Result<Self, Error>
        where
            Self: super::MItem<B>,
            Op: op::BinaryOp1<B, Self>,
        {
            let _ = (policy, input, init, op);
            Err(Error::Launch {
                message: "reduce is not supported for this item shape".to_string(),
            })
        }

        fn inner_product_with_right_item<LeftIter, RightIter, TransformOp, ReduceOp, Output>(
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            left: LeftIter,
            right: RightIter,
            transform_op: TransformOp,
            init: Output,
            reduce_op: ReduceOp,
        ) -> Result<Output, Error>
        where
            Self: super::MItem<B>,
            LeftIter: MIter<B, Item = Self>,
            RightIter: MIter<B>,
            TransformOp: op::BinaryOp2<B, Self, <RightIter as MIter<B>>::Item, Output = Output>,
            Output: super::MItem<B>,
            ReduceOp: op::BinaryOp1<B, Output>,
        {
            let _ = (policy, left, right, transform_op, init, reduce_op);
            Err(Error::Launch {
                message: "inner_product is not supported for this iterator shape".to_string(),
            })
        }

        fn inner_product_with_left_scalar<
            LeftIter,
            RightIter,
            LeftScalar,
            TransformOp,
            ReduceOp,
            Output,
        >(
            policy: &crate::detail::CubePolicy<<B as Backend>::Runtime>,
            left: LeftIter,
            right: RightIter,
            transform_op: TransformOp,
            init: Output,
            reduce_op: ReduceOp,
        ) -> Result<Output, Error>
        where
            Self: super::MItem<B>,
            LeftScalar: super::Scalar + 'static,
            LeftIter: MIter<B, Item = (LeftScalar,)>,
            RightIter: MIter<B, Item = Self>,
            TransformOp: op::BinaryOp2<B, (LeftScalar,), Self, Output = Output>,
            Output: super::MItem<B>,
            ReduceOp: op::BinaryOp1<B, Output>,
        {
            let _ = (policy, left, right, transform_op, init, reduce_op);
            Err(Error::Launch {
                message: "inner_product is not supported for this iterator shape".to_string(),
            })
        }
    }
}

fn array_from_inner<B, Item, Output>(inner: <Item as MItem<B>>::Inner) -> Output
where
    B: Backend,
    Item: MItem<B>,
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

fn column_view_at<B, Iter, T>(
    iter: &Iter,
    index: usize,
    algorithm: &str,
) -> Result<crate::detail::device::DeviceColumnView<<B as sealed::Backend>::Runtime, T>, Error>
where
    B: Backend,
    Iter: MIter<B>,
    T: Scalar + 'static,
{
    <Iter as sealed::MIterDispatch<B>>::column_view_by_index_inner::<T>(iter, index)?.ok_or_else(
        || Error::Launch {
            message: format!("{algorithm} is not supported for this iterator shape"),
        },
    )
}

fn validate_input<B, Input>(exec: &Executor<B>, input: &Input) -> Result<(), Error>
where
    B: Backend,
    Input: MIter<B>,
{
    <Input as sealed::MIterDispatch<B>>::validate_executor(input, exec)
}

fn validate_slice<B, T>(exec: &Executor<B>, slice: &DeviceSlice<'_, B, T>) -> Result<(), Error>
where
    B: Backend,
{
    exec.ensure_policy_id(slice.source.inner.policy_id())
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelOp<B, Op>(PhantomData<fn() -> (B, Op)>);

impl<B, Op> KernelOp<B, Op> {
    fn new() -> Self {
        Self(PhantomData)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelTuple1Op<B, Op>(PhantomData<fn() -> (B, Op)>);

impl<B, Op> KernelTuple1Op<B, Op> {
    fn new() -> Self {
        Self(PhantomData)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelTuple1InnerProductOp<B, Op, Output>(PhantomData<fn() -> (B, Op, Output)>);

impl<B, Op, Output> KernelTuple1InnerProductOp<B, Op, Output> {
    fn new() -> Self {
        Self(PhantomData)
    }
}

#[cubecl::cube]
impl<B, T, Op> crate::detail::op::kernel::UnaryOp<T> for KernelTuple1Op<B, Op>
where
    B: Backend,
    T: Scalar,
    Op: op::UnaryOp<B, (T,), Output = (T,)>,
{
    type Output = T;

    fn apply(input: T) -> T {
        Op::apply((input,)).0
    }
}

#[cubecl::cube]
impl<B, T, Op> crate::detail::op::kernel::BinaryOp2<T> for KernelTuple1Op<B, Op>
where
    B: Backend,
    T: Scalar,
    Op: op::BinaryOp1<B, (T,)>,
{
    fn apply(lhs: T, rhs: T) -> T {
        Op::apply((lhs,), (rhs,)).0
    }
}

#[cubecl::cube]
impl<B, T, Op> crate::detail::op::kernel::PredicateOp1<T> for KernelTuple1Op<B, Op>
where
    B: Backend,
    T: Scalar,
    Op: op::PredicateOp1<B, (T,)>,
{
    fn apply(input: T) -> bool {
        Op::apply((input,))
    }
}

#[cubecl::cube]
impl<B, T, Op> crate::detail::op::kernel::PredicateOp2<T> for KernelTuple1Op<B, Op>
where
    B: Backend,
    T: Scalar,
    Op: op::PredicateOp2<B, (T,)>,
{
    fn apply(lhs: T, rhs: T) -> bool {
        Op::apply((lhs,), (rhs,))
    }
}

#[cubecl::cube]
impl<B, Left, Right, Op, Output> op::UnaryOp<B, (Left, Right)>
    for KernelTuple1InnerProductOp<B, Op, Output>
where
    B: Backend,
    Left: Scalar,
    Right: Scalar,
    Output: MItem<B>,
    Output: 'static,
    Op: op::BinaryOp2<B, (Left,), (Right,), Output = Output>,
{
    type Output = Output;

    fn apply(input: (Left, Right)) -> Self::Output {
        Op::apply((input.0,), (input.1,))
    }
}

#[cubecl::cube]
impl<B, Input, Op> crate::detail::op::kernel::UnaryOp<Input> for KernelOp<B, Op>
where
    B: Backend,
    Input: MItem<B>,
    Op: op::UnaryOp<B, Input>,
{
    type Output = Op::Output;

    fn apply(input: Input) -> Self::Output {
        Op::apply(input)
    }
}

#[cubecl::cube]
impl<B, Item, Op> crate::detail::op::kernel::BinaryOp2<Item> for KernelOp<B, Op>
where
    B: Backend,
    Item: MItem<B>,
    Op: op::BinaryOp1<B, Item>,
{
    fn apply(lhs: Item, rhs: Item) -> Item {
        Op::apply(lhs, rhs)
    }
}

#[cubecl::cube]
impl<B, Item, Op> crate::detail::op::kernel::PredicateOp1<Item> for KernelOp<B, Op>
where
    B: Backend,
    Item: MItem<B>,
    Op: op::PredicateOp1<B, Item>,
{
    fn apply(input: Item) -> bool {
        Op::apply(input)
    }
}

#[cubecl::cube]
impl<B, Item, Op> crate::detail::op::kernel::PredicateOp2<Item> for KernelOp<B, Op>
where
    B: Backend,
    Item: MItem<B>,
    Op: op::PredicateOp2<B, Item>,
{
    fn apply(lhs: Item, rhs: Item) -> bool {
        Op::apply(lhs, rhs)
    }
}

#[doc(hidden)]
pub struct StencilFlag;

#[cubecl::cube]
impl<B> op::PredicateOp1<B, (u32,)> for StencilFlag
where
    B: Backend,
{
    fn apply(input: (u32,)) -> bool {
        input.0 > 0
    }
}

macro_rules! inner_product_left_item_body {
    ($B:ident; ($left_ty:ident); $policy:ident, $left:ident, $right:ident, $transform_op:ident, $init:ident, $reduce_op:ident) => {{
        <<RightIter as MIter<$B>>::Item as sealed::MItemDispatch<$B>>::inner_product_with_left_scalar::<
            LeftIter,
            RightIter,
            $left_ty,
            TransformOp,
            ReduceOp,
            Output,
        >($policy, $left, $right, $transform_op, $init, $reduce_op)
    }};
    ($B:ident; ($first:ident, $( $rest:ident ),+); $policy:ident, $left:ident, $right:ident, $transform_op:ident, $init:ident, $reduce_op:ident) => {{
        let _ = ($policy, $left, $right, $transform_op, $init, $reduce_op);
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! inner_product_right_item_body {
    ($B:ident; ($right_ty:ident); $policy:ident, $left:ident, $right:ident, $transform_op:ident, $init:ident, $reduce_op:ident) => {{
        let left = column_view_at::<$B, LeftIter, LeftScalar>(&$left, 0, "inner_product")?;
        let right = column_view_at::<$B, RightIter, $right_ty>(&$right, 0, "inner_product")?;
        let transformed = <Output as sealed::MItemDispatch<$B>>::transform_binary(
            $policy,
            left,
            right,
            KernelTuple1InnerProductOp::<$B, TransformOp, Output>::new(),
        )?;
        let _ = $transform_op;
        <Output as sealed::MItemDispatch<$B>>::reduce_inner($policy, transformed, $init, $reduce_op)
    }};
    ($B:ident; ($first:ident, $( $rest:ident ),+); $policy:ident, $left:ident, $right:ident, $transform_op:ident, $init:ident, $reduce_op:ident) => {{
        let _ = ($policy, $left, $right, $transform_op, $init, $reduce_op);
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_mitem_tuple {
    ($( $ty:ident : $var:ident ),+) => {
        impl<B, $( $ty ),+> MItem<B> for ($( $ty, )+)
        where
            B: Backend,
            $( $ty: Scalar, )+
        {
            type Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+);
        }

        impl<B, $( $ty ),+> MVec<B> for ($( DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar, )+
        {
            type Item = ($( $ty, )+);

            fn from_inner(inner: <Self::Item as MItem<B>>::Inner) -> Self {
                let ($( $var, )+) = inner;
                ($( DeviceVec::from_inner($var), )+)
            }

            fn len(&self) -> usize {
                self.0.len()
            }
        }

        impl<B, $( $ty ),+> sealed::MItemDispatch<B> for ($( $ty, )+)
        where
            B: Backend,
            $( $ty: Scalar, )+
        {
            fn transform_unary<Input, Op>(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                input: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    Input,
                >,
                op: Op,
            ) -> Result<<Self as MItem<B>>::Inner, Error>
            where
                Input: Scalar,
                Op: op::UnaryOp<B, (Input,), Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    <B as sealed::Backend>::Runtime,
                    Input,
                    KernelOp<B, Op>,
                >,
                <Self as crate::detail::MItemStorage<
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
                        KernelOp<B, Op>,
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
            ) -> Result<<Self as MItem<B>>::Inner, Error>
            where
                Left: Scalar,
                Right: Scalar,
                Op: op::UnaryOp<B, (Left, Right), Output = Self>,
                Self: crate::detail::TransformSoA2Output<
                    <B as sealed::Backend>::Runtime,
                    Left,
                    Right,
                    KernelOp<B, Op>,
                >,
                <Self as crate::detail::MItemStorage<
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
                        KernelOp<B, Op>,
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
            ) -> Result<<Self as MItem<B>>::Inner, Error>
            where
                First: Scalar,
                Second: Scalar,
                Third: Scalar,
                Op: op::UnaryOp<B, (First, Second, Third), Output = Self>,
                Self: crate::detail::TransformSoA3Output<
                    <B as sealed::Backend>::Runtime,
                    First,
                    Second,
                    Third,
                    KernelOp<B, Op>,
                >,
                <Self as crate::detail::MItemStorage<
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
                        KernelOp<B, Op>,
                    >>::run(
                        policy,
                        first,
                        second,
                        third,
                )?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn reduce_inner<Op>(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                input: <Self as MItem<B>>::Inner,
                init: Self,
                op: Op,
            ) -> Result<Self, Error>
            where
                Op: op::BinaryOp1<B, Self>,
            {
                let _ = op;
                crate::detail::reduce(policy, input, init, KernelOp::<B, Op>::new())
            }

            fn inner_product_with_right_item<
                LeftIter,
                RightIter,
                TransformOp,
                ReduceOp,
                Output,
            >(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                left: LeftIter,
                right: RightIter,
                transform_op: TransformOp,
                init: Output,
                reduce_op: ReduceOp,
            ) -> Result<Output, Error>
            where
                LeftIter: MIter<B, Item = Self>,
                RightIter: MIter<B>,
                TransformOp:
                    op::BinaryOp2<B, Self, <RightIter as MIter<B>>::Item, Output = Output>,
                Output: MItem<B>,
                ReduceOp: op::BinaryOp1<B, Output>,
            {
                inner_product_left_item_body!(
                    B;
                    ($( $ty ),+);
                    policy,
                    left,
                    right,
                    transform_op,
                    init,
                    reduce_op
                )
            }

            fn inner_product_with_left_scalar<
                LeftIter,
                RightIter,
                LeftScalar,
                TransformOp,
                ReduceOp,
                Output,
            >(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                left: LeftIter,
                right: RightIter,
                transform_op: TransformOp,
                init: Output,
                reduce_op: ReduceOp,
            ) -> Result<Output, Error>
            where
                LeftScalar: Scalar + 'static,
                LeftIter: MIter<B, Item = (LeftScalar,)>,
                RightIter: MIter<B, Item = Self>,
                TransformOp: op::BinaryOp2<B, (LeftScalar,), Self, Output = Output>,
                Output: MItem<B>,
                ReduceOp: op::BinaryOp1<B, Output>,
            {
                inner_product_right_item_body!(
                    B;
                    ($( $ty ),+);
                    policy,
                    left,
                    right,
                    transform_op,
                    init,
                    reduce_op
                )
            }
        }
    };
}

impl_mitem_tuple!(A: a);
impl_mitem_tuple!(A: a, B0: b);
impl_mitem_tuple!(A: a, B0: b, C: c);

macro_rules! impl_wide_mitem_tuple {
    ($( $ty:ident : $var:ident ),+) => {
        impl<B, $( $ty ),+> MItem<B> for ($( $ty, )+)
        where
            B: Backend,
            $( $ty: Scalar, )+
        {
            type Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+);
        }

        impl<B, $( $ty ),+> sealed::MItemDispatch<B> for ($( $ty, )+)
        where
            B: Backend,
            $( $ty: Scalar, )+
        {
        }

        impl<B, $( $ty ),+> MVec<B> for ($( DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar, )+
        {
            type Item = ($( $ty, )+);

            fn from_inner(inner: <Self::Item as MItem<B>>::Inner) -> Self {
                let ($( $var, )+) = inner;
                ($( DeviceVec::from_inner($var), )+)
            }

            fn len(&self) -> usize {
                self.0.len()
            }
        }
    };
}

impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f, G: g);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f, G: g, H: h);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k);
impl_wide_mitem_tuple!(
    A: a,
    B0: b,
    C: c,
    D: d,
    E: e,
    F: f,
    G: g,
    H: h,
    I: i,
    J: j,
    K: k,
    L: l
);

impl<B, T> MVec<B> for SoA1<DeviceVec<B, T>>
where
    B: Backend,
    T: Scalar,
{
    type Item = (T,);

    fn from_inner(inner: <Self::Item as MItem<B>>::Inner) -> Self {
        Self(DeviceVec::from_inner(inner.0))
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<B, A, C> MVec<B> for SoA2<DeviceVec<B, A>, DeviceVec<B, C>>
where
    B: Backend,
    A: Scalar,
    C: Scalar,
{
    type Item = (A, C);

    fn from_inner(inner: <Self::Item as MItem<B>>::Inner) -> Self {
        Self(
            DeviceVec::from_inner(inner.0),
            DeviceVec::from_inner(inner.1),
        )
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<B, A, C, D> MVec<B> for SoA3<DeviceVec<B, A>, DeviceVec<B, C>, DeviceVec<B, D>>
where
    B: Backend,
    A: Scalar,
    C: Scalar,
    D: Scalar,
{
    type Item = (A, C, D);

    fn from_inner(inner: <Self::Item as MItem<B>>::Inner) -> Self {
        Self(
            DeviceVec::from_inner(inner.0),
            DeviceVec::from_inner(inner.1),
            DeviceVec::from_inner(inner.2),
        )
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, B, T> MIter<B> for SoA1<DeviceSlice<'a, B, T>>
where
    B: Backend,
    T: Scalar + 'static,
    (T,): MItem<B, Inner = (crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>,)>,
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

impl<'a, B, T> sealed::MIterDispatch<B> for SoA1<DeviceSlice<'a, B, T>>
where
    B: Backend,
    T: Scalar + 'static,
    (T,): MItem<B, Inner = (crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, T>,)>,
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
        U: Scalar,
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
        Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
    {
        let stencil = self.into_inner();
        crate::detail::api::PrecomputedSelection::from_stencil_with_policy::<_, KernelOp<B, Pred>>(
            policy, &stencil, invert,
        )
    }

    fn transform_dispatch<Op, Output, Y>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        op: Op,
    ) -> Result<Output, Error>
    where
        Op: op::UnaryOp<B, <Self as MIter<B>>::Item, Output = Y>,
        Y: MItem<B>,
        Output: MVec<B, Item = Y>,
    {
        let input = self.into_inner().0;
        let inner = <Y as sealed::MItemDispatch<B>>::transform_unary(policy, input, op)?;
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
        _less: Less,
    ) -> Result<Output, Error>
    where
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::sort(policy, self.into_inner(), KernelOp::<B, Less>::new())?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K: Scalar + 'static,
        Less: op::PredicateOp2<B, (K,)>,
        KeyOutput: MVec<B, Item = (K,)>,
        ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::sort_by_key(
            policy,
            (keys,),
            self.into_inner(),
            KernelOp::<B, Less>::new(),
        )?;
        Ok((
            array_from_inner::<B, (K,), KeyOutput>(key_inner),
            array_from_inner::<B, (T,), ValueOutput>(value_inner),
        ))
    }

    fn sort_by_key_dispatch<Values, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        values: Values,
        less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Values: MIter<B>,
        <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        KeyOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
        ValueOutput: MVec<B, Item = <Values as MIter<B>>::Item>,
    {
        let keys = self
            .column_vec_inner::<T>(policy)?
            .ok_or_else(|| Error::Launch {
                message: "sort_by_key keys must be backed by one DeviceVec or DeviceSlice"
                    .to_string(),
            })?;
        <Values as sealed::MIterDispatch<B>>::sort_by_single_key_dispatch(
            values, policy, &keys, less,
        )
    }

    fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        _eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K: Scalar + 'static,
        Eq: op::PredicateOp2<B, (K,)>,
        KeyOutput: MVec<B, Item = (K,)>,
        ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::unique_by_key(
            policy,
            (keys,),
            self.into_inner(),
            KernelOp::<B, Eq>::new(),
        )?;
        Ok((
            array_from_inner::<B, (K,), KeyOutput>(key_inner),
            array_from_inner::<B, (T,), ValueOutput>(value_inner),
        ))
    }

    fn unique_by_key_dispatch<Values, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        values: Values,
        eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Values: MIter<B>,
        <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
        Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        KeyOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
        ValueOutput: MVec<B, Item = <Values as MIter<B>>::Item>,
    {
        let keys = self
            .column_vec_inner::<T>(policy)?
            .ok_or_else(|| Error::Launch {
                message: "unique_by_key keys must be backed by one DeviceVec or DeviceSlice"
                    .to_string(),
            })?;
        <Values as sealed::MIterDispatch<B>>::unique_by_single_key_dispatch(
            values, policy, &keys, eq,
        )
    }

    fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        _key_eq: KeyEq,
        _op: Op,
    ) -> Result<Output, Error>
    where
        K: Scalar + 'static,
        KeyEq: op::PredicateOp2<B, (K,)>,
        Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::inclusive_scan_by_key(
            policy,
            (keys,),
            self.into_inner(),
            KernelOp::<B, KeyEq>::new(),
            KernelOp::<B, Op>::new(),
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn inclusive_scan_by_key_dispatch<Values, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        values: Values,
        key_eq: KeyEq,
        op: Op,
    ) -> Result<Output, Error>
    where
        Values: MIter<B>,
        <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        Op: op::BinaryOp1<B, <Values as MIter<B>>::Item>,
        Output: MVec<B, Item = <Values as MIter<B>>::Item>,
    {
        let keys = self
            .column_vec_inner::<T>(policy)?
            .ok_or_else(|| Error::Launch {
                message:
                    "inclusive_scan_by_key keys must be backed by one DeviceVec or DeviceSlice"
                        .to_string(),
            })?;
        <Values as sealed::MIterDispatch<B>>::inclusive_scan_by_single_key_dispatch(
            values, policy, &keys, key_eq, op,
        )
    }

    fn exclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        _key_eq: KeyEq,
        init: <Self as MIter<B>>::Item,
        _op: Op,
    ) -> Result<Output, Error>
    where
        K: Scalar + 'static,
        KeyEq: op::PredicateOp2<B, (K,)>,
        Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::exclusive_scan_by_key(
            policy,
            (keys,),
            self.into_inner(),
            KernelOp::<B, KeyEq>::new(),
            init,
            KernelOp::<B, Op>::new(),
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn exclusive_scan_by_key_dispatch<Values, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        values: Values,
        key_eq: KeyEq,
        init: <Values as MIter<B>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Values: MIter<B>,
        <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        Op: op::BinaryOp1<B, <Values as MIter<B>>::Item>,
        Output: MVec<B, Item = <Values as MIter<B>>::Item>,
    {
        let keys = self
            .column_vec_inner::<T>(policy)?
            .ok_or_else(|| Error::Launch {
                message:
                    "exclusive_scan_by_key keys must be backed by one DeviceVec or DeviceSlice"
                        .to_string(),
            })?;
        <Values as sealed::MIterDispatch<B>>::exclusive_scan_by_single_key_dispatch(
            values, policy, &keys, key_eq, init, op,
        )
    }

    fn reduce_by_single_key_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        _key_eq: KeyEq,
        init: <Self as MIter<B>>::Item,
        _op: Op,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K: Scalar + 'static,
        KeyEq: op::PredicateOp2<B, (K,)>,
        Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
        KeyOutput: MVec<B, Item = (K,)>,
        ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::reduce_by_key(
            policy,
            (keys,),
            self.into_inner(),
            KernelOp::<B, KeyEq>::new(),
            init,
            KernelOp::<B, Op>::new(),
        )?;
        Ok((
            array_from_inner::<B, (K,), KeyOutput>(key_inner),
            array_from_inner::<B, (T,), ValueOutput>(value_inner),
        ))
    }

    fn reduce_by_key_dispatch<Values, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        values: Values,
        key_eq: KeyEq,
        init: <Values as MIter<B>>::Item,
        op: Op,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Values: MIter<B>,
        <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        Op: op::BinaryOp1<B, <Values as MIter<B>>::Item>,
        KeyOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
        ValueOutput: MVec<B, Item = <Values as MIter<B>>::Item>,
    {
        let keys = self
            .column_vec_inner::<T>(policy)?
            .ok_or_else(|| Error::Launch {
                message: "reduce_by_key keys must be backed by one DeviceVec or DeviceSlice"
                    .to_string(),
            })?;
        <Values as sealed::MIterDispatch<B>>::reduce_by_single_key_dispatch(
            values, policy, &keys, key_eq, init, op,
        )
    }

    fn merge_by_single_key_same_dispatch<K, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        left_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        right_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
        right_values: RightValues,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        RightValues: MIter<B, Item = <Self as MIter<B>>::Item>,
        K: Scalar + 'static,
        Less: op::PredicateOp2<B, (K,)>,
        KeyOutput: MVec<B, Item = (K,)>,
        ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let left_value = self.into_inner().0;
        let right_value =
            <RightValues as sealed::MIterDispatch<B>>::column_view_by_index_inner::<T>(
                &right_values,
                0,
            )?
            .ok_or_else(|| Error::Launch {
                message: "merge_by_key right values must match left value shape".to_string(),
            })?;
        let (key_inner, value_inner) = crate::detail::merge_by_key(
            policy,
            crate::detail::device::SoAView1 { source: left_keys },
            crate::detail::device::SoAView1 { source: left_value },
            crate::detail::device::SoAView1 { source: right_keys },
            crate::detail::device::SoAView1 {
                source: right_value,
            },
            KernelTuple1Op::<B, Less>::new(),
        )?;
        Ok((
            array_from_inner::<B, (K,), KeyOutput>(key_inner),
            array_from_inner::<B, (T,), ValueOutput>(value_inner),
        ))
    }

    fn merge_by_key_dispatch<RightKeys, LeftValues, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right_keys: RightKeys,
        left_values: LeftValues,
        right_values: RightValues,
        less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        RightKeys: MIter<B, Item = <Self as MIter<B>>::Item>,
        LeftValues: MIter<B>,
        RightValues: MIter<B, Item = <LeftValues as MIter<B>>::Item>,
        <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        KeyOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
        ValueOutput: MVec<B, Item = <LeftValues as MIter<B>>::Item>,
    {
        let left_keys = self
            .column_vec_inner::<T>(policy)?
            .ok_or_else(|| Error::Launch {
                message: "merge_by_key left keys must be backed by one DeviceVec or DeviceSlice"
                    .to_string(),
            })?;
        let right_keys =
            <RightKeys as sealed::MIterDispatch<B>>::column_vec_inner::<T>(&right_keys, policy)?
                .ok_or_else(|| Error::Launch {
                    message:
                        "merge_by_key right keys must be backed by one DeviceVec or DeviceSlice"
                            .to_string(),
                })?;
        <LeftValues as sealed::MIterDispatch<B>>::merge_by_single_key_same_dispatch(
            left_values,
            policy,
            &left_keys,
            &right_keys,
            right_values,
            less,
        )
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
        _op: Op,
    ) -> Result<<Self as MIter<B>>::Item, Error>
    where
        Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::reduce(policy, self.into_inner(), init, KernelOp::<B, Op>::new())
    }

    fn inclusive_scan_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _op: Op,
    ) -> Result<Output, Error>
    where
        Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner =
            crate::detail::inclusive_scan(policy, self.into_inner(), KernelOp::<B, Op>::new())?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn exclusive_scan_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        init: <Self as MIter<B>>::Item,
        _op: Op,
    ) -> Result<Output, Error>
    where
        Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::exclusive_scan(
            policy,
            self.into_inner(),
            init,
            KernelOp::<B, Op>::new(),
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn adjacent_difference_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _op: Op,
    ) -> Result<Output, Error>
    where
        Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::adjacent_difference(
            policy,
            self.into_inner(),
            KernelOp::<B, Op>::new(),
        )?;
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
        let inner = crate::detail::copy_if(
            policy,
            self.into_inner(),
            stencil,
            KernelOp::<B, StencilFlag>::new(),
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn remove_if_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _pred: Pred,
    ) -> Result<Output, Error>
    where
        Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner =
            crate::detail::remove_if(policy, self.into_inner(), KernelOp::<B, Pred>::new())?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn count_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _pred: Pred,
    ) -> Result<usize, Error>
    where
        Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::count_if(policy, self.into_inner(), KernelOp::<B, Pred>::new())
    }

    fn all_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _pred: Pred,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::all_of(policy, self.into_inner(), KernelOp::<B, Pred>::new())
    }

    fn any_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _pred: Pred,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::any_of(policy, self.into_inner(), KernelOp::<B, Pred>::new())
    }

    fn none_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _pred: Pred,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::none_of(policy, self.into_inner(), KernelOp::<B, Pred>::new())
    }

    fn find_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _pred: Pred,
    ) -> Result<Option<usize>, Error>
    where
        Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::find_if(policy, self.into_inner(), KernelOp::<B, Pred>::new())
    }

    fn partition_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _pred: Pred,
    ) -> Result<(Output, Output), Error>
    where
        Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let (matching, failing) =
            crate::detail::partition(policy, self.into_inner(), KernelOp::<B, Pred>::new())?;
        Ok((
            array_from_inner::<B, (T,), Output>(matching),
            array_from_inner::<B, (T,), Output>(failing),
        ))
    }

    fn is_partitioned_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _pred: Pred,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::is_partitioned(policy, self.into_inner(), KernelOp::<B, Pred>::new())
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
            KernelOp::<B, StencilFlag>::new(),
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn unique_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _pred: Pred,
    ) -> Result<Output, Error>
    where
        Pred: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::unique(policy, self.into_inner(), KernelOp::<B, Pred>::new())?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn min_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _less: Less,
    ) -> Result<Option<usize>, Error>
    where
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::min_element(policy, self.into_inner(), KernelOp::<B, Less>::new())
    }

    fn max_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _less: Less,
    ) -> Result<Option<usize>, Error>
    where
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::max_element(policy, self.into_inner(), KernelOp::<B, Less>::new())
    }

    fn minmax_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _less: Less,
    ) -> Result<Option<(usize, usize)>, Error>
    where
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::minmax_element(policy, self.into_inner(), KernelOp::<B, Less>::new())
    }

    fn adjacent_find_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _pred: Pred,
    ) -> Result<Option<usize>, Error>
    where
        Pred: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::adjacent_find(policy, self.into_inner(), KernelOp::<B, Pred>::new())
    }

    fn lower_bound_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        value: <Self as MIter<B>>::Item,
        _less: Less,
    ) -> Result<usize, Error>
    where
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::lower_bound(policy, self.into_inner(), value, KernelOp::<B, Less>::new())
    }

    fn upper_bound_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        value: <Self as MIter<B>>::Item,
        _less: Less,
    ) -> Result<usize, Error>
    where
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::upper_bound(policy, self.into_inner(), value, KernelOp::<B, Less>::new())
    }

    fn equal_range_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        value: <Self as MIter<B>>::Item,
        _less: Less,
    ) -> Result<(usize, usize), Error>
    where
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::equal_range(policy, self.into_inner(), value, KernelOp::<B, Less>::new())
    }

    fn is_sorted_until_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _less: Less,
    ) -> Result<usize, Error>
    where
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::is_sorted_until(policy, self.into_inner(), KernelOp::<B, Less>::new())
    }

    fn is_sorted_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        _less: Less,
    ) -> Result<bool, Error>
    where
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::is_sorted(policy, self.into_inner(), KernelOp::<B, Less>::new())
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
            KernelOp::<B, StencilFlag>::new(),
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
            KernelOp::<B, StencilFlag>::new(),
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn equal_dispatch<Right, Eq>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Right,
        _eq: Eq,
    ) -> Result<bool, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        let (left,) = self.into_inner();
        let right = column_view_at::<B, Right, T>(&right, 0, "equal")?;
        crate::detail::equal(policy, (left,), (right,), KernelOp::<B, Eq>::new())
    }

    fn mismatch_dispatch<Right, Eq>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Right,
        _eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        let (left,) = self.into_inner();
        let right = column_view_at::<B, Right, T>(&right, 0, "mismatch")?;
        crate::detail::mismatch(policy, (left,), (right,), KernelOp::<B, Eq>::new())
    }

    fn find_first_of_dispatch<Needles, Eq>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        needles: Needles,
        _eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Needles: MIter<B, Item = <Self as MIter<B>>::Item>,
        Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        let (input,) = self.into_inner();
        let needles = column_view_at::<B, Needles, T>(&needles, 0, "find_first_of")?;
        crate::detail::find_first_of(policy, (input,), (needles,), KernelOp::<B, Eq>::new())
    }

    fn lexicographical_compare_dispatch<Right, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Right,
        _less: Less,
    ) -> Result<bool, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        let (left,) = self.into_inner();
        let right = column_view_at::<B, Right, T>(&right, 0, "lexicographical_compare")?;
        crate::detail::lexicographical_compare(
            policy,
            (left,),
            (right,),
            KernelOp::<B, Less>::new(),
        )
    }

    fn merge_dispatch<Right, Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        let (left,) = self.into_inner();
        let right = column_view_at::<B, Right, T>(&right, 0, "merge")?;
        let inner = crate::detail::merge(policy, (left,), (right,), KernelOp::<B, Less>::new())?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn set_union_dispatch<Right, Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        let (left,) = self.into_inner();
        let right = column_view_at::<B, Right, T>(&right, 0, "set_union")?;
        let inner =
            crate::detail::set_union(policy, (left,), (right,), KernelOp::<B, Less>::new())?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn set_intersection_dispatch<Right, Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        let (left,) = self.into_inner();
        let right = column_view_at::<B, Right, T>(&right, 0, "set_intersection")?;
        let inner =
            crate::detail::set_intersection(policy, (left,), (right,), KernelOp::<B, Less>::new())?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn set_difference_dispatch<Right, Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        let (left,) = self.into_inner();
        let right = column_view_at::<B, Right, T>(&right, 0, "set_difference")?;
        let inner =
            crate::detail::set_difference(policy, (left,), (right,), KernelOp::<B, Less>::new())?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn equal_same_dispatch<Eq>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        _eq: Eq,
    ) -> Result<bool, Error>
    where
        Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::equal(
            policy,
            self.into_inner(),
            right.into_inner(),
            KernelOp::<B, Eq>::new(),
        )
    }

    fn mismatch_same_dispatch<Eq>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        _eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::mismatch(
            policy,
            self.into_inner(),
            right.into_inner(),
            KernelOp::<B, Eq>::new(),
        )
    }

    fn find_first_of_same_dispatch<Eq>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        needles: Self,
        _eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::find_first_of(
            policy,
            self.into_inner(),
            needles.into_inner(),
            KernelOp::<B, Eq>::new(),
        )
    }

    fn lexicographical_compare_same_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        _less: Less,
    ) -> Result<bool, Error>
    where
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        crate::detail::lexicographical_compare(
            policy,
            self.into_inner(),
            right.into_inner(),
            KernelOp::<B, Less>::new(),
        )
    }

    fn merge_same_dispatch<Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::merge(
            policy,
            self.into_inner(),
            right.into_inner(),
            KernelOp::<B, Less>::new(),
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn set_union_same_dispatch<Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::set_union(
            policy,
            self.into_inner(),
            right.into_inner(),
            KernelOp::<B, Less>::new(),
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn set_intersection_same_dispatch<Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::set_intersection(
            policy,
            self.into_inner(),
            right.into_inner(),
            KernelOp::<B, Less>::new(),
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
    }

    fn set_difference_same_dispatch<Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
        right: Self,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
    {
        let inner = crate::detail::set_difference(
            policy,
            self.into_inner(),
            right.into_inner(),
            KernelOp::<B, Less>::new(),
        )?;
        Ok(array_from_inner::<B, (T,), Output>(inner))
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

macro_rules! impl_miter_soa {
    ($name:ident; $( $ty:ident : $idx:tt : $tmp:ident ),+ => $transform:ident) => {
        impl<'a, B, $( $ty ),+> MIter<B> for $name<$( DeviceSlice<'a, B, $ty> ),+>
        where
            B: Backend,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
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

        impl<'a, B, $( $ty ),+> sealed::MIterDispatch<B> for $name<$( DeviceSlice<'a, B, $ty> ),+>
        where
            B: Backend,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
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

            fn column_view_by_index_inner<T: 'static>(
                &self,
                index: usize,
            ) -> Result<
                Option<crate::detail::device::DeviceColumnView<<B as sealed::Backend>::Runtime, T>>,
                Error,
            >
            where
                T: Scalar,
            {
                $(
                    if index == $idx {
                        let source = self.$idx.source as &dyn Any;
                        let source = match source.downcast_ref::<DeviceVec<B, T>>() {
                            Some(source) => source,
                            None => return Ok(None),
                        };
                        return Ok(Some(crate::detail::device::DeviceColumnView::from_slice(
                            &source.inner,
                            self.$idx.offset,
                            self.$idx.len,
                        )));
                    }
                )+
                Ok(None)
            }

            fn selection_stencil_dispatch<Pred>(
                &self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                invert: bool,
            ) -> Result<crate::detail::api::PrecomputedSelection<<B as sealed::Backend>::Runtime>, Error>
            where
                Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
            {
                let stencil = self.into_inner();
                let stencil = impl_miter_view!(stencil; $( $idx ),+);
                crate::detail::api::PrecomputedSelection::from_stencil_with_policy::<
                    _,
                    KernelOp<B, Pred>,
                >(
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
                Op: op::UnaryOp<B, <Self as MIter<B>>::Item, Output = Y>,
                Y: MItem<B>,
                Output: MVec<B, Item = Y>,
            {
                let input = self.into_inner();
                let inner = <Y as sealed::MItemDispatch<B>>::$transform(
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
                _less: Less,
            ) -> Result<Output, Error>
            where
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::sort(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Less>::new())?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                Less: op::PredicateOp2<B, (K,)>,
                KeyOutput: MVec<B, Item = (K,)>,
                ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let values = self.into_inner();
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) = crate::detail::sort_by_key(policy, (keys,), (values,), KernelOp::<B, Less>::new())?;
                Ok((
                    array_from_inner::<B, (K,), KeyOutput>(key_inner),
                    array_from_inner::<B, ($( $ty, )+), ValueOutput>(value_inner),
                ))
            }

            fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                _eq: Eq,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                Eq: op::PredicateOp2<B, (K,)>,
                KeyOutput: MVec<B, Item = (K,)>,
                ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let values = self.into_inner();
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) =
                    crate::detail::unique_by_key(policy, (keys,), (values,), KernelOp::<B, Eq>::new())?;
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
                _op: Op,
            ) -> Result<Output, Error>
            where
                K: Scalar + 'static,
                KeyEq: op::PredicateOp2<B, (K,)>,
                Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let values = self.into_inner();
                let values = impl_miter_view!(values; $( $idx ),+);
                let inner = crate::detail::inclusive_scan_by_key(
                    policy,
                    keys,
                    values,
                    KernelTuple1Op::<B, KeyEq>::new(),
                    KernelOp::<B, Op>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn exclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<B>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K: Scalar + 'static,
                KeyEq: op::PredicateOp2<B, (K,)>,
                Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let values = self.into_inner();
                let values = impl_miter_view!(values; $( $idx ),+);
                let inner = crate::detail::exclusive_scan_by_key(
                    policy,
                    keys,
                    values,
                    KernelTuple1Op::<B, KeyEq>::new(),
                    init,
                    KernelOp::<B, Op>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn reduce_by_single_key_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<B>>::Item,
                _op: Op,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                KeyEq: op::PredicateOp2<B, (K,)>,
                Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
                KeyOutput: MVec<B, Item = (K,)>,
                ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let values = self.into_inner();
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) = crate::detail::reduce_by_key(
                    policy,
                    keys,
                    values,
                    KernelTuple1Op::<B, KeyEq>::new(),
                    init,
                    KernelOp::<B, Op>::new(),
                )?;
                Ok((
                    array_from_inner::<B, (K,), KeyOutput>(key_inner),
                    array_from_inner::<B, ($( $ty, )+), ValueOutput>(value_inner),
                ))
            }

            fn merge_by_single_key_same_dispatch<K, RightValues, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                left_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                right_keys: &crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, K>,
                right_values: RightValues,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                RightValues: MIter<B, Item = <Self as MIter<B>>::Item>,
                K: Scalar + 'static,
                Less: op::PredicateOp2<B, (K,)>,
                KeyOutput: MVec<B, Item = (K,)>,
                ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let left_values = self.into_inner();
                let right_values = ($(
                    <RightValues as sealed::MIterDispatch<B>>::column_view_by_index_inner::<$ty>(
                        &right_values,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "merge_by_key right values must match left value shape".to_string(),
                    })?,
                )+);
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    crate::detail::device::SoAView1 { source: left_keys },
                    impl_miter_view!(left_values; $( $idx ),+),
                    crate::detail::device::SoAView1 { source: right_keys },
                    impl_miter_view!(right_values; $( $idx ),+),
                    KernelTuple1Op::<B, Less>::new(),
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
                _op: Op,
            ) -> Result<<Self as MIter<B>>::Item, Error>
            where
                Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::reduce(policy, impl_miter_view!(input; $( $idx ),+), init, KernelOp::<B, Op>::new())
            }

            fn inclusive_scan_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::inclusive_scan(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<B, Op>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn exclusive_scan_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                init: <Self as MIter<B>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::exclusive_scan(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    init,
                    KernelOp::<B, Op>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn adjacent_difference_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::BinaryOp1<B, <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::adjacent_difference(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<B, Op>::new(),
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
                    KernelOp::<B, StencilFlag>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn remove_if_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _pred: Pred,
            ) -> Result<Output, Error>
            where
                Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::remove_if(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<B, Pred>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn count_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _pred: Pred,
            ) -> Result<usize, Error>
            where
                Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::count_if(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn all_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::all_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn any_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::any_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn none_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::none_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn find_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _pred: Pred,
            ) -> Result<Option<usize>, Error>
            where
                Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::find_if(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn partition_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _pred: Pred,
            ) -> Result<(Output, Output), Error>
            where
                Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let (matching, failing) = crate::detail::partition(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<B, Pred>::new(),
                )?;
                Ok((
                    array_from_inner::<B, ($( $ty, )+), Output>(matching),
                    array_from_inner::<B, ($( $ty, )+), Output>(failing),
                ))
            }

            fn is_partitioned_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp1<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::is_partitioned(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
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
                    KernelOp::<B, StencilFlag>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn unique_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _pred: Pred,
            ) -> Result<Output, Error>
            where
                Pred: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::unique(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<B, Pred>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn min_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _less: Less,
            ) -> Result<Option<usize>, Error>
            where
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::min_element(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Less>::new())
            }

            fn max_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _less: Less,
            ) -> Result<Option<usize>, Error>
            where
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::max_element(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Less>::new())
            }

            fn minmax_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _less: Less,
            ) -> Result<Option<(usize, usize)>, Error>
            where
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::minmax_element(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Less>::new())
            }

            fn adjacent_find_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _pred: Pred,
            ) -> Result<Option<usize>, Error>
            where
                Pred: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::adjacent_find(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn lower_bound_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                value: <Self as MIter<B>>::Item,
                _less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::lower_bound(policy, impl_miter_view!(input; $( $idx ),+), value, KernelOp::<B, Less>::new())
            }

            fn upper_bound_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                value: <Self as MIter<B>>::Item,
                _less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::upper_bound(policy, impl_miter_view!(input; $( $idx ),+), value, KernelOp::<B, Less>::new())
            }

            fn equal_range_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                value: <Self as MIter<B>>::Item,
                _less: Less,
            ) -> Result<(usize, usize), Error>
            where
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::equal_range(policy, impl_miter_view!(input; $( $idx ),+), value, KernelOp::<B, Less>::new())
            }

            fn is_sorted_until_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::is_sorted_until(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Less>::new())
            }

            fn is_sorted_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::is_sorted(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Less>::new())
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
                    KernelOp::<B, StencilFlag>::new(),
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
                    KernelOp::<B, StencilFlag>::new(),
                )?;
                Ok(array_from_inner::<B, <Self as MIter<B>>::Item, Output>(inner))
            }

            fn equal_dispatch<Right, Eq>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Right,
                _eq: Eq,
            ) -> Result<bool, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = ($( column_view_at::<B, Right, $ty>(&right, $idx, "equal")?, )+);
                crate::detail::equal(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Eq>::new(),
                )
            }

            fn mismatch_dispatch<Right, Eq>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Right,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = ($( column_view_at::<B, Right, $ty>(&right, $idx, "mismatch")?, )+);
                crate::detail::mismatch(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Eq>::new(),
                )
            }

            fn find_first_of_dispatch<Needles, Eq>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                needles: Needles,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Needles: MIter<B, Item = <Self as MIter<B>>::Item>,
                Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let needles = ($( column_view_at::<B, Needles, $ty>(&needles, $idx, "find_first_of")?, )+);
                crate::detail::find_first_of(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    impl_miter_view!(needles; $( $idx ),+),
                    KernelOp::<B, Eq>::new(),
                )
            }

            fn lexicographical_compare_dispatch<Right, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Right,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = ($( column_view_at::<B, Right, $ty>(&right, $idx, "lexicographical_compare")?, )+);
                crate::detail::lexicographical_compare(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Less>::new(),
                )
            }

            fn merge_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = ($( column_view_at::<B, Right, $ty>(&right, $idx, "merge")?, )+);
                let inner = crate::detail::merge(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Less>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn set_union_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = ($( column_view_at::<B, Right, $ty>(&right, $idx, "set_union")?, )+);
                let inner = crate::detail::set_union(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Less>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn set_intersection_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = ($( column_view_at::<B, Right, $ty>(&right, $idx, "set_intersection")?, )+);
                let inner = crate::detail::set_intersection(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Less>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn set_difference_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = ($( column_view_at::<B, Right, $ty>(&right, $idx, "set_difference")?, )+);
                let inner = crate::detail::set_difference(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Less>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn equal_same_dispatch<Eq>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                _eq: Eq,
            ) -> Result<bool, Error>
            where
                Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::equal(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Eq>::new(),
                )
            }

            fn mismatch_same_dispatch<Eq>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::mismatch(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Eq>::new(),
                )
            }

            fn find_first_of_same_dispatch<Eq>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                needles: Self,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Eq: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let needles = needles.into_inner();
                crate::detail::find_first_of(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    impl_miter_view!(needles; $( $idx ),+),
                    KernelOp::<B, Eq>::new(),
                )
            }

            fn lexicographical_compare_same_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                crate::detail::lexicographical_compare(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Less>::new(),
                )
            }

            fn merge_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                let inner = crate::detail::merge(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Less>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn set_union_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                let inner = crate::detail::set_union(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Less>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn set_intersection_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                let inner = crate::detail::set_intersection(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Less>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn set_difference_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::PredicateOp2<B, <Self as MIter<B>>::Item>,
            {
                let left = self.into_inner();
                let right = right.into_inner();
                let inner = crate::detail::set_difference(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<B, Less>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

        }
    };
}

impl_miter_soa!(SoA2; A: 0: a, C: 1: c => transform_binary);
impl_miter_soa!(SoA3; A: 0: a, C: 1: c, D: 2: d => transform_ternary);

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
    Op: op::BinaryOp1<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::adjacent_difference_dispatch(source, exec.policy(), op)
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
    Pred: op::PredicateOp2<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::adjacent_find_dispatch(source, exec.policy(), pred)
}

/// Returns whether all elements satisfy `pred`.
pub fn all_of<B, Input, Pred>(exec: &Executor<B>, source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp1<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::all_of_dispatch(source, exec.policy(), pred)
}

/// Returns whether any element satisfies `pred`.
pub fn any_of<B, Input, Pred>(exec: &Executor<B>, source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp1<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::any_of_dispatch(source, exec.policy(), pred)
}

/// Copies elements whose `u32` stencil flag is non-zero.
pub fn copy_if<B, Input, Output>(
    exec: &Executor<B>,
    source: Input,
    stencil: DeviceSlice<'_, B, u32>,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &stencil)?;
    <Input as sealed::MIterDispatch<B>>::copy_if_dispatch(source, exec.policy(), SoA1(stencil))
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
    Pred: op::PredicateOp1<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::count_if_dispatch(source, exec.policy(), pred)
}

/// Returns whether two inputs are equal under `eq`.
pub fn equal<B, Left, Right, Eq>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    eq: Eq,
) -> Result<bool, Error>
where
    B: Backend,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Eq: op::PredicateOp2<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::equal_dispatch(left, exec.policy(), right, eq)
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
    Less: op::PredicateOp2<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::equal_range_dispatch(source, exec.policy(), value, less)
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
    Op: op::BinaryOp1<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::exclusive_scan_dispatch(source, exec.policy(), init, op)
}

/// Exclusive scan by key.
pub fn exclusive_scan_by_key<B, Keys, Values, KeyEq, Op, Output>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Values::Item,
    op: Op,
) -> Result<Output, Error>
where
    B: Backend,
    Keys: MIter<B>,
    Values: MIter<B>,
    KeyEq: op::PredicateOp2<B, Keys::Item>,
    Op: op::BinaryOp1<B, Values::Item>,
    Output: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::exclusive_scan_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        init,
        op,
    )
}

/// Finds the first input element equal to any needle.
pub fn find_first_of<B, Input, Needles, Eq>(
    exec: &Executor<B>,
    source: Input,
    needles: Needles,
    eq: Eq,
) -> Result<Option<usize>, Error>
where
    B: Backend,
    Input: MIter<B>,
    Needles: MIter<B, Item = Input::Item>,
    Eq: op::PredicateOp2<B, Input::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &needles)?;
    <Input as sealed::MIterDispatch<B>>::find_first_of_dispatch(source, exec.policy(), needles, eq)
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
    Pred: op::PredicateOp1<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::find_if_dispatch(source, exec.policy(), pred)
}

/// Gathers a massively iterator at index positions into an owned vector.
pub fn gather<B, Input, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: DeviceSlice<'_, B, u32>,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &indices)?;
    <Input as sealed::MIterDispatch<B>>::gather_dispatch(source, exec.policy(), SoA1(indices))
}

/// Gathers elements whose `u32` stencil flag is non-zero.
pub fn gather_if<B, Input, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: DeviceSlice<'_, B, u32>,
    default: Input::Item,
    stencil: DeviceSlice<'_, B, u32>,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &indices)?;
    validate_slice(exec, &stencil)?;
    <Input as sealed::MIterDispatch<B>>::gather_if_dispatch(
        source,
        exec.policy(),
        SoA1(indices),
        default,
        SoA1(stencil),
    )
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
    Op: op::BinaryOp1<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::inclusive_scan_dispatch(source, exec.policy(), op)
}

/// Inclusive scan by key.
pub fn inclusive_scan_by_key<B, Keys, Values, KeyEq, Op, Output>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    op: Op,
) -> Result<Output, Error>
where
    B: Backend,
    Keys: MIter<B>,
    Values: MIter<B>,
    KeyEq: op::PredicateOp2<B, Keys::Item>,
    Op: op::BinaryOp1<B, Values::Item>,
    Output: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::inclusive_scan_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        op,
    )
}

/// Applies a binary transform over two inputs and reduces the result.
pub fn inner_product<B, Left, Right, ZipperOp, ReduceOp>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    transform_op: ZipperOp,
    init: ZipperOp::Output,
    reduce_op: ReduceOp,
) -> Result<ZipperOp::Output, Error>
where
    B: Backend,
    Left: MIter<B>,
    Right: MIter<B>,
    ZipperOp: op::BinaryOp2<B, Left::Item, Right::Item>,
    ReduceOp: op::BinaryOp1<B, ZipperOp::Output>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left::Item as sealed::MItemDispatch<B>>::inner_product_with_right_item::<
        Left,
        Right,
        ZipperOp,
        ReduceOp,
        ZipperOp::Output,
    >(exec.policy(), left, right, transform_op, init, reduce_op)
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
    Pred: op::PredicateOp1<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::is_partitioned_dispatch(source, exec.policy(), pred)
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
    Less: op::PredicateOp2<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::is_sorted_dispatch(source, exec.policy(), less)
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
    Less: op::PredicateOp2<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::is_sorted_until_dispatch(source, exec.policy(), less)
}

/// Lexicographically compares two inputs.
pub fn lexicographical_compare<B, Left, Right, Less>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<bool, Error>
where
    B: Backend,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Less: op::PredicateOp2<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::lexicographical_compare_dispatch(
        left,
        exec.policy(),
        right,
        less,
    )
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
    Less: op::PredicateOp2<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::lower_bound_dispatch(source, exec.policy(), value, less)
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
    Less: op::PredicateOp2<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::max_element_dispatch(source, exec.policy(), less)
}

/// Merges two sorted inputs.
pub fn merge<B, Left, Right, Output, Less>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Output: MVec<B, Item = Left::Item>,
    Less: op::PredicateOp2<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::merge_dispatch(left, exec.policy(), right, less)
}

/// Merges two sorted key-value ranges by key.
pub fn merge_by_key<B, LeftKeys, LeftValues, RightKeys, RightValues, Less, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    LeftKeys: MIter<B>,
    RightKeys: MIter<B, Item = LeftKeys::Item>,
    LeftValues: MIter<B>,
    RightValues: MIter<B, Item = LeftValues::Item>,
    Less: op::PredicateOp2<B, LeftKeys::Item>,
    KeyOutput: MVec<B, Item = LeftKeys::Item>,
    ValueOutput: MVec<B, Item = LeftValues::Item>,
{
    validate_input(exec, &left_keys)?;
    validate_input(exec, &left_values)?;
    validate_input(exec, &right_keys)?;
    validate_input(exec, &right_values)?;
    <LeftKeys as sealed::MIterDispatch<B>>::merge_by_key_dispatch(
        left_keys,
        exec.policy(),
        right_keys,
        left_values,
        right_values,
        less,
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
    Less: op::PredicateOp2<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::min_element_dispatch(source, exec.policy(), less)
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
    Less: op::PredicateOp2<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::minmax_element_dispatch(source, exec.policy(), less)
}

/// Finds the first mismatch between two inputs.
pub fn mismatch<B, Left, Right, Eq>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    eq: Eq,
) -> Result<Option<usize>, Error>
where
    B: Backend,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Eq: op::PredicateOp2<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::mismatch_dispatch(left, exec.policy(), right, eq)
}

/// Returns whether no elements satisfy `pred`.
pub fn none_of<B, Input, Pred>(exec: &Executor<B>, source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Backend,
    Input: MIter<B>,
    Pred: op::PredicateOp1<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::none_of_dispatch(source, exec.policy(), pred)
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
    Pred: op::PredicateOp1<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::partition_dispatch(source, exec.policy(), pred)
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
    Op: op::BinaryOp1<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::reduce_dispatch(source, exec.policy(), init, op)
}

/// Reduces consecutive values with equal keys.
pub fn reduce_by_key<B, Keys, Values, KeyEq, Op, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Values::Item,
    op: Op,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    Keys: MIter<B>,
    Values: MIter<B>,
    KeyEq: op::PredicateOp2<B, Keys::Item>,
    Op: op::BinaryOp1<B, Values::Item>,
    KeyOutput: MVec<B, Item = Keys::Item>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::reduce_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        init,
        op,
    )
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
    Pred: op::PredicateOp1<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::remove_if_dispatch(source, exec.policy(), pred)
}

/// Replaces elements whose `u32` stencil flag is non-zero.
pub fn replace_if<B, Input, Output>(
    exec: &Executor<B>,
    source: Input,
    replacement: Input::Item,
    stencil: DeviceSlice<'_, B, u32>,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &stencil)?;
    <Input as sealed::MIterDispatch<B>>::replace_if_dispatch(
        source,
        exec.policy(),
        replacement,
        SoA1(stencil),
    )
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

/// Scatters values into a newly allocated output.
pub fn scatter<B, Input, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: DeviceSlice<'_, B, u32>,
    len: usize,
    default: Input::Item,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &indices)?;
    <Input as sealed::MIterDispatch<B>>::scatter_dispatch(
        source,
        exec.policy(),
        SoA1(indices),
        len,
        default,
    )
}

/// Scatters values whose `u32` stencil flag is non-zero into a newly allocated output.
pub fn scatter_if<B, Input, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: DeviceSlice<'_, B, u32>,
    len: usize,
    default: Input::Item,
    stencil: DeviceSlice<'_, B, u32>,
) -> Result<Output, Error>
where
    B: Backend,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &indices)?;
    validate_slice(exec, &stencil)?;
    <Input as sealed::MIterDispatch<B>>::scatter_if_dispatch(
        source,
        exec.policy(),
        SoA1(indices),
        len,
        default,
        SoA1(stencil),
    )
}

/// Computes the sorted set difference of two sorted inputs.
pub fn set_difference<B, Left, Right, Output, Less>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Output: MVec<B, Item = Left::Item>,
    Less: op::PredicateOp2<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::set_difference_dispatch(left, exec.policy(), right, less)
}

/// Computes the sorted set intersection of two sorted inputs.
pub fn set_intersection<B, Left, Right, Output, Less>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Output: MVec<B, Item = Left::Item>,
    Less: op::PredicateOp2<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::set_intersection_dispatch(left, exec.policy(), right, less)
}

/// Computes the sorted set union of two sorted inputs.
pub fn set_union<B, Left, Right, Output, Less>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<Output, Error>
where
    B: Backend,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Output: MVec<B, Item = Left::Item>,
    Less: op::PredicateOp2<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::set_union_dispatch(left, exec.policy(), right, less)
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
    Less: op::PredicateOp2<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::sort_dispatch(source, exec.policy(), less)
}

/// Sorts key-value pairs by key.
pub fn sort_by_key<B, Keys, Values, Less, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    Keys: MIter<B>,
    Values: MIter<B>,
    Less: op::PredicateOp2<B, Keys::Item>,
    KeyOutput: MVec<B, Item = Keys::Item>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::sort_by_key_dispatch(keys, exec.policy(), values, less)
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
    Less: op::PredicateOp2<B, Input::Item>,
{
    sort(exec, source, less)
}

/// Stable key-value sort. The current lower implementation is stable.
pub fn stable_sort_by_key<B, Keys, Values, Less, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    Keys: MIter<B>,
    Values: MIter<B>,
    Less: op::PredicateOp2<B, Keys::Item>,
    KeyOutput: MVec<B, Item = Keys::Item>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    sort_by_key(exec, keys, values, less)
}

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
    Op: op::UnaryOp<B, Input::Item, Output = Output::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::transform_dispatch(source, exec.policy(), op)
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
    Pred: op::PredicateOp2<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::unique_dispatch(source, exec.policy(), pred)
}

/// Removes consecutive duplicate keys and keeps their values.
pub fn unique_by_key<B, Keys, Values, Eq, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    eq: Eq,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Backend,
    Keys: MIter<B>,
    Values: MIter<B>,
    Eq: op::PredicateOp2<B, Keys::Item>,
    KeyOutput: MVec<B, Item = Keys::Item>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::unique_by_key_dispatch(keys, exec.policy(), values, eq)
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
    Less: op::PredicateOp2<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::upper_bound_dispatch(source, exec.policy(), value, less)
}
