use super::{Error, Executor, MIter, MIterMut, MVec, op};

use cubecl::prelude::Runtime;

pub trait ToHostDispatch<B: Runtime> {
    type Output;

    fn to_host_with(&self, exec: &Executor<B>) -> Result<Self::Output, Error>;
}

pub trait MIterMutDispatch<B: Runtime>: Sized {
    fn validate_executor(&self, _exec: &Executor<B>) -> Result<(), Error> {
        Ok(())
    }

    fn column_mut_view_inner<T: 'static>(
        &self,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<B, T>>, Error>
    where
        T: super::Scalar,
    {
        Ok(None)
    }

    fn column_mut_view_by_index_inner<T: 'static>(
        &self,
        index: usize,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<B, T>>, Error>
    where
        T: super::Scalar,
    {
        if index == 0 {
            self.column_mut_view_inner::<T>()
        } else {
            Ok(None)
        }
    }

    fn replace_where_dispatch<Stencil>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _replacement: <Self as MIterMut<B>>::Item,
        _stencil: Stencil,
    ) -> Result<(), Error>
    where
        Self: MIterMut<B>,
        Stencil: MIter<B, Item = (u32,)>,
    {
        Err(Error::Launch {
            message: "replace_where is not supported for this output shape".to_string(),
        })
    }
}

pub trait MIterDispatch<B: Runtime>: Sized {
    fn validate_executor(&self, _exec: &Executor<B>) -> Result<(), Error> {
        Ok(())
    }

    fn index_inner(&self) -> Option<(&crate::detail::DeviceVec<B, u32>,)> {
        None
    }

    fn column_inner<T: 'static>(&self) -> Option<&crate::detail::DeviceVec<B, T>> {
        None
    }

    fn column_view_inner<T: 'static>(
        &self,
    ) -> Result<Option<crate::detail::device::DeviceColumnView<B, T>>, Error>
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
    ) -> Result<Option<crate::detail::device::DeviceColumnView<B, T>>, Error>
    where
        T: super::Scalar,
    {
        if index == 0 {
            self.column_view_inner::<T>()
        } else {
            Ok(None)
        }
    }

    fn transform_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<B>,
        Output: MIterMut<B>,
        Op: op::UnaryOp<B, <Self as MIter<B>>::Item, Output = Output::Item>;

    fn transform_where_dispatch<Op, Stencil, Output>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        op: Op,
        stencil: Stencil,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<B>,
        Stencil: MIter<B, Item = (u32,)>,
        Output: MIterMut<B>,
        Op: op::UnaryOp<B, <Self as MIter<B>>::Item, Output = Output::Item>;

    fn reverse_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<B>,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn sort_dispatch<Less, Output>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        keys: crate::detail::device::DeviceColumnView<B, K>,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<B>,
        K: super::Scalar + 'static,
        Less: op::BinaryPredicateOp<B, (K,)>,
        KeyOutput: MVec<B, Item = (K,)>,
        ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn sort_by_key_dispatch<Values, Less, KeyOutput, ValueOutput>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _values: Values,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<B>,
        Values: MIter<B>,
        <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
        KeyOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
        ValueOutput: MVec<B, Item = <Values as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "sort_by_key is not supported for this key iterator shape".to_string(),
        })
    }

    fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        keys: crate::detail::device::DeviceColumnView<B, K>,
        _eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<B>,
        K: super::Scalar + 'static,
        Eq: op::BinaryPredicateOp<B, (K,)>,
        KeyOutput: MVec<B, Item = (K,)>,
        ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn unique_by_key_dispatch<Values, Eq, KeyOutput, ValueOutput>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _values: Values,
        _eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<B>,
        Values: MIter<B>,
        <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
        Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
        KeyOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
        ValueOutput: MVec<B, Item = <Values as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "unique_by_key is not supported for this key iterator shape".to_string(),
        })
    }

    fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        keys: crate::detail::device::DeviceColumnView<B, K>,
        key_eq: KeyEq,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        K: super::Scalar + 'static,
        KeyEq: op::BinaryPredicateOp<B, (K,)>,
        Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn inclusive_scan_by_key_dispatch<Values, KeyEq, Op, Output>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _values: Values,
        _key_eq: KeyEq,
        _op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Values: MIter<B>,
        <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
        Op: op::ReductionOp<B, <Values as MIter<B>>::Item>,
        Output: MVec<B, Item = <Values as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "inclusive_scan_by_key is not supported for this key iterator shape"
                .to_string(),
        })
    }

    fn exclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        keys: crate::detail::device::DeviceColumnView<B, K>,
        key_eq: KeyEq,
        _init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        K: super::Scalar + 'static,
        KeyEq: op::BinaryPredicateOp<B, (K,)>,
        Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn exclusive_scan_by_key_dispatch<Values, KeyEq, Op, Output>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _values: Values,
        _key_eq: KeyEq,
        _init: <Values as MIter<B>>::Item,
        _op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Values: MIter<B>,
        <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
        Op: op::ReductionOp<B, <Values as MIter<B>>::Item>,
        Output: MVec<B, Item = <Values as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "exclusive_scan_by_key is not supported for this key iterator shape"
                .to_string(),
        })
    }

    fn reduce_by_single_key_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        keys: crate::detail::device::DeviceColumnView<B, K>,
        key_eq: KeyEq,
        _init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<B>,
        K: super::Scalar + 'static,
        KeyEq: op::BinaryPredicateOp<B, (K,)>,
        Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
        KeyOutput: MVec<B, Item = (K,)>,
        ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn reduce_by_key_dispatch<Values, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _values: Values,
        _key_eq: KeyEq,
        _init: <Values as MIter<B>>::Item,
        _op: Op,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<B>,
        Values: MIter<B>,
        <Self as MIter<B>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
        Op: op::ReductionOp<B, <Values as MIter<B>>::Item>,
        KeyOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
        ValueOutput: MVec<B, Item = <Values as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "reduce_by_key is not supported for this key iterator shape".to_string(),
        })
    }

    fn merge_by_single_key_same_dispatch<K, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        left_keys: crate::detail::device::DeviceColumnView<B, K>,
        right_keys: crate::detail::device::DeviceColumnView<B, K>,
        _right_values: RightValues,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<B>,
        RightValues: MIter<B, Item = <Self as MIter<B>>::Item>,
        K: super::Scalar + 'static,
        Less: op::BinaryPredicateOp<B, (K,)>,
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
        policy: &crate::detail::CubePolicy<B>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<B>,
        Indices: MIter<B, Item = (u32,)>,
        Output: MIterMut<B, Item = <Self as MIter<B>>::Item>;

    fn gather_where_dispatch<Indices, Stencil, Output>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _indices: Indices,
        _stencil: Stencil,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<B>,
        Indices: MIter<B, Item = (u32,)>,
        Stencil: MIter<B, Item = (u32,)>,
        Output: MIterMut<B, Item = <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "gather_where is not supported for this iterator shape".to_string(),
        })
    }

    fn scatter_dispatch<Indices, Output>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _indices: Indices,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<B>,
        Indices: MIter<B, Item = (u32,)>,
        Output: MIterMut<B, Item = <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "scatter is not supported for this iterator shape".to_string(),
        })
    }

    fn scatter_where_dispatch<Indices, Stencil, Output>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _indices: Indices,
        _stencil: Stencil,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<B>,
        Indices: MIter<B, Item = (u32,)>,
        Stencil: MIter<B, Item = (u32,)>,
        Output: MIterMut<B, Item = <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "scatter_where is not supported for this iterator shape".to_string(),
        })
    }

    fn reduce_dispatch<Op>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        _init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<<Self as MIter<B>>::Item, Error>
    where
        Self: MIter<B>,
        Op: op::ReductionOp<B, <Self as MIter<B>>::Item>;

    fn inclusive_scan_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn exclusive_scan_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        _init: <Self as MIter<B>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn adjacent_difference_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn copy_where_dispatch<Stencil, Output>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _stencil: Stencil,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Stencil: MIter<B, Item = (u32,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn remove_if_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn remove_where_dispatch<Stencil, Output>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _stencil: Stencil,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Stencil: MIter<B, Item = (u32,)>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn count_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        pred: Pred,
    ) -> Result<usize, Error>
    where
        Self: MIter<B>,
        Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>;

    fn all_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Self: MIter<B>,
        Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>;

    fn any_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Self: MIter<B>,
        Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>;

    fn none_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Self: MIter<B>,
        Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>;

    fn find_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        pred: Pred,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<B>,
        Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>;

    fn partition_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        pred: Pred,
    ) -> Result<(Output, Output), Error>
    where
        Self: MIter<B>,
        Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn is_partitioned_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Self: MIter<B>,
        Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>;

    fn replace_where_dispatch<Stencil, Output>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
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
        _policy: &crate::detail::CubePolicy<B>,
        _invert: bool,
    ) -> Result<crate::detail::api::PrecomputedSelection<B>, Error>
    where
        Self: MIter<B>,
        Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "stencil is not supported for this iterator shape".to_string(),
        })
    }

    fn unique_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Pred: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>;

    fn min_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        less: Less,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<B>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>;

    fn max_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        less: Less,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<B>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>;

    fn minmax_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        less: Less,
    ) -> Result<Option<(usize, usize)>, Error>
    where
        Self: MIter<B>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>;

    fn adjacent_find_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        pred: Pred,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<B>,
        Pred: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>;

    fn equal_dispatch<Right, Eq>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Right,
        _eq: Eq,
    ) -> Result<bool, Error>
    where
        Self: MIter<B>,
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "equal is not supported for this iterator shape".to_string(),
        })
    }

    fn mismatch_dispatch<Right, Eq>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Right,
        _eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<B>,
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "mismatch is not supported for this iterator shape".to_string(),
        })
    }

    fn find_first_of_dispatch<Needles, Eq>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _needles: Needles,
        _eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<B>,
        Needles: MIter<B, Item = <Self as MIter<B>>::Item>,
        Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "find_first_of is not supported for this iterator shape".to_string(),
        })
    }

    fn lower_bound_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        value: <Self as MIter<B>>::Item,
        _less: Less,
    ) -> Result<usize, Error>
    where
        Self: MIter<B>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>;

    fn upper_bound_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        value: <Self as MIter<B>>::Item,
        _less: Less,
    ) -> Result<usize, Error>
    where
        Self: MIter<B>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>;

    fn equal_range_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        value: <Self as MIter<B>>::Item,
        _less: Less,
    ) -> Result<(usize, usize), Error>
    where
        Self: MIter<B>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>;

    fn is_sorted_until_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        less: Less,
    ) -> Result<usize, Error>
    where
        Self: MIter<B>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>;

    fn is_sorted_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<B>,
        less: Less,
    ) -> Result<bool, Error>
    where
        Self: MIter<B>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>;

    fn lexicographical_compare_dispatch<Right, Less>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Right,
        _less: Less,
    ) -> Result<bool, Error>
    where
        Self: MIter<B>,
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "lexicographical_compare is not supported for this iterator shape".to_string(),
        })
    }

    fn merge_dispatch<Right, Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "merge is not supported for this iterator shape".to_string(),
        })
    }

    fn set_union_dispatch<Right, Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "set_union is not supported for this iterator shape".to_string(),
        })
    }

    fn set_intersection_dispatch<Right, Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "set_intersection is not supported for this iterator shape".to_string(),
        })
    }

    fn set_difference_dispatch<Right, Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Right: MIter<B, Item = <Self as MIter<B>>::Item>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "set_difference is not supported for this iterator shape".to_string(),
        })
    }

    fn inner_product_dispatch<Right, TransformOp, ReduceOp, Output>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Right,
        _transform_op: TransformOp,
        _init: Output,
        _reduce_op: ReduceOp,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Right: MIter<B>,
        TransformOp:
            op::BinaryOp<B, <Self as MIter<B>>::Item, <Right as MIter<B>>::Item, Output = Output>,
        Output: super::MItem<B>,
        ReduceOp: op::ReductionOp<B, Output>,
    {
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }

    fn equal_same_dispatch<Eq>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Self,
        _eq: Eq,
    ) -> Result<bool, Error>
    where
        Self: MIter<B>,
        Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "equal is not supported for this iterator shape".to_string(),
        })
    }

    fn mismatch_same_dispatch<Eq>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Self,
        _eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<B>,
        Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "mismatch is not supported for this iterator shape".to_string(),
        })
    }

    fn find_first_of_same_dispatch<Eq>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _needles: Self,
        _eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<B>,
        Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "find_first_of is not supported for this iterator shape".to_string(),
        })
    }

    fn lexicographical_compare_same_dispatch<Less>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Self,
        _less: Less,
    ) -> Result<bool, Error>
    where
        Self: MIter<B>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "lexicographical_compare is not supported for this iterator shape".to_string(),
        })
    }

    fn merge_same_dispatch<Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Self,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "merge is not supported for this iterator shape".to_string(),
        })
    }

    fn set_union_same_dispatch<Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Self,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "set_union is not supported for this iterator shape".to_string(),
        })
    }

    fn set_intersection_same_dispatch<Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Self,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "set_intersection is not supported for this iterator shape".to_string(),
        })
    }

    fn set_difference_same_dispatch<Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Self,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        Output: MVec<B, Item = <Self as MIter<B>>::Item>,
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "set_difference is not supported for this iterator shape".to_string(),
        })
    }

    fn inner_product_same_dispatch<TransformOp, ReduceOp, Output>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
        _right: Self,
        _transform_op: TransformOp,
        _init: Output,
        _reduce_op: ReduceOp,
    ) -> Result<Output, Error>
    where
        Self: MIter<B>,
        TransformOp:
            op::BinaryOp<B, <Self as MIter<B>>::Item, <Self as MIter<B>>::Item, Output = Output>,
        Output: super::MItem<B>,
        ReduceOp: op::ReductionOp<B, Output>,
    {
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }

    fn merge_by_key_dispatch<RightKeys, LeftValues, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        _policy: &crate::detail::CubePolicy<B>,
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
        Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
        KeyOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
        ValueOutput: MVec<B, Item = <LeftValues as MIter<B>>::Item>,
    {
        Err(Error::Launch {
            message: "merge_by_key is not supported for this key iterator shape".to_string(),
        })
    }
}

pub trait MItemDispatch<B: Runtime>: Sized {
    fn transform_unary<Input, Op>(
        policy: &crate::detail::CubePolicy<B>,
        input: crate::detail::device::DeviceColumnView<B, Input>,
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
        policy: &crate::detail::CubePolicy<B>,
        left: crate::detail::device::DeviceColumnView<B, Left>,
        right: crate::detail::device::DeviceColumnView<B, Right>,
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
        policy: &crate::detail::CubePolicy<B>,
        first: crate::detail::device::DeviceColumnView<B, First>,
        second: crate::detail::device::DeviceColumnView<B, Second>,
        third: crate::detail::device::DeviceColumnView<B, Third>,
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
        policy: &crate::detail::CubePolicy<B>,
        input: <Self as super::MItem<B>>::Inner,
        init: Self,
        op: Op,
    ) -> Result<Self, Error>
    where
        Self: super::MItem<B>,
        Op: op::ReductionOp<B, Self>,
    {
        let _ = (policy, input, init, op);
        Err(Error::Launch {
            message: "reduce is not supported for this item shape".to_string(),
        })
    }

    fn inner_product_with_right_item<LeftIter, RightIter, TransformOp, ReduceOp, Output>(
        policy: &crate::detail::CubePolicy<B>,
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
        TransformOp: op::BinaryOp<B, Self, <RightIter as MIter<B>>::Item, Output = Output>,
        Output: super::MItem<B>,
        ReduceOp: op::ReductionOp<B, Output>,
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
        policy: &crate::detail::CubePolicy<B>,
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
        TransformOp: op::BinaryOp<B, (LeftScalar,), Self, Output = Output>,
        Output: super::MItem<B>,
        ReduceOp: op::ReductionOp<B, Output>,
    {
        let _ = (policy, left, right, transform_op, init, reduce_op);
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }
}
