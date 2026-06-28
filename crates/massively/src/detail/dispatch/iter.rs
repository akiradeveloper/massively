use super::*;

pub trait MIterDispatch<R: Runtime>: Sized {
    fn validate_executor(&self, _exec: &Executor<R>) -> Result<(), Error> {
        Ok(())
    }

    fn transform_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Output: MIterMut<R>,
        Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>;

    fn transform_where_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Output: MIterMut<R>,
        Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>;

    fn reverse_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn sort_dispatch<Less, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        K: Scalar + 'static,
        Less: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: MVec<R, Item = (K,)>,
        ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn sort_by_key_dispatch<Values, Less, KeyOutput, ValueOutput>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _values: Values,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MVec<R, Item = <Values as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "sort_by_key is not supported for this key iterator shape".to_string(),
        })
    }

    fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        _eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        K: Scalar + 'static,
        Eq: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: MVec<R, Item = (K,)>,
        ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn unique_by_key_dispatch<Values, Eq, KeyOutput, ValueOutput>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _values: Values,
        _eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MVec<R, Item = <Values as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "unique_by_key is not supported for this key iterator shape".to_string(),
        })
    }

    fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        key_eq: KeyEq,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        K: Scalar + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn inclusive_scan_by_key_dispatch<Values, KeyEq, Op, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _values: Values,
        _key_eq: KeyEq,
        _op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
        Output: MVec<R, Item = <Values as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "inclusive_scan_by_key is not supported for this key iterator shape"
                .to_string(),
        })
    }

    fn exclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        key_eq: KeyEq,
        _init: <Self as MIter<R>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        K: Scalar + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn exclusive_scan_by_key_dispatch<Values, KeyEq, Op, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _values: Values,
        _key_eq: KeyEq,
        _init: <Values as MIter<R>>::Item,
        _op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
        Output: MVec<R, Item = <Values as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "exclusive_scan_by_key is not supported for this key iterator shape"
                .to_string(),
        })
    }

    fn reduce_by_single_key_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        key_eq: KeyEq,
        _init: <Self as MIter<R>>::Item,
        op: Op,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        K: Scalar + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MVec<R, Item = (K,)>,
        ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn reduce_by_key_dispatch<Values, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _values: Values,
        _key_eq: KeyEq,
        _init: <Values as MIter<R>>::Item,
        _op: Op,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
        KeyOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MVec<R, Item = <Values as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "reduce_by_key is not supported for this key iterator shape".to_string(),
        })
    }

    fn merge_by_single_key_same_dispatch<K, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        left_keys: crate::detail::device::DeviceColumnView<R, K>,
        right_keys: crate::detail::device::DeviceColumnView<R, K>,
        _right_values: RightValues,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        RightValues: MIter<R, Item = <Self as MIter<R>>::Item, Inner = <Self as MIter<R>>::Inner>,
        K: Scalar + 'static,
        Less: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: MVec<R, Item = (K,)>,
        ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (left_keys, right_keys);
        Err(Error::Launch {
            message: "merge_by_key is not supported for this iterator shape".to_string(),
        })
    }

    fn gather_dispatch<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
            + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
        <Indices as crate::detail::device::KernelColumn>::Expr:
            crate::expr::GpuExpr<u32> + crate::expr::DeviceGpuExpr<u32>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>;

    fn gather_where_dispatch<Indices, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _indices: Indices,
        _stencil: crate::detail::api::PrecomputedSelection<R>,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
            + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
        <Indices as crate::detail::device::KernelColumn>::Expr:
            crate::expr::GpuExpr<u32> + crate::expr::DeviceGpuExpr<u32>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "gather_where is not supported for this iterator shape".to_string(),
        })
    }

    fn scatter_dispatch<Indices, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _indices: Indices,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
            + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
        <Indices as crate::detail::device::KernelColumn>::Expr: crate::expr::GpuExpr<u32>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "scatter is not supported for this iterator shape".to_string(),
        })
    }

    fn scatter_where_dispatch<Indices, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _indices: Indices,
        _stencil: crate::detail::api::PrecomputedSelection<R>,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
            + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
        <Indices as crate::detail::device::KernelColumn>::Expr:
            crate::expr::GpuExpr<u32> + crate::expr::DeviceGpuExpr<u32>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "scatter_where is not supported for this iterator shape".to_string(),
        })
    }

    fn reduce_dispatch<Op>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _init: <Self as MIter<R>>::Item,
        op: Op,
    ) -> Result<<Self as MIter<R>>::Item, Error>
    where
        Self: MIter<R>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>;

    fn inclusive_scan_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn exclusive_scan_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _init: <Self as MIter<R>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn adjacent_difference_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn copy_where_dispatch<Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn remove_if_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn remove_where_dispatch<Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn count_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<usize, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>;

    fn all_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>;

    fn any_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>;

    fn none_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>;

    fn find_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>;

    fn partition_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<(Output, Output), Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn is_partitioned_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>;

    fn replace_where_dispatch<Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        replacement: <Self as MIter<R>>::Item,
        _stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    #[doc(hidden)]
    fn selection_stencil_dispatch<Pred>(
        &self,
        _policy: &crate::detail::CubePolicy<R>,
        _invert: bool,
    ) -> Result<crate::detail::api::PrecomputedSelection<R>, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "stencil is not supported for this iterator shape".to_string(),
        })
    }

    fn unique_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>;

    fn min_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>;

    fn max_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>;

    fn minmax_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<Option<(usize, usize)>, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>;

    fn adjacent_find_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<R>,
        Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>;

    fn equal_dispatch<Right, Eq>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _eq: Eq,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item, Inner = <Self as MIter<R>>::Inner>,
        Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "equal is not supported for this iterator shape".to_string(),
        })
    }

    fn mismatch_dispatch<Right, Eq>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item, Inner = <Self as MIter<R>>::Inner>,
        Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "mismatch is not supported for this iterator shape".to_string(),
        })
    }

    fn find_first_of_dispatch<Needles, Eq>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _needles: Needles,
        _eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<R>,
        Needles: MIter<R, Item = <Self as MIter<R>>::Item, Inner = <Self as MIter<R>>::Inner>,
        Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "find_first_of is not supported for this iterator shape".to_string(),
        })
    }

    fn lower_bound_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        value: <Self as MIter<R>>::Item,
        _less: Less,
    ) -> Result<usize, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>;

    fn upper_bound_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        value: <Self as MIter<R>>::Item,
        _less: Less,
    ) -> Result<usize, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>;

    fn equal_range_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        value: <Self as MIter<R>>::Item,
        _less: Less,
    ) -> Result<(usize, usize), Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>;

    fn is_sorted_until_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<usize, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>;

    fn is_sorted_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>;

    fn lexicographical_compare_dispatch<Right, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _less: Less,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item, Inner = <Self as MIter<R>>::Inner>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "lexicographical_compare is not supported for this iterator shape".to_string(),
        })
    }

    fn merge_dispatch<Right, Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item, Inner = <Self as MIter<R>>::Inner>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "merge is not supported for this iterator shape".to_string(),
        })
    }

    fn set_union_dispatch<Right, Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item, Inner = <Self as MIter<R>>::Inner>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "set_union is not supported for this iterator shape".to_string(),
        })
    }

    fn set_intersection_dispatch<Right, Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item, Inner = <Self as MIter<R>>::Inner>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "set_intersection is not supported for this iterator shape".to_string(),
        })
    }

    fn set_difference_dispatch<Right, Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item, Inner = <Self as MIter<R>>::Inner>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "set_difference is not supported for this iterator shape".to_string(),
        })
    }

    fn inner_product_dispatch<Right, TransformOp, ReduceOp, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _transform_op: TransformOp,
        _init: Output,
        _reduce_op: ReduceOp,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Right: MIter<R>,
        TransformOp:
            op::BinaryOp<R, <Self as MIter<R>>::Item, <Right as MIter<R>>::Item, Output = Output>,
        Output: MItem<R>,
        ReduceOp: op::ReductionOp<R, Output>,
    {
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }

    fn equal_same_dispatch<Eq>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Self,
        _eq: Eq,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "equal is not supported for this iterator shape".to_string(),
        })
    }

    fn mismatch_same_dispatch<Eq>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Self,
        _eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<R>,
        Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "mismatch is not supported for this iterator shape".to_string(),
        })
    }

    fn find_first_of_same_dispatch<Eq>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _needles: Self,
        _eq: Eq,
    ) -> Result<Option<usize>, Error>
    where
        Self: MIter<R>,
        Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "find_first_of is not supported for this iterator shape".to_string(),
        })
    }

    fn lexicographical_compare_same_dispatch<Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Self,
        _less: Less,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "lexicographical_compare is not supported for this iterator shape".to_string(),
        })
    }

    fn merge_same_dispatch<Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Self,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "merge is not supported for this iterator shape".to_string(),
        })
    }

    fn set_union_same_dispatch<Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Self,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "set_union is not supported for this iterator shape".to_string(),
        })
    }

    fn set_intersection_same_dispatch<Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Self,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "set_intersection is not supported for this iterator shape".to_string(),
        })
    }

    fn set_difference_same_dispatch<Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Self,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "set_difference is not supported for this iterator shape".to_string(),
        })
    }

    fn inner_product_same_dispatch<TransformOp, ReduceOp, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Self,
        _transform_op: TransformOp,
        _init: Output,
        _reduce_op: ReduceOp,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        TransformOp:
            op::BinaryOp<R, <Self as MIter<R>>::Item, <Self as MIter<R>>::Item, Output = Output>,
        Output: MItem<R>,
        ReduceOp: op::ReductionOp<R, Output>,
    {
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }

    fn merge_by_key_dispatch<RightKeys, LeftValues, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right_keys: RightKeys,
        _left_values: LeftValues,
        _right_values: RightValues,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        RightKeys: MIter<R, Item = <Self as MIter<R>>::Item, Inner = <Self as MIter<R>>::Inner>,
        LeftValues: MIter<R>,
        RightValues: MIter<R, Item = <LeftValues as MIter<R>>::Item, Inner = <LeftValues as MIter<R>>::Inner>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MVec<R, Item = <LeftValues as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "merge_by_key is not supported for this key iterator shape".to_string(),
        })
    }
}
