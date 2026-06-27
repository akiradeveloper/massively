use super::*;

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
            B: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
                B,
                Inner = ($( crate::detail::DeviceVec<B, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( crate::detail::device::DeviceColumnView<B, $ty>, )+);

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
            B: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
                B,
                Inner = ($( crate::detail::DeviceVec<B, $ty>, )+),
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
                Option<crate::detail::device::DeviceColumnView<B, T>>,
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
                policy: &crate::detail::CubePolicy<B>,
                invert: bool,
            ) -> Result<crate::detail::api::PrecomputedSelection<B>, Error>
            where
                Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>,
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

            fn transform_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<B>,
                Op: op::UnaryOp<B, <Self as MIter<B>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner();
                let inner = <Output::Item as sealed::MItemDispatch<B>>::$transform(
                    policy,
                    $( input.$idx, )+
                    op,
                )?;
                output.write_from_inner(policy, inner)
            }

            fn transform_where_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                op: Op,
                stencil: crate::detail::api::PrecomputedSelection<B>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<B>,
                Op: op::UnaryOp<B, <Self as MIter<B>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner();
                let inner = <Output::Item as sealed::MItemDispatch<B>>::$transform(
                    policy,
                    $( input.$idx, )+
                    op,
                )?;
                output.write_where_from_inner(policy, inner, stencil)
            }

            fn reverse_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
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
                policy: &crate::detail::CubePolicy<B>,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::sort(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Less>::new())?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                keys: crate::detail::device::DeviceColumnView<B, K>,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                Less: op::BinaryPredicateOp<B, (K,)>,
                KeyOutput: MVec<B, Item = (K,)>,
                ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let values = self.into_inner();
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) =
                    crate::detail::sort_by_key(policy, (keys,), (values,), KernelOp::<B, Less>::new())?;
                Ok((
                    array_from_inner::<B, (K,), KeyOutput>(key_inner),
                    array_from_inner::<B, ($( $ty, )+), ValueOutput>(value_inner),
                ))
            }

            fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                keys: crate::detail::device::DeviceColumnView<B, K>,
                _eq: Eq,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                Eq: op::BinaryPredicateOp<B, (K,)>,
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
                policy: &crate::detail::CubePolicy<B>,
                keys: crate::detail::device::DeviceColumnView<B, K>,
                _key_eq: KeyEq,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<B, (K,)>,
                Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                keys: crate::detail::device::DeviceColumnView<B, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<B>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<B, (K,)>,
                Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                keys: crate::detail::device::DeviceColumnView<B, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<B>>::Item,
                _op: Op,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<B, (K,)>,
                Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
                KeyOutput: MVec<B, Item = (K,)>,
                ValueOutput: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let values = self.into_inner();
                let (key_inner, value_inner) = crate::detail::reduce_by_key(
                    policy,
                    (keys,),
                    values,
                    KernelOp::<B, KeyEq>::new(),
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
                policy: &crate::detail::CubePolicy<B>,
                left_keys: crate::detail::device::DeviceColumnView<B, K>,
                right_keys: crate::detail::device::DeviceColumnView<B, K>,
                right_values: RightValues,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                RightValues: MIter<B, Item = <Self as MIter<B>>::Item>,
                K: Scalar + 'static,
                Less: op::BinaryPredicateOp<B, (K,)>,
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

            fn gather_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                indices: crate::detail::device::DeviceColumnView<B, u32>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<B>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "gather output must match input shape".to_string(),
                    })?;
                    crate::detail::api::device_expr_gather_into_with_policy(
                        policy,
                        &input.$idx,
                        &indices,
                        &$tmp,
                    )?;
                )+
                Ok(())
            }

            fn reduce_dispatch<Op>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                init: <Self as MIter<B>>::Item,
                _op: Op,
            ) -> Result<<Self as MIter<B>>::Item, Error>
            where
                Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::reduce(policy, impl_miter_view!(input; $( $idx ),+), init, KernelOp::<B, Op>::new())
            }

            fn inclusive_scan_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                init: <Self as MIter<B>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<B, <Self as MIter<B>>::Item>,
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

            fn copy_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                stencil: crate::detail::api::PrecomputedSelection<B>,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::copy_where(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    stencil,
                    KernelOp::<B, StencilFlag>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn remove_if_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _pred: Pred,
            ) -> Result<Output, Error>
            where
                Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>,
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

            fn remove_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                stencil: crate::detail::api::PrecomputedSelection<B>,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::copy_where(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    stencil,
                    KernelOp::<B, StencilFlag>::new(),
                )?;
                Ok(array_from_inner::<B, ($( $ty, )+), Output>(inner))
            }

            fn count_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _pred: Pred,
            ) -> Result<usize, Error>
            where
                Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::count_if(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn all_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::all_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn any_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::any_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn none_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::none_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn find_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _pred: Pred,
            ) -> Result<Option<usize>, Error>
            where
                Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::find_if(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn partition_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _pred: Pred,
            ) -> Result<(Output, Output), Error>
            where
                Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::is_partitioned(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn replace_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                replacement: <Self as MIter<B>>::Item,
                stencil: crate::detail::api::PrecomputedSelection<B>,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                let inner = crate::detail::replace_where(
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
                policy: &crate::detail::CubePolicy<B>,
                _pred: Pred,
            ) -> Result<Output, Error>
            where
                Pred: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                _less: Less,
            ) -> Result<Option<usize>, Error>
            where
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::min_element(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Less>::new())
            }

            fn max_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _less: Less,
            ) -> Result<Option<usize>, Error>
            where
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::max_element(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Less>::new())
            }

            fn minmax_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _less: Less,
            ) -> Result<Option<(usize, usize)>, Error>
            where
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::minmax_element(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Less>::new())
            }

            fn adjacent_find_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _pred: Pred,
            ) -> Result<Option<usize>, Error>
            where
                Pred: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::adjacent_find(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Pred>::new())
            }

            fn lower_bound_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                value: <Self as MIter<B>>::Item,
                _less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::lower_bound(policy, impl_miter_view!(input; $( $idx ),+), value, KernelOp::<B, Less>::new())
            }

            fn upper_bound_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                value: <Self as MIter<B>>::Item,
                _less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::upper_bound(policy, impl_miter_view!(input; $( $idx ),+), value, KernelOp::<B, Less>::new())
            }

            fn equal_range_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                value: <Self as MIter<B>>::Item,
                _less: Less,
            ) -> Result<(usize, usize), Error>
            where
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::equal_range(policy, impl_miter_view!(input; $( $idx ),+), value, KernelOp::<B, Less>::new())
            }

            fn is_sorted_until_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::is_sorted_until(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Less>::new())
            }

            fn is_sorted_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                crate::detail::is_sorted(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<B, Less>::new())
            }

            fn gather_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                indices: crate::detail::device::DeviceColumnView<B, u32>,
                stencil: crate::detail::api::PrecomputedSelection<B>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<B>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "gather_where output must match input shape".to_string(),
                    })?;
                    crate::detail::api::device_expr_gather_where_into_with_policy(
                        policy,
                        &input.$idx,
                        &indices,
                        &stencil,
                        &$tmp,
                        KernelOp::<B, StencilFlag>::new(),
                    )?;
                )+
                Ok(())
            }

            fn scatter_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                indices: crate::detail::device::DeviceColumnView<B, u32>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<B>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "scatter output must match input shape".to_string(),
                    })?;
                    crate::detail::api::device_expr_scatter_into_with_policy(
                        policy,
                        &input.$idx,
                        &indices,
                        &$tmp,
                    )?;
                )+
                Ok(())
            }

            fn scatter_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                indices: crate::detail::device::DeviceColumnView<B, u32>,
                stencil: crate::detail::api::PrecomputedSelection<B>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<B, Item = <Self as MIter<B>>::Item>,
            {
                let input = self.into_inner();
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<B>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "scatter_where output must match input shape".to_string(),
                    })?;
                    crate::detail::api::device_expr_scatter_where_into_with_policy(
                        policy,
                        &input.$idx,
                        &indices,
                        &stencil,
                        &$tmp,
                        KernelOp::<B, StencilFlag>::new(),
                    )?;
                )+
                Ok(())
            }

            fn equal_dispatch<Right, Eq>(
                self,
                policy: &crate::detail::CubePolicy<B>,
                right: Right,
                _eq: Eq,
            ) -> Result<bool, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Right,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                needles: Needles,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Needles: MIter<B, Item = <Self as MIter<B>>::Item>,
                Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Right,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<B, Item = <Self as MIter<B>>::Item>,
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Self,
                _eq: Eq,
            ) -> Result<bool, Error>
            where
                Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Self,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                needles: Self,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Eq: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Self,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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
                policy: &crate::detail::CubePolicy<B>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<B, Item = <Self as MIter<B>>::Item>,
                Less: op::BinaryPredicateOp<B, <Self as MIter<B>>::Item>,
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

macro_rules! impl_miter_mut_soa {
    ($name:ident; $( $ty:ident : $idx:tt ),+) => {
        impl<'a, B, $( $ty ),+> MIterMut<B> for $name<$( DeviceSliceMut<'a, B, $ty> ),+>
        where
            B: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
                B,
                Inner = ($( crate::detail::DeviceVec<B, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( crate::detail::device::DeviceColumnMutView<B, $ty>, )+);

            fn len(&self) -> usize {
                self.0.len()
            }

            fn into_inner(self) -> Self::Inner {
                ($(
                    crate::detail::device::DeviceColumnMutView::from_slice(
                        &self.$idx.source.inner,
                        self.$idx.offset,
                        self.$idx.len,
                    ),
                )+)
            }

            fn write_from_inner(
                self,
                policy: &crate::detail::CubePolicy<B>,
                inner: <Self::Item as MItem<B>>::Inner,
            ) -> Result<(), Error> {
                let output = self.into_inner();
                $(
                    {
                        let input =
                            crate::detail::device::DeviceColumnView::from_column(&inner.$idx);
                        crate::detail::api::device_expr_collect_into_with_policy(
                            policy,
                            &input,
                            &output.$idx,
                        )?;
                    }
                )+
                Ok(())
            }

            fn write_where_from_inner(
                self,
                policy: &crate::detail::CubePolicy<B>,
                inner: <Self::Item as MItem<B>>::Inner,
                stencil: crate::detail::api::PrecomputedSelection<B>,
            ) -> Result<(), Error>
            {
                let output = self.into_inner();
                $(
                    {
                        let input =
                            crate::detail::device::DeviceColumnView::from_column(&inner.$idx);
                        crate::detail::api::device_expr_copy_where_into_with_policy(
                            policy,
                            &input,
                            &stencil,
                            &output.$idx,
                            KernelOp::<B, StencilFlag>::new(),
                        )?;
                    }
                )+
                Ok(())
            }

            fn replace_where_inner(
                self,
                policy: &crate::detail::CubePolicy<B>,
                replacement: Self::Item,
                stencil: crate::detail::api::PrecomputedSelection<B>,
            ) -> Result<(), Error>
            {
                let output = self.into_inner();
                $(
                    crate::detail::api::replace_where_into_with_policy(
                        policy,
                        replacement.$idx,
                        &stencil,
                        &output.$idx,
                        KernelOp::<B, StencilFlag>::new(),
                    )?;
                )+
                Ok(())
            }
        }

        impl<'a, B, $( $ty ),+> sealed::MIterMutDispatch<B> for $name<$( DeviceSliceMut<'a, B, $ty> ),+>
        where
            B: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
                B,
                Inner = ($( crate::detail::DeviceVec<B, $ty>, )+),
            >,
        {
            fn validate_executor(&self, exec: &Executor<B>) -> Result<(), Error> {
                $(
                    exec.ensure_policy_id(self.$idx.source.inner.policy_id())?;
                )+
                $(
                    ensure_same_len(self.$idx.len(), self.0.len())?;
                )+
                Ok(())
            }

            fn column_mut_view_by_index_inner<U: 'static>(
                &self,
                index: usize,
            ) -> Result<
                Option<crate::detail::device::DeviceColumnMutView<B, U>>,
                Error,
            >
            where
                U: Scalar,
            {
                $(
                    if index == $idx {
                        let source = &*self.$idx.source as &dyn Any;
                        let source = match source.downcast_ref::<DeviceVec<B, U>>() {
                            Some(source) => source,
                            None => return Ok(None),
                        };
                        return Ok(Some(crate::detail::device::DeviceColumnMutView::from_slice(
                            &source.inner,
                            self.$idx.offset,
                            self.$idx.len,
                        )));
                    }
                )+
                Ok(None)
            }

        }
    };
}

impl_miter_soa!(SoA2; A: 0: a, C: 1: c => transform_binary);
impl_miter_soa!(SoA3; A: 0: a, C: 1: c, D: 2: d => transform_ternary);
impl_miter_mut_soa!(SoA2; A: 0, C: 1);
impl_miter_mut_soa!(SoA3; A: 0, C: 1, D: 2);
