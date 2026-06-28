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
        impl<R, $( $ty ),+> MIter<R> for $name<$( $ty ),+>
        where
            R: Runtime,
            $( $ty: MSlice<R>, )+
            ($( <$ty as MSlice<R>>::Item, )+): MItem<
                R,
                Inner = ($( crate::detail::DeviceVec<R, <$ty as MSlice<R>>::Item>, )+),
            >,
        {
            type Item = ($( <$ty as MSlice<R>>::Item, )+);
            type Inner = ($( crate::detail::device::DeviceColumnView<R, <$ty as MSlice<R>>::Item>, )+);

            fn len(&self) -> usize {
                self.0.len()
            }

            fn into_inner(self) -> Self::Inner {
                unreachable!("read-only MIter lowering requires a CubePolicy")
            }

            fn into_inner_with_policy(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<Self::Inner, Error> {
                Ok(($( lower_mslice_column(self.$idx, policy)?, )+))
            }
        }

        impl<R, $( $ty ),+> sealed::MIterDispatch<R> for $name<$( $ty ),+>
        where
            R: Runtime,
            $( $ty: MSlice<R>, )+
            ($( <$ty as MSlice<R>>::Item, )+): MItem<
                R,
                Inner = ($( crate::detail::DeviceVec<R, <$ty as MSlice<R>>::Item>, )+),
            >,
        {
            fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
                $(
                    self.$idx.validate_executor(exec)?;
                )+
                Ok(())
            }

            fn column_view_by_index_inner<T: 'static>(
                &self,
                index: usize,
            ) -> Result<
                Option<crate::detail::device::DeviceColumnView<R, T>>,
                Error,
            >
            where
                T: Scalar,
            {
                $(
                    if index == $idx {
                        return self.$idx.column_view::<T>();
                    }
                )+
                Ok(None)
            }

            fn column_view_by_index_with_policy<T: 'static>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                index: usize,
            ) -> Result<
                Option<crate::detail::device::DeviceColumnView<R, T>>,
                Error,
            >
            where
                Self: MIter<R>,
                T: Scalar,
            {
                $(
                    if index == $idx {
                        return lower_mslice_column_as(self.$idx, policy);
                    }
                )+
                Ok(None)
            }

            fn transform_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = <Output::Item as sealed::MItemDispatch<R>>::$transform(
                    policy,
                    $( input.$idx, )+
                    op,
                )?;
                output.write_from_inner(policy, inner)
            }

            fn transform_where_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                op: Op,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = <Output::Item as sealed::MItemDispatch<R>>::$transform(
                    policy,
                    $( input.$idx, )+
                    op,
                )?;
                output.write_where_from_inner(policy, inner, stencil)
            }

            fn reverse_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::reverse(policy, impl_miter_view!(input; $( $idx ),+))?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn sort_dispatch<Less, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::sort(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MVec<R, Item = (K,)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) =
                    crate::detail::sort_by_key(policy, (keys,), (values,), KernelOp::<R, Less>::new())?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _eq: Eq,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                Eq: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MVec<R, Item = (K,)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) =
                    crate::detail::unique_by_key(policy, (keys,), (values,), KernelOp::<R, Eq>::new())?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let inner = crate::detail::inclusive_scan_by_key(
                    policy,
                    keys,
                    values,
                    KernelTuple1Op::<R, KeyEq>::new(),
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn exclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let inner = crate::detail::exclusive_scan_by_key(
                    policy,
                    keys,
                    values,
                    KernelTuple1Op::<R, KeyEq>::new(),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn reduce_by_single_key_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MVec<R, Item = (K,)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::reduce_by_key(
                    policy,
                    (keys,),
                    values,
                    KernelOp::<R, KeyEq>::new(),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn merge_by_single_key_same_dispatch<K, RightValues, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                left_keys: crate::detail::device::DeviceColumnView<R, K>,
                right_keys: crate::detail::device::DeviceColumnView<R, K>,
                right_values: RightValues,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K: Scalar + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MVec<R, Item = (K,)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_inner_with_policy(policy)?;
                let right_values = ($(
                    <RightValues as sealed::MIterDispatch<R>>::column_view_by_index_inner::<<$ty as MSlice<R>>::Item>(
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
                    KernelTuple1Op::<R, Less>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn gather_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr: crate::expr::GpuExpr<u32>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<<$ty as MSlice<R>>::Item>(
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
                policy: &crate::detail::CubePolicy<R>,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<<Self as MIter<R>>::Item, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::reduce(policy, impl_miter_view!(input; $( $idx ),+), init, KernelOp::<R, Op>::new())
            }

            fn inclusive_scan_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::inclusive_scan(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn exclusive_scan_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::exclusive_scan(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn adjacent_difference_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::adjacent_difference(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn copy_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::copy_where(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    stencil,
                    KernelOp::<R, StencilFlag>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn remove_if_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Output, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::remove_if(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Pred>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn remove_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::copy_where(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    stencil,
                    KernelOp::<R, StencilFlag>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn count_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<usize, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::count_if(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn all_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::all_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn any_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::any_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn none_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::none_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn find_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Option<usize>, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::find_if(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn partition_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<(Output, Output), Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (matching, failing) = crate::detail::partition(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Pred>::new(),
                )?;
                Ok((
                    array_from_inner::<R, <Self as MIter<R>>::Item, Output>(matching),
                    array_from_inner::<R, <Self as MIter<R>>::Item, Output>(failing),
                ))
            }

            fn is_partitioned_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::is_partitioned(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn replace_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                replacement: <Self as MIter<R>>::Item,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::replace_where(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    replacement,
                    stencil,
                    KernelOp::<R, StencilFlag>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn unique_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Output, Error>
            where
                Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::unique(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Pred>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn min_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Option<usize>, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::min_element(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())
            }

            fn max_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Option<usize>, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::max_element(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())
            }

            fn minmax_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Option<(usize, usize)>, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::minmax_element(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())
            }

            fn adjacent_find_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Option<usize>, Error>
            where
                Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::adjacent_find(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn lower_bound_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                value: <Self as MIter<R>>::Item,
                _less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::lower_bound(policy, impl_miter_view!(input; $( $idx ),+), value, KernelOp::<R, Less>::new())
            }

            fn upper_bound_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                value: <Self as MIter<R>>::Item,
                _less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::upper_bound(policy, impl_miter_view!(input; $( $idx ),+), value, KernelOp::<R, Less>::new())
            }

            fn equal_range_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                value: <Self as MIter<R>>::Item,
                _less: Less,
            ) -> Result<(usize, usize), Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::equal_range(policy, impl_miter_view!(input; $( $idx ),+), value, KernelOp::<R, Less>::new())
            }

            fn is_sorted_until_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::is_sorted_until(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())
            }

            fn is_sorted_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::is_sorted(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())
            }

            fn gather_where_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr:
                    crate::expr::GpuExpr<u32> + crate::expr::DeviceGpuExpr<u32>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<<$ty as MSlice<R>>::Item>(
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
                        KernelOp::<R, StencilFlag>::new(),
                    )?;
                )+
                Ok(())
            }

            fn scatter_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr: crate::expr::GpuExpr<u32>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<<$ty as MSlice<R>>::Item>(
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

            fn scatter_where_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr:
                    crate::expr::GpuExpr<u32> + crate::expr::DeviceGpuExpr<u32>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<<$ty as MSlice<R>>::Item>(
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
                        KernelOp::<R, StencilFlag>::new(),
                    )?;
                )+
                Ok(())
            }

            fn equal_dispatch<Right, Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _eq: Eq,
            ) -> Result<bool, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = ($( column_view_at::<R, Right, <$ty as MSlice<R>>::Item>(&right, $idx, "equal")?, )+);
                crate::detail::equal(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Eq>::new(),
                )
            }

            fn mismatch_dispatch<Right, Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = ($( column_view_at::<R, Right, <$ty as MSlice<R>>::Item>(&right, $idx, "mismatch")?, )+);
                crate::detail::mismatch(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Eq>::new(),
                )
            }

            fn find_first_of_dispatch<Needles, Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                needles: Needles,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Needles: MIter<R, Item = <Self as MIter<R>>::Item>,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let needles = ($( column_view_at::<R, Needles, <$ty as MSlice<R>>::Item>(&needles, $idx, "find_first_of")?, )+);
                crate::detail::find_first_of(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    impl_miter_view!(needles; $( $idx ),+),
                    KernelOp::<R, Eq>::new(),
                )
            }

            fn lexicographical_compare_dispatch<Right, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = ($( column_view_at::<R, Right, <$ty as MSlice<R>>::Item>(&right, $idx, "lexicographical_compare")?, )+);
                crate::detail::lexicographical_compare(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )
            }

            fn merge_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = ($( column_view_at::<R, Right, <$ty as MSlice<R>>::Item>(&right, $idx, "merge")?, )+);
                let inner = crate::detail::merge(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_union_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = ($( column_view_at::<R, Right, <$ty as MSlice<R>>::Item>(&right, $idx, "set_union")?, )+);
                let inner = crate::detail::set_union(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_intersection_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = ($( column_view_at::<R, Right, <$ty as MSlice<R>>::Item>(&right, $idx, "set_intersection")?, )+);
                let inner = crate::detail::set_intersection(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_difference_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = ($( column_view_at::<R, Right, <$ty as MSlice<R>>::Item>(&right, $idx, "set_difference")?, )+);
                let inner = crate::detail::set_difference(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn equal_same_dispatch<Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _eq: Eq,
            ) -> Result<bool, Error>
            where
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                crate::detail::equal(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Eq>::new(),
                )
            }

            fn mismatch_same_dispatch<Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                crate::detail::mismatch(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Eq>::new(),
                )
            }

            fn find_first_of_same_dispatch<Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                needles: Self,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let needles = needles.into_inner_with_policy(policy)?;
                crate::detail::find_first_of(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    impl_miter_view!(needles; $( $idx ),+),
                    KernelOp::<R, Eq>::new(),
                )
            }

            fn lexicographical_compare_same_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                crate::detail::lexicographical_compare(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )
            }

            fn merge_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                let inner = crate::detail::merge(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_union_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                let inner = crate::detail::set_union(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_intersection_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                let inner = crate::detail::set_intersection(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_difference_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                let inner = crate::detail::set_difference(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

        }
    };
}

macro_rules! impl_miter_mut_soa {
    ($name:ident; $( $ty:ident : $idx:tt ),+) => {
        impl<'a, R, $( $ty ),+> MIterMut<R> for $name<$( DeviceSliceMut<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
                R,
                Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( crate::detail::device::DeviceColumnMutView<R, $ty>, )+);

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
                policy: &crate::detail::CubePolicy<R>,
                inner: <Self::Item as MItem<R>>::Inner,
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
                policy: &crate::detail::CubePolicy<R>,
                inner: <Self::Item as MItem<R>>::Inner,
                stencil: crate::detail::api::PrecomputedSelection<R>,
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
                            KernelOp::<R, StencilFlag>::new(),
                        )?;
                    }
                )+
                Ok(())
            }

            fn replace_where_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                replacement: Self::Item,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<(), Error>
            {
                let output = self.into_inner();
                $(
                    crate::detail::api::replace_where_into_with_policy(
                        policy,
                        replacement.$idx,
                        &stencil,
                        &output.$idx,
                        KernelOp::<R, StencilFlag>::new(),
                    )?;
                )+
                Ok(())
            }
        }

        impl<'a, R, $( $ty ),+> sealed::MIterMutDispatch<R> for $name<$( DeviceSliceMut<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
                R,
                Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
            >,
        {
            fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
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
                Option<crate::detail::device::DeviceColumnMutView<R, U>>,
                Error,
            >
            where
                U: Scalar,
            {
                $(
                    if index == $idx {
                        let source = &*self.$idx.source as &dyn Any;
                        let source = match source.downcast_ref::<DeviceVec<R, U>>() {
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
