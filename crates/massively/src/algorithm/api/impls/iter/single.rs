use super::*;

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
