use super::*;

impl<'a, R, T> MIter<R> for SoA1<crate::runtime::DeviceSlice<'a, R, T>>
where
    R: Runtime,
    T: Scalar + 'static,
    (T,): MItem<
            R,
            Inner = (crate::detail::DeviceVec<R, T>,),
            View = (crate::detail::device::DeviceColumnView<R, T>,),
        >,
{
    type Item = (T,);
    type Inner = (crate::detail::device::DeviceColumnView<R, T>,);

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
        let _ = policy;
        Ok((self.0.column_view(),))
    }

    fn into_view_with_policy(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<<Self::Item as MItem<R>>::View, Error> {
        let _ = policy;
        Ok((self.0.column_view(),))
    }
}

impl<'a, R, T> sealed::MIterDispatch<R> for SoA1<crate::runtime::DeviceSlice<'a, R, T>>
where
    R: Runtime,
    T: Scalar + 'static,
    (T,): MItem<
            R,
            Inner = (crate::detail::DeviceVec<R, T>,),
            View = (crate::detail::device::DeviceColumnView<R, T>,),
        >,
{
    fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
        exec.ensure_policy_id(self.0.policy_id())
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
        let input = self.into_inner_with_policy(policy)?.0;
        let inner = <Output::Item as sealed::MItemDispatch<R>>::transform_unary(policy, input, op)?;
        output.write_from_inner(policy, inner)
    }

    fn map_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
    ) -> Result<Output, Error>
    where
        Output: MVec<R>,
        Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
    {
        let input = self.into_inner_with_policy(policy)?.0;
        let inner = <Output::Item as sealed::MItemDispatch<R>>::transform_unary(policy, input, op)?;
        Ok(array_from_inner::<R, Output::Item, Output>(inner))
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
        let input = self.into_inner_with_policy(policy)?.0;
        let inner = <Output::Item as sealed::MItemDispatch<R>>::transform_unary(policy, input, op)?;
        output.write_where_from_inner(policy, inner, stencil)
    }

    fn reverse_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<Output, Error>
    where
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::reverse(policy, self.into_inner_with_policy(policy)?)?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        let inner = crate::detail::sort(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        let (key_inner, value_inner) = crate::detail::sort_by_key(
            policy,
            (keys,),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok((
            array_from_inner::<R, (K,), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
        ))
    }

    fn sort_by_three_key_dispatch<K1, K2, K3, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K1: Scalar + 'static,
        K2: Scalar + 'static,
        K3: Scalar + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: MVec<R, Item = (K1, K2, K3)>,
        ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::sort_by_key(
            policy,
            (first_key, second_key, third_key),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok((
            array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
        ))
    }

    fn sort_by_key_dispatch<Values, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MVec<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::sort_by_single_key_dispatch(
            values, policy, keys, less,
        )
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
        let (key_inner, value_inner) = crate::detail::unique_by_key(
            policy,
            (keys,),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Eq>::new(),
        )?;
        Ok((
            array_from_inner::<R, (K,), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
        ))
    }

    fn unique_by_key_dispatch<Values, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MVec<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::unique_by_single_key_dispatch(
            values, policy, keys, eq,
        )
    }

    fn unique_by_three_key_dispatch<K1, K2, K3, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        _eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K1: Scalar + 'static,
        K2: Scalar + 'static,
        K3: Scalar + 'static,
        Eq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: MVec<R, Item = (K1, K2, K3)>,
        ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::unique_by_key(
            policy,
            crate::detail::device::SoAView3 {
                first: first_key,
                second: second_key,
                third: third_key,
            },
            crate::detail::device::SoAView1 {
                source: self.into_inner_with_policy(policy)?.0,
            },
            KernelOp::<R, Eq>::new(),
        )?;
        Ok((
            array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
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
        let values = self.into_inner_with_policy(policy)?.0;
        let inner = crate::detail::inclusive_scan_by_key(
            policy,
            keys,
            crate::detail::device::SoAView1 { source: values },
            KernelTuple1Op::<R, KeyEq>::new(),
            KernelOp::<R, Op>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }

    fn inclusive_scan_by_three_key_dispatch<K1, K2, K3, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        _key_eq: KeyEq,
        _op: Op,
    ) -> Result<Output, Error>
    where
        K1: Scalar + 'static,
        K2: Scalar + 'static,
        K3: Scalar + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?.0;
        ensure_same_len(values.len, first_key.len)?;
        let head_flags = crate::detail::read::unique_tuple3_flags_read::<
            _,
            _,
            _,
            KernelOp<R, KeyEq>,
        >(policy, &first_key, &second_key, &third_key)?;
        let len_u32 = u32::try_from(first_key.len)
            .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
        let control = crate::detail::control::ScanByKeyControl {
            key_bindings: crate::detail::device::KernelColumnBindings::empty(policy.client()),
            head_flags,
            len: first_key.len,
            len_u32,
            _marker: std::marker::PhantomData,
        };
        let inner = crate::detail::read::inclusive_scan_by_flags_one::<
            _,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
        >(policy, &values, &control)?;
        Ok(array_from_inner::<R, (T,), Output>((inner,)))
    }

    fn inclusive_scan_by_key_dispatch<Values, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        op: Op,
    ) -> Result<Output, Error>
    where
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
        Output: MVec<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::inclusive_scan_by_single_key_dispatch(
            values, policy, keys, key_eq, op,
        )
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
        let values = self.into_inner_with_policy(policy)?.0;
        let inner = crate::detail::exclusive_scan_by_key(
            policy,
            keys,
            crate::detail::device::SoAView1 { source: values },
            KernelTuple1Op::<R, KeyEq>::new(),
            init.0,
            KernelOp::<R, Op>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }

    fn exclusive_scan_by_three_key_dispatch<K1, K2, K3, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        _key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        _op: Op,
    ) -> Result<Output, Error>
    where
        K1: Scalar + 'static,
        K2: Scalar + 'static,
        K3: Scalar + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?.0;
        ensure_same_len(values.len, first_key.len)?;
        let head_flags = crate::detail::read::unique_tuple3_flags_read::<
            _,
            _,
            _,
            KernelOp<R, KeyEq>,
        >(policy, &first_key, &second_key, &third_key)?;
        let len_u32 = u32::try_from(first_key.len)
            .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
        let control = crate::detail::control::ScanByKeyControl {
            key_bindings: crate::detail::device::KernelColumnBindings::empty(policy.client()),
            head_flags,
            len: first_key.len,
            len_u32,
            _marker: std::marker::PhantomData,
        };
        let inner = crate::detail::read::exclusive_scan_by_flags_one::<
            _,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
        >(policy, &values, &control, init.0)?;
        Ok(array_from_inner::<R, (T,), Output>((inner,)))
    }

    fn exclusive_scan_by_key_dispatch<Values, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        init: <Values as MIter<R>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
        Output: MVec<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::exclusive_scan_by_single_key_dispatch(
            values, policy, keys, key_eq, init, op,
        )
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
        let (key_inner, value_inner) = crate::detail::reduce_by_key(
            policy,
            (keys,),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, KeyEq>::new(),
            init,
            KernelOp::<R, Op>::new(),
        )?;
        Ok((
            array_from_inner::<R, (K,), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
        ))
    }

    fn reduce_by_key_dispatch<Values, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        init: <Values as MIter<R>>::Item,
        op: Op,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
        KeyOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MVec<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::reduce_by_single_key_dispatch(
            values, policy, keys, key_eq, init, op,
        )
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
        let left_value = self.into_view_with_policy(policy)?.0;
        let right_value = right_values.into_view_with_policy(policy)?.0;
        let (key_inner, value_inner) = crate::detail::merge_by_key(
            policy,
            crate::detail::device::SoAView1 { source: left_keys },
            crate::detail::device::SoAView1 { source: left_value },
            crate::detail::device::SoAView1 { source: right_keys },
            crate::detail::device::SoAView1 {
                source: right_value,
            },
            KernelTuple1Op::<R, Less>::new(),
        )?;
        Ok((
            array_from_inner::<R, (K,), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
        ))
    }

    fn merge_by_three_key_same_dispatch<K1, K2, K3, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        left_first_key: crate::detail::device::DeviceColumnView<R, K1>,
        left_second_key: crate::detail::device::DeviceColumnView<R, K2>,
        left_third_key: crate::detail::device::DeviceColumnView<R, K3>,
        right_first_key: crate::detail::device::DeviceColumnView<R, K1>,
        right_second_key: crate::detail::device::DeviceColumnView<R, K2>,
        right_third_key: crate::detail::device::DeviceColumnView<R, K3>,
        right_values: RightValues,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
        K1: Scalar + 'static,
        K2: Scalar + 'static,
        K3: Scalar + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: MVec<R, Item = (K1, K2, K3)>,
        ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
    {
        let left_value = self.into_view_with_policy(policy)?.0;
        let right_value = right_values.into_view_with_policy(policy)?.0;
        let (key_inner, value_inner) = crate::detail::merge_by_key(
            policy,
            (left_first_key, left_second_key, left_third_key),
            crate::detail::device::SoAView1 { source: left_value },
            (right_first_key, right_second_key, right_third_key),
            crate::detail::device::SoAView1 {
                source: right_value,
            },
            KernelOp::<R, Less>::new(),
        )?;
        Ok((
            array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
        ))
    }

    fn reduce_by_three_key_dispatch<K1, K2, K3, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        _key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        _op: Op,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K1: Scalar + 'static,
        K2: Scalar + 'static,
        K3: Scalar + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MVec<R, Item = (K1, K2, K3)>,
        ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?.0;
        ensure_same_len(values.len, first_key.len)?;
        if first_key.len == 0 {
            let key_inner = (
                policy.empty_device_vec(),
                policy.empty_device_vec(),
                policy.empty_device_vec(),
            );
            let value_inner = (policy.empty_device_vec(),);
            return Ok((
                array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                array_from_inner::<R, (T,), ValueOutput>(value_inner),
            ));
        }
        let head_flags = crate::detail::read::unique_tuple3_flags_read::<
            crate::detail::device::DeviceColumnView<R, K1>,
            crate::detail::device::DeviceColumnView<R, K2>,
            crate::detail::device::DeviceColumnView<R, K3>,
            KernelOp<R, KeyEq>,
        >(policy, &first_key, &second_key, &third_key)?;
        let end_flags = end_flags_from_head_flags(policy, head_flags.clone(), first_key.len)?;
        let len_u32 = u32::try_from(first_key.len)
            .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
        let control: crate::detail::control::ScanByKeyControl<
            R,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
        > = crate::detail::control::ScanByKeyControl {
            key_bindings: crate::detail::device::KernelColumnBindings::empty(policy.client()),
            head_flags,
            len: first_key.len,
            len_u32,
            _marker: std::marker::PhantomData,
        };
        let inclusive = crate::detail::read::inclusive_scan_by_flags_one::<
            _,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
        >(policy, &values, &control)?;

        let client = policy.client();
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
        let init_handle = client.create_from_slice(T::as_bytes(&[init.0]));
        let reduced_handle = client.empty(first_key.len * std::mem::size_of::<T>());
        let num_blocks = first_key
            .len
            .div_ceil(crate::detail::primitives::scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
        unsafe {
            crate::kernels::reduce_by_key_apply_init_kernel::launch_unchecked::<
                T,
                KernelOp<R, Op>,
                R,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(crate::detail::primitives::scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts(inclusive.handle.clone(), first_key.len),
                BufferArg::from_raw_parts(init_handle.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_handle.clone(), first_key.len),
            );
        }

        let key_inner = (
            crate::detail::api::device_expr_compact_with_flags_with_policy(
                policy,
                &first_key,
                end_flags.clone(),
            )?,
            crate::detail::api::device_expr_compact_with_flags_with_policy(
                policy,
                &second_key,
                end_flags.clone(),
            )?,
            crate::detail::api::device_expr_compact_with_flags_with_policy(
                policy,
                &third_key,
                end_flags.clone(),
            )?,
        );
        let value_handles = crate::detail::primitives::select::handles_from_flags(
            policy,
            first_key.len,
            len_u32,
            end_flags,
            reduced_handle,
        )?;
        let value_inner = (crate::detail::primitives::select::compact::<R, T>(
            policy,
            value_handles,
        )?,);
        Ok((
            array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
        ))
    }

    fn merge_by_key_dispatch<RightKeys, LeftValues, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right_keys: RightKeys,
        left_values: LeftValues,
        right_values: RightValues,
        less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        RightKeys: MIter<R, Item = <Self as MIter<R>>::Item>,
        LeftValues: MIter<R>,
        RightValues: MIter<R, Item = <LeftValues as MIter<R>>::Item>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MVec<R, Item = <LeftValues as MIter<R>>::Item>,
    {
        let (left_keys,) = self.into_view_with_policy(policy)?;
        let (right_keys,) = right_keys.into_view_with_policy(policy)?;
        <LeftValues as sealed::MIterDispatch<R>>::merge_by_single_key_same_dispatch(
            left_values,
            policy,
            left_keys,
            right_keys,
            right_values,
            less,
        )
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
        let input = self.into_inner_with_policy(policy)?.0;
        let output = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_inner::<T>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "gather output must match input shape".to_string(),
            })?;
        crate::detail::api::device_expr_gather_into_with_policy(policy, &input, &indices, &output)
    }

    fn permute_dispatch<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        indices: Indices,
    ) -> Result<Output, Error>
    where
        Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
            + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
        <Indices as crate::detail::device::KernelColumn>::Expr: crate::expr::GpuExpr<u32>,
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
    {
        let input = self.into_inner_with_policy(policy)?.0;
        let inner = crate::detail::api::device_expr_gather_with_policy(policy, &input, &indices)?;
        Ok(array_from_inner::<R, (T,), Output>((inner,)))
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
        crate::detail::reduce(
            policy,
            self.into_inner_with_policy(policy)?,
            init,
            KernelOp::<R, Op>::new(),
        )
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
        let inner = crate::detail::inclusive_scan(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Op>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        let inner = crate::detail::exclusive_scan(
            policy,
            self.into_inner_with_policy(policy)?,
            init,
            KernelOp::<R, Op>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        let inner = crate::detail::adjacent_difference(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Op>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }

    fn copy_where_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<Output, Error>
    where
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::copy_where(
            policy,
            self.into_inner_with_policy(policy)?,
            stencil,
            KernelOp::<R, StencilFlag>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        let inner = crate::detail::remove_if(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }

    fn remove_where_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<Output, Error>
    where
        Output: MVec<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::copy_where(
            policy,
            self.into_inner_with_policy(policy)?,
            stencil,
            KernelOp::<R, StencilFlag>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }

    fn count_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
    ) -> Result<usize, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::count_if(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )
    }

    fn all_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::all_of(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )
    }

    fn any_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::any_of(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )
    }

    fn none_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::none_of(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )
    }

    fn find_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
    ) -> Result<Option<usize>, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::find_if(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )
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
        let (matching, failing) = crate::detail::partition(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )?;
        Ok((
            array_from_inner::<R, (T,), Output>(matching),
            array_from_inner::<R, (T,), Output>(failing),
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
        crate::detail::is_partitioned(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )
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
        let inner = crate::detail::replace_where(
            policy,
            self.into_inner_with_policy(policy)?,
            replacement,
            stencil,
            KernelOp::<R, StencilFlag>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        let inner = crate::detail::unique(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }

    fn min_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _less: Less,
    ) -> Result<Option<usize>, Error>
    where
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::min_element(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )
    }

    fn max_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _less: Less,
    ) -> Result<Option<usize>, Error>
    where
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::max_element(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )
    }

    fn minmax_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _less: Less,
    ) -> Result<Option<(usize, usize)>, Error>
    where
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::minmax_element(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )
    }

    fn adjacent_find_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
    ) -> Result<Option<usize>, Error>
    where
        Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::adjacent_find(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )
    }

    fn lower_bound_dispatch<Values, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        _less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, u32>, Error>
    where
        Values: MIter<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::lower_bound_many(
            policy,
            self.into_view_with_policy(policy)?,
            values.into_view_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok(crate::runtime::DeviceVec::from_inner(inner))
    }

    fn upper_bound_dispatch<Values, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        _less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, u32>, Error>
    where
        Values: MIter<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::upper_bound_many(
            policy,
            self.into_view_with_policy(policy)?,
            values.into_view_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok(crate::runtime::DeviceVec::from_inner(inner))
    }

    fn is_sorted_until_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _less: Less,
    ) -> Result<usize, Error>
    where
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::is_sorted_until(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )
    }

    fn is_sorted_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _less: Less,
    ) -> Result<bool, Error>
    where
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::is_sorted(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )
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
        let input = self.into_inner_with_policy(policy)?.0;
        let output = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_inner::<T>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "gather_where output must match input shape".to_string(),
            })?;
        crate::detail::api::device_expr_gather_where_into_with_control(
            policy,
            &input,
            &indices,
            stencil.control(),
            &output,
        )
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
        let input = self.into_inner_with_policy(policy)?.0;
        let output = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_inner::<T>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "scatter output must match input shape".to_string(),
            })?;
        crate::detail::api::device_expr_scatter_into_with_policy(policy, &input, &indices, &output)
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
        let input = self.into_inner_with_policy(policy)?.0;
        let output = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_inner::<T>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "scatter_where output must match input shape".to_string(),
            })?;
        crate::detail::api::device_expr_scatter_where_into_with_control(
            policy,
            &input,
            &indices,
            stencil.control(),
            &output,
        )
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
        crate::detail::equal(
            policy,
            self.into_view_with_policy(policy)?,
            right.into_view_with_policy(policy)?,
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
        crate::detail::mismatch(
            policy,
            self.into_view_with_policy(policy)?,
            right.into_view_with_policy(policy)?,
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
        crate::detail::find_first_of(
            policy,
            self.into_view_with_policy(policy)?,
            needles.into_view_with_policy(policy)?,
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
        crate::detail::lexicographical_compare(
            policy,
            self.into_view_with_policy(policy)?,
            right.into_view_with_policy(policy)?,
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
        let inner = crate::detail::merge(
            policy,
            self.into_view_with_policy(policy)?,
            right.into_view_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        let inner = crate::detail::set_union(
            policy,
            self.into_view_with_policy(policy)?,
            right.into_view_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        let inner = crate::detail::set_intersection(
            policy,
            self.into_view_with_policy(policy)?,
            right.into_view_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        let inner = crate::detail::set_difference(
            policy,
            self.into_view_with_policy(policy)?,
            right.into_view_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        crate::detail::equal(
            policy,
            self.into_inner_with_policy(policy)?,
            right.into_inner_with_policy(policy)?,
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
        crate::detail::mismatch(
            policy,
            self.into_inner_with_policy(policy)?,
            right.into_inner_with_policy(policy)?,
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
        crate::detail::find_first_of(
            policy,
            self.into_inner_with_policy(policy)?,
            needles.into_inner_with_policy(policy)?,
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
        crate::detail::lexicographical_compare(
            policy,
            self.into_inner_with_policy(policy)?,
            right.into_inner_with_policy(policy)?,
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
        let inner = crate::detail::merge(
            policy,
            self.into_inner_with_policy(policy)?,
            right.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        let inner = crate::detail::set_union(
            policy,
            self.into_inner_with_policy(policy)?,
            right.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        let inner = crate::detail::set_intersection(
            policy,
            self.into_inner_with_policy(policy)?,
            right.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
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
        let inner = crate::detail::set_difference(
            policy,
            self.into_inner_with_policy(policy)?,
            right.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }
}

impl<'a, R, T> MIterMut<R> for SoA1<DeviceSliceMut<'a, R, T>>
where
    R: Runtime,
    T: Scalar + 'static,
    (T,): MItem<R, Inner = (crate::detail::DeviceVec<R, T>,)>,
{
    type Item = (T,);
    type Inner = (crate::detail::device::DeviceColumnMutView<R, T>,);

    fn len(&self) -> usize {
        self.0.len()
    }

    fn into_inner(self) -> Self::Inner {
        (crate::detail::device::DeviceColumnMutView::from_slice(
            &self.0.source.inner,
            self.0.offset,
            self.0.len,
        ),)
    }

    fn write_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MItem<R>>::Inner,
    ) -> Result<(), Error> {
        let output = self.into_inner().0;
        let input = crate::detail::device::DeviceColumnView::from_column(&inner.0);
        crate::detail::api::device_expr_collect_into_with_policy(policy, &input, &output)
    }

    fn write_where_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MItem<R>>::Inner,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error> {
        let output = self.into_inner().0;
        let input = crate::detail::device::DeviceColumnView::from_column(&inner.0);
        crate::detail::api::device_expr_copy_where_into_with_policy(
            policy,
            &input,
            &stencil,
            &output,
            KernelOp::<R, StencilFlag>::new(),
        )
    }

    fn replace_where_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        replacement: Self::Item,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error> {
        let output = self.into_inner().0;
        crate::detail::api::replace_where_into_with_control(
            policy,
            replacement.0,
            stencil.control(),
            &output,
        )
    }

    fn fill_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        value: Self::Item,
    ) -> Result<(), Error> {
        let output = self.into_inner().0;
        crate::detail::primitives::fill_slice_with_policy(policy, value.0, &output)
    }
}

impl<'a, R, T> sealed::MIterMutDispatch<R> for SoA1<DeviceSliceMut<'a, R, T>>
where
    R: Runtime,
    T: Scalar + 'static,
    (T,): MItem<R, Inner = (crate::detail::DeviceVec<R, T>,)>,
{
    fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
        exec.ensure_policy_id(self.0.source.inner.policy_id())
    }

    fn column_mut_view_inner<U: 'static>(
        &self,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<R, U>>, Error>
    where
        U: Scalar,
    {
        let source = &*self.0.source as &dyn Any;
        let source = match source.downcast_ref::<DeviceVec<R, U>>() {
            Some(source) => source,
            None => return Ok(None),
        };
        Ok(Some(
            crate::detail::device::DeviceColumnMutView::from_slice(
                &source.inner,
                self.0.offset,
                self.0.len,
            ),
        ))
    }
}
