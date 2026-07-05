use super::*;

impl<'a, R, T> MIter<R> for crate::runtime::DeviceSlice<'a, R, T>
where
    R: Runtime,
    T: MStorageElement + MItem<R> + 'static,
{
    type Item = T;
    type Inner = crate::detail::device::DeviceColumnView<R, T>;

    fn len(&self) -> MIndex {
        self.len()
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("read-only MIter lowering requires a CubePolicy")
    }

    fn into_inner_with_policy(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<Self::Inner, Error> {
        let _ = policy;
        Ok(self.column_view())
    }

    fn into_view_with_policy(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<<Self::Item as MAlloc<R>>::View, Error>
    where
        Self::Item: MAlloc<R>,
    {
        let _ = policy;
        unreachable!("scalar DeviceSlice is not an allocatable SoA view")
    }
}

impl<'a, R, T> sealed::MIterDispatch<R> for crate::runtime::DeviceSlice<'a, R, T>
where
    R: Runtime,
    T: MStorageElement + MItem<R> + 'static,
{
    fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
        exec.ensure_policy_id(self.policy_id())
    }

    fn index_column_dispatch(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<crate::detail::device::DeviceColumnView<R, MIndex>, Error>
    where
        Self: MIter<R, Item = MIndex>,
    {
        let _ = policy;
        Ok(self.aux_u32_column_view())
    }

    fn stencil_selection_dispatch(
        self,
        policy: &crate::detail::CubePolicy<R>,
        invert: bool,
        flags_only: bool,
    ) -> Result<crate::detail::api::PrecomputedSelection<R>, Error>
    where
        Self: MIter<R, Item = u32>,
    {
        let stencil = self.aux_u32_column_view();
        if flags_only {
            crate::detail::api::PrecomputedSelection::from_stencil_flags_with_policy::<
                _,
                KernelOp<R, StencilFlag>,
            >(policy, &(stencil,), invert)
        } else {
            crate::detail::api::PrecomputedSelection::from_stencil_with_policy::<
                _,
                KernelOp<R, StencilFlag>,
            >(policy, &(stencil,), invert)
        }
    }

    fn transform_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
    {
        let input = self.into_inner_with_policy(policy)?;
        let inner = <Output::Item as sealed::MItemDispatch<R>>::transform_scalar_input(
            policy, input, op, env,
        )?;
        output.write_from_inner(policy, inner)
    }

    fn map_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<Output, Error>
    where
        Output: StorageFromInner<R>,
        Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
    {
        let input = self.into_inner_with_policy(policy)?;
        let inner = <Output::Item as sealed::MItemDispatch<R>>::transform_scalar_input(
            policy, input, op, env,
        )?;
        Ok(array_from_inner::<R, Output::Item, Output>(inner))
    }

    fn transform_where_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
    {
        let input = self.into_inner_with_policy(policy)?;
        let inner = <Output::Item as sealed::MItemDispatch<R>>::transform_scalar_input(
            policy, input, op, env,
        )?;
        output.write_where_from_inner(policy, inner, stencil)
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
        let result = crate::detail::reduce(
            policy,
            (self.into_inner_with_policy(policy)?,),
            (init,),
            KernelScalarTuple1Op::<R, Op>::new(),
        )?;
        Ok(result.0)
    }

    fn count_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<MIndex, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::count_if(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )
    }

    fn all_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::all_of(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )
    }

    fn any_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::any_of(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )
    }

    fn none_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::none_of(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )
    }

    fn find_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<Option<MIndex>, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::find_if(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )
    }

    fn is_partitioned_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::is_partitioned(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )
    }

    fn min_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _less: Less,
    ) -> Result<Option<MIndex>, Error>
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
    ) -> Result<Option<MIndex>, Error>
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
    ) -> Result<Option<(MIndex, MIndex)>, Error>
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
    ) -> Result<Option<MIndex>, Error>
    where
        Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::adjacent_find(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )
    }

    fn is_sorted_until_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _less: Less,
    ) -> Result<MIndex, Error>
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
}

impl<'a, R, T> MIter<R> for SoA1<crate::runtime::DeviceSlice<'a, R, T>>
where
    R: Runtime,
    T: MStorageElement + 'static,
    (T,): MAlloc<
            R,
            Inner = (crate::detail::DeviceVec<R, T>,),
            View = (crate::detail::device::DeviceColumnView<R, T>,),
        >,
{
    type Item = (T,);
    type Inner = (crate::detail::device::DeviceColumnView<R, T>,);

    fn len(&self) -> MIndex {
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
    ) -> Result<<Self::Item as MAlloc<R>>::View, Error> {
        let _ = policy;
        Ok((self.0.column_view(),))
    }
}

impl<'a, R, T> sealed::MIterDispatch<R> for SoA1<crate::runtime::DeviceSlice<'a, R, T>>
where
    R: Runtime,
    T: MStorageElement + 'static,
    (T,): MAlloc<
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
        env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
    {
        let input = self.into_inner_with_policy(policy)?.0;
        let inner =
            <Output::Item as sealed::MItemDispatch<R>>::transform_unary(policy, input, op, env)?;
        output.write_from_inner(policy, inner)
    }

    fn map_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<Output, Error>
    where
        Output: StorageFromInner<R>,
        Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
    {
        let input = self.into_inner_with_policy(policy)?.0;
        let inner =
            <Output::Item as sealed::MItemDispatch<R>>::transform_unary(policy, input, op, env)?;
        Ok(array_from_inner::<R, Output::Item, Output>(inner))
    }

    fn transform_where_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
    {
        let input = self.into_inner_with_policy(policy)?.0;
        let inner =
            <Output::Item as sealed::MItemDispatch<R>>::transform_unary(policy, input, op, env)?;
        output.write_where_from_inner(policy, inner, stencil)
    }

    fn reverse_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<Output, Error>
    where
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::sort(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }

    fn reverse_into_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::reverse(policy, self.into_inner_with_policy(policy)?)?;
        output.write_from_inner(policy, inner)
    }

    fn sort_into_dispatch<Less, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::sort(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        output.write_from_inner(policy, inner)
    }

    fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: StorageFromInner<R, Item = (K,)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn sort_by_single_key_into_dispatch<K, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        _less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        K: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: MIterMut<R, Item = (K,)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::sort_by_key(
            policy,
            (keys,),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        key_output.write_from_inner(policy, key_inner)?;
        value_output.write_from_inner(policy, value_inner)
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
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn sort_by_three_key_into_dispatch<K1, K2, K3, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        _less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::sort_by_key(
            policy,
            (first_key, second_key, third_key),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        key_output.write_from_inner(policy, key_inner)?;
        value_output.write_from_inner(policy, value_inner)
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
        KeyOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::sort_by_single_key_dispatch(
            values, policy, keys, less,
        )
    }

    fn sort_by_key_into_dispatch<Values, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MIterMut<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::sort_by_single_key_into_dispatch(
            values,
            policy,
            keys,
            less,
            key_output,
            value_output,
        )
    }

    fn sort_by_two_key_dispatch<K1, K2, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2)>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::sort_by_key(
            policy,
            (first_key, second_key),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        Ok((
            array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
        ))
    }

    fn sort_by_two_key_into_dispatch<K1, K2, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        _less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2)>,
        KeyOutput: MIterMut<R, Item = (K1, K2)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::sort_by_key(
            policy,
            (first_key, second_key),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        key_output.write_from_inner(policy, key_inner)?;
        value_output.write_from_inner(policy, value_inner)
    }

    fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        _eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K: MStorageElement + 'static,
        Eq: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: StorageFromInner<R, Item = (K,)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn unique_by_single_key_into_dispatch<K, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        _eq: Eq,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        K: MStorageElement + 'static,
        Eq: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: MIterMut<R, Item = (K,)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::unique_by_key(
            policy,
            (keys,),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Eq>::new(),
        )?;
        let len = key_inner.0.len() as MIndex;
        key_output.write_prefix_from_inner(policy, key_inner)?;
        value_output.write_prefix_from_inner(policy, value_inner)?;
        Ok(len)
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
        KeyOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::unique_by_single_key_dispatch(
            values, policy, keys, eq,
        )
    }

    fn unique_by_key_into_dispatch<Values, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        eq: Eq,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MIterMut<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::unique_by_single_key_into_dispatch(
            values,
            policy,
            keys,
            eq,
            key_output,
            value_output,
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
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        Eq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn unique_by_three_key_into_dispatch<K1, K2, K3, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        _eq: Eq,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        Eq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::unique_by_key(
            policy,
            (first_key, second_key, third_key),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Eq>::new(),
        )?;
        let len = key_inner.0.len() as MIndex;
        key_output.write_prefix_from_inner(policy, key_inner)?;
        value_output.write_prefix_from_inner(policy, value_inner)?;
        Ok(len)
    }

    fn unique_by_two_key_dispatch<K1, K2, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        _eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        Eq: op::BinaryPredicateOp<R, (K1, K2)>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::unique_by_key(
            policy,
            (first_key, second_key),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Eq>::new(),
        )?;
        Ok((
            array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
        ))
    }

    fn unique_by_two_key_into_dispatch<K1, K2, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        _eq: Eq,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        Eq: op::BinaryPredicateOp<R, (K1, K2)>,
        KeyOutput: MIterMut<R, Item = (K1, K2)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::unique_by_key(
            policy,
            (first_key, second_key),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Eq>::new(),
        )?;
        let len = key_inner.0.len() as MIndex;
        key_output.write_prefix_from_inner(policy, key_inner)?;
        value_output.write_prefix_from_inner(policy, value_inner)?;
        Ok(len)
    }

    fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        _key_eq: KeyEq,
        _op: Op,
    ) -> Result<Output, Error>
    where
        K: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn inclusive_scan_by_single_key_into_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        _key_eq: KeyEq,
        _op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        K: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?.0;
        let inner = crate::detail::inclusive_scan_by_key(
            policy,
            keys,
            crate::detail::device::SoAView1 { source: values },
            KernelTuple1Op::<R, KeyEq>::new(),
            KernelOp::<R, Op>::new(),
        )?;
        output.write_from_inner(policy, inner)
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
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
            head_flags,
            len: first_key.len,
            len_u32,
            _runtime: std::marker::PhantomData,
        };
        let inner = crate::detail::read::inclusive_scan_by_flags_one::<_, KernelOp<R, Op>>(
            policy, &values, &control,
        )?;
        Ok(array_from_inner::<R, (T,), Output>((inner,)))
    }

    fn inclusive_scan_by_three_key_into_dispatch<K1, K2, K3, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        _key_eq: KeyEq,
        _op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?.0;
        let inner = crate::detail::inclusive_scan_by_key(
            policy,
            (first_key, second_key, third_key),
            crate::detail::device::SoAView1 { source: values },
            KernelOp::<R, KeyEq>::new(),
            KernelOp::<R, Op>::new(),
        )?;
        output.write_from_inner(policy, inner)
    }

    fn inclusive_scan_by_two_key_dispatch<K1, K2, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        _key_eq: KeyEq,
        _op: Op,
    ) -> Result<Output, Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?.0;
        ensure_same_len(values.len, first_key.len)?;
        let head_flags = crate::detail::read::unique_tuple2_flags_read::<_, _, KernelOp<R, KeyEq>>(
            policy,
            &first_key,
            &second_key,
        )?;
        let len_u32 = u32::try_from(first_key.len)
            .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
        let control = crate::detail::control::ScanByKeyControl {
            head_flags,
            len: first_key.len,
            len_u32,
            _runtime: std::marker::PhantomData,
        };
        let inner = crate::detail::read::inclusive_scan_by_flags_one::<_, KernelOp<R, Op>>(
            policy, &values, &control,
        )?;
        Ok(array_from_inner::<R, (T,), Output>((inner,)))
    }

    fn inclusive_scan_by_two_key_into_dispatch<K1, K2, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        _key_eq: KeyEq,
        _op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?.0;
        let inner = crate::detail::inclusive_scan_by_key(
            policy,
            (first_key, second_key),
            crate::detail::device::SoAView1 { source: values },
            KernelOp::<R, KeyEq>::new(),
            KernelOp::<R, Op>::new(),
        )?;
        output.write_from_inner(policy, inner)
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
        Output: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::inclusive_scan_by_single_key_dispatch(
            values, policy, keys, key_eq, op,
        )
    }

    fn inclusive_scan_by_key_into_dispatch<Values, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::inclusive_scan_by_single_key_into_dispatch(
            values, policy, keys, key_eq, op, output,
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
        K: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn exclusive_scan_by_single_key_into_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        _key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        _op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        K: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
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
        output.write_from_inner(policy, inner)
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
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
            head_flags,
            len: first_key.len,
            len_u32,
            _runtime: std::marker::PhantomData,
        };
        let inner = crate::detail::read::exclusive_scan_by_flags_one::<_, KernelOp<R, Op>>(
            policy, &values, &control, init.0,
        )?;
        Ok(array_from_inner::<R, (T,), Output>((inner,)))
    }

    fn exclusive_scan_by_three_key_into_dispatch<K1, K2, K3, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        _key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        _op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?.0;
        let inner = crate::detail::exclusive_scan_by_key(
            policy,
            (first_key, second_key, third_key),
            crate::detail::device::SoAView1 { source: values },
            KernelOp::<R, KeyEq>::new(),
            init.0,
            KernelOp::<R, Op>::new(),
        )?;
        output.write_from_inner(policy, inner)
    }

    fn exclusive_scan_by_two_key_dispatch<K1, K2, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        _key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        _op: Op,
    ) -> Result<Output, Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?.0;
        ensure_same_len(values.len, first_key.len)?;
        let head_flags = crate::detail::read::unique_tuple2_flags_read::<_, _, KernelOp<R, KeyEq>>(
            policy,
            &first_key,
            &second_key,
        )?;
        let len_u32 = u32::try_from(first_key.len)
            .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
        let control = crate::detail::control::ScanByKeyControl {
            head_flags,
            len: first_key.len,
            len_u32,
            _runtime: std::marker::PhantomData,
        };
        let inner = crate::detail::read::exclusive_scan_by_flags_one::<_, KernelOp<R, Op>>(
            policy, &values, &control, init.0,
        )?;
        Ok(array_from_inner::<R, (T,), Output>((inner,)))
    }

    fn exclusive_scan_by_two_key_into_dispatch<K1, K2, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        _key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        _op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?.0;
        let inner = crate::detail::exclusive_scan_by_key(
            policy,
            (first_key, second_key),
            crate::detail::device::SoAView1 { source: values },
            KernelOp::<R, KeyEq>::new(),
            init.0,
            KernelOp::<R, Op>::new(),
        )?;
        output.write_from_inner(policy, inner)
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
        Output: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::exclusive_scan_by_single_key_dispatch(
            values, policy, keys, key_eq, init, op,
        )
    }

    fn exclusive_scan_by_key_into_dispatch<Values, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        init: <Values as MIter<R>>::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::exclusive_scan_by_single_key_into_dispatch(
            values, policy, keys, key_eq, init, op, output,
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
        K: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: StorageFromInner<R, Item = (K,)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn reduce_by_single_key_into_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        _key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        _op: Op,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        K: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = (K,)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let (key_inner, value_inner) = crate::detail::reduce_by_key(
            policy,
            (keys,),
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, KeyEq>::new(),
            init,
            KernelOp::<R, Op>::new(),
        )?;
        let len = key_inner.0.len() as MIndex;
        key_output.write_prefix_from_inner(policy, key_inner)?;
        value_output.write_prefix_from_inner(policy, value_inner)?;
        Ok(len)
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
        KeyOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::reduce_by_single_key_dispatch(
            values, policy, keys, key_eq, init, op,
        )
    }

    fn reduce_by_key_into_dispatch<Values, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        init: <Values as MIter<R>>::Item,
        op: Op,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MIterMut<R, Item = <Values as MIter<R>>::Item>,
    {
        let (keys,) = self.into_inner_with_policy(policy)?;
        <Values as sealed::MIterDispatch<R>>::reduce_by_single_key_into_dispatch(
            values,
            policy,
            keys,
            key_eq,
            init,
            op,
            key_output,
            value_output,
        )
    }

    fn reduce_by_two_key_dispatch<K1, K2, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        _key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        _op: Op,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?.0;
        ensure_same_len(values.len, first_key.len)?;
        if first_key.len == 0 {
            let key_inner = (policy.empty_device_vec(), policy.empty_device_vec());
            let value_inner = (policy.empty_device_vec(),);
            return Ok((
                array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
                array_from_inner::<R, (T,), ValueOutput>(value_inner),
            ));
        }
        let head_flags = crate::detail::read::unique_tuple2_flags_read::<_, _, KernelOp<R, KeyEq>>(
            policy,
            &first_key,
            &second_key,
        )?;
        let end_flags = end_flags_from_head_flags(policy, head_flags.clone(), first_key.len)?;
        let len_u32 = u32::try_from(first_key.len)
            .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
        let value_selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            policy,
            first_key.len,
            len_u32,
            end_flags.clone(),
        )?;
        let value_count =
            crate::detail::primitives::select::selected_count(policy, &value_selected_rank)?;
        let segment = crate::detail::control::SegmentControl::from_head_end_flags(
            head_flags,
            end_flags,
            first_key.len,
            len_u32,
        );
        let reduce_control = crate::detail::control::ReduceByKeyControl::from_segment(
            segment,
            value_selected_rank,
            value_count,
        );
        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(
            &reduce_control.output_selection,
            reduce_control.output_count,
        );
        let key_inner = payload_apply.apply_expr2(policy, &first_key, &second_key)?;
        let reduce_apply = crate::detail::apply::SegmentedReduceApply::new(&reduce_control);
        let value_inner =
            (reduce_apply.apply_expr::<_, KernelOp<R, Op>>(policy, &values, init.0)?,);
        Ok((
            array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
        ))
    }

    fn reduce_by_two_key_into_dispatch<K1, K2, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        _key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        _op: Op,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = (K1, K2)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?;
        let (key_inner, value_inner) = crate::detail::reduce_by_key(
            policy,
            (first_key, second_key),
            values,
            KernelOp::<R, KeyEq>::new(),
            init,
            KernelOp::<R, Op>::new(),
        )?;
        let len = mindex_from_usize(key_inner.0.len())?;
        key_output.write_prefix_from_inner(policy, key_inner)?;
        value_output.write_prefix_from_inner(policy, value_inner)?;
        Ok(len)
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
        K: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: StorageFromInner<R, Item = (K,)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn merge_by_single_key_same_into_dispatch<K, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        left_keys: crate::detail::device::DeviceColumnView<R, K>,
        right_keys: crate::detail::device::DeviceColumnView<R, K>,
        right_values: RightValues,
        _less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
        K: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: MIterMut<R, Item = (K,)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
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
        key_output.write_from_inner(policy, key_inner)?;
        value_output.write_from_inner(policy, value_inner)
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
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn merge_by_three_key_same_into_dispatch<
        K1,
        K2,
        K3,
        RightValues,
        Less,
        KeyOutput,
        ValueOutput,
    >(
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
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
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
        key_output.write_from_inner(policy, key_inner)?;
        value_output.write_from_inner(policy, value_inner)
    }

    fn merge_by_two_key_same_dispatch<K1, K2, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        left_first_key: crate::detail::device::DeviceColumnView<R, K1>,
        left_second_key: crate::detail::device::DeviceColumnView<R, K2>,
        right_first_key: crate::detail::device::DeviceColumnView<R, K1>,
        right_second_key: crate::detail::device::DeviceColumnView<R, K2>,
        right_values: RightValues,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2)>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let left_value = self.into_view_with_policy(policy)?.0;
        let right_value = right_values.into_view_with_policy(policy)?.0;
        let (key_inner, value_inner) = crate::detail::merge_by_key(
            policy,
            (left_first_key, left_second_key),
            crate::detail::device::SoAView1 { source: left_value },
            (right_first_key, right_second_key),
            crate::detail::device::SoAView1 {
                source: right_value,
            },
            KernelOp::<R, Less>::new(),
        )?;
        Ok((
            array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
        ))
    }

    fn merge_by_two_key_same_into_dispatch<K1, K2, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        left_first_key: crate::detail::device::DeviceColumnView<R, K1>,
        left_second_key: crate::detail::device::DeviceColumnView<R, K2>,
        right_first_key: crate::detail::device::DeviceColumnView<R, K1>,
        right_second_key: crate::detail::device::DeviceColumnView<R, K2>,
        right_values: RightValues,
        _less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2)>,
        KeyOutput: MIterMut<R, Item = (K1, K2)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let left_value = self.into_view_with_policy(policy)?.0;
        let right_value = right_values.into_view_with_policy(policy)?.0;
        let (key_inner, value_inner) = crate::detail::merge_by_key(
            policy,
            (left_first_key, left_second_key),
            crate::detail::device::SoAView1 { source: left_value },
            (right_first_key, right_second_key),
            crate::detail::device::SoAView1 {
                source: right_value,
            },
            KernelOp::<R, Less>::new(),
        )?;
        key_output.write_from_inner(policy, key_inner)?;
        value_output.write_from_inner(policy, value_inner)
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
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        let value_selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            policy,
            first_key.len,
            len_u32,
            end_flags.clone(),
        )?;
        let value_count =
            crate::detail::primitives::select::selected_count(policy, &value_selected_rank)?;
        let segment = crate::detail::control::SegmentControl::from_head_end_flags(
            head_flags,
            end_flags,
            first_key.len,
            len_u32,
        );
        let reduce_control = crate::detail::control::ReduceByKeyControl::from_segment(
            segment,
            value_selected_rank,
            value_count,
        );
        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(
            &reduce_control.output_selection,
            reduce_control.output_count,
        );
        let key_inner = payload_apply.apply_expr3(policy, &first_key, &second_key, &third_key)?;
        let reduce_apply = crate::detail::apply::SegmentedReduceApply::new(&reduce_control);
        let value_inner =
            (reduce_apply.apply_expr::<_, KernelOp<R, Op>>(policy, &values, init.0)?,);
        Ok((
            array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
            array_from_inner::<R, (T,), ValueOutput>(value_inner),
        ))
    }

    fn reduce_by_three_key_into_dispatch<K1, K2, K3, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        _key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        _op: Op,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let values = self.into_inner_with_policy(policy)?;
        let (key_inner, value_inner) = crate::detail::reduce_by_key(
            policy,
            (first_key, second_key, third_key),
            values,
            KernelOp::<R, KeyEq>::new(),
            init,
            KernelOp::<R, Op>::new(),
        )?;
        let len = mindex_from_usize(key_inner.0.len())?;
        key_output.write_prefix_from_inner(policy, key_inner)?;
        value_output.write_prefix_from_inner(policy, value_inner)?;
        Ok(len)
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
        KeyOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: StorageFromInner<R, Item = <LeftValues as MIter<R>>::Item>,
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

    fn merge_by_key_into_dispatch<
        RightKeys,
        LeftValues,
        RightValues,
        Less,
        KeyOutput,
        ValueOutput,
    >(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right_keys: RightKeys,
        left_values: LeftValues,
        right_values: RightValues,
        less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        RightKeys: MIter<R, Item = <Self as MIter<R>>::Item>,
        LeftValues: MIter<R>,
        RightValues: MIter<R, Item = <LeftValues as MIter<R>>::Item>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MIterMut<R, Item = <LeftValues as MIter<R>>::Item>,
    {
        let (left_keys,) = self.into_view_with_policy(policy)?;
        let (right_keys,) = right_keys.into_view_with_policy(policy)?;
        <LeftValues as sealed::MIterDispatch<R>>::merge_by_single_key_same_into_dispatch(
            left_values,
            policy,
            left_keys,
            right_keys,
            right_values,
            less,
            key_output,
            value_output,
        )
    }

    fn gather_dispatch<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>
    where
        Indices: MIter<R, Item = MIndex>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let indices =
            <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
        let input = self.into_inner_with_policy(policy)?.0;
        let output = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_inner::<T>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "gather output must match input shape".to_string(),
            })?;
        crate::detail::apply::IndexedExprApply::gather_expr_into(policy, &input, &indices, &output)
    }

    fn permute_dispatch<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        indices: Indices,
    ) -> Result<Output, Error>
    where
        Indices: MIter<R, Item = MIndex>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let indices =
            <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
        let input = self.into_inner_with_policy(policy)?.0;
        let inner = crate::detail::apply::IndexedExprApply::gather_expr(policy, &input, &indices)?;
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::adjacent_difference(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Op>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }

    fn inclusive_scan_into_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::inclusive_scan(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Op>::new(),
        )?;
        output.write_from_inner(policy, inner)
    }

    fn exclusive_scan_into_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        init: <Self as MIter<R>>::Item,
        _op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::exclusive_scan(
            policy,
            self.into_inner_with_policy(policy)?,
            init,
            KernelOp::<R, Op>::new(),
        )?;
        output.write_from_inner(policy, inner)
    }

    fn adjacent_difference_into_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::adjacent_difference(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Op>::new(),
        )?;
        output.write_from_inner(policy, inner)
    }

    fn copy_where_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<Output, Error>
    where
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::copy_where(
            policy,
            self.into_inner_with_policy(policy)?,
            stencil,
            KernelOp::<R, StencilFlag>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }

    fn copy_where_into_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::copy_where(
            policy,
            self.into_inner_with_policy(policy)?,
            stencil,
            KernelOp::<R, StencilFlag>::new(),
        )?;
        let len = inner.0.len() as MIndex;
        output.write_prefix_from_inner(policy, inner)?;
        Ok(len)
    }

    fn remove_if_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<Output, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::remove_if(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }

    fn remove_where_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<Output, Error>
    where
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::copy_where(
            policy,
            self.into_inner_with_policy(policy)?,
            stencil,
            KernelOp::<R, StencilFlag>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }

    fn remove_where_into_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::copy_where(
            policy,
            self.into_inner_with_policy(policy)?,
            stencil,
            KernelOp::<R, StencilFlag>::new(),
        )?;
        let len = inner.0.len() as MIndex;
        output.write_prefix_from_inner(policy, inner)?;
        Ok(len)
    }

    fn count_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<MIndex, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::count_if(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )
    }

    fn all_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::all_of(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )
    }

    fn any_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::any_of(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )
    }

    fn none_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::none_of(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )
    }

    fn find_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<Option<MIndex>, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::find_if(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )
    }

    fn partition_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<(Output, Output), Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let (matching, failing) = crate::detail::partition(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )?;
        Ok((
            array_from_inner::<R, (T,), Output>(matching),
            array_from_inner::<R, (T,), Output>(failing),
        ))
    }

    fn partition_into_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let (matching, failing) = crate::detail::partition(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )?;
        let split = matching.0.len() as MIndex;
        output.write_split_from_inner(policy, matching, failing)?;
        Ok(split)
    }

    fn is_partitioned_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<bool, Error>
    where
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        crate::detail::is_partitioned(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
            env,
        )
    }

    fn replace_where_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        replacement: <Self as MIter<R>>::Item,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<Output, Error>
    where
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::unique(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )?;
        Ok(array_from_inner::<R, (T,), Output>(inner))
    }

    fn unique_into_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::unique(
            policy,
            self.into_inner_with_policy(policy)?,
            KernelOp::<R, Pred>::new(),
        )?;
        let len = inner.0.len() as MIndex;
        output.write_prefix_from_inner(policy, inner)?;
        Ok(len)
    }

    fn min_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _less: Less,
    ) -> Result<Option<MIndex>, Error>
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
    ) -> Result<Option<MIndex>, Error>
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
    ) -> Result<Option<(MIndex, MIndex)>, Error>
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
    ) -> Result<Option<MIndex>, Error>
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
    ) -> Result<crate::runtime::DeviceVec<R, MIndex>, Error>
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
    ) -> Result<crate::runtime::DeviceVec<R, MIndex>, Error>
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
    ) -> Result<MIndex, Error>
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
        Indices: MIter<R, Item = MIndex>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let indices =
            <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
        let input = self.into_inner_with_policy(policy)?.0;
        let mask = stencil.mask();
        let output = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_inner::<T>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "gather_where output must match input shape".to_string(),
            })?;
        crate::detail::apply::MaskedIndexedExprApply::gather_where_expr_into(
            policy, &input, &indices, &mask, &output,
        )
    }

    fn scatter_dispatch<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>
    where
        Indices: MIter<R, Item = MIndex>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let indices =
            <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
        let input = self.into_inner_with_policy(policy)?.0;
        let output = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_inner::<T>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "scatter output must match input shape".to_string(),
            })?;
        crate::detail::apply::IndexedExprApply::scatter_expr_into(policy, &input, &indices, &output)
    }

    fn scatter_where_dispatch<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        indices: Indices,
        stencil: crate::detail::api::PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Indices: MIter<R, Item = MIndex>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let indices =
            <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
        let input = self.into_inner_with_policy(policy)?.0;
        let mask = stencil.mask();
        let output = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_inner::<T>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "scatter_where output must match input shape".to_string(),
            })?;
        crate::detail::apply::MaskedIndexedExprApply::scatter_where_expr_into(
            policy, &input, &indices, &mask, &output,
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
    ) -> Result<Option<MIndex>, Error>
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
    ) -> Result<Option<MIndex>, Error>
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn merge_into_dispatch<Right, Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        _less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::merge(
            policy,
            self.into_view_with_policy(policy)?,
            right.into_view_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        output.write_from_inner(policy, inner)
    }

    fn set_union_dispatch<Right, Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn set_union_into_dispatch<Right, Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        _less: Less,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::set_union(
            policy,
            self.into_view_with_policy(policy)?,
            right.into_view_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        let len = inner.0.len() as MIndex;
        output.write_prefix_from_inner(policy, inner)?;
        Ok(len)
    }

    fn set_intersection_dispatch<Right, Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn set_intersection_into_dispatch<Right, Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        _less: Less,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::set_intersection(
            policy,
            self.into_view_with_policy(policy)?,
            right.into_view_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        let len = inner.0.len() as MIndex;
        output.write_prefix_from_inner(policy, inner)?;
        Ok(len)
    }

    fn set_difference_dispatch<Right, Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        _less: Less,
    ) -> Result<Output, Error>
    where
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

    fn set_difference_into_dispatch<Right, Output, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        right: Right,
        _less: Less,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let inner = crate::detail::set_difference(
            policy,
            self.into_view_with_policy(policy)?,
            right.into_view_with_policy(policy)?,
            KernelOp::<R, Less>::new(),
        )?;
        let len = inner.0.len() as MIndex;
        output.write_prefix_from_inner(policy, inner)?;
        Ok(len)
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
    ) -> Result<Option<MIndex>, Error>
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
    ) -> Result<Option<MIndex>, Error>
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
    T: MStorageElement + 'static,
    (T,): MAlloc<R, Inner = (crate::detail::DeviceVec<R, T>,)>,
{
    type Item = (T,);
    type Inner = (crate::detail::device::DeviceColumnMutView<R, T>,);

    fn len(&self) -> MIndex {
        self.0.len()
    }

    fn into_inner(self) -> Self::Inner {
        (crate::detail::device::DeviceColumnMutView::from_slice(
            &self.0.source.inner,
            usize_from_mindex(self.0.offset),
            usize_from_mindex(self.0.len),
        ),)
    }

    fn write_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error> {
        let output = self.into_inner().0;
        let input = crate::detail::device::DeviceColumnView::from_column(&inner.0);
        crate::detail::apply::MaterializeWriteApply::new(&output).collect_expr(policy, &input)
    }

    fn write_prefix_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error> {
        let mut output = self.into_inner().0;
        let input = crate::detail::device::DeviceColumnView::from_column(&inner.0);
        if input.len > output.len {
            return Err(Error::LengthMismatch {
                input: input.len,
                output: output.len,
            });
        }
        output.len = input.len;
        crate::detail::apply::MaterializeWriteApply::new(&output).collect_expr(policy, &input)
    }

    fn write_split_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        selected: <Self::Item as MAlloc<R>>::Inner,
        rejected: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error> {
        let output = self.into_inner().0;
        let selected_input = crate::detail::device::DeviceColumnView::from_column(&selected.0);
        let rejected_input = crate::detail::device::DeviceColumnView::from_column(&rejected.0);
        let input_len = selected_input.len + rejected_input.len;
        if input_len > output.len {
            return Err(Error::LengthMismatch {
                input: input_len,
                output: output.len,
            });
        }
        let mut selected_output = output.clone();
        selected_output.len = selected_input.len;
        crate::detail::apply::MaterializeWriteApply::new(&selected_output)
            .collect_expr(policy, &selected_input)?;

        let mut rejected_output = output;
        rejected_output.offset += selected_input.len;
        rejected_output.len = rejected_input.len;
        crate::detail::apply::MaterializeWriteApply::new(&rejected_output)
            .collect_expr(policy, &rejected_input)
    }

    fn write_where_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error> {
        let output = self.into_inner().0;
        let input = crate::detail::device::DeviceColumnView::from_column(&inner.0);
        crate::detail::apply::MaterializeWriteApply::new(&output).copy_where_expr(
            policy,
            &input,
            &stencil,
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
        let mask = stencil.mask();
        crate::detail::apply::MaskWriteApply::new(&mask, &output)
            .replace_value(policy, replacement.0)
    }

    fn fill_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        value: Self::Item,
    ) -> Result<(), Error> {
        let output = self.into_inner().0;
        crate::detail::apply::FillWriteApply::new(&output).fill_value(policy, value.0)
    }
}

impl<'a, R, T> sealed::MIterMutDispatch<R> for SoA1<DeviceSliceMut<'a, R, T>>
where
    R: Runtime,
    T: MStorageElement + 'static,
    (T,): MAlloc<R, Inner = (crate::detail::DeviceVec<R, T>,)>,
{
    fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
        exec.ensure_policy_id(self.0.source.inner.policy_id())
    }

    fn column_mut_view_inner<U: 'static>(
        &self,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<R, U>>, Error>
    where
        U: MStorageElement,
    {
        let source = &*self.0.source as &dyn Any;
        let source = match source.downcast_ref::<DeviceVec<R, U>>() {
            Some(source) => source,
            None => return Ok(None),
        };
        Ok(Some(
            crate::detail::device::DeviceColumnMutView::from_slice(
                &source.inner,
                usize_from_mindex(self.0.offset),
                usize_from_mindex(self.0.len),
            ),
        ))
    }
}
