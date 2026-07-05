use super::*;

fn unsupported<T>(name: &str) -> Result<T, Error> {
    Err(Error::Launch {
        message: format!("{name} is not supported for this iterator shape"),
    })
}

pub trait MIterDispatch<R: Runtime>: Sized {
    fn validate_executor(&self, _exec: &Executor<R>) -> Result<(), Error> {
        Ok(())
    }

    fn index_column_dispatch(
        self,
        _policy: &crate::detail::CubePolicy<R>,
    ) -> Result<crate::detail::device::DeviceColumnView<R, MIndex>, Error>
    where
        Self: MIter<R, Item = MIndex>,
    {
        unsupported("index iterator")
    }

    fn stencil_selection_dispatch(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _invert: bool,
        _flags_only: bool,
    ) -> Result<crate::detail::api::PrecomputedSelection<R>, Error>
    where
        Self: MIter<R, Item = u32>,
    {
        unsupported("stencil iterator")
    }

    fn transform_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Output: MIterMut<R>,
        Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
    {
        let _ = (policy, op, env, output);
        unsupported("transform")
    }

    fn map_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
        env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: StorageFromInner<R>,
        Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
    {
        let _ = (policy, op, env);
        unsupported("map")
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
        Self: MIter<R>,
        Output: MIterMut<R>,
        Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
    {
        let _ = (policy, op, env, stencil, output);
        unsupported("transform_where")
    }

    fn reverse_dispatch<Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = policy;
        unsupported("reverse")
    }

    fn sort_dispatch<Less, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, less);
        unsupported("sort")
    }

    fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        _less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        K: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: StorageFromInner<R, Item = (K,)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, keys, _less);
        unsupported("sort_by_key")
    }

    fn sort_by_single_key_into_dispatch<K, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        K: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: MIterMut<R, Item = (K,)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, keys, less, key_output, value_output);
        unsupported("sort_by_key")
    }

    fn sort_by_three_key_dispatch<K1, K2, K3, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, third_key, less);
        unsupported("sort_by_key")
    }

    fn sort_by_three_key_into_dispatch<K1, K2, K3, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (
            policy,
            first_key,
            second_key,
            third_key,
            less,
            key_output,
            value_output,
        );
        unsupported("sort_by_key")
    }

    fn sort_by_two_key_dispatch<K1, K2, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2)>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, less);
        unsupported("sort_by_key")
    }

    fn sort_by_two_key_into_dispatch<K1, K2, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2)>,
        KeyOutput: MIterMut<R, Item = (K1, K2)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (
            policy,
            first_key,
            second_key,
            less,
            key_output,
            value_output,
        );
        unsupported("sort_by_key")
    }

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
        KeyOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
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
        K: MStorageElement + 'static,
        Eq: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: StorageFromInner<R, Item = (K,)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, keys, _eq);
        unsupported("unique_by_key")
    }

    fn unique_by_single_key_into_dispatch<K, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        eq: Eq,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        K: MStorageElement + 'static,
        Eq: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: MIterMut<R, Item = (K,)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, keys, eq, key_output, value_output);
        unsupported("unique_by_key")
    }

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
        KeyOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "unique_by_key is not supported for this key iterator shape".to_string(),
        })
    }

    fn unique_by_three_key_dispatch<K1, K2, K3, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        Eq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, third_key, eq);
        unsupported("unique_by_key")
    }

    fn unique_by_three_key_into_dispatch<K1, K2, K3, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        eq: Eq,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        Eq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (
            policy,
            first_key,
            second_key,
            third_key,
            eq,
            key_output,
            value_output,
        );
        unsupported("unique_by_key")
    }

    fn unique_by_two_key_dispatch<K1, K2, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        eq: Eq,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        Eq: op::BinaryPredicateOp<R, (K1, K2)>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, eq);
        unsupported("unique_by_key")
    }

    fn unique_by_two_key_into_dispatch<K1, K2, Eq, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        eq: Eq,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        Eq: op::BinaryPredicateOp<R, (K1, K2)>,
        KeyOutput: MIterMut<R, Item = (K1, K2)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, eq, key_output, value_output);
        unsupported("unique_by_key")
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
        K: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, keys, key_eq, op);
        unsupported("inclusive_scan_by_key")
    }

    fn inclusive_scan_by_single_key_into_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        key_eq: KeyEq,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        K: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, keys, key_eq, op, output);
        unsupported("inclusive_scan_by_key")
    }

    fn inclusive_scan_by_three_key_dispatch<K1, K2, K3, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        key_eq: KeyEq,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, third_key, key_eq, op);
        unsupported("inclusive_scan_by_key")
    }

    fn inclusive_scan_by_three_key_into_dispatch<K1, K2, K3, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        key_eq: KeyEq,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, third_key, key_eq, op, output);
        unsupported("inclusive_scan_by_key")
    }

    fn inclusive_scan_by_two_key_dispatch<K1, K2, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        key_eq: KeyEq,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, key_eq, op);
        unsupported("inclusive_scan_by_key")
    }

    fn inclusive_scan_by_two_key_into_dispatch<K1, K2, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        key_eq: KeyEq,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, key_eq, op, output);
        unsupported("inclusive_scan_by_key")
    }

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
        Output: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
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
        K: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, keys, key_eq, op);
        unsupported("exclusive_scan_by_key")
    }

    fn exclusive_scan_by_single_key_into_dispatch<K, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        K: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, keys, key_eq, init, op, output);
        unsupported("exclusive_scan_by_key")
    }

    fn exclusive_scan_by_three_key_dispatch<K1, K2, K3, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, third_key, key_eq, init, op);
        unsupported("exclusive_scan_by_key")
    }

    fn exclusive_scan_by_three_key_into_dispatch<K1, K2, K3, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (
            policy, first_key, second_key, third_key, key_eq, init, op, output,
        );
        unsupported("exclusive_scan_by_key")
    }

    fn exclusive_scan_by_two_key_dispatch<K1, K2, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, key_eq, init, op);
        unsupported("exclusive_scan_by_key")
    }

    fn exclusive_scan_by_two_key_into_dispatch<K1, K2, KeyEq, Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, key_eq, init, op, output);
        unsupported("exclusive_scan_by_key")
    }

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
        Output: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
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
        K: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: StorageFromInner<R, Item = (K,)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, keys, key_eq, op);
        unsupported("reduce_by_key")
    }

    fn reduce_by_single_key_into_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        keys: crate::detail::device::DeviceColumnView<R, K>,
        key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        op: Op,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        K: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K,)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = (K,)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, keys, key_eq, init, op, key_output, value_output);
        unsupported("reduce_by_key")
    }

    fn reduce_by_three_key_dispatch<K1, K2, K3, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        op: Op,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, third_key, key_eq, init, op);
        unsupported("reduce_by_key")
    }

    fn reduce_by_three_key_into_dispatch<K1, K2, K3, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        third_key: crate::detail::device::DeviceColumnView<R, K3>,
        key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        op: Op,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (
            policy,
            first_key,
            second_key,
            third_key,
            key_eq,
            init,
            op,
            key_output,
            value_output,
        );
        unsupported("reduce_by_key")
    }

    fn reduce_by_two_key_dispatch<K1, K2, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        op: Op,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, first_key, second_key, key_eq, init, op);
        unsupported("reduce_by_key")
    }

    fn reduce_by_two_key_into_dispatch<K1, K2, KeyEq, Op, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        first_key: crate::detail::device::DeviceColumnView<R, K1>,
        second_key: crate::detail::device::DeviceColumnView<R, K2>,
        key_eq: KeyEq,
        init: <Self as MIter<R>>::Item,
        op: Op,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = (K1, K2)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (
            policy,
            first_key,
            second_key,
            key_eq,
            init,
            op,
            key_output,
            value_output,
        );
        unsupported("reduce_by_key")
    }

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
        KeyOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
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
        RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
        K: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: StorageFromInner<R, Item = (K,)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (left_keys, right_keys);
        Err(Error::Launch {
            message: "merge_by_key is not supported for this iterator shape".to_string(),
        })
    }

    fn merge_by_single_key_same_into_dispatch<K, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        left_keys: crate::detail::device::DeviceColumnView<R, K>,
        right_keys: crate::detail::device::DeviceColumnView<R, K>,
        right_values: RightValues,
        less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
        K: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K,)>,
        KeyOutput: MIterMut<R, Item = (K,)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (
            policy,
            left_keys,
            right_keys,
            right_values,
            less,
            key_output,
            value_output,
        );
        unsupported("merge_by_key")
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
        less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (
            policy,
            left_first_key,
            left_second_key,
            left_third_key,
            right_first_key,
            right_second_key,
            right_third_key,
            right_values,
            less,
        );
        unsupported("merge_by_key")
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
        less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        K3: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
        KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (
            policy,
            left_first_key,
            left_second_key,
            left_third_key,
            right_first_key,
            right_second_key,
            right_third_key,
            right_values,
            less,
            key_output,
            value_output,
        );
        unsupported("merge_by_key")
    }

    fn merge_by_two_key_same_dispatch<K1, K2, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        left_first_key: crate::detail::device::DeviceColumnView<R, K1>,
        left_second_key: crate::detail::device::DeviceColumnView<R, K2>,
        right_first_key: crate::detail::device::DeviceColumnView<R, K1>,
        right_second_key: crate::detail::device::DeviceColumnView<R, K2>,
        right_values: RightValues,
        less: Less,
    ) -> Result<(KeyOutput, ValueOutput), Error>
    where
        Self: MIter<R>,
        RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2)>,
        KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
        ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (
            policy,
            left_first_key,
            left_second_key,
            right_first_key,
            right_second_key,
            right_values,
            less,
        );
        unsupported("merge_by_key")
    }

    fn merge_by_two_key_same_into_dispatch<K1, K2, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        left_first_key: crate::detail::device::DeviceColumnView<R, K1>,
        left_second_key: crate::detail::device::DeviceColumnView<R, K2>,
        right_first_key: crate::detail::device::DeviceColumnView<R, K1>,
        right_second_key: crate::detail::device::DeviceColumnView<R, K2>,
        right_values: RightValues,
        less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
        K1: MStorageElement + 'static,
        K2: MStorageElement + 'static,
        Less: op::BinaryPredicateOp<R, (K1, K2)>,
        KeyOutput: MIterMut<R, Item = (K1, K2)>,
        ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (
            policy,
            left_first_key,
            left_second_key,
            right_first_key,
            right_second_key,
            right_values,
            less,
            key_output,
            value_output,
        );
        unsupported("merge_by_key")
    }

    fn gather_dispatch<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Indices: MIter<R, Item = MIndex>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, indices, output);
        unsupported("gather")
    }

    fn permute_dispatch<Indices, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        indices: Indices,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Indices: MIter<R, Item = MIndex>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, indices);
        unsupported("permute")
    }

    fn gather_where_dispatch<Indices, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _indices: Indices,
        _stencil: crate::detail::api::PrecomputedSelection<R>,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Indices: MIter<R, Item = MIndex>,
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
        Indices: MIter<R, Item = MIndex>,
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
        Indices: MIter<R, Item = MIndex>,
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
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, _init, op);
        unsupported("reduce")
    }

    fn inclusive_scan_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, op);
        unsupported("inclusive_scan")
    }

    fn exclusive_scan_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        _init: <Self as MIter<R>>::Item,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, _init, op);
        unsupported("exclusive_scan")
    }

    fn adjacent_difference_dispatch<Op, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        op: Op,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, op);
        unsupported("adjacent_difference")
    }

    fn copy_where_dispatch<Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (_policy, _stencil);
        unsupported("copy_where")
    }

    fn remove_if_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, pred, env);
        unsupported("remove_if")
    }

    fn remove_where_dispatch<Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (_policy, _stencil);
        unsupported("remove_where")
    }

    fn count_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, pred, env);
        unsupported("count_if")
    }

    fn all_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, pred, env);
        unsupported("all_of")
    }

    fn any_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, pred, env);
        unsupported("any_of")
    }

    fn none_of_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, pred, env);
        unsupported("none_of")
    }

    fn find_if_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<Option<MIndex>, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, pred, env);
        unsupported("find_if")
    }

    fn partition_dispatch<Pred, Output>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<(Output, Output), Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, pred, env);
        unsupported("partition")
    }

    fn is_partitioned_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, pred, env);
        unsupported("is_partitioned")
    }

    fn replace_where_dispatch<Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        replacement: <Self as MIter<R>>::Item,
        _stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<Output, Error>
    where
        Self: MIter<R>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (_policy, replacement, _stencil);
        unsupported("replace_where")
    }

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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
    {
        let _ = (policy, pred);
        unsupported("unique")
    }

    fn min_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<Option<MIndex>, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, less);
        unsupported("min_element")
    }

    fn max_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<Option<MIndex>, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, less);
        unsupported("max_element")
    }

    fn minmax_element_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<Option<(MIndex, MIndex)>, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, less);
        unsupported("minmax_element")
    }

    fn adjacent_find_dispatch<Pred>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        pred: Pred,
    ) -> Result<Option<MIndex>, Error>
    where
        Self: MIter<R>,
        Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, pred);
        unsupported("adjacent_find")
    }

    fn equal_dispatch<Right, Eq>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _eq: Eq,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
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
    ) -> Result<Option<MIndex>, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
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
    ) -> Result<Option<MIndex>, Error>
    where
        Self: MIter<R>,
        Needles: MIter<R, Item = <Self as MIter<R>>::Item>,
        Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "find_first_of is not supported for this iterator shape".to_string(),
        })
    }

    fn lower_bound_dispatch<Values, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        _less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, MIndex>, Error>
    where
        Self: MIter<R>,
        Values: MIter<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, values, _less);
        unsupported("lower_bound")
    }

    fn upper_bound_dispatch<Values, Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        values: Values,
        _less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, MIndex>, Error>
    where
        Self: MIter<R>,
        Values: MIter<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, values, _less);
        unsupported("upper_bound")
    }

    fn is_sorted_until_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, less);
        unsupported("is_sorted_until")
    }

    fn is_sorted_dispatch<Less>(
        self,
        policy: &crate::detail::CubePolicy<R>,
        less: Less,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        let _ = (policy, less);
        unsupported("is_sorted")
    }

    fn lexicographical_compare_dispatch<Right, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _less: Less,
    ) -> Result<bool, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
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
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "set_difference is not supported for this iterator shape".to_string(),
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
    ) -> Result<Option<MIndex>, Error>
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
    ) -> Result<Option<MIndex>, Error>
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
        Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "set_difference is not supported for this iterator shape".to_string(),
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
        RightKeys: MIter<R, Item = <Self as MIter<R>>::Item>,
        LeftValues: MIter<R>,
        RightValues: MIter<R, Item = <LeftValues as MIter<R>>::Item>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: StorageFromInner<R, Item = <LeftValues as MIter<R>>::Item>,
    {
        Err(Error::Launch {
            message: "merge_by_key is not supported for this key iterator shape".to_string(),
        })
    }

    fn reverse_into_dispatch<Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        unsupported("reverse")
    }

    fn sort_into_dispatch<Less, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _less: Less,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        unsupported("sort")
    }

    fn merge_into_dispatch<Right, Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _less: Less,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        unsupported("merge")
    }

    fn set_union_into_dispatch<Right, Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _less: Less,
        _output: Output,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        unsupported("set_union")
    }

    fn set_intersection_into_dispatch<Right, Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _less: Less,
        _output: Output,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        unsupported("set_intersection")
    }

    fn set_difference_into_dispatch<Right, Output, Less>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _right: Right,
        _less: Less,
        _output: Output,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        Right: MIter<R, Item = <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
    {
        unsupported("set_difference")
    }

    fn inclusive_scan_into_dispatch<Op, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _op: Op,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        unsupported("inclusive_scan")
    }

    fn exclusive_scan_into_dispatch<Op, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _init: <Self as MIter<R>>::Item,
        _op: Op,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        unsupported("exclusive_scan")
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
        Self: MIter<R>,
        Values: MIter<R>,
        KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Values as MIter<R>>::Item>,
    {
        let _ = (policy, values, key_eq, init, op, output);
        unsupported("exclusive_scan_by_key")
    }

    fn adjacent_difference_into_dispatch<Op, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _op: Op,
        _output: Output,
    ) -> Result<(), Error>
    where
        Self: MIter<R>,
        Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        unsupported("adjacent_difference")
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
        Self: MIter<R>,
        Values: MIter<R>,
        KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Values as MIter<R>>::Item>,
    {
        let _ = (policy, values, key_eq, op, output);
        unsupported("inclusive_scan_by_key")
    }

    fn copy_where_into_dispatch<Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _stencil: crate::detail::api::PrecomputedSelection<R>,
        _output: Output,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        unsupported("copy_where")
    }

    fn remove_where_into_dispatch<Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _stencil: crate::detail::api::PrecomputedSelection<R>,
        _output: Output,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        unsupported("remove_where")
    }

    fn partition_into_dispatch<Pred, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        _env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
        _output: Output,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        unsupported("partition")
    }

    fn unique_into_dispatch<Pred, Output>(
        self,
        _policy: &crate::detail::CubePolicy<R>,
        _pred: Pred,
        _output: Output,
    ) -> Result<MIndex, Error>
    where
        Self: MIter<R>,
        Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
    {
        unsupported("unique")
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
        Self: MIter<R>,
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MIterMut<R, Item = <Values as MIter<R>>::Item>,
    {
        let _ = (policy, values, key_eq, init, op, key_output, value_output);
        unsupported("reduce_by_key")
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
        Self: MIter<R>,
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MIterMut<R, Item = <Values as MIter<R>>::Item>,
    {
        let _ = (policy, values, eq, key_output, value_output);
        unsupported("unique_by_key")
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
        Self: MIter<R>,
        Values: MIter<R>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MIterMut<R, Item = <Values as MIter<R>>::Item>,
    {
        let _ = (policy, values, less, key_output, value_output);
        unsupported("sort_by_key")
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
        Self: MIter<R>,
        RightKeys: MIter<R, Item = <Self as MIter<R>>::Item>,
        LeftValues: MIter<R>,
        RightValues: MIter<R, Item = <LeftValues as MIter<R>>::Item>,
        <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
        Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
        KeyOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
        ValueOutput: MIterMut<R, Item = <LeftValues as MIter<R>>::Item>,
    {
        let _ = (
            policy,
            right_keys,
            left_values,
            right_values,
            less,
            key_output,
            value_output,
        );
        unsupported("merge_by_key")
    }
}
