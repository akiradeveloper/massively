use super::*;
impl<Keys, Values, Less> crate::detail::read::KernelSortByKeyCall<Values, Less> for Keys
where
    Keys: crate::detail::read::KernelSortByKeyKeys<Less>,
    Values: crate::detail::read::KernelSortByKeyValues<
            DeviceVec<Keys::Runtime, u32>,
            Runtime = Keys::Runtime,
        >,
{
    type Runtime = Keys::Runtime;
    type Output = (
        <Keys as crate::detail::read::KernelSortByKeyKeys<Less>>::OutputKeys,
        <Values as crate::detail::read::KernelSortByKeyValues<
            DeviceVec<Keys::Runtime, u32>,
        >>::OutputValues,
    );

    fn sort_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let (keys, indices) = self.sort_by_key_control(policy)?;
        let control = crate::detail::control::PermutationControl::from_indices(&indices)?;
        let indices = control.indices(policy);
        let values = values.sort_by_key_values(policy, &indices)?;
        Ok((keys, values))
    }
}

impl<LeftKeys, LeftValues, RightKeys, RightValues, Less>
    crate::detail::read::KernelMergeByKeyCall<LeftValues, RightKeys, RightValues, Less> for LeftKeys
where
    LeftKeys: crate::detail::read::KernelMergeByKeyKeys<RightKeys, Less>,
    LeftValues:
        crate::detail::read::KernelMergeByKeyValues<RightValues, Runtime = LeftKeys::Runtime>,
{
    type Runtime = LeftKeys::Runtime;
    type Output = (
        <LeftKeys as crate::detail::read::KernelMergeByKeyKeys<RightKeys, Less>>::OutputKeys,
        <LeftValues as crate::detail::read::KernelMergeByKeyValues<RightValues>>::OutputValues,
    );

    fn merge_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        left_values: LeftValues,
        right_keys: RightKeys,
        right_values: RightValues,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let (keys, control) = self.merge_by_key_control(policy, right_keys)?;
        let values = left_values.merge_by_key_values(policy, right_values, &control)?;
        Ok((keys, values))
    }
}

/// Sorts read-only key-value pairs by key and returns owned SoA outputs.
pub fn sort_by_key<R, Keys, Values, Less>(
    policy: &CubePolicy<R>,
    keys: Keys,
    values: Values,
    _less: Less,
) -> Result<<<Keys as crate::detail::read::KernelSortByKeyCall<Values, Less>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Keys: crate::detail::read::KernelSortByKeyCall<Values, Less, Runtime = R>,
    <Keys as crate::detail::read::KernelSortByKeyCall<Values, Less>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        keys.sort_by_key_call(policy, values, GpuOp::<Less>::new())?,
    )
}

/// Merges two sorted key-value ranges by key.
pub fn merge_by_key<R, LeftKeys, LeftValues, RightKeys, RightValues, Less>(
    policy: &CubePolicy<R>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    _less: Less,
) -> Result<
    <<LeftKeys as crate::detail::read::KernelMergeByKeyCall<
        LeftValues,
        RightKeys,
        RightValues,
        Less,
    >>::Output as MaterializeOutput>::Output,
    Error,
>
where
    R: Runtime,
    LeftKeys: crate::detail::read::KernelMergeByKeyCall<
            LeftValues,
            RightKeys,
            RightValues,
            Less,
            Runtime = R,
        >,
    <LeftKeys as crate::detail::read::KernelMergeByKeyCall<
        LeftValues,
        RightKeys,
        RightValues,
        Less,
    >>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        left_keys.merge_by_key_call(
            policy,
            left_values,
            right_keys,
            right_values,
            GpuOp::<Less>::new(),
        )?,
    )
}
