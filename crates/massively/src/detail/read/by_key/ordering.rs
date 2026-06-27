use super::super::*;

pub(crate) trait KernelSortByKeyKeys<Less>: Sized {
    type Runtime: Runtime;
    type OutputKeys;

    fn sort_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, DeviceVec<Self::Runtime, u32>), Error>;
}

pub(crate) trait KernelSortByKeyValues<IndexSource>: Sized
where
    IndexSource: KernelColumn<Item = u32> + KernelColumnAt<S0>,
{
    type Runtime: Runtime;
    type OutputValues;

    fn sort_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
    ) -> Result<Self::OutputValues, Error>;
}

pub(crate) trait KernelMergeByKeyKeys<RightKeys, Less>: Sized
where
    RightKeys: Sized,
{
    type Runtime: Runtime;
    type OutputKeys;

    fn merge_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
        right_keys: RightKeys,
    ) -> Result<(Self::OutputKeys, primitive_ordering::MergeByKeyControl), Error>;
}

pub(crate) trait KernelMergeByKeyValues<RightValues>: Sized {
    type Runtime: Runtime;
    type OutputValues;

    fn merge_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        right_values: RightValues,
        control: &primitive_ordering::MergeByKeyControl,
    ) -> Result<Self::OutputValues, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelSortByKeyCall<Values, Less>: Sized {
    type Runtime: Runtime;
    type Output;

    fn sort_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelMergeByKeyCall<LeftValues, RightKeys, RightValues, Less>:
    Sized
{
    type Runtime: Runtime;
    type Output;

    fn merge_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        left_values: LeftValues,
        right_keys: RightKeys,
        right_values: RightValues,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;
}
impl<KeySource, Less> KernelSortByKeyKeys<Less> for KeySource
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    KeySource::Item: Scalar + 'static,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    Less: BinaryPredicateOp<KeySource::Item>,
{
    type Runtime = KeySource::Runtime;
    type OutputKeys = DeviceSoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>;

    fn sort_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, DeviceVec<Self::Runtime, u32>), Error> {
        let indices =
            primitive_range::indices_u32(policy, <KeySource as KernelColumn>::len(&self))?;
        let (keys, indices) = primitive_ordering::sort_by_key_input_with_policy(
            policy,
            &self,
            &indices,
            crate::op::GpuOp::<Less>::new(),
        )?;
        Ok((DeviceSoA1 { source: keys }, indices))
    }
}

macro_rules! impl_kernel_sort_by_key_keys_tuple1 {
    ($target:ty, $field:tt) => {
        impl<KeySource, Less> KernelSortByKeyKeys<Less> for $target
        where
            KeySource: KernelColumn + KernelColumnAt<S0>,
            KeySource::Item: Scalar + 'static,
            KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
            Less: BinaryPredicateOp<KeySource::Item>,
        {
            type Runtime = KeySource::Runtime;
            type OutputKeys = DeviceSoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>;

            fn sort_by_key_control(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<(Self::OutputKeys, DeviceVec<Self::Runtime, u32>), Error> {
                <KeySource as KernelSortByKeyKeys<Less>>::sort_by_key_control(self.$field, policy)
            }
        }
    };
}

impl_kernel_sort_by_key_keys_tuple1!(SoAView1<KeySource>, source);
impl_kernel_sort_by_key_keys_tuple1!(DeviceSoA1<KeySource>, source);

impl<KeySource, Less> KernelSortByKeyKeys<Less> for (KeySource,)
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    KeySource::Item: Scalar + 'static,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    Less: BinaryPredicateOp<(KeySource::Item,)>,
    crate::detail::api::Tuple1Less<Less>: BinaryPredicateOp<KeySource::Item>,
{
    type Runtime = KeySource::Runtime;
    type OutputKeys = DeviceSoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>;

    fn sort_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, DeviceVec<Self::Runtime, u32>), Error> {
        <KeySource as KernelSortByKeyKeys<crate::detail::api::Tuple1Less<Less>>>::sort_by_key_control(
            self.0,
            policy,
        )
    }
}

impl<ValueSource, IndexSource> KernelSortByKeyValues<IndexSource> for ValueSource
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    ValueSource::Item: Scalar + 'static,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    type Runtime = ValueSource::Runtime;
    type OutputValues = DeviceSoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn sort_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
    ) -> Result<Self::OutputValues, Error> {
        validate_key_column(indices, <ValueSource as KernelColumn>::len(&self))?;
        self.gather_read(policy, indices)
    }
}

macro_rules! impl_kernel_sort_by_key_values_tuple1 {
    ($target:ty, $field:tt) => {
        impl<ValueSource, IndexSource> KernelSortByKeyValues<IndexSource> for $target
        where
            ValueSource: KernelColumn + KernelColumnAt<S0>,
            IndexSource:
                KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
            ValueSource::Item: Scalar + 'static,
            ValueSource::Expr: GpuExpr<ValueSource::Item>,
            IndexSource::Expr: GpuExpr<u32>,
        {
            type Runtime = ValueSource::Runtime;
            type OutputValues = DeviceSoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

            fn sort_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
            ) -> Result<Self::OutputValues, Error> {
                self.$field.sort_by_key_values(policy, indices)
            }
        }
    };
}

impl_kernel_sort_by_key_values_tuple1!(SoAView1<ValueSource>, source);
impl_kernel_sort_by_key_values_tuple1!(DeviceSoA1<ValueSource>, source);

impl<ValueSource, IndexSource> KernelSortByKeyValues<IndexSource> for (ValueSource,)
where
    ValueSource: KernelSortByKeyValues<IndexSource>,
    IndexSource: KernelColumn<Item = u32> + KernelColumnAt<S0>,
{
    type Runtime = <ValueSource as KernelSortByKeyValues<IndexSource>>::Runtime;
    type OutputValues = <ValueSource as KernelSortByKeyValues<IndexSource>>::OutputValues;

    fn sort_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
    ) -> Result<Self::OutputValues, Error> {
        self.0.sort_by_key_values(policy, indices)
    }
}

macro_rules! impl_kernel_sort_by_key_values_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right, IndexSource> KernelSortByKeyValues<IndexSource> for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            IndexSource: KernelColumn<Runtime = Left::Runtime, Item = u32> + KernelColumnAt<S0>,
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: GpuExpr<Left::Item>,
            Right::Expr: GpuExpr<Right::Item>,
            IndexSource::Expr: GpuExpr<u32>,
        {
            type Runtime = Left::Runtime;
            type OutputValues =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

            fn sort_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
            ) -> Result<Self::OutputValues, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                validate_key_column(indices, <Left as KernelColumn>::len(&self.$left))?;
                self.gather_read(policy, indices)
            }
        }
    };
}

impl_kernel_sort_by_key_values_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_sort_by_key_values_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right, IndexSource> KernelSortByKeyValues<IndexSource> for (Left, Right)
where
    IndexSource: KernelColumn<Item = u32> + KernelColumnAt<S0>,
    SoAView2<Left, Right>: KernelSortByKeyValues<IndexSource>,
{
    type Runtime = <SoAView2<Left, Right> as KernelSortByKeyValues<IndexSource>>::Runtime;
    type OutputValues = <SoAView2<Left, Right> as KernelSortByKeyValues<IndexSource>>::OutputValues;

    fn sort_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
    ) -> Result<Self::OutputValues, Error> {
        <SoAView2<Left, Right> as KernelSortByKeyValues<IndexSource>>::sort_by_key_values(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            indices,
        )
    }
}

macro_rules! impl_kernel_sort_by_key_values_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, IndexSource> KernelSortByKeyValues<IndexSource> for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            IndexSource: KernelColumn<Runtime = First::Runtime, Item = u32> + KernelColumnAt<S0>,
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
            First::Expr: GpuExpr<First::Item>,
            Second::Expr: GpuExpr<Second::Item>,
            Third::Expr: GpuExpr<Third::Item>,
            IndexSource::Expr: GpuExpr<u32>,
        {
            type Runtime = First::Runtime;
            type OutputValues = $out<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;

            fn sort_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
            ) -> Result<Self::OutputValues, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                validate_key_column(indices, <First as KernelColumn>::len(&self.$first))?;
                self.gather_read(policy, indices)
            }
        }
    };
}

impl_kernel_sort_by_key_values_tuple3!(SoAView3<First, Second, Third>, DeviceSoA3, first, second, third);
impl_kernel_sort_by_key_values_tuple3!(DeviceSoA3<First, Second, Third>, DeviceSoA3, first, second, third);

impl<First, Second, Third, IndexSource> KernelSortByKeyValues<IndexSource>
    for (First, Second, Third)
where
    IndexSource: KernelColumn<Item = u32> + KernelColumnAt<S0>,
    SoAView3<First, Second, Third>: KernelSortByKeyValues<IndexSource>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelSortByKeyValues<IndexSource>>::Runtime;
    type OutputValues =
        <SoAView3<First, Second, Third> as KernelSortByKeyValues<IndexSource>>::OutputValues;

    fn sort_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
    ) -> Result<Self::OutputValues, Error> {
        <SoAView3<First, Second, Third> as KernelSortByKeyValues<IndexSource>>::sort_by_key_values(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            indices,
        )
    }
}

impl<LeftKey, RightKey, Less> KernelMergeByKeyKeys<RightKey, Less> for LeftKey
where
    LeftKey: KernelColumn + KernelColumnAt<S0>,
    RightKey: KernelColumn<Runtime = LeftKey::Runtime, Item = LeftKey::Item> + KernelColumnAt<S0>,
    LeftKey::Item: Scalar + 'static,
    LeftKey::Expr: DeviceGpuExpr<LeftKey::Item>,
    RightKey::Expr: DeviceGpuExpr<RightKey::Item>,
    Less: BinaryPredicateOp<LeftKey::Item>,
{
    type Runtime = LeftKey::Runtime;
    type OutputKeys = DeviceSoA1<DeviceVec<LeftKey::Runtime, LeftKey::Item>>;

    fn merge_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
        right_keys: RightKey,
    ) -> Result<(Self::OutputKeys, primitive_ordering::MergeByKeyControl), Error> {
        let (keys, control) = crate::detail::api::device_expr_merge_by_key_control_with_policy::<
            LeftKey,
            RightKey,
            Less,
        >(policy, &self, &right_keys)?;
        Ok((DeviceSoA1 { source: keys }, control))
    }
}

macro_rules! impl_kernel_merge_by_key_keys_tuple1 {
    ($left_target:ty, $right_target:ty, $left_field:tt, $right_field:tt) => {
        impl<LeftKey, RightKey, Less> KernelMergeByKeyKeys<$right_target, Less> for $left_target
        where
            LeftKey: KernelMergeByKeyKeys<RightKey, Less>,
        {
            type Runtime = LeftKey::Runtime;
            type OutputKeys = LeftKey::OutputKeys;

            fn merge_by_key_control(
                self,
                policy: &CubePolicy<Self::Runtime>,
                right_keys: $right_target,
            ) -> Result<(Self::OutputKeys, primitive_ordering::MergeByKeyControl), Error> {
                self.$left_field
                    .merge_by_key_control(policy, right_keys.$right_field)
            }
        }
    };
}

impl_kernel_merge_by_key_keys_tuple1!(SoAView1<LeftKey>, SoAView1<RightKey>, source, source);
impl_kernel_merge_by_key_keys_tuple1!(DeviceSoA1<LeftKey>, DeviceSoA1<RightKey>, source, source);

impl<LeftKey, RightKey, Less> KernelMergeByKeyKeys<(RightKey,), Less> for (LeftKey,)
where
    LeftKey: KernelColumn + KernelColumnAt<S0>,
    RightKey: KernelColumn<Runtime = LeftKey::Runtime, Item = LeftKey::Item> + KernelColumnAt<S0>,
    LeftKey::Item: Scalar + 'static,
    LeftKey::Expr: DeviceGpuExpr<LeftKey::Item>,
    RightKey::Expr: DeviceGpuExpr<RightKey::Item>,
    Less: BinaryPredicateOp<(LeftKey::Item,)>,
{
    type Runtime = LeftKey::Runtime;
    type OutputKeys = DeviceSoA1<DeviceVec<LeftKey::Runtime, LeftKey::Item>>;

    fn merge_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
        right_keys: (RightKey,),
    ) -> Result<(Self::OutputKeys, primitive_ordering::MergeByKeyControl), Error> {
        <LeftKey as KernelMergeByKeyKeys<RightKey, crate::detail::api::Tuple1Less<Less>>>::merge_by_key_control(
            self.0,
            policy,
            right_keys.0,
        )
    }
}

impl<LeftValue, RightValue> KernelMergeByKeyValues<RightValue> for LeftValue
where
    LeftValue: KernelColumn + KernelColumnAt<S0>,
    RightValue:
        KernelColumn<Runtime = LeftValue::Runtime, Item = LeftValue::Item> + KernelColumnAt<S0>,
    LeftValue::Item: Scalar + 'static,
    LeftValue::Expr: DeviceGpuExpr<LeftValue::Item>,
    RightValue::Expr: DeviceGpuExpr<RightValue::Item>,
{
    type Runtime = LeftValue::Runtime;
    type OutputValues = DeviceSoA1<DeviceVec<LeftValue::Runtime, LeftValue::Item>>;

    fn merge_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        right_values: RightValue,
        control: &primitive_ordering::MergeByKeyControl,
    ) -> Result<Self::OutputValues, Error> {
        let values = crate::detail::api::device_expr_merge_by_key_values_with_control_with_policy(
            policy,
            &self,
            &right_values,
            control,
        )?;
        Ok(DeviceSoA1 { source: values })
    }
}

macro_rules! impl_kernel_merge_by_key_values_tuple1 {
    ($left_target:ty, $right_target:ty, $left_field:tt, $right_field:tt) => {
        impl<LeftValue, RightValue> KernelMergeByKeyValues<$right_target> for $left_target
        where
            LeftValue: KernelColumn + KernelColumnAt<S0>,
            RightValue: KernelColumn<Runtime = LeftValue::Runtime, Item = LeftValue::Item>
                + KernelColumnAt<S0>,
            LeftValue::Item: Scalar + 'static,
            LeftValue::Expr: DeviceGpuExpr<LeftValue::Item>,
            RightValue::Expr: DeviceGpuExpr<RightValue::Item>,
        {
            type Runtime = LeftValue::Runtime;
            type OutputValues = DeviceSoA1<DeviceVec<LeftValue::Runtime, LeftValue::Item>>;

            fn merge_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                right_values: $right_target,
                control: &primitive_ordering::MergeByKeyControl,
            ) -> Result<Self::OutputValues, Error> {
                self.$left_field
                    .merge_by_key_values(policy, right_values.$right_field, control)
            }
        }
    };
}

impl_kernel_merge_by_key_values_tuple1!(SoAView1<LeftValue>, SoAView1<RightValue>, source, source);
impl_kernel_merge_by_key_values_tuple1!(
    DeviceSoA1<LeftValue>,
    DeviceSoA1<RightValue>,
    source,
    source
);

impl<LeftValue, RightValue> KernelMergeByKeyValues<(RightValue,)> for (LeftValue,)
where
    LeftValue: KernelMergeByKeyValues<RightValue>,
{
    type Runtime = LeftValue::Runtime;
    type OutputValues = LeftValue::OutputValues;

    fn merge_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        right_values: (RightValue,),
        control: &primitive_ordering::MergeByKeyControl,
    ) -> Result<Self::OutputValues, Error> {
        self.0.merge_by_key_values(policy, right_values.0, control)
    }
}

macro_rules! impl_kernel_merge_by_key_values_tuple2 {
    ($left_target:ty, $right_target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<LeftA, LeftB, RightA, RightB> KernelMergeByKeyValues<$right_target> for $left_target
        where
            LeftA: KernelColumn + KernelColumnAt<S0>,
            LeftB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
            RightA: KernelColumn<Runtime = LeftA::Runtime, Item = LeftA::Item> + KernelColumnAt<S0>,
            RightB: KernelColumn<Runtime = LeftA::Runtime, Item = LeftB::Item> + KernelColumnAt<S0>,
            LeftA::Item: Scalar + 'static,
            LeftB::Item: Scalar + 'static,
            LeftA::Expr: DeviceGpuExpr<LeftA::Item>,
            LeftB::Expr: DeviceGpuExpr<LeftB::Item>,
            RightA::Expr: DeviceGpuExpr<RightA::Item>,
            RightB::Expr: DeviceGpuExpr<RightB::Item>,
        {
            type Runtime = LeftA::Runtime;
            type OutputValues = $out<
                DeviceVec<LeftA::Runtime, LeftA::Item>,
                DeviceVec<LeftA::Runtime, LeftB::Item>,
            >;

            fn merge_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                right_values: $right_target,
                control: &primitive_ordering::MergeByKeyControl,
            ) -> Result<Self::OutputValues, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                validate_columns2(&right_values.$left, &right_values.$right)?;
                let left =
                    crate::detail::api::device_expr_merge_by_key_values_with_control_with_policy(
                        policy,
                        &self.$left,
                        &right_values.$left,
                        control,
                    )?;
                let right =
                    crate::detail::api::device_expr_merge_by_key_values_with_control_with_policy(
                        policy,
                        &self.$right,
                        &right_values.$right,
                        control,
                    )?;
                Ok($out { left, right })
            }
        }
    };
}

impl_kernel_merge_by_key_values_tuple2!(SoAView2<LeftA, LeftB>, SoAView2<RightA, RightB>, DeviceSoA2, left, right);
impl_kernel_merge_by_key_values_tuple2!(DeviceSoA2<LeftA, LeftB>, DeviceSoA2<RightA, RightB>, DeviceSoA2, left, right);

impl<LeftA, LeftB, RightA, RightB> KernelMergeByKeyValues<(RightA, RightB)> for (LeftA, LeftB)
where
    SoAView2<LeftA, LeftB>: KernelMergeByKeyValues<SoAView2<RightA, RightB>>,
{
    type Runtime =
        <SoAView2<LeftA, LeftB> as KernelMergeByKeyValues<SoAView2<RightA, RightB>>>::Runtime;
    type OutputValues =
        <SoAView2<LeftA, LeftB> as KernelMergeByKeyValues<SoAView2<RightA, RightB>>>::OutputValues;

    fn merge_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        right_values: (RightA, RightB),
        control: &primitive_ordering::MergeByKeyControl,
    ) -> Result<Self::OutputValues, Error> {
        SoAView2 {
            left: self.0,
            right: self.1,
        }
        .merge_by_key_values(
            policy,
            SoAView2 {
                left: right_values.0,
                right: right_values.1,
            },
            control,
        )
    }
}

macro_rules! impl_kernel_merge_by_key_values_tuple3 {
    ($left_target:ty, $right_target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<LeftA, LeftB, LeftC, RightA, RightB, RightC> KernelMergeByKeyValues<$right_target>
            for $left_target
        where
            LeftA: KernelColumn + KernelColumnAt<S0>,
            LeftB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
            LeftC: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
            RightA: KernelColumn<Runtime = LeftA::Runtime, Item = LeftA::Item> + KernelColumnAt<S0>,
            RightB: KernelColumn<Runtime = LeftA::Runtime, Item = LeftB::Item> + KernelColumnAt<S0>,
            RightC: KernelColumn<Runtime = LeftA::Runtime, Item = LeftC::Item> + KernelColumnAt<S0>,
            LeftA::Item: Scalar + 'static,
            LeftB::Item: Scalar + 'static,
            LeftC::Item: Scalar + 'static,
            LeftA::Expr: DeviceGpuExpr<LeftA::Item>,
            LeftB::Expr: DeviceGpuExpr<LeftB::Item>,
            LeftC::Expr: DeviceGpuExpr<LeftC::Item>,
            RightA::Expr: DeviceGpuExpr<RightA::Item>,
            RightB::Expr: DeviceGpuExpr<RightB::Item>,
            RightC::Expr: DeviceGpuExpr<RightC::Item>,
        {
            type Runtime = LeftA::Runtime;
            type OutputValues = $out<
                DeviceVec<LeftA::Runtime, LeftA::Item>,
                DeviceVec<LeftA::Runtime, LeftB::Item>,
                DeviceVec<LeftA::Runtime, LeftC::Item>,
            >;

            fn merge_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                right_values: $right_target,
                control: &primitive_ordering::MergeByKeyControl,
            ) -> Result<Self::OutputValues, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                validate_columns3(
                    &right_values.$first,
                    &right_values.$second,
                    &right_values.$third,
                )?;
                let first =
                    crate::detail::api::device_expr_merge_by_key_values_with_control_with_policy(
                        policy,
                        &self.$first,
                        &right_values.$first,
                        control,
                    )?;
                let second =
                    crate::detail::api::device_expr_merge_by_key_values_with_control_with_policy(
                        policy,
                        &self.$second,
                        &right_values.$second,
                        control,
                    )?;
                let third =
                    crate::detail::api::device_expr_merge_by_key_values_with_control_with_policy(
                        policy,
                        &self.$third,
                        &right_values.$third,
                        control,
                    )?;
                Ok($out {
                    first,
                    second,
                    third,
                })
            }
        }
    };
}

impl_kernel_merge_by_key_values_tuple3!(SoAView3<LeftA, LeftB, LeftC>, SoAView3<RightA, RightB, RightC>, DeviceSoA3, first, second, third);
impl_kernel_merge_by_key_values_tuple3!(DeviceSoA3<LeftA, LeftB, LeftC>, DeviceSoA3<RightA, RightB, RightC>, DeviceSoA3, first, second, third);

impl<LeftA, LeftB, LeftC, RightA, RightB, RightC> KernelMergeByKeyValues<(RightA, RightB, RightC)>
    for (LeftA, LeftB, LeftC)
where
    SoAView3<LeftA, LeftB, LeftC>: KernelMergeByKeyValues<SoAView3<RightA, RightB, RightC>>,
{
    type Runtime = <SoAView3<LeftA, LeftB, LeftC> as KernelMergeByKeyValues<
        SoAView3<RightA, RightB, RightC>,
    >>::Runtime;
    type OutputValues = <SoAView3<LeftA, LeftB, LeftC> as KernelMergeByKeyValues<
        SoAView3<RightA, RightB, RightC>,
    >>::OutputValues;

    fn merge_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        right_values: (RightA, RightB, RightC),
        control: &primitive_ordering::MergeByKeyControl,
    ) -> Result<Self::OutputValues, Error> {
        SoAView3 {
            first: self.0,
            second: self.1,
            third: self.2,
        }
        .merge_by_key_values(
            policy,
            SoAView3 {
                first: right_values.0,
                second: right_values.1,
                third: right_values.2,
            },
            control,
        )
    }
}
