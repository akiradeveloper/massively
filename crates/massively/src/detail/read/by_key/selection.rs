use super::super::selection::unique_one_flags_read;
use super::super::*;

#[allow(dead_code)]
pub(crate) struct UniqueByKeyControl {
    pub(crate) flags: cubecl::server::Handle,
    pub(crate) len: usize,
}

#[allow(dead_code)]
pub(crate) trait KernelUniqueByKeyKeys<Eq>: Sized {
    type Runtime: Runtime;
    type OutputKeys;

    fn unique_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, UniqueByKeyControl), Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelUniqueByKeyValues: Sized {
    type Runtime: Runtime;
    type OutputValues;

    fn unique_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &UniqueByKeyControl,
    ) -> Result<Self::OutputValues, Error>;
}

#[allow(dead_code)]

pub(crate) trait KernelUniqueByKeyCall<Values, Eq>: Sized {
    type Runtime: Runtime;
    type Output;

    fn unique_by_key_call(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error>;
}
impl<KeySource, Eq> KernelUniqueByKeyKeys<Eq> for KeySource
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    KeySource::Item: Scalar + 'static,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    Eq: BinaryPredicateOp<KeySource::Item>,
{
    type Runtime = KeySource::Runtime;
    type OutputKeys = DeviceSoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>;

    fn unique_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, UniqueByKeyControl), Error> {
        <KeySource as KernelColumn>::validate(&self)?;
        let len = <KeySource as KernelColumn>::len(&self);
        let flags = unique_one_flags_read::<KeySource, Eq>(policy, &self)?;
        let out_keys = crate::detail::api::device_expr_compact_with_flags_with_policy(
            policy,
            &self,
            flags.clone(),
        )?;
        Ok((
            DeviceSoA1 { source: out_keys },
            UniqueByKeyControl { flags, len },
        ))
    }
}

macro_rules! impl_kernel_unique_by_key_keys_tuple1 {
    ($target:ty, $field:tt) => {
        impl<KeySource, Eq> KernelUniqueByKeyKeys<Eq> for $target
        where
            KeySource: KernelColumn + KernelColumnAt<S0>,
            KeySource::Item: Scalar + 'static,
            KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
            Eq: BinaryPredicateOp<KeySource::Item>,
        {
            type Runtime = KeySource::Runtime;
            type OutputKeys = DeviceSoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>;

            fn unique_by_key_control(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<(Self::OutputKeys, UniqueByKeyControl), Error> {
                <KeySource as KernelUniqueByKeyKeys<Eq>>::unique_by_key_control(self.$field, policy)
            }
        }
    };
}

impl_kernel_unique_by_key_keys_tuple1!(SoAView1<KeySource>, source);
impl_kernel_unique_by_key_keys_tuple1!(DeviceSoA1<KeySource>, source);

impl<KeySource, Eq> KernelUniqueByKeyKeys<Eq> for (KeySource,)
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    KeySource::Item: Scalar + 'static,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    Eq: BinaryPredicateOp<(KeySource::Item,)>,
    crate::detail::api::Tuple1Less<Eq>: BinaryPredicateOp<KeySource::Item>,
{
    type Runtime = KeySource::Runtime;
    type OutputKeys = DeviceSoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>;

    fn unique_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, UniqueByKeyControl), Error> {
        <KeySource as KernelUniqueByKeyKeys<crate::detail::api::Tuple1Less<Eq>>>::unique_by_key_control(
            self.0,
            policy,
        )
    }
}

macro_rules! impl_kernel_unique_by_key_keys_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right, Eq> KernelUniqueByKeyKeys<Eq> for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: DeviceGpuExpr<Left::Item>,
            Right::Expr: DeviceGpuExpr<Right::Item>,
            Eq: BinaryPredicateOp<(Left::Item, Right::Item)>,
        {
            type Runtime = Left::Runtime;
            type OutputKeys =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

            fn unique_by_key_control(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<(Self::OutputKeys, UniqueByKeyControl), Error> {
                validate_columns2(&self.$left, &self.$right)?;
                let len = <Left as KernelColumn>::len(&self.$left);
                let flags = super::super::selection::unique_tuple2_flags_read::<Left, Right, Eq>(
                    policy,
                    &self.$left,
                    &self.$right,
                )?;
                let left = crate::detail::api::device_expr_compact_with_flags_with_policy(
                    policy,
                    &self.$left,
                    flags.clone(),
                )?;
                let right = crate::detail::api::device_expr_compact_with_flags_with_policy(
                    policy,
                    &self.$right,
                    flags.clone(),
                )?;
                Ok(($out { left, right }, UniqueByKeyControl { flags, len }))
            }
        }
    };
}

impl_kernel_unique_by_key_keys_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_unique_by_key_keys_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right, Eq> KernelUniqueByKeyKeys<Eq> for (Left, Right)
where
    SoAView2<Left, Right>: KernelUniqueByKeyKeys<Eq>,
{
    type Runtime = <SoAView2<Left, Right> as KernelUniqueByKeyKeys<Eq>>::Runtime;
    type OutputKeys = <SoAView2<Left, Right> as KernelUniqueByKeyKeys<Eq>>::OutputKeys;

    fn unique_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, UniqueByKeyControl), Error> {
        <SoAView2<Left, Right> as KernelUniqueByKeyKeys<Eq>>::unique_by_key_control(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
        )
    }
}

macro_rules! impl_kernel_unique_by_key_keys_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, Eq> KernelUniqueByKeyKeys<Eq> for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
            First::Expr: DeviceGpuExpr<First::Item>,
            Second::Expr: DeviceGpuExpr<Second::Item>,
            Third::Expr: DeviceGpuExpr<Third::Item>,
            Eq: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
        {
            type Runtime = First::Runtime;
            type OutputKeys = $out<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;

            fn unique_by_key_control(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<(Self::OutputKeys, UniqueByKeyControl), Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                let len = <First as KernelColumn>::len(&self.$first);
                let flags = super::super::selection::unique_tuple3_flags_read::<
                    First,
                    Second,
                    Third,
                    Eq,
                >(policy, &self.$first, &self.$second, &self.$third)?;
                let first = crate::detail::api::device_expr_compact_with_flags_with_policy(
                    policy,
                    &self.$first,
                    flags.clone(),
                )?;
                let second = crate::detail::api::device_expr_compact_with_flags_with_policy(
                    policy,
                    &self.$second,
                    flags.clone(),
                )?;
                let third = crate::detail::api::device_expr_compact_with_flags_with_policy(
                    policy,
                    &self.$third,
                    flags.clone(),
                )?;
                Ok((
                    $out {
                        first,
                        second,
                        third,
                    },
                    UniqueByKeyControl { flags, len },
                ))
            }
        }
    };
}

impl_kernel_unique_by_key_keys_tuple3!(
    SoAView3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);
impl_kernel_unique_by_key_keys_tuple3!(
    DeviceSoA3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);

impl<First, Second, Third, Eq> KernelUniqueByKeyKeys<Eq> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelUniqueByKeyKeys<Eq>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelUniqueByKeyKeys<Eq>>::Runtime;
    type OutputKeys = <SoAView3<First, Second, Third> as KernelUniqueByKeyKeys<Eq>>::OutputKeys;

    fn unique_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, UniqueByKeyControl), Error> {
        <SoAView3<First, Second, Third> as KernelUniqueByKeyKeys<Eq>>::unique_by_key_control(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
        )
    }
}

impl<ValueSource> KernelUniqueByKeyValues for ValueSource
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Item: Scalar + 'static,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
{
    type Runtime = ValueSource::Runtime;
    type OutputValues = DeviceSoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn unique_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &UniqueByKeyControl,
    ) -> Result<Self::OutputValues, Error> {
        <ValueSource as KernelColumn>::validate(&self)?;
        ensure_same_len(<ValueSource as KernelColumn>::len(&self), control.len)?;
        Ok(DeviceSoA1 {
            source: crate::detail::api::device_expr_compact_with_flags_with_policy(
                policy,
                &self,
                control.flags.clone(),
            )?,
        })
    }
}

macro_rules! impl_kernel_unique_by_key_values_tuple1 {
    ($target:ty, $field:tt) => {
        impl<ValueSource> KernelUniqueByKeyValues for $target
        where
            ValueSource: KernelColumn + KernelColumnAt<S0>,
            ValueSource::Item: Scalar + 'static,
            ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
        {
            type Runtime = ValueSource::Runtime;
            type OutputValues = DeviceSoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

            fn unique_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &UniqueByKeyControl,
            ) -> Result<Self::OutputValues, Error> {
                self.$field.unique_by_key_values(policy, control)
            }
        }
    };
}

impl_kernel_unique_by_key_values_tuple1!(SoAView1<ValueSource>, source);
impl_kernel_unique_by_key_values_tuple1!(DeviceSoA1<ValueSource>, source);

impl<ValueSource> KernelUniqueByKeyValues for (ValueSource,)
where
    ValueSource: KernelUniqueByKeyValues,
{
    type Runtime = <ValueSource as KernelUniqueByKeyValues>::Runtime;
    type OutputValues = <ValueSource as KernelUniqueByKeyValues>::OutputValues;

    fn unique_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &UniqueByKeyControl,
    ) -> Result<Self::OutputValues, Error> {
        self.0.unique_by_key_values(policy, control)
    }
}

macro_rules! impl_kernel_unique_by_key_values_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right> KernelUniqueByKeyValues for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: DeviceGpuExpr<Left::Item>,
            Right::Expr: DeviceGpuExpr<Right::Item>,
        {
            type Runtime = Left::Runtime;
            type OutputValues =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

            fn unique_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &UniqueByKeyControl,
            ) -> Result<Self::OutputValues, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                ensure_same_len(<Left as KernelColumn>::len(&self.$left), control.len)?;
                let left = crate::detail::api::device_expr_compact_with_flags_with_policy(
                    policy,
                    &self.$left,
                    control.flags.clone(),
                )?;
                let right = crate::detail::api::device_expr_compact_with_flags_with_policy(
                    policy,
                    &self.$right,
                    control.flags.clone(),
                )?;
                Ok($out { left, right })
            }
        }
    };
}

impl_kernel_unique_by_key_values_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_unique_by_key_values_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right> KernelUniqueByKeyValues for (Left, Right)
where
    SoAView2<Left, Right>: KernelUniqueByKeyValues,
{
    type Runtime = <SoAView2<Left, Right> as KernelUniqueByKeyValues>::Runtime;
    type OutputValues = <SoAView2<Left, Right> as KernelUniqueByKeyValues>::OutputValues;

    fn unique_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &UniqueByKeyControl,
    ) -> Result<Self::OutputValues, Error> {
        <SoAView2<Left, Right> as KernelUniqueByKeyValues>::unique_by_key_values(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            control,
        )
    }
}

macro_rules! impl_kernel_unique_by_key_values_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third> KernelUniqueByKeyValues for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
            First::Expr: DeviceGpuExpr<First::Item>,
            Second::Expr: DeviceGpuExpr<Second::Item>,
            Third::Expr: DeviceGpuExpr<Third::Item>,
        {
            type Runtime = First::Runtime;
            type OutputValues = $out<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;

            fn unique_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &UniqueByKeyControl,
            ) -> Result<Self::OutputValues, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                ensure_same_len(<First as KernelColumn>::len(&self.$first), control.len)?;
                let first = crate::detail::api::device_expr_compact_with_flags_with_policy(
                    policy,
                    &self.$first,
                    control.flags.clone(),
                )?;
                let second = crate::detail::api::device_expr_compact_with_flags_with_policy(
                    policy,
                    &self.$second,
                    control.flags.clone(),
                )?;
                let third = crate::detail::api::device_expr_compact_with_flags_with_policy(
                    policy,
                    &self.$third,
                    control.flags.clone(),
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

impl_kernel_unique_by_key_values_tuple3!(
    SoAView3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);
impl_kernel_unique_by_key_values_tuple3!(
    DeviceSoA3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);

impl<First, Second, Third> KernelUniqueByKeyValues for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelUniqueByKeyValues,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelUniqueByKeyValues>::Runtime;
    type OutputValues = <SoAView3<First, Second, Third> as KernelUniqueByKeyValues>::OutputValues;

    fn unique_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &UniqueByKeyControl,
    ) -> Result<Self::OutputValues, Error> {
        <SoAView3<First, Second, Third> as KernelUniqueByKeyValues>::unique_by_key_values(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            control,
        )
    }
}
