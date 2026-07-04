use super::super::selection::unique_one_flags_read;
use super::super::*;
use crate::detail::control::{SegmentControl, UniqueByKeyControl};

fn unique_by_key_control_from_flags<R: Runtime>(
    policy: &CubePolicy<R>,
    len: usize,
    flags: cubecl::server::Handle,
) -> Result<UniqueByKeyControl, Error> {
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let segment: SegmentControl<R> = SegmentControl::from_head_flags(flags, len, len_u32);
    let selection =
        select::selected_rank_from_flags(policy, len, len_u32, segment.head_flags.clone())?;
    let count = select::selected_count(policy, &selection)?;
    Ok(UniqueByKeyControl { selection, count })
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
        let control = unique_by_key_control_from_flags(policy, len, flags)?;
        let payload_apply =
            crate::detail::api::SelectedPayloadApply::new(&control.selection, control.count);
        let out_keys = payload_apply.apply_expr(policy, &self)?;
        Ok((DeviceSoA1 { source: out_keys }, control))
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
                let control = unique_by_key_control_from_flags(policy, len, flags)?;
                let payload_apply = crate::detail::api::SelectedPayloadApply::new(
                    &control.selection,
                    control.count,
                );
                let left = payload_apply.apply_expr(policy, &self.$left)?;
                let right = payload_apply.apply_expr(policy, &self.$right)?;
                Ok(($out { left, right }, control))
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
                let control = unique_by_key_control_from_flags(policy, len, flags)?;
                let payload_apply = crate::detail::api::SelectedPayloadApply::new(
                    &control.selection,
                    control.count,
                );
                let first = payload_apply.apply_expr(policy, &self.$first)?;
                let second = payload_apply.apply_expr(policy, &self.$second)?;
                let third = payload_apply.apply_expr(policy, &self.$third)?;
                Ok((
                    $out {
                        first,
                        second,
                        third,
                    },
                    control,
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
        ensure_same_len(
            <ValueSource as KernelColumn>::len(&self),
            control.selection.len,
        )?;
        let payload_apply =
            crate::detail::api::SelectedPayloadApply::new(&control.selection, control.count);
        Ok(DeviceSoA1 {
            source: payload_apply.apply_expr(policy, &self)?,
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
                ensure_same_len(
                    <Left as KernelColumn>::len(&self.$left),
                    control.selection.len,
                )?;
                let payload_apply = crate::detail::api::SelectedPayloadApply::new(
                    &control.selection,
                    control.count,
                );
                let left = payload_apply.apply_expr(policy, &self.$left)?;
                let right = payload_apply.apply_expr(policy, &self.$right)?;
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
                ensure_same_len(
                    <First as KernelColumn>::len(&self.$first),
                    control.selection.len,
                )?;
                let payload_apply = crate::detail::api::SelectedPayloadApply::new(
                    &control.selection,
                    control.count,
                );
                let first = payload_apply.apply_expr(policy, &self.$first)?;
                let second = payload_apply.apply_expr(policy, &self.$second)?;
                let third = payload_apply.apply_expr(policy, &self.$third)?;
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
