use super::super::*;

pub(crate) trait KernelSortByKeyKeys<Less>: Sized {
    type Runtime: Runtime;
    type OutputKeys;

    fn sort_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, DeviceVec<Self::Runtime, MIndex>), Error>;
}

pub(crate) trait KernelSortByKeyValues: Sized {
    type Runtime: Runtime;
    type OutputValues;

    fn sort_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &crate::detail::control::PermutationControl<Self::Runtime>,
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
    ) -> Result<(Self::OutputKeys, DeviceVec<Self::Runtime, MIndex>), Error> {
        let (keys, indices) =
            crate::detail::apply::SortByKeyApply::apply_keys1::<KeySource, Less>(policy, &self)?;
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
            ) -> Result<(Self::OutputKeys, DeviceVec<Self::Runtime, MIndex>), Error> {
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
    ) -> Result<(Self::OutputKeys, DeviceVec<Self::Runtime, MIndex>), Error> {
        <KeySource as KernelSortByKeyKeys<crate::detail::api::Tuple1Less<Less>>>::sort_by_key_control(
            self.0,
            policy,
        )
    }
}

impl<First, Second, Less> KernelSortByKeyKeys<Less> for (First, Second)
where
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Item: Scalar + 'static,
    Second::Item: Scalar + 'static,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Less: BinaryPredicateOp<(First::Item, Second::Item)>,
    crate::detail::api::Tuple2AsTuple3Less<Less>:
        BinaryPredicateOp<(First::Item, Second::Item, u32)>,
{
    type Runtime = First::Runtime;
    type OutputKeys =
        DeviceSoA2<DeviceVec<First::Runtime, First::Item>, DeviceVec<First::Runtime, Second::Item>>;

    fn sort_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, DeviceVec<Self::Runtime, MIndex>), Error> {
        let (first, second, indices) =
            crate::detail::apply::SortByKeyApply::apply_keys2::<First, Second, Less>(
                policy, &self.0, &self.1,
            )?;
        Ok((
            DeviceSoA2 {
                left: first,
                right: second,
            },
            indices,
        ))
    }
}

impl<First, Second, Third, Less> KernelSortByKeyKeys<Less> for (First, Second, Third)
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
    Less: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
{
    type Runtime = First::Runtime;
    type OutputKeys = DeviceSoA3<
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    >;

    fn sort_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<(Self::OutputKeys, DeviceVec<Self::Runtime, MIndex>), Error> {
        let (first, second, third, indices) =
            crate::detail::apply::SortByKeyApply::apply_keys3::<First, Second, Third, Less>(
                policy, &self.0, &self.1, &self.2,
            )?;
        Ok((
            DeviceSoA3 {
                first,
                second,
                third,
            },
            indices,
        ))
    }
}

impl<ValueSource> KernelSortByKeyValues for ValueSource
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Item: Scalar + 'static,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
{
    type Runtime = ValueSource::Runtime;
    type OutputValues = DeviceSoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn sort_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &crate::detail::control::PermutationControl<Self::Runtime>,
    ) -> Result<Self::OutputValues, Error> {
        ensure_same_len(control.len, <ValueSource as KernelColumn>::len(&self))?;
        let apply = crate::detail::apply::PermutationPayloadApply::new(control);
        Ok(DeviceSoA1 {
            source: apply.apply_expr(policy, &self)?,
        })
    }
}

macro_rules! impl_kernel_sort_by_key_values_tuple1 {
    ($target:ty, $field:tt) => {
        impl<ValueSource> KernelSortByKeyValues for $target
        where
            ValueSource: KernelColumn + KernelColumnAt<S0>,
            ValueSource::Item: Scalar + 'static,
            ValueSource::Expr: GpuExpr<ValueSource::Item>,
        {
            type Runtime = ValueSource::Runtime;
            type OutputValues = DeviceSoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

            fn sort_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &crate::detail::control::PermutationControl<Self::Runtime>,
            ) -> Result<Self::OutputValues, Error> {
                self.$field.sort_by_key_values(policy, control)
            }
        }
    };
}

impl_kernel_sort_by_key_values_tuple1!(SoAView1<ValueSource>, source);
impl_kernel_sort_by_key_values_tuple1!(DeviceSoA1<ValueSource>, source);

impl<ValueSource> KernelSortByKeyValues for (ValueSource,)
where
    ValueSource: KernelSortByKeyValues,
{
    type Runtime = <ValueSource as KernelSortByKeyValues>::Runtime;
    type OutputValues = <ValueSource as KernelSortByKeyValues>::OutputValues;

    fn sort_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &crate::detail::control::PermutationControl<Self::Runtime>,
    ) -> Result<Self::OutputValues, Error> {
        self.0.sort_by_key_values(policy, control)
    }
}

macro_rules! impl_kernel_sort_by_key_values_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right> KernelSortByKeyValues for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: GpuExpr<Left::Item>,
            Right::Expr: GpuExpr<Right::Item>,
        {
            type Runtime = Left::Runtime;
            type OutputValues =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

            fn sort_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &crate::detail::control::PermutationControl<Self::Runtime>,
            ) -> Result<Self::OutputValues, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                ensure_same_len(control.len, <Left as KernelColumn>::len(&self.$left))?;
                let apply = crate::detail::apply::PermutationPayloadApply::new(control);
                let (left, right) = apply.apply_expr2(policy, &self.$left, &self.$right)?;
                Ok($out { left, right })
            }
        }
    };
}

impl_kernel_sort_by_key_values_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_sort_by_key_values_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right> KernelSortByKeyValues for (Left, Right)
where
    SoAView2<Left, Right>: KernelSortByKeyValues,
{
    type Runtime = <SoAView2<Left, Right> as KernelSortByKeyValues>::Runtime;
    type OutputValues = <SoAView2<Left, Right> as KernelSortByKeyValues>::OutputValues;

    fn sort_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &crate::detail::control::PermutationControl<Self::Runtime>,
    ) -> Result<Self::OutputValues, Error> {
        <SoAView2<Left, Right> as KernelSortByKeyValues>::sort_by_key_values(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            control,
        )
    }
}

macro_rules! impl_kernel_sort_by_key_values_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third> KernelSortByKeyValues for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
            First::Expr: GpuExpr<First::Item>,
            Second::Expr: GpuExpr<Second::Item>,
            Third::Expr: GpuExpr<Third::Item>,
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
                control: &crate::detail::control::PermutationControl<Self::Runtime>,
            ) -> Result<Self::OutputValues, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                ensure_same_len(control.len, <First as KernelColumn>::len(&self.$first))?;
                let apply = crate::detail::apply::PermutationPayloadApply::new(control);
                let (first, second, third) =
                    apply.apply_expr3(policy, &self.$first, &self.$second, &self.$third)?;
                Ok($out {
                    first,
                    second,
                    third,
                })
            }
        }
    };
}

impl_kernel_sort_by_key_values_tuple3!(SoAView3<First, Second, Third>, DeviceSoA3, first, second, third);
impl_kernel_sort_by_key_values_tuple3!(DeviceSoA3<First, Second, Third>, DeviceSoA3, first, second, third);

impl<First, Second, Third> KernelSortByKeyValues for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelSortByKeyValues,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelSortByKeyValues>::Runtime;
    type OutputValues = <SoAView3<First, Second, Third> as KernelSortByKeyValues>::OutputValues;

    fn sort_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &crate::detail::control::PermutationControl<Self::Runtime>,
    ) -> Result<Self::OutputValues, Error> {
        <SoAView3<First, Second, Third> as KernelSortByKeyValues>::sort_by_key_values(
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

macro_rules! impl_kernel_sort_by_key_values_wide_tuple {
    ($apply:ident; $first_ty:ident : $first_idx:tt $(, $ty:ident : $idx:tt )+) => {
        impl<$first_ty, $( $ty, )+> KernelSortByKeyValues
            for ($first_ty, $( $ty, )+)
        where
            $first_ty: KernelColumn + KernelColumnAt<S0>,
            $(
                $ty: KernelColumn<Runtime = $first_ty::Runtime> + KernelColumnAt<S0>,
            )+
            $first_ty::Item: Scalar + 'static,
            $first_ty::Expr: GpuExpr<$first_ty::Item>,
            $(
                $ty::Runtime: Runtime,
                $ty::Item: Scalar + 'static,
                $ty::Expr: GpuExpr<$ty::Item>,
            )+
        {
            type Runtime = $first_ty::Runtime;
            type OutputValues = (
                DeviceVec<Self::Runtime, $first_ty::Item>,
                $( DeviceVec<Self::Runtime, $ty::Item>, )+
            );

            fn sort_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                control: &crate::detail::control::PermutationControl<Self::Runtime>,
            ) -> Result<Self::OutputValues, Error> {
                self.$first_idx.validate()?;
                $( self.$idx.validate()?; )+
                ensure_same_len(control.len, self.0.len())?;
                let apply = crate::detail::apply::PermutationPayloadApply::new(control);
                apply.$apply(policy, &self.$first_idx, $( &self.$idx, )+)
            }
        }
    };
}

impl_kernel_sort_by_key_values_wide_tuple!(apply_expr4; A: 0, B: 1, C: 2, D: 3);
impl_kernel_sort_by_key_values_wide_tuple!(apply_expr5; A: 0, B: 1, C: 2, D: 3, E: 4);
impl_kernel_sort_by_key_values_wide_tuple!(apply_expr6; A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);

impl<A, B, C, D, E, F, G> KernelSortByKeyValues for (A, B, C, D, E, F, G)
where
    A: KernelColumn + KernelColumnAt<S0>,
    B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    G: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: Scalar + 'static,
    B::Item: Scalar + 'static,
    C::Item: Scalar + 'static,
    D::Item: Scalar + 'static,
    E::Item: Scalar + 'static,
    F::Item: Scalar + 'static,
    G::Item: Scalar + 'static,
    A::Expr: GpuExpr<A::Item>,
    B::Expr: GpuExpr<B::Item>,
    C::Expr: GpuExpr<C::Item>,
    D::Expr: GpuExpr<D::Item>,
    E::Expr: GpuExpr<E::Item>,
    F::Expr: GpuExpr<F::Item>,
    G::Expr: GpuExpr<G::Item>,
{
    type Runtime = A::Runtime;
    type OutputValues = (
        DeviceVec<A::Runtime, A::Item>,
        DeviceVec<A::Runtime, B::Item>,
        DeviceVec<A::Runtime, C::Item>,
        DeviceVec<A::Runtime, D::Item>,
        DeviceVec<A::Runtime, E::Item>,
        DeviceVec<A::Runtime, F::Item>,
        DeviceVec<A::Runtime, G::Item>,
    );

    fn sort_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        control: &crate::detail::control::PermutationControl<Self::Runtime>,
    ) -> Result<Self::OutputValues, Error> {
        self.0.validate()?;
        self.1.validate()?;
        self.2.validate()?;
        self.3.validate()?;
        self.4.validate()?;
        self.5.validate()?;
        self.6.validate()?;
        ensure_same_len(self.1.len(), self.0.len())?;
        ensure_same_len(self.2.len(), self.0.len())?;
        ensure_same_len(self.3.len(), self.0.len())?;
        ensure_same_len(self.4.len(), self.0.len())?;
        ensure_same_len(self.5.len(), self.0.len())?;
        ensure_same_len(self.6.len(), self.0.len())?;
        ensure_same_len(control.len, self.0.len())?;
        let apply = crate::detail::apply::PermutationPayloadApply::new(control);
        apply.apply_expr7(
            policy, &self.0, &self.1, &self.2, &self.3, &self.4, &self.5, &self.6,
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
        let (keys, control) = crate::detail::apply::MergeByKeyControlApply::apply_keys1::<
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

impl<LeftA, LeftB, RightA, RightB, Less> KernelMergeByKeyKeys<(RightA, RightB), Less>
    for (LeftA, LeftB)
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
    Less: BinaryPredicateOp<(LeftA::Item, LeftB::Item)>,
{
    type Runtime = LeftA::Runtime;
    type OutputKeys =
        DeviceSoA2<DeviceVec<LeftA::Runtime, LeftA::Item>, DeviceVec<LeftA::Runtime, LeftB::Item>>;

    fn merge_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
        right_keys: (RightA, RightB),
    ) -> Result<(Self::OutputKeys, primitive_ordering::MergeByKeyControl), Error> {
        crate::detail::apply::MergeByKeyControlApply::apply_keys2::<
            LeftA,
            LeftB,
            RightA,
            RightB,
            Less,
        >(policy, &self.0, &self.1, &right_keys.0, &right_keys.1)
    }
}

impl<LeftA, LeftB, LeftC, RightA, RightB, RightC, Less>
    KernelMergeByKeyKeys<(RightA, RightB, RightC), Less> for (LeftA, LeftB, LeftC)
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
    Less: BinaryPredicateOp<(LeftA::Item, LeftB::Item, LeftC::Item)>,
{
    type Runtime = LeftA::Runtime;
    type OutputKeys = DeviceSoA3<
        DeviceVec<LeftA::Runtime, LeftA::Item>,
        DeviceVec<LeftA::Runtime, LeftB::Item>,
        DeviceVec<LeftA::Runtime, LeftC::Item>,
    >;

    fn merge_by_key_control(
        self,
        policy: &CubePolicy<Self::Runtime>,
        right_keys: (RightA, RightB, RightC),
    ) -> Result<(Self::OutputKeys, primitive_ordering::MergeByKeyControl), Error> {
        crate::detail::apply::MergeByKeyControlApply::apply_keys3::<
            LeftA,
            LeftB,
            LeftC,
            RightA,
            RightB,
            RightC,
            Less,
        >(
            policy,
            &self.0,
            &self.1,
            &self.2,
            &right_keys.0,
            &right_keys.1,
            &right_keys.2,
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
        let apply = crate::detail::apply::MergePayloadApply::new(control);
        let values = apply.apply_expr(policy, &self, &right_values)?;
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
                let apply = crate::detail::apply::MergePayloadApply::new(control);
                let (left, right) = apply.apply_expr2(
                    policy,
                    &self.$left,
                    &self.$right,
                    &right_values.$left,
                    &right_values.$right,
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
                let apply = crate::detail::apply::MergePayloadApply::new(control);
                let (first, second, third) = apply.apply_expr3(
                    policy,
                    &self.$first,
                    &self.$second,
                    &self.$third,
                    &right_values.$first,
                    &right_values.$second,
                    &right_values.$third,
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

macro_rules! impl_kernel_merge_by_key_values_tuple_wide {
    ($first_left:ident : $first_right:ident : $first_idx:tt, $($left:ident : $right:ident : $idx:tt),+) => {
        impl<$first_left, $($left,)+ $first_right, $($right,)+>
            KernelMergeByKeyValues<($first_right, $($right,)+)>
            for ($first_left, $($left,)+)
        where
            $first_left: KernelColumn + KernelColumnAt<S0>,
            $first_right: KernelColumn<
                Runtime = $first_left::Runtime,
                Item = $first_left::Item,
            > + KernelColumnAt<S0>,
            $first_left::Item: Scalar + 'static,
            $first_left::Expr: DeviceGpuExpr<$first_left::Item>,
            $first_right::Expr: DeviceGpuExpr<$first_right::Item>,
            $(
                $left: KernelColumn<Runtime = $first_left::Runtime> + KernelColumnAt<S0>,
                $right: KernelColumn<
                    Runtime = $first_left::Runtime,
                    Item = $left::Item,
                > + KernelColumnAt<S0>,
                $left::Item: Scalar + 'static,
                $left::Expr: DeviceGpuExpr<$left::Item>,
                $right::Expr: DeviceGpuExpr<$right::Item>,
            )+
        {
            type Runtime = $first_left::Runtime;
            type OutputValues = (
                DeviceVec<Self::Runtime, $first_left::Item>,
                $(DeviceVec<Self::Runtime, $left::Item>,)+
            );

            fn merge_by_key_values(
                self,
                policy: &CubePolicy<Self::Runtime>,
                right_values: ($first_right, $($right,)+),
                control: &primitive_ordering::MergeByKeyControl,
            ) -> Result<Self::OutputValues, Error> {
                self.$first_idx.validate()?;
                right_values.$first_idx.validate()?;
                $(
                    self.$idx.validate()?;
                    right_values.$idx.validate()?;
                )+
                let apply = crate::detail::apply::MergePayloadApply::new(control);
                impl_kernel_merge_by_key_values_tuple_wide!(@apply apply, policy, self, right_values; $first_idx, $($idx),+)
            }
        }
    };
    (@apply $apply:ident, $policy:ident, $self_value:ident, $right_values:ident; 0, 1, 2, 3) => {
        $apply.apply_expr4(
            $policy,
            (&$self_value.0, &$self_value.1, &$self_value.2, &$self_value.3),
            (&$right_values.0, &$right_values.1, &$right_values.2, &$right_values.3),
        )
    };
    (@apply $apply:ident, $policy:ident, $self_value:ident, $right_values:ident; 0, 1, 2, 3, 4) => {
        $apply.apply_expr5(
            $policy,
            (&$self_value.0, &$self_value.1, &$self_value.2, &$self_value.3, &$self_value.4),
            (&$right_values.0, &$right_values.1, &$right_values.2, &$right_values.3, &$right_values.4),
        )
    };
    (@apply $apply:ident, $policy:ident, $self_value:ident, $right_values:ident; 0, 1, 2, 3, 4, 5) => {
        $apply.apply_expr6(
            $policy,
            (
                &$self_value.0,
                &$self_value.1,
                &$self_value.2,
                &$self_value.3,
                &$self_value.4,
                &$self_value.5,
            ),
            (
                &$right_values.0,
                &$right_values.1,
                &$right_values.2,
                &$right_values.3,
                &$right_values.4,
                &$right_values.5,
            ),
        )
    };
}

impl_kernel_merge_by_key_values_tuple_wide!(LA: RA: 0, LB: RB: 1, LC: RC: 2, LD: RD: 3);
impl_kernel_merge_by_key_values_tuple_wide!(LA: RA: 0, LB: RB: 1, LC: RC: 2, LD: RD: 3, LE: RE: 4);
impl_kernel_merge_by_key_values_tuple_wide!(LA: RA: 0, LB: RB: 1, LC: RC: 2, LD: RD: 3, LE: RE: 4, LF: RF: 5);

impl<LA, LB, LC, LD, LE, LF, LG, RA, RB, RC, RD, RE, RF, RG>
    KernelMergeByKeyValues<(RA, RB, RC, RD, RE, RF, RG)> for (LA, LB, LC, LD, LE, LF, LG)
where
    LA: KernelColumn + KernelColumnAt<S0>,
    LB: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
    LC: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
    LD: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
    LE: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
    LF: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
    LG: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
    RA: KernelColumn<Runtime = LA::Runtime, Item = LA::Item> + KernelColumnAt<S0>,
    RB: KernelColumn<Runtime = LA::Runtime, Item = LB::Item> + KernelColumnAt<S0>,
    RC: KernelColumn<Runtime = LA::Runtime, Item = LC::Item> + KernelColumnAt<S0>,
    RD: KernelColumn<Runtime = LA::Runtime, Item = LD::Item> + KernelColumnAt<S0>,
    RE: KernelColumn<Runtime = LA::Runtime, Item = LE::Item> + KernelColumnAt<S0>,
    RF: KernelColumn<Runtime = LA::Runtime, Item = LF::Item> + KernelColumnAt<S0>,
    RG: KernelColumn<Runtime = LA::Runtime, Item = LG::Item> + KernelColumnAt<S0>,
    LA::Item: Scalar + 'static,
    LB::Item: Scalar + 'static,
    LC::Item: Scalar + 'static,
    LD::Item: Scalar + 'static,
    LE::Item: Scalar + 'static,
    LF::Item: Scalar + 'static,
    LG::Item: Scalar + 'static,
    LA::Expr: DeviceGpuExpr<LA::Item>,
    LB::Expr: DeviceGpuExpr<LB::Item>,
    LC::Expr: DeviceGpuExpr<LC::Item>,
    LD::Expr: DeviceGpuExpr<LD::Item>,
    LE::Expr: DeviceGpuExpr<LE::Item>,
    LF::Expr: DeviceGpuExpr<LF::Item>,
    LG::Expr: DeviceGpuExpr<LG::Item>,
    RA::Expr: DeviceGpuExpr<RA::Item>,
    RB::Expr: DeviceGpuExpr<RB::Item>,
    RC::Expr: DeviceGpuExpr<RC::Item>,
    RD::Expr: DeviceGpuExpr<RD::Item>,
    RE::Expr: DeviceGpuExpr<RE::Item>,
    RF::Expr: DeviceGpuExpr<RF::Item>,
    RG::Expr: DeviceGpuExpr<RG::Item>,
{
    type Runtime = LA::Runtime;
    type OutputValues = (
        DeviceVec<LA::Runtime, LA::Item>,
        DeviceVec<LA::Runtime, LB::Item>,
        DeviceVec<LA::Runtime, LC::Item>,
        DeviceVec<LA::Runtime, LD::Item>,
        DeviceVec<LA::Runtime, LE::Item>,
        DeviceVec<LA::Runtime, LF::Item>,
        DeviceVec<LA::Runtime, LG::Item>,
    );

    fn merge_by_key_values(
        self,
        policy: &CubePolicy<Self::Runtime>,
        right_values: (RA, RB, RC, RD, RE, RF, RG),
        control: &primitive_ordering::MergeByKeyControl,
    ) -> Result<Self::OutputValues, Error> {
        self.0.validate()?;
        self.1.validate()?;
        self.2.validate()?;
        self.3.validate()?;
        self.4.validate()?;
        self.5.validate()?;
        self.6.validate()?;
        right_values.0.validate()?;
        right_values.1.validate()?;
        right_values.2.validate()?;
        right_values.3.validate()?;
        right_values.4.validate()?;
        right_values.5.validate()?;
        right_values.6.validate()?;
        let apply = crate::detail::apply::MergePayloadApply::new(control);
        apply.apply_expr7(
            policy,
            (
                &self.0, &self.1, &self.2, &self.3, &self.4, &self.5, &self.6,
            ),
            (
                &right_values.0,
                &right_values.1,
                &right_values.2,
                &right_values.3,
                &right_values.4,
                &right_values.5,
                &right_values.6,
            ),
        )
    }
}
