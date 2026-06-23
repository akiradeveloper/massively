use super::*;
use crate::detail::api::{Tuple1Less, device_expr_gather_with_policy};
use crate::error::ensure_same_len;

/// Key/value input accepted by [`sort_by_key`].
#[doc(hidden)]
pub trait SortByKeyInput<Values, Less> {
    /// Runtime used by this input.
    type Runtime: Runtime;
    /// Output produced by key-value sorting.
    type Output;

    /// Sorts key-value pairs by key.
    fn sort_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;

    /// Sorts key-value pairs by key with an explicit executor policy.
    fn sort_by_key_input_with_policy(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>
    where
        Self: Sized,
    {
        self.sort_by_key_input(policy, values, less)
    }
}

impl<KeySource, ValueSource, Less> SortByKeyInput<SoA1<ValueSource>, Less> for SoAView1<KeySource>
where
    Self: ReadOnlySoA<Item = (KeySource::Item,), Scalar = KeySource::Item>,
    SoA1<ValueSource>: SoA<Item = (ValueSource::Item,), Scalar = ValueSource::Item>,
    KeySource: KernelColumn + KernelColumnAt<S0>,
    ValueSource: ReadOnlyKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Less: BinaryPredicateOp<KeySource::Item>,
{
    type Runtime = KeySource::Runtime;
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA1<DeviceVec<KeySource::Runtime, ValueSource::Item>>,
    );

    fn sort_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: SoA1<ValueSource>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let (keys, values) = ordering::sort_by_key_input_with_policy(
            policy,
            &self.source,
            &values.source,
            GpuOp::<Less>::new(),
        )?;
        Ok((SoA1 { source: keys }, SoA1 { source: values }))
    }
}

impl<KeySource, ValueSource, Less> SortByKeyInput<ValueSource, Less> for KeySource
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    ValueSource: ReadOnlyKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Less: BinaryPredicateOp<KeySource::Item>,
{
    type Runtime = KeySource::Runtime;
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA1<DeviceVec<KeySource::Runtime, ValueSource::Item>>,
    );

    fn sort_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: ValueSource,
        op: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<KeySource> as SortByKeyInput<SoA1<ValueSource>, Less>>::sort_by_key_input(
            SoAView1 { source: self },
            policy,
            SoA1 { source: values },
            op,
        )
    }
}

impl<KeySource, ValueSource, Less> SortByKeyInput<(ValueSource,), Less> for (KeySource,)
where
    KeySource: SortByKeyInput<ValueSource, Tuple1Less<Less>>,
{
    type Runtime = <KeySource as SortByKeyInput<ValueSource, Tuple1Less<Less>>>::Runtime;
    type Output = <KeySource as SortByKeyInput<ValueSource, Tuple1Less<Less>>>::Output;

    fn sort_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: (ValueSource,),
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <KeySource as SortByKeyInput<ValueSource, Tuple1Less<Less>>>::sort_by_key_input(
            self.0,
            policy,
            values.0,
            GpuOp::<Tuple1Less<Less>>::new(),
        )
    }
}

macro_rules! impl_sort_by_key_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<KeySource, $first, $( $rest ),+, Less> SortByKeyInput<$name<$first, $( $rest ),+>, Less>
            for SoAView1<KeySource>
        where
            Self: ReadOnlySoA<Item = (KeySource::Item,), Scalar = KeySource::Item>,
            $name<$first, $( $rest ),+>: SoA,
            KeySource: KernelColumn + KernelColumnAt<S0>,
            $first: ReadOnlyKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
            $(
                $rest: ReadOnlyKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
            )+
            KeySource::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            <$first as KernelColumn>::Expr: GpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
                <$rest as KernelColumn>::Expr: GpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Less: BinaryPredicateOp<KeySource::Item>,
        {
            type Runtime = KeySource::Runtime;
            type Output = (
                SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
                $name<
                    DeviceVec<KeySource::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<KeySource::Runtime, <$rest as KernelColumn>::Item> ),+
                >,
            );

            fn sort_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                values: $name<$first, $( $rest ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                SoA::validate(&values)?;
                let indices = primitive_range::indices_u32(policy, self.source.len())?;
                let (out_keys, sorted_indices) =
                    ordering::sort_by_key_input_with_policy(policy, &self.source, &indices, GpuOp::<Less>::new())?;
                let $first_field = device_expr_gather_with_policy(policy, &values.$first_field, &sorted_indices)?;
                $(
                    let $field = device_expr_gather_with_policy(policy, &values.$field, &sorted_indices)?;
                )+
                Ok((SoA1 { source: out_keys }, $name { $first_field, $( $field ),+ }))
            }
        }
    };
}

impl_sort_by_key_input!(SoA2<A, B> { left, right });
impl_sort_by_key_input!(SoA3<A, B, C> { first, second, third });

macro_rules! impl_sort_by_key_input_key_source {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<KeySource, $( $field_ty ),+, Less> SortByKeyInput<$name<$( $field_ty ),+>, Less>
            for KeySource
        where
            KeySource: KernelColumn + KernelColumnAt<S0>,
            SoAView1<KeySource>: SortByKeyInput<$name<$( $field_ty ),+>, Less>,
        {
            type Runtime =
                <SoAView1<KeySource> as SortByKeyInput<$name<$( $field_ty ),+>, Less>>::Runtime;
            type Output = <SoAView1<KeySource> as SortByKeyInput<$name<$( $field_ty ),+>, Less>>::Output;

            fn sort_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                values: $name<$( $field_ty ),+>,
                less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                <SoAView1<KeySource> as SortByKeyInput<$name<$( $field_ty ),+>, Less>>::sort_by_key_input(
                    SoAView1 { source: self },
                    policy,
                    values,
                    less,
                )
            }
        }
    };
}

impl_sort_by_key_input_key_source!(SoA2<A, B>);
impl_sort_by_key_input_key_source!(SoA3<A, B, C>);

macro_rules! impl_sort_by_key_view_values {
    ($view:ident -> $out:ident < $( $value:ident: $field:ident ),+ >) => {
        impl<KeySource, $( $value ),+, Less> SortByKeyInput<$view<$( $value ),+>, Less>
            for SoAView1<KeySource>
        where
            Self: ReadOnlySoA<Item = (KeySource::Item,), Scalar = KeySource::Item>,
            $view<$( $value ),+>: ReadOnlySoA,
            KeySource: KernelColumn + KernelColumnAt<S0>,
            $( $value: ReadOnlyKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>, )+
            KeySource::Item: CubePrimitive + CubeElement,
            $( <$value as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
            $( <$value as KernelColumn>::Expr: DeviceGpuExpr<<$value as KernelColumn>::Item>, )+
            $( <$value as KernelColumn>::Expr: GpuExpr<<$value as KernelColumn>::Item>, )+
            Less: BinaryPredicateOp<KeySource::Item>,
        {
            type Runtime = KeySource::Runtime;
            type Output = (
                SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
                $out<$( DeviceVec<KeySource::Runtime, <$value as KernelColumn>::Item> ),+>,
            );

            fn sort_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                values: $view<$( $value ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&values)?;
                let indices = primitive_range::indices_u32(policy, self.source.len())?;
                let (out_keys, sorted_indices) =
                    ordering::sort_by_key_input_with_policy(policy, &self.source, &indices, GpuOp::<Less>::new())?;
                $(
                    let $field = device_expr_gather_with_policy(policy, &values.$field, &sorted_indices)?;
                )+
                Ok((SoA1 { source: out_keys }, $out { $( $field ),+ }))
            }
        }

        impl<KeySource, $( $value ),+, Less> SortByKeyInput<$view<$( $value ),+>, Less>
            for KeySource
        where
            KeySource: KernelColumn + KernelColumnAt<S0>,
            SoAView1<KeySource>: SortByKeyInput<$view<$( $value ),+>, Less>,
        {
            type Runtime =
                <SoAView1<KeySource> as SortByKeyInput<$view<$( $value ),+>, Less>>::Runtime;
            type Output =
                <SoAView1<KeySource> as SortByKeyInput<$view<$( $value ),+>, Less>>::Output;

            fn sort_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                values: $view<$( $value ),+>,
                less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                <SoAView1<KeySource> as SortByKeyInput<$view<$( $value ),+>, Less>>::sort_by_key_input(
                    SoAView1 { source: self },
                    policy,
                    values,
                    less,
                )
            }
        }
    };
}

impl_sort_by_key_view_values!(SoAView2 -> SoA2<A: left, B: right>);
impl_sort_by_key_view_values!(SoAView3 -> SoA3<A: first, B: second, C: third>);

/// Key/value inputs accepted by [`merge_by_key`].
#[doc(hidden)]
pub trait MergeByKeyInput<LeftValues, RightKeys, RightValues, Less> {
    /// Runtime used by this input.
    type Runtime: Runtime;
    /// Output produced by key-value merge.
    type Output;

    /// Merges two sorted key-value ranges by key.
    fn merge_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        left_values: LeftValues,
        right_keys: RightKeys,
        right_values: RightValues,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;
}

impl<LeftKey, LeftValue, RightKey, RightValue, Less>
    MergeByKeyInput<SoAView1<LeftValue>, SoAView1<RightKey>, SoAView1<RightValue>, Less>
    for SoAView1<LeftKey>
where
    Self: ReadOnlySoA<Item = (LeftKey::Item,), Scalar = LeftKey::Item>,
    SoAView1<LeftValue>: ReadOnlySoA<Item = (LeftValue::Item,), Scalar = LeftValue::Item>,
    SoAView1<RightKey>: ReadOnlySoA<Item = (RightKey::Item,), Scalar = RightKey::Item>,
    SoAView1<RightValue>: ReadOnlySoA<Item = (RightValue::Item,), Scalar = RightValue::Item>,
    LeftKey: KernelColumn + KernelColumnAt<S0>,
    LeftValue: KernelColumn<Runtime = LeftKey::Runtime> + KernelColumnAt<S0>,
    RightKey: KernelColumn<Runtime = LeftKey::Runtime, Item = LeftKey::Item> + KernelColumnAt<S0>,
    RightValue:
        KernelColumn<Runtime = LeftKey::Runtime, Item = LeftValue::Item> + KernelColumnAt<S0>,
    LeftKey::Item: CubePrimitive + CubeElement,
    LeftValue::Item: CubePrimitive + CubeElement,
    LeftKey::Expr: DeviceGpuExpr<LeftKey::Item>,
    LeftValue::Expr: DeviceGpuExpr<LeftValue::Item>,
    RightKey::Expr: DeviceGpuExpr<RightKey::Item>,
    RightValue::Expr: DeviceGpuExpr<RightValue::Item>,
    Less: BinaryPredicateOp<LeftKey::Item>,
{
    type Runtime = LeftKey::Runtime;
    type Output = (
        SoA1<DeviceVec<LeftKey::Runtime, LeftKey::Item>>,
        SoA1<DeviceVec<LeftKey::Runtime, LeftValue::Item>>,
    );

    fn merge_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        left_values: SoAView1<LeftValue>,
        right_keys: SoAView1<RightKey>,
        right_values: SoAView1<RightValue>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&left_values)?;
        ReadOnlySoA::validate(&right_keys)?;
        ReadOnlySoA::validate(&right_values)?;
        ensure_same_len(self.source.len(), left_values.source.len())?;
        ensure_same_len(right_keys.source.len(), right_values.source.len())?;

        let (keys, control) = device_expr_merge_by_key_control_with_policy::<
            LeftKey,
            RightKey,
            Less,
        >(policy, &self.source, &right_keys.source)?;
        let values = device_expr_merge_by_key_values_with_control_with_policy(
            policy,
            &left_values.source,
            &right_values.source,
            &control,
        )?;
        Ok((SoA1 { source: keys }, SoA1 { source: values }))
    }
}

macro_rules! impl_merge_by_key_input {
    ($name:ident < $first_left:ident, $( $left:ident ),+ >,
     $right_name:ident < $first_right:ident, $( $right:ident ),+ >,
     $output:ident { $first_field:ident, $( $field:ident ),+ }) => {
        impl<LeftKey, RightKey, $first_left, $( $left ),+, $first_right, $( $right ),+, Less>
            MergeByKeyInput<
                $name<$first_left, $( $left ),+>,
                SoAView1<RightKey>,
                $right_name<$first_right, $( $right ),+>,
                Less,
            > for SoAView1<LeftKey>
        where
            Self: ReadOnlySoA<Item = (LeftKey::Item,), Scalar = LeftKey::Item>,
            SoAView1<RightKey>: ReadOnlySoA<Item = (RightKey::Item,), Scalar = RightKey::Item>,
            LeftKey: KernelColumn + KernelColumnAt<S0>,
            RightKey: KernelColumn<Runtime = LeftKey::Runtime, Item = LeftKey::Item> + KernelColumnAt<S0>,
            $first_left: KernelColumn<Runtime = LeftKey::Runtime> + KernelColumnAt<S0>,
            $first_right: KernelColumn<Runtime = LeftKey::Runtime, Item = <$first_left as KernelColumn>::Item>
                + KernelColumnAt<S0>,
            $(
                $left: KernelColumn<Runtime = LeftKey::Runtime> + KernelColumnAt<S0>,
                $right: KernelColumn<Runtime = LeftKey::Runtime, Item = <$left as KernelColumn>::Item>
                    + KernelColumnAt<S0>,
            )+
            LeftKey::Item: CubePrimitive + CubeElement,
            <$first_left as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$left as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            LeftKey::Expr: DeviceGpuExpr<LeftKey::Item>,
            RightKey::Expr: DeviceGpuExpr<RightKey::Item>,
            <$first_left as KernelColumn>::Expr: DeviceGpuExpr<<$first_left as KernelColumn>::Item>,
            <$first_right as KernelColumn>::Expr: DeviceGpuExpr<<$first_right as KernelColumn>::Item>,
            $(
                <$left as KernelColumn>::Expr: DeviceGpuExpr<<$left as KernelColumn>::Item>,
                <$right as KernelColumn>::Expr: DeviceGpuExpr<<$right as KernelColumn>::Item>,
            )+
            Less: BinaryPredicateOp<LeftKey::Item>,
        {
            type Runtime = LeftKey::Runtime;
            type Output = (
                SoA1<DeviceVec<LeftKey::Runtime, LeftKey::Item>>,
                $output<
                    DeviceVec<LeftKey::Runtime, <$first_left as KernelColumn>::Item>,
                    $( DeviceVec<LeftKey::Runtime, <$left as KernelColumn>::Item> ),+
                >,
            );

            fn merge_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                left_values: $name<$first_left, $( $left ),+>,
                right_keys: SoAView1<RightKey>,
                right_values: $right_name<$first_right, $( $right ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&right_keys)?;
                left_values.$first_field.validate()?;
                right_values.$first_field.validate()?;
                ensure_same_len(self.source.len(), left_values.$first_field.len())?;
                ensure_same_len(right_keys.source.len(), right_values.$first_field.len())?;
                $(
                    left_values.$field.validate()?;
                    right_values.$field.validate()?;
                    ensure_same_len(self.source.len(), left_values.$field.len())?;
                    ensure_same_len(right_keys.source.len(), right_values.$field.len())?;
                )+

                // Compute merge-path control once and apply the same source
                // side/index stream to every value column.
                let (keys, control) =
                    device_expr_merge_by_key_control_with_policy::<LeftKey, RightKey, Less>(
                        policy,
                        &self.source,
                        &right_keys.source,
                    )?;
                let $first_field =
                    device_expr_merge_by_key_values_with_control_with_policy(policy, &left_values.$first_field, &right_values.$first_field, &control)?;
                $(
                    let $field =
                        device_expr_merge_by_key_values_with_control_with_policy(policy, &left_values.$field, &right_values.$field, &control)?;
                )+

                Ok((SoA1 { source: keys }, $output { $first_field, $( $field ),+ }))
            }
        }
    };
}

impl_merge_by_key_input!(SoAView2<A, B>, SoAView2<C, D>, SoA2 { left, right });
impl_merge_by_key_input!(SoAView3<A, B, C>, SoAView3<D, E, F>, SoA3 { first, second, third });
impl_merge_by_key_input!(SoA2<A, B>, SoA2<C, D>, SoA2 { left, right });
impl_merge_by_key_input!(SoA3<A, B, C>, SoA3<D, E, F>, SoA3 { first, second, third });

impl<LeftKey, LeftValue, RightKey, RightValue, Less>
    MergeByKeyInput<LeftValue, RightKey, RightValue, Less> for LeftKey
where
    LeftKey: KernelColumn + KernelColumnAt<S0>,
    LeftValue: KernelColumn<Runtime = LeftKey::Runtime> + KernelColumnAt<S0>,
    RightKey: KernelColumn<Runtime = LeftKey::Runtime, Item = LeftKey::Item> + KernelColumnAt<S0>,
    RightValue:
        KernelColumn<Runtime = LeftKey::Runtime, Item = LeftValue::Item> + KernelColumnAt<S0>,
    LeftKey::Item: CubePrimitive + CubeElement,
    LeftValue::Item: CubePrimitive + CubeElement,
    LeftKey::Expr: DeviceGpuExpr<LeftKey::Item>,
    LeftValue::Expr: DeviceGpuExpr<LeftValue::Item>,
    RightKey::Expr: DeviceGpuExpr<RightKey::Item>,
    RightValue::Expr: DeviceGpuExpr<RightValue::Item>,
    Less: BinaryPredicateOp<LeftKey::Item>,
{
    type Runtime = LeftKey::Runtime;
    type Output = (
        SoA1<DeviceVec<LeftKey::Runtime, LeftKey::Item>>,
        SoA1<DeviceVec<LeftKey::Runtime, LeftValue::Item>>,
    );

    fn merge_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        left_values: LeftValue,
        right_keys: RightKey,
        right_values: RightValue,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<LeftKey> as MergeByKeyInput<
            SoAView1<LeftValue>,
            SoAView1<RightKey>,
            SoAView1<RightValue>,
            Less,
        >>::merge_by_key_input(
            SoAView1 { source: self },
            policy,
            SoAView1 {
                source: left_values,
            },
            SoAView1 { source: right_keys },
            SoAView1 {
                source: right_values,
            },
            less,
        )
    }
}

impl<LeftKey, LeftValue, RightKey, RightValue, Less>
    MergeByKeyInput<(LeftValue,), (RightKey,), (RightValue,), Less> for (LeftKey,)
where
    LeftKey: MergeByKeyInput<LeftValue, RightKey, RightValue, Tuple1Less<Less>>,
{
    type Runtime =
        <LeftKey as MergeByKeyInput<LeftValue, RightKey, RightValue, Tuple1Less<Less>>>::Runtime;
    type Output =
        <LeftKey as MergeByKeyInput<LeftValue, RightKey, RightValue, Tuple1Less<Less>>>::Output;

    fn merge_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        left_values: (LeftValue,),
        right_keys: (RightKey,),
        right_values: (RightValue,),
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <LeftKey as MergeByKeyInput<
            LeftValue,
            RightKey,
            RightValue,
            Tuple1Less<Less>,
        >>::merge_by_key_input(
            self.0,
            policy,
            left_values.0,
            right_keys.0,
            right_values.0,
            GpuOp::<Tuple1Less<Less>>::new(),
        )
    }
}

/// Sorts read-only key-value pairs by key and returns owned SoA outputs.
pub fn sort_by_key<R, Keys, Values, Less>(
    policy: &CubePolicy<R>,
    keys: Keys,
    values: Values,
    _less: Less,
) -> Result<<<Keys as SortByKeyInput<Values, Less>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Keys: SortByKeyInput<Values, Less, Runtime = R>,
    <Keys as SortByKeyInput<Values, Less>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        keys.sort_by_key_input_with_policy(policy, values, GpuOp::<Less>::new())?,
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
    <<LeftKeys as MergeByKeyInput<LeftValues, RightKeys, RightValues, Less>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    R: Runtime,
    LeftKeys: MergeByKeyInput<LeftValues, RightKeys, RightValues, Less, Runtime = R>,
    <LeftKeys as MergeByKeyInput<LeftValues, RightKeys, RightValues, Less>>::Output:
        MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        left_keys.merge_by_key_input(
            policy,
            left_values,
            right_keys,
            right_values,
            GpuOp::<Less>::new(),
        )?,
    )
}
