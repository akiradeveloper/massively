use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, OwnedKernelColumn, S0, SoA, SoA1, SoA2, SoA3,
        SoA4, SoA5, SoA6, SoA7, SoA8, SoA9, SoA10, SoA11, SoA12, SoVA, SoVA1,
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::{BinaryPredicateOp, GpuOp},
    primitives::{ordering, range as primitive_range},
};
use cubecl::prelude::*;

const BLOCK_ORDERING_SIZE: u32 = 256;

fn materialize_one<Source>(
    input: SoA1<Source>,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    input.source.validate()?;
    super::device_expr_collect(&input.source)
}

fn materialize_sova_one<Source>(
    input: SoVA1<Source>,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    SoVA1<Source>: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    SoVA::validate(&input)?;
    super::device_expr_collect(&input.source)
}

/// Pair input accepted by sorted binary ordering algorithms.
#[doc(hidden)]
pub trait PairOrderingInput<Other, Less> {
    /// Output produced by this algorithm.
    type Output;

    /// Merges two sorted inputs.
    fn merge_input(self, other: Other, less: GpuOp<Less>) -> Result<Self::Output, Error>;
    /// Computes the sorted set union.
    fn set_union_input(self, other: Other, less: GpuOp<Less>) -> Result<Self::Output, Error>;
    /// Computes the sorted set intersection.
    fn set_intersection_input(self, other: Other, less: GpuOp<Less>)
    -> Result<Self::Output, Error>;
    /// Computes the sorted set difference.
    fn set_difference_input(self, other: Other, less: GpuOp<Less>) -> Result<Self::Output, Error>;
    /// Computes the sorted set symmetric difference.
    fn set_symmetric_difference_input(
        self,
        other: Other,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;
}

impl<Left, Right, Less> PairOrderingInput<SoVA1<Right>, Less> for SoVA1<Left>
where
    Self: SoVA<Item = Left::Item, Scalar = Left::Item>,
    SoVA1<Right>: SoVA<Item = Right::Item, Scalar = Right::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<Left::Item>,
{
    type Output = SoA1<DeviceVec<Left::Runtime, Left::Item>>;

    fn merge_input(self, other: SoVA1<Right>, _less: GpuOp<Less>) -> Result<Self::Output, Error> {
        let left = materialize_sova_one(self)?;
        let right = materialize_sova_one(other)?;
        Ok(SoA1 {
            source: ordering::merge(&left, &right, GpuOp::<Less>::new())?,
        })
    }

    fn set_union_input(
        self,
        other: SoVA1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_sova_one(self)?;
        let right = materialize_sova_one(other)?;
        Ok(SoA1 {
            source: ordering::set_union(&left, &right, GpuOp::<Less>::new())?,
        })
    }

    fn set_intersection_input(
        self,
        other: SoVA1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_sova_one(self)?;
        let right = materialize_sova_one(other)?;
        Ok(SoA1 {
            source: ordering::set_intersection(&left, &right, GpuOp::<Less>::new())?,
        })
    }

    fn set_difference_input(
        self,
        other: SoVA1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_sova_one(self)?;
        let right = materialize_sova_one(other)?;
        Ok(SoA1 {
            source: ordering::set_difference(&left, &right, GpuOp::<Less>::new())?,
        })
    }

    fn set_symmetric_difference_input(
        self,
        other: SoVA1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_sova_one(self)?;
        let right = materialize_sova_one(other)?;
        Ok(SoA1 {
            source: ordering::set_symmetric_difference(&left, &right, GpuOp::<Less>::new())?,
        })
    }
}

impl<Left, Right, Less> PairOrderingInput<Right, Less> for Left
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<Left::Item>,
{
    type Output = SoA1<DeviceVec<Left::Runtime, Left::Item>>;

    fn merge_input(self, other: Right, less: GpuOp<Less>) -> Result<Self::Output, Error> {
        <SoVA1<Left> as PairOrderingInput<SoVA1<Right>, Less>>::merge_input(
            SoVA1 { source: self },
            SoVA1 { source: other },
            less,
        )
    }

    fn set_union_input(self, other: Right, less: GpuOp<Less>) -> Result<Self::Output, Error> {
        <SoVA1<Left> as PairOrderingInput<SoVA1<Right>, Less>>::set_union_input(
            SoVA1 { source: self },
            SoVA1 { source: other },
            less,
        )
    }

    fn set_intersection_input(
        self,
        other: Right,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoVA1<Left> as PairOrderingInput<SoVA1<Right>, Less>>::set_intersection_input(
            SoVA1 { source: self },
            SoVA1 { source: other },
            less,
        )
    }

    fn set_difference_input(self, other: Right, less: GpuOp<Less>) -> Result<Self::Output, Error> {
        <SoVA1<Left> as PairOrderingInput<SoVA1<Right>, Less>>::set_difference_input(
            SoVA1 { source: self },
            SoVA1 { source: other },
            less,
        )
    }

    fn set_symmetric_difference_input(
        self,
        other: Right,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoVA1<Left> as PairOrderingInput<SoVA1<Right>, Less>>::set_symmetric_difference_input(
            SoVA1 { source: self },
            SoVA1 { source: other },
            less,
        )
    }
}

/// Reverses a device vector and returns new device storage.
fn reverse_device_vec<R, T>(input: &DeviceVec<R, T>) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    let num_blocks = input.len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = input.policy.client();
    let output_handle = client.empty(input.len * std::mem::size_of::<T>());

    if input.len != 0 {
        unsafe {
            reverse_kernel::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                ArrayArg::from_raw_parts::<T>(&input.handle, input.len, 1),
                ArrayArg::from_raw_parts::<T>(&output_handle, input.len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(DeviceVec::from_handle(
        input.policy.clone(),
        output_handle,
        input.len,
    ))
}

/// Input accepted by [`reverse`].
#[doc(hidden)]
pub trait ReverseInput {
    /// Output produced by reversing this input.
    type Output;

    /// Reverses this input.
    fn reverse_input(self) -> Result<Self::Output, Error>;
}

impl<Source> ReverseInput for SoA1<Source>
where
    Self: SoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn reverse_input(self) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let input = super::device_expr_collect(&self.source)?;
        Ok(SoA1 {
            source: reverse_device_vec(&input)?,
        })
    }
}

impl<Source> ReverseInput for Source
where
    Source: OwnedKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn reverse_input(self) -> Result<Self::Output, Error> {
        <SoA1<Source> as ReverseInput>::reverse_input(SoA1 { source: self })
    }
}

/// Reverses owned SoA input and returns new device storage.
pub fn reverse<Input>(input: Input) -> Result<<Input as ReverseInput>::Output, Error>
where
    Input: ReverseInput,
{
    input.reverse_input()
}

/// Input accepted by [`sort`].
#[doc(hidden)]
pub trait SortInput<Less> {
    /// Output produced by sorting this input.
    type Output;

    /// Sorts this input.
    fn sort_input(self, less: GpuOp<Less>) -> Result<Self::Output, Error>;
}

/// Key/value input accepted by [`sort_by_key`].
#[doc(hidden)]
pub trait SortByKeyInput<Values, Less> {
    /// Output produced by key-value sorting.
    type Output;

    /// Sorts key-value pairs by key.
    fn sort_by_key_input(self, values: Values, less: GpuOp<Less>) -> Result<Self::Output, Error>;
}

impl<KeySource, ValueSource, Less> SortByKeyInput<SoA1<ValueSource>, Less> for SoVA1<KeySource>
where
    Self: SoVA<Item = KeySource::Item, Scalar = KeySource::Item>,
    SoA1<ValueSource>: SoA<Item = ValueSource::Item, Scalar = ValueSource::Item>,
    KeySource: KernelColumn + KernelColumnAt<S0>,
    ValueSource: OwnedKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Less: BinaryPredicateOp<KeySource::Item>,
{
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA1<DeviceVec<KeySource::Runtime, ValueSource::Item>>,
    );

    fn sort_by_key_input(
        self,
        values: SoA1<ValueSource>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let keys = materialize_sova_one(self)?;
        let values = materialize_one(values)?;
        let (keys, values) = ordering::sort_by_key(&keys, &values, GpuOp::<Less>::new())?;
        Ok((SoA1 { source: keys }, SoA1 { source: values }))
    }
}

impl<KeySource, ValueSource, Less> SortByKeyInput<ValueSource, Less> for KeySource
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    ValueSource: OwnedKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Less: BinaryPredicateOp<KeySource::Item>,
{
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA1<DeviceVec<KeySource::Runtime, ValueSource::Item>>,
    );

    fn sort_by_key_input(
        self,
        values: ValueSource,
        op: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoVA1<KeySource> as SortByKeyInput<SoA1<ValueSource>, Less>>::sort_by_key_input(
            SoVA1 { source: self },
            SoA1 { source: values },
            op,
        )
    }
}

macro_rules! impl_sort_by_key_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<KeySource, $first, $( $rest ),+, Less> SortByKeyInput<$name<$first, $( $rest ),+>, Less>
            for SoVA1<KeySource>
        where
            Self: SoVA<Item = KeySource::Item, Scalar = KeySource::Item>,
            $name<$first, $( $rest ),+>: SoA,
            KeySource: KernelColumn + KernelColumnAt<S0>,
            $first: OwnedKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
            $(
                $rest: OwnedKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
            )+
            KeySource::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Less: BinaryPredicateOp<KeySource::Item>,
        {
            type Output = (
                SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
                $name<
                    DeviceVec<KeySource::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<KeySource::Runtime, <$rest as KernelColumn>::Item> ),+
                >,
            );

            fn sort_by_key_input(
                self,
                values: $name<$first, $( $rest ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                SoVA::validate(&self)?;
                SoA::validate(&values)?;
                let keys = super::device_expr_collect(&self.source)?;
                let indices = primitive_range::indices_u32(keys.policy(), keys.len())?;
                let (out_keys, sorted_indices) =
                    ordering::sort_by_key(&keys, &indices, GpuOp::<Less>::new())?;
                let $first_field = super::device_expr_collect(&values.$first_field)?;
                let $first_field = primitive_range::gather_device(&$first_field, &sorted_indices)?;
                $(
                    let $field = super::device_expr_collect(&values.$field)?;
                    let $field = primitive_range::gather_device(&$field, &sorted_indices)?;
                )+
                Ok((SoA1 { source: out_keys }, $name { $first_field, $( $field ),+ }))
            }
        }
    };
}

impl_sort_by_key_input!(SoA2<A, B> { left, right });
impl_sort_by_key_input!(SoA3<A, B, C> { first, second, third });
impl_sort_by_key_input!(SoA4<A, B, C, D> { a, b, c, d });
impl_sort_by_key_input!(SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_sort_by_key_input!(SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_sort_by_key_input!(SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_sort_by_key_input!(SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_sort_by_key_input!(SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_sort_by_key_input!(SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_sort_by_key_input!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_sort_by_key_input!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_sort_by_key_input_key_source {
    ($name:ident < $( $field_ty:ident ),+ >) => {
        impl<KeySource, $( $field_ty ),+, Less> SortByKeyInput<$name<$( $field_ty ),+>, Less>
            for KeySource
        where
            KeySource: KernelColumn + KernelColumnAt<S0>,
            SoVA1<KeySource>: SortByKeyInput<$name<$( $field_ty ),+>, Less>,
        {
            type Output = <SoVA1<KeySource> as SortByKeyInput<$name<$( $field_ty ),+>, Less>>::Output;

            fn sort_by_key_input(
                self,
                values: $name<$( $field_ty ),+>,
                less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                <SoVA1<KeySource> as SortByKeyInput<$name<$( $field_ty ),+>, Less>>::sort_by_key_input(
                    SoVA1 { source: self },
                    values,
                    less,
                )
            }
        }
    };
}

impl_sort_by_key_input_key_source!(SoA2<A, B>);
impl_sort_by_key_input_key_source!(SoA3<A, B, C>);
impl_sort_by_key_input_key_source!(SoA4<A, B, C, D>);
impl_sort_by_key_input_key_source!(SoA5<A, B, C, D, E>);
impl_sort_by_key_input_key_source!(SoA6<A, B, C, D, E, F>);
impl_sort_by_key_input_key_source!(SoA7<A, B, C, D, E, F, G>);
impl_sort_by_key_input_key_source!(SoA8<A, B, C, D, E, F, G, H>);
impl_sort_by_key_input_key_source!(SoA9<A, B, C, D, E, F, G, H, I>);
impl_sort_by_key_input_key_source!(SoA10<A, B, C, D, E, F, G, H, I, J>);
impl_sort_by_key_input_key_source!(SoA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_sort_by_key_input_key_source!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>);

/// Key/value inputs accepted by [`merge_by_key`].
#[doc(hidden)]
pub trait MergeByKeyInput<LeftValues, RightKeys, RightValues, Less> {
    /// Output produced by key-value merge.
    type Output;

    /// Merges two sorted key-value ranges by key.
    fn merge_by_key_input(
        self,
        left_values: LeftValues,
        right_keys: RightKeys,
        right_values: RightValues,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;
}

impl<LeftKey, LeftValue, RightKey, RightValue, Less>
    MergeByKeyInput<SoVA1<LeftValue>, SoVA1<RightKey>, SoVA1<RightValue>, Less> for SoVA1<LeftKey>
where
    Self: SoVA<Item = LeftKey::Item, Scalar = LeftKey::Item>,
    SoVA1<LeftValue>: SoVA<Item = LeftValue::Item, Scalar = LeftValue::Item>,
    SoVA1<RightKey>: SoVA<Item = RightKey::Item, Scalar = RightKey::Item>,
    SoVA1<RightValue>: SoVA<Item = RightValue::Item, Scalar = RightValue::Item>,
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
    type Output = (
        SoA1<DeviceVec<LeftKey::Runtime, LeftKey::Item>>,
        SoA1<DeviceVec<LeftKey::Runtime, LeftValue::Item>>,
    );

    fn merge_by_key_input(
        self,
        left_values: SoVA1<LeftValue>,
        right_keys: SoVA1<RightKey>,
        right_values: SoVA1<RightValue>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left_keys = materialize_sova_one(self)?;
        let left_values = materialize_sova_one(left_values)?;
        let right_keys = materialize_sova_one(right_keys)?;
        let right_values = materialize_sova_one(right_values)?;
        let (keys, values) = ordering::merge_by_key(
            &left_keys,
            &left_values,
            &right_keys,
            &right_values,
            GpuOp::<Less>::new(),
        )?;
        Ok((SoA1 { source: keys }, SoA1 { source: values }))
    }
}

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
    type Output = (
        SoA1<DeviceVec<LeftKey::Runtime, LeftKey::Item>>,
        SoA1<DeviceVec<LeftKey::Runtime, LeftValue::Item>>,
    );

    fn merge_by_key_input(
        self,
        left_values: LeftValue,
        right_keys: RightKey,
        right_values: RightValue,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoVA1<LeftKey> as MergeByKeyInput<
            SoVA1<LeftValue>,
            SoVA1<RightKey>,
            SoVA1<RightValue>,
            Less,
        >>::merge_by_key_input(
            SoVA1 { source: self },
            SoVA1 {
                source: left_values,
            },
            SoVA1 { source: right_keys },
            SoVA1 {
                source: right_values,
            },
            less,
        )
    }
}

impl<Source, Less> SortInput<Less> for SoA1<Source>
where
    Self: SoA<Item = Source::Item, Scalar = Source::Item>,
    Source: OwnedKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn sort_input(self, _less: GpuOp<Less>) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let source = super::device_expr_collect(&self.source)?;
        Ok(SoA1 {
            source: ordering::sort(&source, GpuOp::<Less>::new())?,
        })
    }
}

impl<Source, Less> SortInput<Less> for Source
where
    Source: OwnedKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn sort_input(self, less: GpuOp<Less>) -> Result<Self::Output, Error> {
        <SoA1<Source> as SortInput<Less>>::sort_input(SoA1 { source: self }, less)
    }
}

impl<Left, Right, Less> SortInput<Less> for SoA2<Left, Right>
where
    Self: SoA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: OwnedKernelColumn + KernelColumnAt<S0>,
    Right: OwnedKernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<(Left::Item, Right::Item)>,
{
    type Output = SoA2<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

    fn sort_input(self, _less: GpuOp<Less>) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let left = super::device_expr_collect(&self.left)?;
        let right = super::device_expr_collect(&self.right)?;
        let (first, second) = ordering::sort_tuple2(&left, &right, GpuOp::<Less>::new())?;
        Ok(SoA2 {
            left: first,
            right: second,
        })
    }
}

impl<First, Second, Third, Less> SortInput<Less> for SoA3<First, Second, Third>
where
    Self: SoA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
    First: OwnedKernelColumn + KernelColumnAt<S0>,
    Second: OwnedKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: OwnedKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    Less: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
{
    type Output = SoA3<
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    >;

    fn sort_input(self, _less: GpuOp<Less>) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let first = super::device_expr_collect(&self.first)?;
        let second = super::device_expr_collect(&self.second)?;
        let third = super::device_expr_collect(&self.third)?;
        let (first, second, third) =
            ordering::sort_tuple3(&first, &second, &third, GpuOp::<Less>::new())?;
        Ok(SoA3 {
            first,
            second,
            third,
        })
    }
}

macro_rules! impl_sort_input {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ },
        $sort_fn:ident
    ) => {
        impl<$first, $( $rest ),+, Less> SortInput<Less> for $name<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: OwnedKernelColumn + KernelColumnAt<S0>,
            $(
                $rest: OwnedKernelColumn<
                        Runtime = <$first as KernelColumn>::Runtime,
                        Item = <$first as KernelColumn>::Item,
                    > + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Less: BinaryPredicateOp<(
                impl_sort_input!(@item_ty $first),
                $( impl_sort_input!(@item_ty $rest) ),+
            )>,
        {
            type Output = $name<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$first as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn sort_input(self, _less: GpuOp<Less>) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let ($first_field, $( $field ),+) =
                    ordering::$sort_fn(&$first_field, $( &$field, )+ GpuOp::<Less>::new())?;
                Ok($name { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_sort_input!(SoA4<A, B, C, D> { a, b, c, d }, sort_tuple4);
impl_sort_input!(SoA5<A, B, C, D, E> { a, b, c, d, e }, sort_tuple5);
impl_sort_input!(SoA6<A, B, C, D, E, F> { a, b, c, d, e, f }, sort_tuple6);
impl_sort_input!(SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g }, sort_tuple7);
impl_sort_input!(SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h }, sort_tuple8);
impl_sort_input!(SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i }, sort_tuple9);
impl_sort_input!(SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j }, sort_tuple10);
impl_sort_input!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k }, sort_tuple11);
impl_sort_input!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l }, sort_tuple12);

/// Sorts owned SoA input.
///
/// This is a consuming algorithm: pass a [`DeviceVec`](crate::DeviceVec) for one
/// column or [`zip`](crate::zip) for multiple owned columns. Borrowed views from
/// [`vzip`](crate::vzip) are intentionally not accepted.
pub fn sort<Input, Less>(
    input: Input,
    _less: Less,
) -> Result<<Input as SortInput<Less>>::Output, Error>
where
    Input: SortInput<Less>,
{
    input.sort_input(GpuOp::<Less>::new())
}

/// Merges two sorted read-only inputs into owned device storage.
///
/// This is a borrowing algorithm. Both inputs are read, and the merged output is
/// newly materialized.
pub fn merge<Left, Right, Less>(
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<Left as PairOrderingInput<Right, Less>>::Output, Error>
where
    Left: PairOrderingInput<Right, Less>,
{
    left.merge_input(right, GpuOp::<Less>::new())
}

/// Sorts key-value pairs by key and returns owned SoA outputs.
///
/// This is a mixed algorithm: keys may be read-only, while values are owned SoA
/// payload storage consumed by the operation.
pub fn sort_by_key<Keys, Values, Less>(
    keys: Keys,
    values: Values,
    _less: Less,
) -> Result<<Keys as SortByKeyInput<Values, Less>>::Output, Error>
where
    Keys: SortByKeyInput<Values, Less>,
{
    keys.sort_by_key_input(values, GpuOp::<Less>::new())
}

/// Stable sort. The current device implementation is stable.
pub fn stable_sort<Input, Less>(
    input: Input,
    less: Less,
) -> Result<<Input as SortInput<Less>>::Output, Error>
where
    Input: SortInput<Less>,
{
    sort(input, less)
}

/// Stable key-value sort. The current device implementation is stable.
pub fn stable_sort_by_key<Keys, Values, Less>(
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<<Keys as SortByKeyInput<Values, Less>>::Output, Error>
where
    Keys: SortByKeyInput<Values, Less>,
{
    sort_by_key(keys, values, less)
}

/// Merges two sorted key-value ranges by key.
pub fn merge_by_key<LeftKeys, LeftValues, RightKeys, RightValues, Less>(
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    _less: Less,
) -> Result<<LeftKeys as MergeByKeyInput<LeftValues, RightKeys, RightValues, Less>>::Output, Error>
where
    LeftKeys: MergeByKeyInput<LeftValues, RightKeys, RightValues, Less>,
{
    left_keys.merge_by_key_input(left_values, right_keys, right_values, GpuOp::<Less>::new())
}

/// Computes the sorted set union of two sorted device vectors.
pub fn set_union<Left, Right, Less>(
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<Left as PairOrderingInput<Right, Less>>::Output, Error>
where
    Left: PairOrderingInput<Right, Less>,
{
    left.set_union_input(right, GpuOp::<Less>::new())
}

/// Computes the sorted set intersection of two sorted device vectors.
pub fn set_intersection<Left, Right, Less>(
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<Left as PairOrderingInput<Right, Less>>::Output, Error>
where
    Left: PairOrderingInput<Right, Less>,
{
    left.set_intersection_input(right, GpuOp::<Less>::new())
}

/// Computes the sorted set difference `left - right`.
pub fn set_difference<Left, Right, Less>(
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<Left as PairOrderingInput<Right, Less>>::Output, Error>
where
    Left: PairOrderingInput<Right, Less>,
{
    left.set_difference_input(right, GpuOp::<Less>::new())
}

/// Computes the sorted set symmetric difference.
pub fn set_symmetric_difference<Left, Right, Less>(
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<Left as PairOrderingInput<Right, Less>>::Output, Error>
where
    Left: PairOrderingInput<Right, Less>,
{
    left.set_symmetric_difference_input(right, GpuOp::<Less>::new())
}
