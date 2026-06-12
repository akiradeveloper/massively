use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, OwnedKernelColumn, S0, SoA, SoA1, SoA2, SoA3,
        SoA4, SoA5, SoA6, SoA7, SoA8, SoA9, SoA10, SoA11, SoA12,
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::{BinaryPredicateOp, GpuOp, PredicateOp},
    primitives::select,
};
use cubecl::prelude::*;

const BLOCK_MUTATION_SIZE: u32 = 256;

fn mutation_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_MUTATION_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

#[doc(hidden)]
pub trait ReplaceIfInput<Pred> {
    type Item;
    type Output;

    fn replace_if_input(
        self,
        replacement: Self::Item,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, Pred> ReplaceIfInput<Pred> for SoA1<Source>
where
    Self: SoA<Item = Source::Item, Scalar = Source::Item>,
    Source: OwnedKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Item = Source::Item;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn replace_if_input(
        self,
        replacement: Self::Item,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let input = super::device_expr_collect(&self.source)?;
        Ok(SoA1 {
            source: replace_if_device_vec(&input, replacement, GpuOp::<Pred>::new())?,
        })
    }
}

impl<Source, Pred> ReplaceIfInput<Pred> for Source
where
    Source: OwnedKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Item = Source::Item;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn replace_if_input(
        self,
        replacement: Self::Item,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoA1<Source> as ReplaceIfInput<Pred>>::replace_if_input(
            SoA1 { source: self },
            replacement,
            pred,
        )
    }
}

#[doc(hidden)]
pub trait UniqueInput<Pred> {
    type Output;

    fn unique_input(self, pred: GpuOp<Pred>) -> Result<Self::Output, Error>;
}

#[doc(hidden)]
pub trait UniqueByKeyInput<Values, Eq> {
    type Output;

    fn unique_by_key_input(self, values: Values, eq: GpuOp<Eq>) -> Result<Self::Output, Error>;
}

impl<KeySource, ValueSource, Eq> UniqueByKeyInput<SoA1<ValueSource>, Eq> for SoA1<KeySource>
where
    Self: SoA<Item = KeySource::Item, Scalar = KeySource::Item>,
    SoA1<ValueSource>: SoA<Item = ValueSource::Item, Scalar = ValueSource::Item>,
    KeySource: OwnedKernelColumn + KernelColumnAt<S0>,
    ValueSource: OwnedKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Eq: BinaryPredicateOp<KeySource::Item>,
{
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA1<DeviceVec<KeySource::Runtime, ValueSource::Item>>,
    );

    fn unique_by_key_input(
        self,
        values: SoA1<ValueSource>,
        _eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        SoA::validate(&values)?;
        let keys = super::device_expr_collect(&self.source)?;
        let values = super::device_expr_collect(&values.source)?;
        let (keys, values) = unique_by_key_device_vec(&keys, &values, GpuOp::<Eq>::new())?;
        Ok((SoA1 { source: keys }, SoA1 { source: values }))
    }
}

impl<KeySource, R, T, Eq> UniqueByKeyInput<DeviceVec<R, T>, Eq> for KeySource
where
    KeySource: OwnedKernelColumn + KernelColumnAt<S0>,
    SoA1<KeySource>: UniqueByKeyInput<SoA1<DeviceVec<R, T>>, Eq>,
    R: Runtime,
    T: CubePrimitive + CubeElement,
    KeySource::Item: CubePrimitive + CubeElement,
{
    type Output = <SoA1<KeySource> as UniqueByKeyInput<SoA1<DeviceVec<R, T>>, Eq>>::Output;

    fn unique_by_key_input(
        self,
        values: DeviceVec<R, T>,
        eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        <SoA1<KeySource> as UniqueByKeyInput<SoA1<DeviceVec<R, T>>, Eq>>::unique_by_key_input(
            SoA1 { source: self },
            SoA1 { source: values },
            eq,
        )
    }
}

macro_rules! impl_unique_by_key_key_forward {
    ($name:ident < $first:ident, $( $rest:ident ),+ >) => {
        impl<KeySource, $first, $( $rest ),+, Eq> UniqueByKeyInput<$name<$first, $( $rest ),+>, Eq>
            for KeySource
        where
            KeySource: OwnedKernelColumn + KernelColumnAt<S0>,
            SoA1<KeySource>: UniqueByKeyInput<$name<$first, $( $rest ),+>, Eq>,
            KeySource::Item: CubePrimitive + CubeElement,
        {
            type Output = <SoA1<KeySource> as UniqueByKeyInput<$name<$first, $( $rest ),+>, Eq>>::Output;

            fn unique_by_key_input(
                self,
                values: $name<$first, $( $rest ),+>,
                eq: GpuOp<Eq>,
            ) -> Result<Self::Output, Error> {
                <SoA1<KeySource> as UniqueByKeyInput<$name<$first, $( $rest ),+>, Eq>>::unique_by_key_input(
                    SoA1 { source: self },
            values,
            eq,
        )
    }
        }
    };
}

impl_unique_by_key_key_forward!(SoA2<A, B>);
impl_unique_by_key_key_forward!(SoA3<A, B, C>);
impl_unique_by_key_key_forward!(SoA4<A, B, C, D>);
impl_unique_by_key_key_forward!(SoA5<A, B, C, D, E>);
impl_unique_by_key_key_forward!(SoA6<A, B, C, D, E, F>);
impl_unique_by_key_key_forward!(SoA7<A, B, C, D, E, F, G>);
impl_unique_by_key_key_forward!(SoA8<A, B, C, D, E, F, G, H>);
impl_unique_by_key_key_forward!(SoA9<A, B, C, D, E, F, G, H, I>);
impl_unique_by_key_key_forward!(SoA10<A, B, C, D, E, F, G, H, I, J>);
impl_unique_by_key_key_forward!(SoA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_unique_by_key_key_forward!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>);

macro_rules! impl_unique_by_key_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<KeySource, $first, $( $rest ),+, Eq> UniqueByKeyInput<$name<$first, $( $rest ),+>, Eq>
            for SoA1<KeySource>
        where
            Self: SoA<Item = KeySource::Item, Scalar = KeySource::Item>,
            $name<$first, $( $rest ),+>: SoA,
            KeySource: OwnedKernelColumn + KernelColumnAt<S0>,
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
            Eq: BinaryPredicateOp<KeySource::Item>,
        {
            type Output = (
                SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
                $name<
                    DeviceVec<KeySource::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<KeySource::Runtime, <$rest as KernelColumn>::Item> ),+
                >,
            );

            fn unique_by_key_input(
                self,
                values: $name<$first, $( $rest ),+>,
                _eq: GpuOp<Eq>,
            ) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                SoA::validate(&values)?;
                let keys = super::device_expr_collect(&self.source)?;
                let $first_field = super::device_expr_collect(&values.$first_field)?;
                let key_handles = unique_by_key_handles::<_, _, Eq>(&keys)?;
                let count = select::selected_count(keys.policy(), &key_handles)?;
                let value_handles = select::handles_for_value(&key_handles, $first_field.handle.clone());
                let out_keys = select::compact_with_count::<
                    KeySource::Runtime,
                    KeySource::Item,
                >(keys.policy(), key_handles.clone(), count)?;
                let $first_field = select::compact_with_count::<
                    KeySource::Runtime,
                    <$first as KernelColumn>::Item,
                >($first_field.policy(), value_handles, count)?;
                $(
                    let $field = super::device_expr_collect(&values.$field)?;
                    let $field = select::compact_with_count::<
                        KeySource::Runtime,
                        <$rest as KernelColumn>::Item,
                    >(
                        $field.policy(),
                        select::handles_for_value(&key_handles, $field.handle.clone()),
                        count,
                    )?;
                )+
                Ok((SoA1 { source: out_keys }, $name { $first_field, $( $field ),+ }))
            }
        }
    };
}

impl_unique_by_key_input!(SoA2<A, B> { left, right });
impl_unique_by_key_input!(SoA3<A, B, C> { first, second, third });
impl_unique_by_key_input!(SoA4<A, B, C, D> { a, b, c, d });
impl_unique_by_key_input!(SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_unique_by_key_input!(SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_unique_by_key_input!(SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_unique_by_key_input!(SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_unique_by_key_input!(SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_unique_by_key_input!(SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_unique_by_key_input!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_unique_by_key_input!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

impl<Source, Pred> UniqueInput<Pred> for SoA1<Source>
where
    Self: SoA<Item = Source::Item, Scalar = Source::Item>,
    Source: OwnedKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn unique_input(self, _pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let input = super::device_expr_collect(&self.source)?;
        Ok(SoA1 {
            source: select::unique(&input, GpuOp::<Pred>::new())?,
        })
    }
}

impl<Source, Pred> UniqueInput<Pred> for Source
where
    Source: OwnedKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn unique_input(self, pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
        <SoA1<Source> as UniqueInput<Pred>>::unique_input(SoA1 { source: self }, pred)
    }
}

fn replace_if_device_vec<R, T, Pred>(
    input: &DeviceVec<R, T>,
    replacement: T,
    _pred: GpuOp<Pred>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Pred: PredicateOp<T>,
{
    u32::try_from(input.len).map_err(|_| Error::LengthTooLarge { len: input.len })?;
    let client = input.policy.client();
    let output_handle = client.empty(input.len * std::mem::size_of::<T>());

    if input.len != 0 {
        let block_count_u32 = mutation_block_count(input.len)?;
        let replacement_values = [replacement];
        let replacement_handle = client.create_from_slice(T::as_bytes(&replacement_values));

        unsafe {
            replace_if_value_kernel::launch_unchecked::<T, Pred, R>(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_MUTATION_SIZE),
                ArrayArg::from_raw_parts::<T>(&input.handle, input.len, 1),
                ArrayArg::from_raw_parts::<T>(&replacement_handle, 1, 1),
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

/// Replaces elements that satisfy `Pred`.
pub fn replace_if<Input, Pred>(
    input: Input,
    replacement: <Input as ReplaceIfInput<Pred>>::Item,
    _pred: Pred,
) -> Result<<Input as ReplaceIfInput<Pred>>::Output, Error>
where
    Input: ReplaceIfInput<Pred>,
{
    input.replace_if_input(replacement, GpuOp::<Pred>::new())
}

/// Removes consecutive duplicates.
pub fn unique<Input, Pred>(
    input: Input,
    _pred: Pred,
) -> Result<<Input as UniqueInput<Pred>>::Output, Error>
where
    Input: UniqueInput<Pred>,
{
    input.unique_input(GpuOp::<Pred>::new())
}

fn unique_by_key_device_vec<R, K, T, Eq>(
    keys: &DeviceVec<R, K>,
    values: &DeviceVec<R, T>,
    _eq: GpuOp<Eq>,
) -> Result<(DeviceVec<R, K>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    Eq: BinaryPredicateOp<K>,
{
    if keys.len != values.len {
        return Err(Error::LengthMismatch {
            input: values.len,
            output: keys.len,
        });
    }
    if keys.len == 0 {
        return Ok((
            DeviceVec::from_handle(keys.policy.clone(), keys.policy.client().empty(0), 0),
            DeviceVec::from_handle(values.policy.clone(), values.policy.client().empty(0), 0),
        ));
    }

    let key_handles = unique_by_key_handles::<R, K, Eq>(keys)?;
    let count = select::selected_count(keys.policy(), &key_handles)?;
    let value_handles = select::handles_for_value(&key_handles, values.handle.clone());
    let out_keys = select::compact_with_count::<R, K>(keys.policy(), key_handles, count)?;
    let out_values = select::compact_with_count::<R, T>(values.policy(), value_handles, count)?;

    Ok((out_keys, out_values))
}

fn unique_by_key_handles<R, K, Eq>(
    keys: &DeviceVec<R, K>,
) -> Result<select::SelectionHandles, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    Eq: BinaryPredicateOp<K>,
{
    if keys.len == 0 {
        return select::handles_from_flags(
            keys.policy(),
            0,
            0,
            keys.policy.client().empty(0),
            keys.handle.clone(),
        );
    }

    let len_u32 = u32::try_from(keys.len).map_err(|_| Error::LengthTooLarge { len: keys.len })?;
    let client = keys.policy.client();
    let block_count_u32 = mutation_block_count(keys.len)?;
    let flag_handle = client.empty(keys.len * std::mem::size_of::<u32>());

    unsafe {
        unique_by_key_flags_kernel::launch_unchecked::<K, Eq, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_MUTATION_SIZE),
            ArrayArg::from_raw_parts::<K>(&keys.handle, keys.len, 1),
            ArrayArg::from_raw_parts::<u32>(&flag_handle, keys.len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    select::handles_from_flags(
        keys.policy(),
        keys.len,
        len_u32,
        flag_handle,
        keys.handle.clone(),
    )
}

/// Removes consecutive duplicate keys and carries the first value for each key.
pub fn unique_by_key<Keys, Values, Eq>(
    keys: Keys,
    values: Values,
    _eq: Eq,
) -> Result<<Keys as UniqueByKeyInput<Values, Eq>>::Output, Error>
where
    Keys: UniqueByKeyInput<Values, Eq>,
{
    keys.unique_by_key_input(values, GpuOp::<Eq>::new())
}
