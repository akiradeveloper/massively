use super::memory::{MaterializeOutput, materialize};
use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlyKernelColumn, ReadOnlySoA, S0, SoA, SoA1,
        SoA2, SoA3, SoA4, SoA5, SoA6, SoA7, SoA8, SoA9, SoA10, SoA11, SoA12, SoAView1, SoAView2,
        SoAView3, SoAView4, SoAView5, SoAView6, SoAView7, SoAView8, SoAView9, SoAView10, SoAView11,
        SoAView12, StorageKernelColumn,
    },
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr},
    kernels::*,
    op::{BinaryPredicateOp, GpuOp, PredicateOp},
    primitives::{segmented, select},
};
use cubecl::prelude::*;

const BLOCK_SEQUENCE_SIZE: u32 = 256;

fn sequence_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_SEQUENCE_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

#[doc(hidden)]
pub trait ReplaceIfInput<Stencil, Pred> {
    type Item;
    type Output;

    fn replace_if_input(
        self,
        replacement: Self::Item,
        stencil: Stencil,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, Stencil, Pred> ReplaceIfInput<Stencil, Pred> for SoA1<Source>
where
    Self: SoA<Item = Source::Item, Scalar = Source::Item>,
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Stencil: KernelColumn<Runtime = Source::Runtime> + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<Stencil::Item>,
{
    type Item = Source::Item;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn replace_if_input(
        self,
        replacement: Self::Item,
        stencil: Stencil,
        _pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        stencil.validate()?;
        super::ensure_same_len(self.source.len(), stencil.len())?;
        let input = super::device_expr_collect(&self.source)?;
        let flags = super::device_expr_selection_handles::<Stencil, Pred>(&stencil, false)?;
        Ok(SoA1 {
            source: replace_with_flags_device_vec(&input, replacement, &flags.flag)?,
        })
    }
}

impl<Source, Stencil, Pred> ReplaceIfInput<Stencil, Pred> for Source
where
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Stencil: KernelColumn<Runtime = Source::Runtime> + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<Stencil::Item>,
{
    type Item = Source::Item;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn replace_if_input(
        self,
        replacement: Self::Item,
        stencil: Stencil,
        pred: GpuOp<Pred>,
    ) -> Result<Self::Output, Error> {
        <SoA1<Source> as ReplaceIfInput<Stencil, Pred>>::replace_if_input(
            SoA1 { source: self },
            replacement,
            stencil,
            pred,
        )
    }
}

macro_rules! impl_replace_if_tuple {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident: $first_index:tt, $( $field:ident: $index:tt ),+ }
    ) => {
        impl<$first, $( $rest ),+, Stencil, Pred> ReplaceIfInput<Stencil, Pred>
            for $name<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: StorageKernelColumn + KernelColumnAt<S0>,
            $(
                $rest: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            Stencil: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Stencil::Item: CubePrimitive + CubeElement,
            Stencil::Expr: GpuExpr<Stencil::Item>,
            Pred: PredicateOp<Stencil::Item>,
        {
            type Item = (
                impl_replace_if_tuple!(@item_ty $first),
                $( impl_replace_if_tuple!(@item_ty $rest) ),+
            );
            type Output = $name<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn replace_if_input(
                self,
                replacement: Self::Item,
                stencil: Stencil,
                _pred: GpuOp<Pred>,
            ) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                stencil.validate()?;
                super::ensure_same_len(self.$first_field.len(), stencil.len())?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let flags = super::device_expr_selection_handles::<Stencil, Pred>(&stencil, false)?;
                Ok($name {
                    $first_field: replace_with_flags_device_vec(
                        &$first_field,
                        replacement.$first_index,
                        &flags.flag,
                    )?,
                    $(
                        $field: replace_with_flags_device_vec(
                            &$field,
                            replacement.$index,
                            &flags.flag,
                        )?,
                    )+
                })
            }
        }
    };
}

impl_replace_if_tuple!(SoA2<A, B> { left: 0, right: 1 });
impl_replace_if_tuple!(SoA3<A, B, C> { first: 0, second: 1, third: 2 });
impl_replace_if_tuple!(SoA4<A, B, C, D> { a: 0, b: 1, c: 2, d: 3 });
impl_replace_if_tuple!(SoA5<A, B, C, D, E> { a: 0, b: 1, c: 2, d: 3, e: 4 });
impl_replace_if_tuple!(SoA6<A, B, C, D, E, F> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5 });
impl_replace_if_tuple!(SoA7<A, B, C, D, E, F, G> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6 });
impl_replace_if_tuple!(SoA8<A, B, C, D, E, F, G, H> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7 });
impl_replace_if_tuple!(SoA9<A, B, C, D, E, F, G, H, I> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8 });
impl_replace_if_tuple!(SoA10<A, B, C, D, E, F, G, H, I, J> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8, j: 9 });
impl_replace_if_tuple!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8, j: 9, k: 10 });
impl_replace_if_tuple!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8, j: 9, k: 10, l: 11 });

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
    KeySource: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    ValueSource: ReadOnlyKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
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

impl<KeySource, ValueSource, Eq> UniqueByKeyInput<ValueSource, Eq> for KeySource
where
    KeySource: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    ValueSource: ReadOnlyKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    SoA1<KeySource>: UniqueByKeyInput<SoA1<ValueSource>, Eq>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
{
    type Output = <SoA1<KeySource> as UniqueByKeyInput<SoA1<ValueSource>, Eq>>::Output;

    fn unique_by_key_input(
        self,
        values: ValueSource,
        eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        <SoA1<KeySource> as UniqueByKeyInput<SoA1<ValueSource>, Eq>>::unique_by_key_input(
            SoA1 { source: self },
            SoA1 { source: values },
            eq,
        )
    }
}

macro_rules! impl_unique_by_tuple_key_scalar_value {
    (
        $storage:ident,
        $keys:ident -> $out_keys:ident,
        $kernel:ident,
        ( $first:ident: $first_field:ident: $first_out:ident: $first_handles:ident, $( $key:ident: $field:ident: $out:ident: $handles:ident ),+ )
    ) => {
        impl<$first, $( $key ),+, ValueSource, Eq> UniqueByKeyInput<ValueSource, Eq>
            for $keys<$first, $( $key ),+>
        where
            Self: $storage<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $key: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            ValueSource: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            ValueSource::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
            Eq: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
        {
            type Output = (
                $out_keys<
                    DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<<$first as KernelColumn>::Runtime, <$key as KernelColumn>::Item> ),+
                >,
                SoA1<DeviceVec<<$first as KernelColumn>::Runtime, ValueSource::Item>>,
            );

            fn unique_by_key_input(
                self,
                values: ValueSource,
                _eq: GpuOp<Eq>,
            ) -> Result<Self::Output, Error> {
                $storage::validate(&self)?;
                values.validate()?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let values = super::device_expr_collect(&values)?;
                $(
                    super::ensure_same_len($field.len, $first_field.len)?;
                )+
                super::ensure_same_len(values.len, $first_field.len)?;
                if $first_field.len == 0 {
                    return Ok((
                        $out_keys {
                            $first_field: DeviceVec::empty($first_field.policy.clone()),
                            $( $field: DeviceVec::empty($field.policy.clone()), )+
                        },
                        SoA1 {
                            source: DeviceVec::empty(values.policy.clone()),
                        },
                    ));
                }

                let len = $first_field.len;
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let client = $first_field.policy.client();
                let block_count_u32 = sequence_block_count(len)?;
                let flag_handle = client.empty(len * std::mem::size_of::<u32>());
                unsafe {
                    $kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$key as KernelColumn>::Item, )+
                        Eq,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&$first_field.handle, len, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$key as KernelColumn>::Item>(&$field.handle, len, 1),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }

                let control = segmented::SegmentControl::from_end_flags(
                    $first_field.policy(),
                    len,
                    len_u32,
                    flag_handle,
                    $first_field.handle.clone(),
                )?;
                let $first_out = control.compact_first::<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>(
                    $first_field.policy(),
                )?;
                $(
                    let $handles = $field.handle.clone();
                    let $out = control.compact_value::<<$first as KernelColumn>::Runtime, <$key as KernelColumn>::Item>(
                        $field.policy(),
                        $handles,
                    )?;
                )+
                let source = control.compact_value::<<$first as KernelColumn>::Runtime, ValueSource::Item>(
                    values.policy(),
                    values.handle.clone(),
                )?;

                Ok((
                    $out_keys { $first_field: $first_out, $( $field: $out ),+ },
                    SoA1 { source },
                ))
            }
        }
    };
}

impl_unique_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView4 -> SoA4, tuple4_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles));
impl_unique_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView5 -> SoA5, tuple5_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles));
impl_unique_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView6 -> SoA6, tuple6_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles));
impl_unique_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView7 -> SoA7, tuple7_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles));
impl_unique_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView8 -> SoA8, tuple8_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles));
impl_unique_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView9 -> SoA9, tuple9_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles));
impl_unique_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView10 -> SoA10, tuple10_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles, J: j: out_j: key_j_handles));
impl_unique_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView11 -> SoA11, tuple11_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles, J: j: out_j: key_j_handles, K: k: out_k: key_k_handles));
impl_unique_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView12 -> SoA12, tuple12_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles, J: j: out_j: key_j_handles, K: k: out_k: key_k_handles, L: l: out_l: key_l_handles));
impl_unique_by_tuple_key_scalar_value!(SoA, SoA2 -> SoA2, tuple2_unique_flags_kernel, (A: left: out_left: key_left_handles, B: right: out_right: key_right_handles));
impl_unique_by_tuple_key_scalar_value!(SoA, SoA3 -> SoA3, tuple3_unique_flags_kernel, (A: first: out_first: key_first_handles, B: second: out_second: key_second_handles, C: third: out_third: key_third_handles));
impl_unique_by_tuple_key_scalar_value!(SoA, SoA4 -> SoA4, tuple4_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles));
impl_unique_by_tuple_key_scalar_value!(SoA, SoA5 -> SoA5, tuple5_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles));
impl_unique_by_tuple_key_scalar_value!(SoA, SoA6 -> SoA6, tuple6_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles));
impl_unique_by_tuple_key_scalar_value!(SoA, SoA7 -> SoA7, tuple7_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles));
impl_unique_by_tuple_key_scalar_value!(SoA, SoA8 -> SoA8, tuple8_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles));
impl_unique_by_tuple_key_scalar_value!(SoA, SoA9 -> SoA9, tuple9_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles));
impl_unique_by_tuple_key_scalar_value!(SoA, SoA10 -> SoA10, tuple10_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles, J: j: out_j: key_j_handles));
impl_unique_by_tuple_key_scalar_value!(SoA, SoA11 -> SoA11, tuple11_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles, J: j: out_j: key_j_handles, K: k: out_k: key_k_handles));
impl_unique_by_tuple_key_scalar_value!(SoA, SoA12 -> SoA12, tuple12_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles, J: j: out_j: key_j_handles, K: k: out_k: key_k_handles, L: l: out_l: key_l_handles));

macro_rules! impl_unique_by_tuple_key_soa2_values {
    (
        $storage:ident,
        $keys:ident -> $out_keys:ident,
        $kernel:ident,
        ( $first:ident: $first_field:ident: $first_out:ident: $first_handles:ident, $( $key:ident: $field:ident: $out:ident: $handles:ident ),+ )
    ) => {
        impl<$first, $( $key ),+, ValueA, ValueB, Eq> UniqueByKeyInput<SoA2<ValueA, ValueB>, Eq>
            for $keys<$first, $( $key ),+>
        where
            Self: $storage<Scalar = <$first as KernelColumn>::Item>,
            SoA2<ValueA, ValueB>: SoA,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $key: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            ValueA: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            ValueB: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            ValueA::Item: CubePrimitive + CubeElement,
            ValueB::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
            ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
            Eq: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
        {
            type Output = (
                $out_keys<
                    DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<<$first as KernelColumn>::Runtime, <$key as KernelColumn>::Item> ),+
                >,
                SoA2<
                    DeviceVec<<$first as KernelColumn>::Runtime, ValueA::Item>,
                    DeviceVec<<$first as KernelColumn>::Runtime, ValueB::Item>,
                >,
            );

            fn unique_by_key_input(
                self,
                values: SoA2<ValueA, ValueB>,
                _eq: GpuOp<Eq>,
            ) -> Result<Self::Output, Error> {
                $storage::validate(&self)?;
                SoA::validate(&values)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let value_a = super::device_expr_collect(&values.left)?;
                let value_b = super::device_expr_collect(&values.right)?;
                $(
                    super::ensure_same_len($field.len, $first_field.len)?;
                )+
                super::ensure_same_len(value_a.len, $first_field.len)?;
                super::ensure_same_len(value_b.len, $first_field.len)?;
                if $first_field.len == 0 {
                    return Ok((
                        $out_keys {
                            $first_field: DeviceVec::empty($first_field.policy.clone()),
                            $( $field: DeviceVec::empty($field.policy.clone()), )+
                        },
                        SoA2 {
                            left: DeviceVec::empty(value_a.policy.clone()),
                            right: DeviceVec::empty(value_b.policy.clone()),
                        },
                    ));
                }

                let len = $first_field.len;
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let client = $first_field.policy.client();
                let block_count_u32 = sequence_block_count(len)?;
                let flag_handle = client.empty(len * std::mem::size_of::<u32>());
                unsafe {
                    $kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$key as KernelColumn>::Item, )+
                        Eq,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&$first_field.handle, len, 1),
                        $(
                            ArrayArg::from_raw_parts::<<$key as KernelColumn>::Item>(&$field.handle, len, 1),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }

                let control = segmented::SegmentControl::from_end_flags(
                    $first_field.policy(),
                    len,
                    len_u32,
                    flag_handle,
                    $first_field.handle.clone(),
                )?;
                let $first_out = control.compact_first::<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>(
                    $first_field.policy(),
                )?;
                $(
                    let $handles = $field.handle.clone();
                    let $out = control.compact_value::<<$first as KernelColumn>::Runtime, <$key as KernelColumn>::Item>(
                        $field.policy(),
                        $handles,
                    )?;
                )+
                let left = control.compact_value::<<$first as KernelColumn>::Runtime, ValueA::Item>(
                    value_a.policy(),
                    value_a.handle.clone(),
                )?;
                let right = control.compact_value::<<$first as KernelColumn>::Runtime, ValueB::Item>(
                    value_b.policy(),
                    value_b.handle.clone(),
                )?;

                Ok((
                    $out_keys { $first_field: $first_out, $( $field: $out ),+ },
                    SoA2 { left, right },
                ))
            }
        }
    };
}

impl_unique_by_tuple_key_soa2_values!(ReadOnlySoA, SoAView4 -> SoA4, tuple4_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa2_values!(SoA, SoA2 -> SoA2, tuple2_unique_flags_kernel, (A: left: out_left: key_left_handles, B: right: out_right: key_right_handles));
impl_unique_by_tuple_key_soa2_values!(SoA, SoA3 -> SoA3, tuple3_unique_flags_kernel, (A: first: out_first: key_first_handles, B: second: out_second: key_second_handles, C: third: out_third: key_third_handles));
impl_unique_by_tuple_key_soa2_values!(SoA, SoA4 -> SoA4, tuple4_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa2_values!(SoA, SoA5 -> SoA5, tuple5_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa2_values!(SoA, SoA6 -> SoA6, tuple6_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa2_values!(SoA, SoA7 -> SoA7, tuple7_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa2_values!(SoA, SoA8 -> SoA8, tuple8_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa2_values!(SoA, SoA9 -> SoA9, tuple9_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa2_values!(SoA, SoA10 -> SoA10, tuple10_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles, J: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa2_values!(SoA, SoA11 -> SoA11, tuple11_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles, J: j: out_j: key_j_handles, K: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa2_values!(SoA, SoA12 -> SoA12, tuple12_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles, J: j: out_j: key_j_handles, K: k: out_k: key_k_handles, L: l: out_l: key_l_handles));

macro_rules! impl_unique_by_tuple_key_soa_values {
    (
        $storage:ident,
        $values:ident -> $out_values:ident < $( $value:ident: $value_field:ident: $value_vec:ident: $value_handles:ident: $value_out:ident ),+ >,
        $keys:ident -> $out_keys:ident,
        $kernel:ident,
        ( $first:ident: $first_field:ident: $first_out:ident: $first_handles:ident,
          $( $key:ident: $field:ident: $out:ident: $handles:ident ),+ )
    ) => {
        impl<$first, $( $key ),+, $( $value ),+, Eq> UniqueByKeyInput<$values<$( $value ),+>, Eq>
            for $keys<$first, $( $key ),+>
        where
            Self: $storage<Scalar = <$first as KernelColumn>::Item>,
            $values<$( $value ),+>: SoA,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $key: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            $( $value: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            $( <$value as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            $( <$value as KernelColumn>::Expr: DeviceGpuExpr<<$value as KernelColumn>::Item>, )+
            Eq: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
        {
            type Output = (
                $out_keys<
                    DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<<$first as KernelColumn>::Runtime, <$key as KernelColumn>::Item> ),+
                >,
                $out_values<$( DeviceVec<<$first as KernelColumn>::Runtime, <$value as KernelColumn>::Item> ),+>,
            );

            fn unique_by_key_input(
                self,
                values: $values<$( $value ),+>,
                _eq: GpuOp<Eq>,
            ) -> Result<Self::Output, Error> {
                $storage::validate(&self)?;
                SoA::validate(&values)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                $( let $value_vec = super::device_expr_collect(&values.$value_field)?; )+
                $(
                    super::ensure_same_len($field.len, $first_field.len)?;
                )+
                $(
                    super::ensure_same_len($value_vec.len, $first_field.len)?;
                )+
                if $first_field.len == 0 {
                    return Ok((
                        $out_keys {
                            $first_field: DeviceVec::empty($first_field.policy.clone()),
                            $( $field: DeviceVec::empty($field.policy.clone()), )+
                        },
                        $out_values {
                            $( $value_field: DeviceVec::empty($value_vec.policy.clone()), )+
                        },
                    ));
                }
                let len = $first_field.len;
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let client = $first_field.policy.client();
                let block_count_u32 = sequence_block_count(len)?;
                let flag_handle = client.empty(len * std::mem::size_of::<u32>());
                unsafe {
                    $kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$key as KernelColumn>::Item, )+
                        Eq,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
                        ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(&$first_field.handle, len, 1),
                        $( ArrayArg::from_raw_parts::<<$key as KernelColumn>::Item>(&$field.handle, len, 1), )+
                        ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
                    ).map_err(|err| Error::Launch { message: format!("{err:?}") })?;
                }
                let control = segmented::SegmentControl::from_end_flags(
                    $first_field.policy(),
                    len,
                    len_u32,
                    flag_handle,
                    $first_field.handle.clone(),
                )?;
                let $first_out = control.compact_first::<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>(
                    $first_field.policy(),
                )?;
                $(
                    let $handles = $field.handle.clone();
                    let $out = control.compact_value::<<$first as KernelColumn>::Runtime, <$key as KernelColumn>::Item>(
                        $field.policy(),
                        $handles,
                    )?;
                )+
                $(
                    let $value_handles = $value_vec.handle.clone();
                    let $value_out = control.compact_value::<<$first as KernelColumn>::Runtime, <$value as KernelColumn>::Item>(
                        $value_vec.policy(),
                        $value_handles,
                    )?;
                )+
                Ok((
                    $out_keys { $first_field: $first_out, $( $field: $out ),+ },
                    $out_values { $( $value_field: $value_out ),+ },
                ))
            }
        }
    };
}

impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA4 -> SoA4 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d >, SoAView2 -> SoA2, tuple2_unique_flags_kernel, (KA: left: out_left: key_left_handles, KB: right: out_right: key_right_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA5 -> SoA5 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e >, SoAView2 -> SoA2, tuple2_unique_flags_kernel, (KA: left: out_left: key_left_handles, KB: right: out_right: key_right_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA6 -> SoA6 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f >, SoAView2 -> SoA2, tuple2_unique_flags_kernel, (KA: left: out_left: key_left_handles, KB: right: out_right: key_right_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA7 -> SoA7 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g >, SoAView2 -> SoA2, tuple2_unique_flags_kernel, (KA: left: out_left: key_left_handles, KB: right: out_right: key_right_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA8 -> SoA8 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h >, SoAView2 -> SoA2, tuple2_unique_flags_kernel, (KA: left: out_left: key_left_handles, KB: right: out_right: key_right_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA9 -> SoA9 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i >, SoAView2 -> SoA2, tuple2_unique_flags_kernel, (KA: left: out_left: key_left_handles, KB: right: out_right: key_right_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA10 -> SoA10 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j >, SoAView2 -> SoA2, tuple2_unique_flags_kernel, (KA: left: out_left: key_left_handles, KB: right: out_right: key_right_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA11 -> SoA11 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k >, SoAView2 -> SoA2, tuple2_unique_flags_kernel, (KA: left: out_left: key_left_handles, KB: right: out_right: key_right_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA12 -> SoA12 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k, VL: l: value_l: value_l_handles: out_value_l >, SoAView2 -> SoA2, tuple2_unique_flags_kernel, (KA: left: out_left: key_left_handles, KB: right: out_right: key_right_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA4 -> SoA4 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d >, SoAView3 -> SoA3, tuple3_unique_flags_kernel, (KA: first: out_first: key_first_handles, KB: second: out_second: key_second_handles, KC: third: out_third: key_third_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA5 -> SoA5 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e >, SoAView3 -> SoA3, tuple3_unique_flags_kernel, (KA: first: out_first: key_first_handles, KB: second: out_second: key_second_handles, KC: third: out_third: key_third_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA6 -> SoA6 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f >, SoAView3 -> SoA3, tuple3_unique_flags_kernel, (KA: first: out_first: key_first_handles, KB: second: out_second: key_second_handles, KC: third: out_third: key_third_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA7 -> SoA7 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g >, SoAView3 -> SoA3, tuple3_unique_flags_kernel, (KA: first: out_first: key_first_handles, KB: second: out_second: key_second_handles, KC: third: out_third: key_third_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA8 -> SoA8 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h >, SoAView3 -> SoA3, tuple3_unique_flags_kernel, (KA: first: out_first: key_first_handles, KB: second: out_second: key_second_handles, KC: third: out_third: key_third_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA9 -> SoA9 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i >, SoAView3 -> SoA3, tuple3_unique_flags_kernel, (KA: first: out_first: key_first_handles, KB: second: out_second: key_second_handles, KC: third: out_third: key_third_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA10 -> SoA10 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j >, SoAView3 -> SoA3, tuple3_unique_flags_kernel, (KA: first: out_first: key_first_handles, KB: second: out_second: key_second_handles, KC: third: out_third: key_third_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA11 -> SoA11 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k >, SoAView3 -> SoA3, tuple3_unique_flags_kernel, (KA: first: out_first: key_first_handles, KB: second: out_second: key_second_handles, KC: third: out_third: key_third_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA12 -> SoA12 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k, VL: l: value_l: value_l_handles: out_value_l >, SoAView3 -> SoA3, tuple3_unique_flags_kernel, (KA: first: out_first: key_first_handles, KB: second: out_second: key_second_handles, KC: third: out_third: key_third_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA3 -> SoA3 < VA: first: value_first: value_first_handles: out_value_first, VB: second: value_second: value_second_handles: out_value_second, VC: third: value_third: value_third_handles: out_value_third >, SoAView4 -> SoA4, tuple4_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA4 -> SoA4 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d >, SoAView4 -> SoA4, tuple4_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA5 -> SoA5 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e >, SoAView4 -> SoA4, tuple4_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA6 -> SoA6 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f >, SoAView4 -> SoA4, tuple4_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA7 -> SoA7 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g >, SoAView4 -> SoA4, tuple4_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA8 -> SoA8 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h >, SoAView4 -> SoA4, tuple4_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA9 -> SoA9 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i >, SoAView4 -> SoA4, tuple4_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA10 -> SoA10 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j >, SoAView4 -> SoA4, tuple4_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA11 -> SoA11 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k >, SoAView4 -> SoA4, tuple4_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA12 -> SoA12 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k, VL: l: value_l: value_l_handles: out_value_l >, SoAView4 -> SoA4, tuple4_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA3 -> SoA3 < VA: first: value_first: value_first_handles: out_value_first, VB: second: value_second: value_second_handles: out_value_second, VC: third: value_third: value_third_handles: out_value_third >, SoAView5 -> SoA5, tuple5_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA4 -> SoA4 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d >, SoAView5 -> SoA5, tuple5_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA5 -> SoA5 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e >, SoAView5 -> SoA5, tuple5_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA6 -> SoA6 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f >, SoAView5 -> SoA5, tuple5_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA7 -> SoA7 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g >, SoAView5 -> SoA5, tuple5_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA8 -> SoA8 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h >, SoAView5 -> SoA5, tuple5_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA9 -> SoA9 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i >, SoAView5 -> SoA5, tuple5_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA10 -> SoA10 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j >, SoAView5 -> SoA5, tuple5_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA11 -> SoA11 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k >, SoAView5 -> SoA5, tuple5_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA12 -> SoA12 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k, VL: l: value_l: value_l_handles: out_value_l >, SoAView5 -> SoA5, tuple5_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA3 -> SoA3 < VA: first: value_first: value_first_handles: out_value_first, VB: second: value_second: value_second_handles: out_value_second, VC: third: value_third: value_third_handles: out_value_third >, SoAView6 -> SoA6, tuple6_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA4 -> SoA4 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d >, SoAView6 -> SoA6, tuple6_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA5 -> SoA5 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e >, SoAView6 -> SoA6, tuple6_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA6 -> SoA6 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f >, SoAView6 -> SoA6, tuple6_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA7 -> SoA7 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g >, SoAView6 -> SoA6, tuple6_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA8 -> SoA8 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h >, SoAView6 -> SoA6, tuple6_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA9 -> SoA9 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i >, SoAView6 -> SoA6, tuple6_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA10 -> SoA10 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j >, SoAView6 -> SoA6, tuple6_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA11 -> SoA11 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k >, SoAView6 -> SoA6, tuple6_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA12 -> SoA12 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k, VL: l: value_l: value_l_handles: out_value_l >, SoAView6 -> SoA6, tuple6_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA3 -> SoA3 < VA: first: value_first: value_first_handles: out_value_first, VB: second: value_second: value_second_handles: out_value_second, VC: third: value_third: value_third_handles: out_value_third >, SoAView7 -> SoA7, tuple7_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA4 -> SoA4 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d >, SoAView7 -> SoA7, tuple7_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA5 -> SoA5 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e >, SoAView7 -> SoA7, tuple7_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA6 -> SoA6 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f >, SoAView7 -> SoA7, tuple7_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA7 -> SoA7 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g >, SoAView7 -> SoA7, tuple7_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA8 -> SoA8 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h >, SoAView7 -> SoA7, tuple7_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA9 -> SoA9 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i >, SoAView7 -> SoA7, tuple7_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA10 -> SoA10 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j >, SoAView7 -> SoA7, tuple7_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA11 -> SoA11 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k >, SoAView7 -> SoA7, tuple7_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA12 -> SoA12 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k, VL: l: value_l: value_l_handles: out_value_l >, SoAView7 -> SoA7, tuple7_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA3 -> SoA3 < VA: first: value_first: value_first_handles: out_value_first, VB: second: value_second: value_second_handles: out_value_second, VC: third: value_third: value_third_handles: out_value_third >, SoAView8 -> SoA8, tuple8_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA4 -> SoA4 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d >, SoAView8 -> SoA8, tuple8_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA5 -> SoA5 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e >, SoAView8 -> SoA8, tuple8_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA6 -> SoA6 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f >, SoAView8 -> SoA8, tuple8_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA7 -> SoA7 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g >, SoAView8 -> SoA8, tuple8_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA8 -> SoA8 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h >, SoAView8 -> SoA8, tuple8_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA9 -> SoA9 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i >, SoAView8 -> SoA8, tuple8_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA10 -> SoA10 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j >, SoAView8 -> SoA8, tuple8_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA11 -> SoA11 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k >, SoAView8 -> SoA8, tuple8_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA12 -> SoA12 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k, VL: l: value_l: value_l_handles: out_value_l >, SoAView8 -> SoA8, tuple8_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA3 -> SoA3 < VA: first: value_first: value_first_handles: out_value_first, VB: second: value_second: value_second_handles: out_value_second, VC: third: value_third: value_third_handles: out_value_third >, SoAView9 -> SoA9, tuple9_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA4 -> SoA4 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d >, SoAView9 -> SoA9, tuple9_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA5 -> SoA5 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e >, SoAView9 -> SoA9, tuple9_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA6 -> SoA6 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f >, SoAView9 -> SoA9, tuple9_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA7 -> SoA7 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g >, SoAView9 -> SoA9, tuple9_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA8 -> SoA8 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h >, SoAView9 -> SoA9, tuple9_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA9 -> SoA9 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i >, SoAView9 -> SoA9, tuple9_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA10 -> SoA10 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j >, SoAView9 -> SoA9, tuple9_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA11 -> SoA11 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k >, SoAView9 -> SoA9, tuple9_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA12 -> SoA12 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k, VL: l: value_l: value_l_handles: out_value_l >, SoAView9 -> SoA9, tuple9_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA3 -> SoA3 < VA: first: value_first: value_first_handles: out_value_first, VB: second: value_second: value_second_handles: out_value_second, VC: third: value_third: value_third_handles: out_value_third >, SoAView10 -> SoA10, tuple10_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA4 -> SoA4 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d >, SoAView10 -> SoA10, tuple10_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA5 -> SoA5 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e >, SoAView10 -> SoA10, tuple10_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA6 -> SoA6 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f >, SoAView10 -> SoA10, tuple10_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA7 -> SoA7 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g >, SoAView10 -> SoA10, tuple10_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA8 -> SoA8 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h >, SoAView10 -> SoA10, tuple10_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA9 -> SoA9 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i >, SoAView10 -> SoA10, tuple10_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA10 -> SoA10 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j >, SoAView10 -> SoA10, tuple10_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA11 -> SoA11 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k >, SoAView10 -> SoA10, tuple10_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA12 -> SoA12 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k, VL: l: value_l: value_l_handles: out_value_l >, SoAView10 -> SoA10, tuple10_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA3 -> SoA3 < VA: first: value_first: value_first_handles: out_value_first, VB: second: value_second: value_second_handles: out_value_second, VC: third: value_third: value_third_handles: out_value_third >, SoAView11 -> SoA11, tuple11_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA4 -> SoA4 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d >, SoAView11 -> SoA11, tuple11_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA5 -> SoA5 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e >, SoAView11 -> SoA11, tuple11_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA6 -> SoA6 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f >, SoAView11 -> SoA11, tuple11_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA7 -> SoA7 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g >, SoAView11 -> SoA11, tuple11_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA8 -> SoA8 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h >, SoAView11 -> SoA11, tuple11_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA9 -> SoA9 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i >, SoAView11 -> SoA11, tuple11_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA10 -> SoA10 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j >, SoAView11 -> SoA11, tuple11_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA11 -> SoA11 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k >, SoAView11 -> SoA11, tuple11_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA12 -> SoA12 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k, VL: l: value_l: value_l_handles: out_value_l >, SoAView11 -> SoA11, tuple11_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA3 -> SoA3 < VA: first: value_first: value_first_handles: out_value_first, VB: second: value_second: value_second_handles: out_value_second, VC: third: value_third: value_third_handles: out_value_third >, SoAView12 -> SoA12, tuple12_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles, KL: l: out_l: key_l_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA4 -> SoA4 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d >, SoAView12 -> SoA12, tuple12_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles, KL: l: out_l: key_l_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA5 -> SoA5 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e >, SoAView12 -> SoA12, tuple12_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles, KL: l: out_l: key_l_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA6 -> SoA6 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f >, SoAView12 -> SoA12, tuple12_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles, KL: l: out_l: key_l_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA7 -> SoA7 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g >, SoAView12 -> SoA12, tuple12_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles, KL: l: out_l: key_l_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA8 -> SoA8 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h >, SoAView12 -> SoA12, tuple12_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles, KL: l: out_l: key_l_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA9 -> SoA9 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i >, SoAView12 -> SoA12, tuple12_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles, KL: l: out_l: key_l_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA10 -> SoA10 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j >, SoAView12 -> SoA12, tuple12_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles, KL: l: out_l: key_l_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA11 -> SoA11 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k >, SoAView12 -> SoA12, tuple12_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles, KL: l: out_l: key_l_handles));
impl_unique_by_tuple_key_soa_values!(ReadOnlySoA, SoA12 -> SoA12 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k, VL: l: value_l: value_l_handles: out_value_l >, SoAView12 -> SoA12, tuple12_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles, KL: l: out_l: key_l_handles));

macro_rules! impl_unique_by_tuple_key_soa_values_for_soa_key {
    ($keys:ident -> $out_keys:ident, $kernel:ident, ( $first:ident: $first_field:ident: $first_out:ident: $first_handles:ident, $( $key:ident: $field:ident: $out:ident: $handles:ident ),+ )) => {
        impl_unique_by_tuple_key_soa_values!(SoA, SoA3 -> SoA3 < VA: first: value_first: value_first_handles: out_value_first, VB: second: value_second: value_second_handles: out_value_second, VC: third: value_third: value_third_handles: out_value_third >, $keys -> $out_keys, $kernel, ( $first: $first_field: $first_out: $first_handles, $( $key: $field: $out: $handles ),+ ));
        impl_unique_by_tuple_key_soa_values!(SoA, SoA4 -> SoA4 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d >, $keys -> $out_keys, $kernel, ( $first: $first_field: $first_out: $first_handles, $( $key: $field: $out: $handles ),+ ));
        impl_unique_by_tuple_key_soa_values!(SoA, SoA5 -> SoA5 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e >, $keys -> $out_keys, $kernel, ( $first: $first_field: $first_out: $first_handles, $( $key: $field: $out: $handles ),+ ));
        impl_unique_by_tuple_key_soa_values!(SoA, SoA6 -> SoA6 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f >, $keys -> $out_keys, $kernel, ( $first: $first_field: $first_out: $first_handles, $( $key: $field: $out: $handles ),+ ));
        impl_unique_by_tuple_key_soa_values!(SoA, SoA7 -> SoA7 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g >, $keys -> $out_keys, $kernel, ( $first: $first_field: $first_out: $first_handles, $( $key: $field: $out: $handles ),+ ));
        impl_unique_by_tuple_key_soa_values!(SoA, SoA8 -> SoA8 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h >, $keys -> $out_keys, $kernel, ( $first: $first_field: $first_out: $first_handles, $( $key: $field: $out: $handles ),+ ));
        impl_unique_by_tuple_key_soa_values!(SoA, SoA9 -> SoA9 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i >, $keys -> $out_keys, $kernel, ( $first: $first_field: $first_out: $first_handles, $( $key: $field: $out: $handles ),+ ));
        impl_unique_by_tuple_key_soa_values!(SoA, SoA10 -> SoA10 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j >, $keys -> $out_keys, $kernel, ( $first: $first_field: $first_out: $first_handles, $( $key: $field: $out: $handles ),+ ));
        impl_unique_by_tuple_key_soa_values!(SoA, SoA11 -> SoA11 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k >, $keys -> $out_keys, $kernel, ( $first: $first_field: $first_out: $first_handles, $( $key: $field: $out: $handles ),+ ));
        impl_unique_by_tuple_key_soa_values!(SoA, SoA12 -> SoA12 < VA: a: value_a: value_a_handles: out_value_a, VB: b: value_b: value_b_handles: out_value_b, VC: c: value_c: value_c_handles: out_value_c, VD: d: value_d: value_d_handles: out_value_d, VE: e: value_e: value_e_handles: out_value_e, VF: f: value_f: value_f_handles: out_value_f, VG: g: value_g: value_g_handles: out_value_g, VH: h: value_h: value_h_handles: out_value_h, VI: i: value_i: value_i_handles: out_value_i, VJ: j: value_j: value_j_handles: out_value_j, VK: k: value_k: value_k_handles: out_value_k, VL: l: value_l: value_l_handles: out_value_l >, $keys -> $out_keys, $kernel, ( $first: $first_field: $first_out: $first_handles, $( $key: $field: $out: $handles ),+ ));
    };
}

impl_unique_by_tuple_key_soa_values_for_soa_key!(SoA2 -> SoA2, tuple2_unique_flags_kernel, (KA: left: out_left: key_left_handles, KB: right: out_right: key_right_handles));
impl_unique_by_tuple_key_soa_values_for_soa_key!(SoA3 -> SoA3, tuple3_unique_flags_kernel, (KA: first: out_first: key_first_handles, KB: second: out_second: key_second_handles, KC: third: out_third: key_third_handles));
impl_unique_by_tuple_key_soa_values_for_soa_key!(SoA4 -> SoA4, tuple4_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles));
impl_unique_by_tuple_key_soa_values_for_soa_key!(SoA5 -> SoA5, tuple5_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa_values_for_soa_key!(SoA6 -> SoA6, tuple6_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa_values_for_soa_key!(SoA7 -> SoA7, tuple7_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa_values_for_soa_key!(SoA8 -> SoA8, tuple8_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa_values_for_soa_key!(SoA9 -> SoA9, tuple9_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa_values_for_soa_key!(SoA10 -> SoA10, tuple10_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa_values_for_soa_key!(SoA11 -> SoA11, tuple11_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa_values_for_soa_key!(SoA12 -> SoA12, tuple12_unique_flags_kernel, (KA: a: out_a: key_a_handles, KB: b: out_b: key_b_handles, KC: c: out_c: key_c_handles, KD: d: out_d: key_d_handles, KE: e: out_e: key_e_handles, KF: f: out_f: key_f_handles, KG: g: out_g: key_g_handles, KH: h: out_h: key_h_handles, KI: i: out_i: key_i_handles, KJ: j: out_j: key_j_handles, KK: k: out_k: key_k_handles, KL: l: out_l: key_l_handles));

impl_unique_by_tuple_key_soa2_values!(ReadOnlySoA, SoAView5 -> SoA5, tuple5_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles));
impl_unique_by_tuple_key_soa2_values!(ReadOnlySoA, SoAView6 -> SoA6, tuple6_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles));
impl_unique_by_tuple_key_soa2_values!(ReadOnlySoA, SoAView7 -> SoA7, tuple7_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles));
impl_unique_by_tuple_key_soa2_values!(ReadOnlySoA, SoAView8 -> SoA8, tuple8_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles));
impl_unique_by_tuple_key_soa2_values!(ReadOnlySoA, SoAView9 -> SoA9, tuple9_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles));
impl_unique_by_tuple_key_soa2_values!(ReadOnlySoA, SoAView10 -> SoA10, tuple10_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles, J: j: out_j: key_j_handles));
impl_unique_by_tuple_key_soa2_values!(ReadOnlySoA, SoAView11 -> SoA11, tuple11_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles, J: j: out_j: key_j_handles, K: k: out_k: key_k_handles));
impl_unique_by_tuple_key_soa2_values!(ReadOnlySoA, SoAView12 -> SoA12, tuple12_unique_flags_kernel, (A: a: out_a: key_a_handles, B: b: out_b: key_b_handles, C: c: out_c: key_c_handles, D: d: out_d: key_d_handles, E: e: out_e: key_e_handles, F: f: out_f: key_f_handles, G: g: out_g: key_g_handles, H: h: out_h: key_h_handles, I: i: out_i: key_i_handles, J: j: out_j: key_j_handles, K: k: out_k: key_k_handles, L: l: out_l: key_l_handles));

impl<KeyA, KeyB, ValueSource, Eq> UniqueByKeyInput<ValueSource, Eq> for SoAView2<KeyA, KeyB>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item), Scalar = KeyA::Item>,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueSource: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Eq: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
{
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA1<DeviceVec<KeyA::Runtime, ValueSource::Item>>,
    );

    fn unique_by_key_input(
        self,
        values: ValueSource,
        _eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let values = SoAView1 { source: values };
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let values = super::device_expr_collect(&values.source)?;
        super::ensure_same_len(key_b.len, key_a.len)?;
        super::ensure_same_len(values.len, key_a.len)?;
        if key_a.len == 0 {
            return Ok((
                SoA2 {
                    left: DeviceVec::empty(key_a.policy.clone()),
                    right: DeviceVec::empty(key_b.policy.clone()),
                },
                SoA1 {
                    source: DeviceVec::empty(values.policy.clone()),
                },
            ));
        }

        let len_u32 =
            u32::try_from(key_a.len).map_err(|_| Error::LengthTooLarge { len: key_a.len })?;
        let client = key_a.policy.client();
        let block_count_u32 = sequence_block_count(key_a.len)?;
        let flag_handle = client.empty(key_a.len * std::mem::size_of::<u32>());

        unsafe {
            tuple2_unique_flags_kernel::launch_unchecked::<KeyA::Item, KeyB::Item, Eq, KeyA::Runtime>(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
                ArrayArg::from_raw_parts::<KeyA::Item>(&key_a.handle, key_a.len, 1),
                ArrayArg::from_raw_parts::<KeyB::Item>(&key_b.handle, key_b.len, 1),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, key_a.len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }

        let control = segmented::SegmentControl::from_end_flags(
            key_a.policy(),
            key_a.len,
            len_u32,
            flag_handle,
            key_a.handle.clone(),
        )?;
        let left = control.compact_first::<KeyA::Runtime, KeyA::Item>(key_a.policy())?;
        let right = control
            .compact_value::<KeyA::Runtime, KeyB::Item>(key_b.policy(), key_b.handle.clone())?;
        let source = control.compact_value::<KeyA::Runtime, ValueSource::Item>(
            values.policy(),
            values.handle.clone(),
        )?;

        Ok((SoA2 { left, right }, SoA1 { source }))
    }
}

impl<KeyA, KeyB, ValueA, ValueB, Eq> UniqueByKeyInput<SoA2<ValueA, ValueB>, Eq>
    for SoAView2<KeyA, KeyB>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item), Scalar = KeyA::Item>,
    SoA2<ValueA, ValueB>: SoA,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: StorageKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: StorageKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    Eq: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
{
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA2<DeviceVec<KeyA::Runtime, ValueA::Item>, DeviceVec<KeyA::Runtime, ValueB::Item>>,
    );

    fn unique_by_key_input(
        self,
        values: SoA2<ValueA, ValueB>,
        _eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let value_a = super::device_expr_collect(&values.left)?;
        let value_b = super::device_expr_collect(&values.right)?;
        super::ensure_same_len(key_b.len, key_a.len)?;
        super::ensure_same_len(value_a.len, key_a.len)?;
        super::ensure_same_len(value_b.len, key_a.len)?;
        if key_a.len == 0 {
            return Ok((
                SoA2 {
                    left: DeviceVec::empty(key_a.policy.clone()),
                    right: DeviceVec::empty(key_b.policy.clone()),
                },
                SoA2 {
                    left: DeviceVec::empty(value_a.policy.clone()),
                    right: DeviceVec::empty(value_b.policy.clone()),
                },
            ));
        }

        let len_u32 =
            u32::try_from(key_a.len).map_err(|_| Error::LengthTooLarge { len: key_a.len })?;
        let client = key_a.policy.client();
        let block_count_u32 = sequence_block_count(key_a.len)?;
        let flag_handle = client.empty(key_a.len * std::mem::size_of::<u32>());

        unsafe {
            tuple2_unique_flags_kernel::launch_unchecked::<
                KeyA::Item,
                KeyB::Item,
                Eq,
                KeyA::Runtime,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
                ArrayArg::from_raw_parts::<KeyA::Item>(&key_a.handle, key_a.len, 1),
                ArrayArg::from_raw_parts::<KeyB::Item>(&key_b.handle, key_b.len, 1),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, key_a.len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }

        let control = segmented::SegmentControl::from_end_flags(
            key_a.policy(),
            key_a.len,
            len_u32,
            flag_handle,
            key_a.handle.clone(),
        )?;
        let left = control.compact_first::<KeyA::Runtime, KeyA::Item>(key_a.policy())?;
        let right = control
            .compact_value::<KeyA::Runtime, KeyB::Item>(key_b.policy(), key_b.handle.clone())?;
        let value_a = control.compact_value::<KeyA::Runtime, ValueA::Item>(
            value_a.policy(),
            value_a.handle.clone(),
        )?;
        let value_b = control.compact_value::<KeyA::Runtime, ValueB::Item>(
            value_b.policy(),
            value_b.handle.clone(),
        )?;

        Ok((
            SoA2 { left, right },
            SoA2 {
                left: value_a,
                right: value_b,
            },
        ))
    }
}

impl<KeyA, KeyB, ValueA, ValueB, ValueC, Eq> UniqueByKeyInput<SoA3<ValueA, ValueB, ValueC>, Eq>
    for SoAView2<KeyA, KeyB>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item), Scalar = KeyA::Item>,
    SoA3<ValueA, ValueB, ValueC>: SoA,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: StorageKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: StorageKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueC: StorageKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    ValueC::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
    Eq: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
{
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA3<
            DeviceVec<KeyA::Runtime, ValueA::Item>,
            DeviceVec<KeyA::Runtime, ValueB::Item>,
            DeviceVec<KeyA::Runtime, ValueC::Item>,
        >,
    );

    fn unique_by_key_input(
        self,
        values: SoA3<ValueA, ValueB, ValueC>,
        _eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let value_a = super::device_expr_collect(&values.first)?;
        let value_b = super::device_expr_collect(&values.second)?;
        let value_c = super::device_expr_collect(&values.third)?;
        super::ensure_same_len(key_b.len, key_a.len)?;
        super::ensure_same_len(value_a.len, key_a.len)?;
        super::ensure_same_len(value_b.len, key_a.len)?;
        super::ensure_same_len(value_c.len, key_a.len)?;
        if key_a.len == 0 {
            return Ok((
                SoA2 {
                    left: DeviceVec::empty(key_a.policy.clone()),
                    right: DeviceVec::empty(key_b.policy.clone()),
                },
                SoA3 {
                    first: DeviceVec::empty(value_a.policy.clone()),
                    second: DeviceVec::empty(value_b.policy.clone()),
                    third: DeviceVec::empty(value_c.policy.clone()),
                },
            ));
        }

        let len_u32 =
            u32::try_from(key_a.len).map_err(|_| Error::LengthTooLarge { len: key_a.len })?;
        let client = key_a.policy.client();
        let block_count_u32 = sequence_block_count(key_a.len)?;
        let flag_handle = client.empty(key_a.len * std::mem::size_of::<u32>());

        unsafe {
            tuple2_unique_flags_kernel::launch_unchecked::<
                KeyA::Item,
                KeyB::Item,
                Eq,
                KeyA::Runtime,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
                ArrayArg::from_raw_parts::<KeyA::Item>(&key_a.handle, key_a.len, 1),
                ArrayArg::from_raw_parts::<KeyB::Item>(&key_b.handle, key_b.len, 1),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, key_a.len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }

        let control = segmented::SegmentControl::from_end_flags(
            key_a.policy(),
            key_a.len,
            len_u32,
            flag_handle,
            key_a.handle.clone(),
        )?;
        let left = control.compact_first::<KeyA::Runtime, KeyA::Item>(key_a.policy())?;
        let right = control
            .compact_value::<KeyA::Runtime, KeyB::Item>(key_b.policy(), key_b.handle.clone())?;
        let value_a = control.compact_value::<KeyA::Runtime, ValueA::Item>(
            value_a.policy(),
            value_a.handle.clone(),
        )?;
        let value_b = control.compact_value::<KeyA::Runtime, ValueB::Item>(
            value_b.policy(),
            value_b.handle.clone(),
        )?;
        let value_c = control.compact_value::<KeyA::Runtime, ValueC::Item>(
            value_c.policy(),
            value_c.handle.clone(),
        )?;

        Ok((
            SoA2 { left, right },
            SoA3 {
                first: value_a,
                second: value_b,
                third: value_c,
            },
        ))
    }
}

impl<KeyA, KeyB, KeyC, ValueSource, Eq> UniqueByKeyInput<ValueSource, Eq>
    for SoAView3<KeyA, KeyB, KeyC>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item, KeyC::Item), Scalar = KeyA::Item>,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyC: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueSource: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    KeyC::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    KeyC::Expr: DeviceGpuExpr<KeyC::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Eq: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
{
    type Output = (
        SoA3<
            DeviceVec<KeyA::Runtime, KeyA::Item>,
            DeviceVec<KeyA::Runtime, KeyB::Item>,
            DeviceVec<KeyA::Runtime, KeyC::Item>,
        >,
        SoA1<DeviceVec<KeyA::Runtime, ValueSource::Item>>,
    );

    fn unique_by_key_input(
        self,
        values: ValueSource,
        _eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let values = SoAView1 { source: values };
        ReadOnlySoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let values = super::device_expr_collect(&values.source)?;
        super::ensure_same_len(key_b.len, key_a.len)?;
        super::ensure_same_len(key_c.len, key_a.len)?;
        super::ensure_same_len(values.len, key_a.len)?;
        if key_a.len == 0 {
            return Ok((
                SoA3 {
                    first: DeviceVec::empty(key_a.policy.clone()),
                    second: DeviceVec::empty(key_b.policy.clone()),
                    third: DeviceVec::empty(key_c.policy.clone()),
                },
                SoA1 {
                    source: DeviceVec::empty(values.policy.clone()),
                },
            ));
        }

        let len_u32 =
            u32::try_from(key_a.len).map_err(|_| Error::LengthTooLarge { len: key_a.len })?;
        let client = key_a.policy.client();
        let block_count_u32 = sequence_block_count(key_a.len)?;
        let flag_handle = client.empty(key_a.len * std::mem::size_of::<u32>());

        unsafe {
            tuple3_unique_flags_kernel::launch_unchecked::<
                KeyA::Item,
                KeyB::Item,
                KeyC::Item,
                Eq,
                KeyA::Runtime,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
                ArrayArg::from_raw_parts::<KeyA::Item>(&key_a.handle, key_a.len, 1),
                ArrayArg::from_raw_parts::<KeyB::Item>(&key_b.handle, key_b.len, 1),
                ArrayArg::from_raw_parts::<KeyC::Item>(&key_c.handle, key_c.len, 1),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, key_a.len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }

        let control = segmented::SegmentControl::from_end_flags(
            key_a.policy(),
            key_a.len,
            len_u32,
            flag_handle,
            key_a.handle.clone(),
        )?;
        let first = control.compact_first::<KeyA::Runtime, KeyA::Item>(key_a.policy())?;
        let second = control
            .compact_value::<KeyA::Runtime, KeyB::Item>(key_b.policy(), key_b.handle.clone())?;
        let third = control
            .compact_value::<KeyA::Runtime, KeyC::Item>(key_c.policy(), key_c.handle.clone())?;
        let source = control.compact_value::<KeyA::Runtime, ValueSource::Item>(
            values.policy(),
            values.handle.clone(),
        )?;

        Ok((
            SoA3 {
                first,
                second,
                third,
            },
            SoA1 { source },
        ))
    }
}

impl<KeyA, KeyB, KeyC, ValueA, ValueB, Eq> UniqueByKeyInput<SoA2<ValueA, ValueB>, Eq>
    for SoAView3<KeyA, KeyB, KeyC>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item, KeyC::Item), Scalar = KeyA::Item>,
    SoA2<ValueA, ValueB>: SoA,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyC: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: StorageKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: StorageKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    KeyC::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    KeyC::Expr: DeviceGpuExpr<KeyC::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    Eq: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
{
    type Output = (
        SoA3<
            DeviceVec<KeyA::Runtime, KeyA::Item>,
            DeviceVec<KeyA::Runtime, KeyB::Item>,
            DeviceVec<KeyA::Runtime, KeyC::Item>,
        >,
        SoA2<DeviceVec<KeyA::Runtime, ValueA::Item>, DeviceVec<KeyA::Runtime, ValueB::Item>>,
    );

    fn unique_by_key_input(
        self,
        values: SoA2<ValueA, ValueB>,
        _eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let value_a = super::device_expr_collect(&values.left)?;
        let value_b = super::device_expr_collect(&values.right)?;
        super::ensure_same_len(key_b.len, key_a.len)?;
        super::ensure_same_len(key_c.len, key_a.len)?;
        super::ensure_same_len(value_a.len, key_a.len)?;
        super::ensure_same_len(value_b.len, key_a.len)?;
        if key_a.len == 0 {
            return Ok((
                SoA3 {
                    first: DeviceVec::empty(key_a.policy.clone()),
                    second: DeviceVec::empty(key_b.policy.clone()),
                    third: DeviceVec::empty(key_c.policy.clone()),
                },
                SoA2 {
                    left: DeviceVec::empty(value_a.policy.clone()),
                    right: DeviceVec::empty(value_b.policy.clone()),
                },
            ));
        }

        let len_u32 =
            u32::try_from(key_a.len).map_err(|_| Error::LengthTooLarge { len: key_a.len })?;
        let client = key_a.policy.client();
        let block_count_u32 = sequence_block_count(key_a.len)?;
        let flag_handle = client.empty(key_a.len * std::mem::size_of::<u32>());

        unsafe {
            tuple3_unique_flags_kernel::launch_unchecked::<
                KeyA::Item,
                KeyB::Item,
                KeyC::Item,
                Eq,
                KeyA::Runtime,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
                ArrayArg::from_raw_parts::<KeyA::Item>(&key_a.handle, key_a.len, 1),
                ArrayArg::from_raw_parts::<KeyB::Item>(&key_b.handle, key_b.len, 1),
                ArrayArg::from_raw_parts::<KeyC::Item>(&key_c.handle, key_c.len, 1),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, key_a.len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }

        let control = segmented::SegmentControl::from_end_flags(
            key_a.policy(),
            key_a.len,
            len_u32,
            flag_handle,
            key_a.handle.clone(),
        )?;
        let first = control.compact_first::<KeyA::Runtime, KeyA::Item>(key_a.policy())?;
        let second = control
            .compact_value::<KeyA::Runtime, KeyB::Item>(key_b.policy(), key_b.handle.clone())?;
        let third = control
            .compact_value::<KeyA::Runtime, KeyC::Item>(key_c.policy(), key_c.handle.clone())?;
        let value_a = control.compact_value::<KeyA::Runtime, ValueA::Item>(
            value_a.policy(),
            value_a.handle.clone(),
        )?;
        let value_b = control.compact_value::<KeyA::Runtime, ValueB::Item>(
            value_b.policy(),
            value_b.handle.clone(),
        )?;

        Ok((
            SoA3 {
                first,
                second,
                third,
            },
            SoA2 {
                left: value_a,
                right: value_b,
            },
        ))
    }
}

impl<KeyA, KeyB, KeyC, ValueA, ValueB, ValueC, Eq>
    UniqueByKeyInput<SoA3<ValueA, ValueB, ValueC>, Eq> for SoAView3<KeyA, KeyB, KeyC>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item, KeyC::Item), Scalar = KeyA::Item>,
    SoA3<ValueA, ValueB, ValueC>: SoA,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyC: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: StorageKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: StorageKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueC: StorageKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    KeyC::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    ValueC::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    KeyC::Expr: DeviceGpuExpr<KeyC::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
    Eq: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
{
    type Output = (
        SoA3<
            DeviceVec<KeyA::Runtime, KeyA::Item>,
            DeviceVec<KeyA::Runtime, KeyB::Item>,
            DeviceVec<KeyA::Runtime, KeyC::Item>,
        >,
        SoA3<
            DeviceVec<KeyA::Runtime, ValueA::Item>,
            DeviceVec<KeyA::Runtime, ValueB::Item>,
            DeviceVec<KeyA::Runtime, ValueC::Item>,
        >,
    );

    fn unique_by_key_input(
        self,
        values: SoA3<ValueA, ValueB, ValueC>,
        _eq: GpuOp<Eq>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let value_a = super::device_expr_collect(&values.first)?;
        let value_b = super::device_expr_collect(&values.second)?;
        let value_c = super::device_expr_collect(&values.third)?;
        super::ensure_same_len(key_b.len, key_a.len)?;
        super::ensure_same_len(key_c.len, key_a.len)?;
        super::ensure_same_len(value_a.len, key_a.len)?;
        super::ensure_same_len(value_b.len, key_a.len)?;
        super::ensure_same_len(value_c.len, key_a.len)?;
        if key_a.len == 0 {
            return Ok((
                SoA3 {
                    first: DeviceVec::empty(key_a.policy.clone()),
                    second: DeviceVec::empty(key_b.policy.clone()),
                    third: DeviceVec::empty(key_c.policy.clone()),
                },
                SoA3 {
                    first: DeviceVec::empty(value_a.policy.clone()),
                    second: DeviceVec::empty(value_b.policy.clone()),
                    third: DeviceVec::empty(value_c.policy.clone()),
                },
            ));
        }

        let len_u32 =
            u32::try_from(key_a.len).map_err(|_| Error::LengthTooLarge { len: key_a.len })?;
        let client = key_a.policy.client();
        let block_count_u32 = sequence_block_count(key_a.len)?;
        let flag_handle = client.empty(key_a.len * std::mem::size_of::<u32>());

        unsafe {
            tuple3_unique_flags_kernel::launch_unchecked::<
                KeyA::Item,
                KeyB::Item,
                KeyC::Item,
                Eq,
                KeyA::Runtime,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
                ArrayArg::from_raw_parts::<KeyA::Item>(&key_a.handle, key_a.len, 1),
                ArrayArg::from_raw_parts::<KeyB::Item>(&key_b.handle, key_b.len, 1),
                ArrayArg::from_raw_parts::<KeyC::Item>(&key_c.handle, key_c.len, 1),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, key_a.len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }

        let control = segmented::SegmentControl::from_end_flags(
            key_a.policy(),
            key_a.len,
            len_u32,
            flag_handle,
            key_a.handle.clone(),
        )?;
        let first = control.compact_first::<KeyA::Runtime, KeyA::Item>(key_a.policy())?;
        let second = control
            .compact_value::<KeyA::Runtime, KeyB::Item>(key_b.policy(), key_b.handle.clone())?;
        let third = control
            .compact_value::<KeyA::Runtime, KeyC::Item>(key_c.policy(), key_c.handle.clone())?;
        let value_a = control.compact_value::<KeyA::Runtime, ValueA::Item>(
            value_a.policy(),
            value_a.handle.clone(),
        )?;
        let value_b = control.compact_value::<KeyA::Runtime, ValueB::Item>(
            value_b.policy(),
            value_b.handle.clone(),
        )?;
        let value_c = control.compact_value::<KeyA::Runtime, ValueC::Item>(
            value_c.policy(),
            value_c.handle.clone(),
        )?;

        Ok((
            SoA3 {
                first,
                second,
                third,
            },
            SoA3 {
                first: value_a,
                second: value_b,
                third: value_c,
            },
        ))
    }
}

macro_rules! impl_unique_by_key_key_forward {
    ($name:ident < $first:ident, $( $rest:ident ),+ >) => {
        impl<KeySource, $first, $( $rest ),+, Eq> UniqueByKeyInput<$name<$first, $( $rest ),+>, Eq>
            for KeySource
        where
            KeySource: StorageKernelColumn + KernelColumnAt<S0>,
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
            KeySource: StorageKernelColumn + KernelColumnAt<S0>,
            $first: StorageKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
            $(
                $rest: StorageKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
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
                let control = segmented::key_run_control::<_, _, Eq>(&keys)?;
                let out_keys = control.compact_first::<
                    KeySource::Runtime,
                    KeySource::Item,
                >(keys.policy())?;
                let $first_field = control.compact_value::<
                    KeySource::Runtime,
                    <$first as KernelColumn>::Item,
                >($first_field.policy(), $first_field.handle.clone())?;
                $(
                    let $field = super::device_expr_collect(&values.$field)?;
                    let $field = control.compact_value::<
                        KeySource::Runtime,
                        <$rest as KernelColumn>::Item,
                    >(
                        $field.policy(),
                        $field.handle.clone(),
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
    Source: StorageKernelColumn + KernelColumnAt<S0>,
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
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn unique_input(self, pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
        <SoA1<Source> as UniqueInput<Pred>>::unique_input(SoA1 { source: self }, pred)
    }
}

macro_rules! impl_unique_tuple {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ },
        $kernel_name:ident
    ) => {
        impl<$first, $( $rest ),+, Pred> UniqueInput<Pred> for $name<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: StorageKernelColumn + KernelColumnAt<S0>,
            $(
                $rest: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Pred: BinaryPredicateOp<(
                impl_unique_tuple!(@item_ty $first),
                $( impl_unique_tuple!(@item_ty $rest) ),+
            )>,
        {
            type Output = $name<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$first as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn unique_input(self, _pred: GpuOp<Pred>) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+

                let len = $first_field.len();
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let client = $first_field.policy().client();
                let flag = client.empty(len * std::mem::size_of::<u32>());

                if len != 0 {
                    let block_count_u32 = sequence_block_count(len)?;
                    unsafe {
                        $kernel_name::launch_unchecked::<
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item, )+
                            Pred,
                            <$first as KernelColumn>::Runtime,
                        >(
                            client,
                            CubeCount::Static(block_count_u32, 1, 1),
                            CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
                            ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(
                                &$first_field.handle,
                                len,
                                1,
                            ),
                            $(
                                ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(
                                    &$field.handle,
                                    len,
                                    1,
                                ),
                            )+
                            ArrayArg::from_raw_parts::<u32>(&flag, len, 1),
                        )
                        .map_err(|err| Error::Launch {
                            message: format!("{err:?}"),
                        })?;
                    }
                }

                let handles = select::handles_from_flags(
                    $first_field.policy(),
                    len,
                    len_u32,
                    flag,
                    $first_field.handle.clone(),
                )?;
                let count = select::selected_count($first_field.policy(), &handles)?;

                let $first_field = select::compact_with_count::<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                >($first_field.policy(), handles.clone(), count)?;
                $(
                    let $field = select::compact_with_count::<
                        <$first as KernelColumn>::Runtime,
                        <$rest as KernelColumn>::Item,
                    >(
                        $field.policy(),
                        select::handles_for_value(&handles, $field.handle.clone()),
                        count,
                    )?;
                )+

                Ok($name { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_unique_tuple!(SoA2<A, B> { left, right }, tuple2_unique_flags_kernel);
impl_unique_tuple!(SoA3<A, B, C> { first, second, third }, tuple3_unique_flags_kernel);
impl_unique_tuple!(SoA4<A, B, C, D> { a, b, c, d }, tuple4_unique_flags_kernel);
impl_unique_tuple!(SoA5<A, B, C, D, E> { a, b, c, d, e }, tuple5_unique_flags_kernel);
impl_unique_tuple!(SoA6<A, B, C, D, E, F> { a, b, c, d, e, f }, tuple6_unique_flags_kernel);
impl_unique_tuple!(SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g }, tuple7_unique_flags_kernel);
impl_unique_tuple!(SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h }, tuple8_unique_flags_kernel);
impl_unique_tuple!(SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i }, tuple9_unique_flags_kernel);
impl_unique_tuple!(SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j }, tuple10_unique_flags_kernel);
impl_unique_tuple!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k }, tuple11_unique_flags_kernel);
impl_unique_tuple!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l }, tuple12_unique_flags_kernel);

fn replace_with_flags_device_vec<R, T>(
    input: &DeviceVec<R, T>,
    replacement: T,
    flag: &cubecl::server::Handle,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    u32::try_from(input.len).map_err(|_| Error::LengthTooLarge { len: input.len })?;
    if input.len == 0 {
        return Ok(DeviceVec::empty(input.policy.clone()));
    }

    let client = input.policy.client();
    let output_handle = client.empty(input.len * std::mem::size_of::<T>());

    let block_count_u32 = sequence_block_count(input.len)?;
    let replacement_values = [replacement];
    let replacement_handle = client.create_from_slice(T::as_bytes(&replacement_values));

    unsafe {
        replace_with_flags_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SEQUENCE_SIZE),
            ArrayArg::from_raw_parts::<T>(&input.handle, input.len, 1),
            ArrayArg::from_raw_parts::<T>(&replacement_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(flag, input.len, 1),
            ArrayArg::from_raw_parts::<T>(&output_handle, input.len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    Ok(DeviceVec::from_handle(
        input.policy.clone(),
        output_handle,
        input.len,
    ))
}

/// Replaces elements whose stencil satisfies `Pred`.
pub fn replace_if<Input, Stencil, Pred>(
    input: Input,
    replacement: <Input as ReplaceIfInput<Stencil, Pred>>::Item,
    stencil: Stencil,
    _pred: Pred,
) -> Result<<<Input as ReplaceIfInput<Stencil, Pred>>::Output as MaterializeOutput>::Output, Error>
where
    Input: ReplaceIfInput<Stencil, Pred>,
    <Input as ReplaceIfInput<Stencil, Pred>>::Output: MaterializeOutput,
{
    materialize(input.replace_if_input(replacement, stencil, GpuOp::<Pred>::new())?)
}

/// Removes consecutive duplicates.
pub fn unique<Input, Pred>(
    input: Input,
    _pred: Pred,
) -> Result<<<Input as UniqueInput<Pred>>::Output as MaterializeOutput>::Output, Error>
where
    Input: UniqueInput<Pred>,
    <Input as UniqueInput<Pred>>::Output: MaterializeOutput,
{
    materialize(input.unique_input(GpuOp::<Pred>::new())?)
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
    super::ensure_same_len(values.len, keys.len)?;
    if keys.len == 0 {
        return Ok((
            DeviceVec::empty(keys.policy.clone()),
            DeviceVec::empty(values.policy.clone()),
        ));
    }

    let control = segmented::key_run_control::<R, K, Eq>(keys)?;
    let out_keys = control.compact_first::<R, K>(keys.policy())?;
    let out_values = control.compact_value::<R, T>(values.policy(), values.handle.clone())?;

    Ok((out_keys, out_values))
}

/// Removes consecutive duplicate keys and carries the first value for each key.
pub fn unique_by_key<Keys, Values, Eq>(
    keys: Keys,
    values: Values,
    _eq: Eq,
) -> Result<<<Keys as UniqueByKeyInput<Values, Eq>>::Output as MaterializeOutput>::Output, Error>
where
    Keys: UniqueByKeyInput<Values, Eq>,
    <Keys as UniqueByKeyInput<Values, Eq>>::Output: MaterializeOutput,
{
    materialize(keys.unique_by_key_input(values, GpuOp::<Eq>::new())?)
}
