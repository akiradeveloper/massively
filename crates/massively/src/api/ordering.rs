use super::memory::{MaterializeOutput, materialize};
use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlyKernelColumn, ReadOnlySoA, S0, SoA, SoA1,
        SoA2, SoA3, SoA4, SoA5, SoA6, SoA7, SoA8, SoA9, SoA10, SoA11, SoA12, SoAView1, SoAView2,
        SoAView3, SoAView4, SoAView5, SoAView6, SoAView7, SoAView8, SoAView9, SoAView10, SoAView11,
        SoAView12,
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::{BinaryPredicateOp, GpuOp},
    primitives::{ordering, range as primitive_range, select},
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

fn materialize_soa_view_one<Source>(
    input: SoAView1<Source>,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    SoAView1<Source>: ReadOnlySoA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    ReadOnlySoA::validate(&input)?;
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

impl<Left, Right, Less> PairOrderingInput<SoAView1<Right>, Less> for SoAView1<Left>
where
    Self: ReadOnlySoA<Item = Left::Item, Scalar = Left::Item>,
    SoAView1<Right>: ReadOnlySoA<Item = Right::Item, Scalar = Right::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<Left::Item>,
{
    type Output = SoA1<DeviceVec<Left::Runtime, Left::Item>>;

    fn merge_input(
        self,
        other: SoAView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_soa_view_one(self)?;
        let right = materialize_soa_view_one(other)?;
        Ok(SoA1 {
            source: ordering::merge(&left, &right, GpuOp::<Less>::new())?,
        })
    }

    fn set_union_input(
        self,
        other: SoAView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_soa_view_one(self)?;
        let right = materialize_soa_view_one(other)?;
        Ok(SoA1 {
            source: ordering::set_union(&left, &right, GpuOp::<Less>::new())?,
        })
    }

    fn set_intersection_input(
        self,
        other: SoAView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_soa_view_one(self)?;
        let right = materialize_soa_view_one(other)?;
        Ok(SoA1 {
            source: ordering::set_intersection(&left, &right, GpuOp::<Less>::new())?,
        })
    }

    fn set_difference_input(
        self,
        other: SoAView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_soa_view_one(self)?;
        let right = materialize_soa_view_one(other)?;
        Ok(SoA1 {
            source: ordering::set_difference(&left, &right, GpuOp::<Less>::new())?,
        })
    }

    fn set_symmetric_difference_input(
        self,
        other: SoAView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_soa_view_one(self)?;
        let right = materialize_soa_view_one(other)?;
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
        <SoAView1<Left> as PairOrderingInput<SoAView1<Right>, Less>>::merge_input(
            SoAView1 { source: self },
            SoAView1 { source: other },
            less,
        )
    }

    fn set_union_input(self, other: Right, less: GpuOp<Less>) -> Result<Self::Output, Error> {
        <SoAView1<Left> as PairOrderingInput<SoAView1<Right>, Less>>::set_union_input(
            SoAView1 { source: self },
            SoAView1 { source: other },
            less,
        )
    }

    fn set_intersection_input(
        self,
        other: Right,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Left> as PairOrderingInput<SoAView1<Right>, Less>>::set_intersection_input(
            SoAView1 { source: self },
            SoAView1 { source: other },
            less,
        )
    }

    fn set_difference_input(self, other: Right, less: GpuOp<Less>) -> Result<Self::Output, Error> {
        <SoAView1<Left> as PairOrderingInput<SoAView1<Right>, Less>>::set_difference_input(
            SoAView1 { source: self },
            SoAView1 { source: other },
            less,
        )
    }

    fn set_symmetric_difference_input(
        self,
        other: Right,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Left> as PairOrderingInput<SoAView1<Right>, Less>>::set_symmetric_difference_input(
            SoAView1 { source: self },
            SoAView1 { source: other },
            less,
        )
    }
}

macro_rules! tuple_membership_handles {
    (
        $kernel_name:ident,
        ($first_item_ty:ty, $( $item_ty:ty ),+),
        $runtime_ty:ty,
        $less_ty:ty,
        ($first_candidate:ident, $( $candidate:ident ),+),
        ($first_sorted:ident, $( $sorted:ident ),+),
        $keep_present:expr
    ) => {{
        let len = $first_candidate.len();
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let client = $first_candidate.policy().client();
        let flag = client.empty(len * std::mem::size_of::<u32>());

        if len != 0 {
            let block_count = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            let keep = [if $keep_present { 1_u32 } else { 0_u32 }];
            let keep_handle = client.create_from_slice(u32::as_bytes(&keep));
            unsafe {
                $kernel_name::launch_unchecked::<
                    $first_item_ty,
                    $( $item_ty, )+
                    $less_ty,
                    $runtime_ty,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                    ArrayArg::from_raw_parts::<$first_item_ty>(
                        &$first_candidate.handle,
                        len,
                        1,
                    ),
                    $(
                        ArrayArg::from_raw_parts::<$item_ty>(
                            &$candidate.handle,
                            len,
                            1,
                        ),
                    )+
                    ArrayArg::from_raw_parts::<$first_item_ty>(
                        &$first_sorted.handle,
                        $first_sorted.len(),
                        1,
                    ),
                    $(
                        ArrayArg::from_raw_parts::<$item_ty>(
                            &$sorted.handle,
                            $sorted.len(),
                            1,
                        ),
                    )+
                    ArrayArg::from_raw_parts::<u32>(&keep_handle, 1, 1),
                    ArrayArg::from_raw_parts::<u32>(&flag, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }

        select::handles_from_flags(
            $first_candidate.policy(),
            len,
            len_u32,
            flag,
            $first_candidate.handle.clone(),
        )
    }};
}

macro_rules! compact_tuple_from_handles {
    (
        $name:ident,
        $runtime_ty:ty,
        $handles:ident,
        $count:ident,
        ($first_item_ty:ty, $( $item_ty:ty ),+),
        { $first_output_field:ident : $first_source:ident, $( $output_field:ident : $source:ident ),+ }
    ) => {{
        let $first_source = select::compact_with_count::<$runtime_ty, $first_item_ty>(
            $first_source.policy(),
            $handles.clone(),
            $count,
        )?;
        $(
            let $source = select::compact_with_count::<$runtime_ty, $item_ty>(
                $source.policy(),
                select::handles_for_value(&$handles, $source.handle.clone()),
                $count,
            )?;
        )+
        $name { $first_output_field: $first_source, $( $output_field: $source ),+ }
    }};
}

macro_rules! impl_tuple_pair_ordering {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ ; $right_first_ty:ident, $( $right_rest_ty:ident ),+ >
        { $first_field:ident / $right_first_var:ident, $( $field:ident / $right_var:ident ),+ },
        $sort_fn:ident,
        $membership_kernel:ident
    ) => {
        impl<$first, $( $rest ),+, $right_first_ty, $( $right_rest_ty ),+, Less>
            PairOrderingInput<$input<$right_first_ty, $( $right_rest_ty ),+>, Less>
            for $input<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $input<$right_first_ty, $( $right_rest_ty ),+>: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $right_first_ty:
                KernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = <$first as KernelColumn>::Item>
                + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
                $right_rest_ty:
                    KernelColumn<Runtime = <$first as KernelColumn>::Runtime, Item = <$rest as KernelColumn>::Item>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            <$right_first_ty as KernelColumn>::Expr: DeviceGpuExpr<<$right_first_ty as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
                <$right_rest_ty as KernelColumn>::Expr: DeviceGpuExpr<<$right_rest_ty as KernelColumn>::Item>,
            )+
            Less: BinaryPredicateOp<(
                impl_tuple_pair_ordering!(@item_ty $first),
                $( impl_tuple_pair_ordering!(@item_ty $rest) ),+
            )>,
        {
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$first as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn merge_input(
                self,
                other: $input<$right_first_ty, $( $right_rest_ty ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let $right_first_var = super::device_expr_collect(&other.$first_field)?;
                $(
                    let $right_var = super::device_expr_collect(&other.$field)?;
                )+
                let $first_field = primitive_range::concat_device(&$first_field, &$right_first_var)?;
                $(
                    let $field = primitive_range::concat_device(&$field, &$right_var)?;
                )+
                let ($first_field, $( $field ),+) = ordering::$sort_fn(
                    &$first_field,
                    $( &$field, )+
                    GpuOp::<Less>::new(),
                )?;
                Ok($output { $first_field, $( $field ),+ })
            }

            fn set_union_input(
                self,
                other: $input<$right_first_ty, $( $right_rest_ty ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let $right_first_var = super::device_expr_collect(&other.$first_field)?;
                $(
                    let $right_var = super::device_expr_collect(&other.$field)?;
                )+
                let handles = tuple_membership_handles!(
                    $membership_kernel,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    <$first as KernelColumn>::Runtime,
                    Less,
                    ($right_first_var, $( $right_var ),+),
                    ($first_field, $( $field ),+),
                    false
                )?;
                let count = select::selected_count($right_first_var.policy(), &handles)?;
                let right_only = compact_tuple_from_handles!(
                    $output,
                    <$first as KernelColumn>::Runtime,
                    handles,
                    count,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    { $first_field: $right_first_var, $( $field: $right_var ),+ }
                );
                let $output { $first_field: $right_first_var, $( $field: $right_var ),+ } = right_only;
                let $first_field = primitive_range::concat_device(&$first_field, &$right_first_var)?;
                $(
                    let $field = primitive_range::concat_device(&$field, &$right_var)?;
                )+
                let ($first_field, $( $field ),+) = ordering::$sort_fn(
                    &$first_field,
                    $( &$field, )+
                    GpuOp::<Less>::new(),
                )?;
                Ok($output { $first_field, $( $field ),+ })
            }

            fn set_intersection_input(
                self,
                other: $input<$right_first_ty, $( $right_rest_ty ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let $right_first_var = super::device_expr_collect(&other.$first_field)?;
                $(
                    let $right_var = super::device_expr_collect(&other.$field)?;
                )+
                let handles = tuple_membership_handles!(
                    $membership_kernel,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    <$first as KernelColumn>::Runtime,
                    Less,
                    ($first_field, $( $field ),+),
                    ($right_first_var, $( $right_var ),+),
                    true
                )?;
                let count = select::selected_count($first_field.policy(), &handles)?;
                Ok(compact_tuple_from_handles!(
                    $output,
                    <$first as KernelColumn>::Runtime,
                    handles,
                    count,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    { $first_field: $first_field, $( $field: $field ),+ }
                ))
            }

            fn set_difference_input(
                self,
                other: $input<$right_first_ty, $( $right_rest_ty ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let $right_first_var = super::device_expr_collect(&other.$first_field)?;
                $(
                    let $right_var = super::device_expr_collect(&other.$field)?;
                )+
                let handles = tuple_membership_handles!(
                    $membership_kernel,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    <$first as KernelColumn>::Runtime,
                    Less,
                    ($first_field, $( $field ),+),
                    ($right_first_var, $( $right_var ),+),
                    false
                )?;
                let count = select::selected_count($first_field.policy(), &handles)?;
                Ok(compact_tuple_from_handles!(
                    $output,
                    <$first as KernelColumn>::Runtime,
                    handles,
                    count,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    { $first_field: $first_field, $( $field: $field ),+ }
                ))
            }

            fn set_symmetric_difference_input(
                self,
                other: $input<$right_first_ty, $( $right_rest_ty ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                let $right_first_var = super::device_expr_collect(&other.$first_field)?;
                $(
                    let $right_var = super::device_expr_collect(&other.$field)?;
                )+
                let left_handles = tuple_membership_handles!(
                    $membership_kernel,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    <$first as KernelColumn>::Runtime,
                    Less,
                    ($first_field, $( $field ),+),
                    ($right_first_var, $( $right_var ),+),
                    false
                )?;
                let left_count = select::selected_count($first_field.policy(), &left_handles)?;
                let right_handles = tuple_membership_handles!(
                    $membership_kernel,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    <$first as KernelColumn>::Runtime,
                    Less,
                    ($right_first_var, $( $right_var ),+),
                    ($first_field, $( $field ),+),
                    false
                )?;
                let right_count = select::selected_count($right_first_var.policy(), &right_handles)?;
                let left_only = compact_tuple_from_handles!(
                    $output,
                    <$first as KernelColumn>::Runtime,
                    left_handles,
                    left_count,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    { $first_field: $first_field, $( $field: $field ),+ }
                );
                let right_only = compact_tuple_from_handles!(
                    $output,
                    <$first as KernelColumn>::Runtime,
                    right_handles,
                    right_count,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    { $first_field: $right_first_var, $( $field: $right_var ),+ }
                );
                let $output { $first_field, $( $field ),+ } = left_only;
                let $output { $first_field: $right_first_var, $( $field: $right_var ),+ } = right_only;
                let $first_field = primitive_range::concat_device(&$first_field, &$right_first_var)?;
                $(
                    let $field = primitive_range::concat_device(&$field, &$right_var)?;
                )+
                let ($first_field, $( $field ),+) = ordering::$sort_fn(
                    &$first_field,
                    $( &$field, )+
                    GpuOp::<Less>::new(),
                )?;
                Ok($output { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_tuple_pair_ordering!(SoAView2 -> SoA2<A, B; RA, RB> { left / right_left, right / right_right }, sort_tuple2, tuple2_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA2 -> SoA2<A, B; RA, RB> { left / right_left, right / right_right }, sort_tuple2, tuple2_membership_flags_kernel);
impl_tuple_pair_ordering!(SoAView3 -> SoA3<A, B, C; RA, RB, RC> { first / right_first, second / right_second, third / right_third }, sort_tuple3, tuple3_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA3 -> SoA3<A, B, C; RA, RB, RC> { first / right_first, second / right_second, third / right_third }, sort_tuple3, tuple3_membership_flags_kernel);
impl_tuple_pair_ordering!(SoAView4 -> SoA4<A, B, C, D; RA, RB, RC, RD> { a / right_a, b / right_b, c / right_c, d / right_d }, sort_tuple4, tuple4_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA4 -> SoA4<A, B, C, D; RA, RB, RC, RD> { a / right_a, b / right_b, c / right_c, d / right_d }, sort_tuple4, tuple4_membership_flags_kernel);
impl_tuple_pair_ordering!(SoAView5 -> SoA5<A, B, C, D, E; RA, RB, RC, RD, RE> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e }, sort_tuple5, tuple5_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA5 -> SoA5<A, B, C, D, E; RA, RB, RC, RD, RE> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e }, sort_tuple5, tuple5_membership_flags_kernel);
impl_tuple_pair_ordering!(SoAView6 -> SoA6<A, B, C, D, E, F; RA, RB, RC, RD, RE, RF> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f }, sort_tuple6, tuple6_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA6 -> SoA6<A, B, C, D, E, F; RA, RB, RC, RD, RE, RF> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f }, sort_tuple6, tuple6_membership_flags_kernel);
impl_tuple_pair_ordering!(SoAView7 -> SoA7<A, B, C, D, E, F, G; RA, RB, RC, RD, RE, RF, RG> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f, g / right_g }, sort_tuple7, tuple7_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA7 -> SoA7<A, B, C, D, E, F, G; RA, RB, RC, RD, RE, RF, RG> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f, g / right_g }, sort_tuple7, tuple7_membership_flags_kernel);
impl_tuple_pair_ordering!(SoAView8 -> SoA8<A, B, C, D, E, F, G, H; RA, RB, RC, RD, RE, RF, RG, RH> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f, g / right_g, h / right_h }, sort_tuple8, tuple8_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA8 -> SoA8<A, B, C, D, E, F, G, H; RA, RB, RC, RD, RE, RF, RG, RH> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f, g / right_g, h / right_h }, sort_tuple8, tuple8_membership_flags_kernel);
impl_tuple_pair_ordering!(SoAView9 -> SoA9<A, B, C, D, E, F, G, H, I; RA, RB, RC, RD, RE, RF, RG, RH, RI> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f, g / right_g, h / right_h, i / right_i }, sort_tuple9, tuple9_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA9 -> SoA9<A, B, C, D, E, F, G, H, I; RA, RB, RC, RD, RE, RF, RG, RH, RI> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f, g / right_g, h / right_h, i / right_i }, sort_tuple9, tuple9_membership_flags_kernel);
impl_tuple_pair_ordering!(SoAView10 -> SoA10<A, B, C, D, E, F, G, H, I, J; RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f, g / right_g, h / right_h, i / right_i, j / right_j }, sort_tuple10, tuple10_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J; RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f, g / right_g, h / right_h, i / right_i, j / right_j }, sort_tuple10, tuple10_membership_flags_kernel);
impl_tuple_pair_ordering!(SoAView11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K; RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ, RK> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f, g / right_g, h / right_h, i / right_i, j / right_j, k / right_k }, sort_tuple11, tuple11_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K; RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ, RK> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f, g / right_g, h / right_h, i / right_i, j / right_j, k / right_k }, sort_tuple11, tuple11_membership_flags_kernel);
impl_tuple_pair_ordering!(SoAView12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L; RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ, RK, RL> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f, g / right_g, h / right_h, i / right_i, j / right_j, k / right_k, l / right_l }, sort_tuple12, tuple12_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L; RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ, RK, RL> { a / right_a, b / right_b, c / right_c, d / right_d, e / right_e, f / right_f, g / right_g, h / right_h, i / right_i, j / right_j, k / right_k, l / right_l }, sort_tuple12, tuple12_membership_flags_kernel);

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
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn reverse_input(self) -> Result<Self::Output, Error> {
        <SoA1<Source> as ReverseInput>::reverse_input(SoA1 { source: self })
    }
}

macro_rules! impl_reverse_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+> ReverseInput for $name<$first, $( $rest ),+>
        where
            Self: SoA<Scalar = <$first as KernelColumn>::Item>,
            $first: ReadOnlyKernelColumn + KernelColumnAt<S0>,
            $(
                $rest: ReadOnlyKernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
        {
            type Output = $name<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$first as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn reverse_input(self) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+

                let $first_field = reverse_device_vec(&$first_field)?;
                $(
                    let $field = reverse_device_vec(&$field)?;
                )+

                Ok($name { $first_field, $( $field ),+ })
            }
        }
    };
}

impl_reverse_input!(SoA2<A, B> { left, right });
impl_reverse_input!(SoA3<A, B, C> { first, second, third });
impl_reverse_input!(SoA4<A, B, C, D> { a, b, c, d });
impl_reverse_input!(SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_reverse_input!(SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_reverse_input!(SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_reverse_input!(SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_reverse_input!(SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_reverse_input!(SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_reverse_input!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_reverse_input!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Reverses read-only SoA input and returns new device storage.
pub fn reverse<Input>(
    input: Input,
) -> Result<<<Input as ReverseInput>::Output as MaterializeOutput>::Output, Error>
where
    Input: ReverseInput,
    <Input as ReverseInput>::Output: MaterializeOutput,
{
    materialize(input.reverse_input()?)
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

impl<KeySource, ValueSource, Less> SortByKeyInput<SoA1<ValueSource>, Less> for SoAView1<KeySource>
where
    Self: ReadOnlySoA<Item = KeySource::Item, Scalar = KeySource::Item>,
    SoA1<ValueSource>: SoA<Item = ValueSource::Item, Scalar = ValueSource::Item>,
    KeySource: KernelColumn + KernelColumnAt<S0>,
    ValueSource: ReadOnlyKernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
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
        let keys = materialize_soa_view_one(self)?;
        let values = materialize_one(values)?;
        let (keys, values) = ordering::sort_by_key(&keys, &values, GpuOp::<Less>::new())?;
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
    type Output = (
        SoA1<DeviceVec<KeySource::Runtime, KeySource::Item>>,
        SoA1<DeviceVec<KeySource::Runtime, ValueSource::Item>>,
    );

    fn sort_by_key_input(
        self,
        values: ValueSource,
        op: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<KeySource> as SortByKeyInput<SoA1<ValueSource>, Less>>::sort_by_key_input(
            SoAView1 { source: self },
            SoA1 { source: values },
            op,
        )
    }
}

macro_rules! impl_sort_by_key_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<KeySource, $first, $( $rest ),+, Less> SortByKeyInput<$name<$first, $( $rest ),+>, Less>
            for SoAView1<KeySource>
        where
            Self: ReadOnlySoA<Item = KeySource::Item, Scalar = KeySource::Item>,
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
                ReadOnlySoA::validate(&self)?;
                SoA::validate(&values)?;
                let keys = super::device_expr_collect(&self.source)?;
                let (out_keys, permutation) =
                    ordering::sort_by_key_permutation(&keys, GpuOp::<Less>::new())?;
                let $first_field = super::device_expr_collect(&values.$first_field)?;
                let $first_field = primitive_range::gather_device(&$first_field, permutation.indices())?;
                $(
                    let $field = super::device_expr_collect(&values.$field)?;
                    let $field = primitive_range::gather_device(&$field, permutation.indices())?;
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
            SoAView1<KeySource>: SortByKeyInput<$name<$( $field_ty ),+>, Less>,
        {
            type Output = <SoAView1<KeySource> as SortByKeyInput<$name<$( $field_ty ),+>, Less>>::Output;

            fn sort_by_key_input(
                self,
                values: $name<$( $field_ty ),+>,
                less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                <SoAView1<KeySource> as SortByKeyInput<$name<$( $field_ty ),+>, Less>>::sort_by_key_input(
                    SoAView1 { source: self },
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

impl<KeyA, KeyB, ValueSource, Less> SortByKeyInput<ValueSource, Less> for SoAView2<KeyA, KeyB>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item), Scalar = KeyA::Item>,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueSource: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Less: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
{
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA1<DeviceVec<KeyA::Runtime, ValueSource::Item>>,
    );

    fn sort_by_key_input(
        self,
        values: ValueSource,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        values.validate()?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let values = super::device_expr_collect(&values)?;
        let (left, right, source) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &values, GpuOp::<Less>::new())?;
        Ok((SoA2 { left, right }, SoA1 { source }))
    }
}

impl<KeyA, KeyB, ValueA, ValueB, Less> SortByKeyInput<SoA2<ValueA, ValueB>, Less>
    for SoAView2<KeyA, KeyB>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item), Scalar = KeyA::Item>,
    SoA2<ValueA, ValueB>: SoA,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    Less: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
{
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA2<DeviceVec<KeyA::Runtime, ValueA::Item>, DeviceVec<KeyA::Runtime, ValueB::Item>>,
    );

    fn sort_by_key_input(
        self,
        values: SoA2<ValueA, ValueB>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let value_a = super::device_expr_collect(&values.left)?;
        let value_b = super::device_expr_collect(&values.right)?;
        let (left, right, value_a) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &value_a, GpuOp::<Less>::new())?;
        let (_, _, value_b) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &value_b, GpuOp::<Less>::new())?;
        Ok((
            SoA2 { left, right },
            SoA2 {
                left: value_a,
                right: value_b,
            },
        ))
    }
}

impl<KeyA, KeyB, ValueA, ValueB, ValueC, Less> SortByKeyInput<SoA3<ValueA, ValueB, ValueC>, Less>
    for SoAView2<KeyA, KeyB>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item), Scalar = KeyA::Item>,
    SoA3<ValueA, ValueB, ValueC>: SoA,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueC: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
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
    Less: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
{
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA3<
            DeviceVec<KeyA::Runtime, ValueA::Item>,
            DeviceVec<KeyA::Runtime, ValueB::Item>,
            DeviceVec<KeyA::Runtime, ValueC::Item>,
        >,
    );

    fn sort_by_key_input(
        self,
        values: SoA3<ValueA, ValueB, ValueC>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let value_a = super::device_expr_collect(&values.first)?;
        let value_b = super::device_expr_collect(&values.second)?;
        let value_c = super::device_expr_collect(&values.third)?;
        let (left, right, value_a) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &value_a, GpuOp::<Less>::new())?;
        let (_, _, value_b) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &value_b, GpuOp::<Less>::new())?;
        let (_, _, value_c) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &value_c, GpuOp::<Less>::new())?;
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

impl<KeyA, KeyB, ValueA, ValueB, ValueC, ValueD, Less>
    SortByKeyInput<SoA4<ValueA, ValueB, ValueC, ValueD>, Less> for SoAView2<KeyA, KeyB>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item), Scalar = KeyA::Item>,
    SoA4<ValueA, ValueB, ValueC, ValueD>: SoA,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueC: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueD: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    ValueA::Item: CubePrimitive + CubeElement,
    ValueB::Item: CubePrimitive + CubeElement,
    ValueC::Item: CubePrimitive + CubeElement,
    ValueD::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
    ValueD::Expr: DeviceGpuExpr<ValueD::Item>,
    Less: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
{
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA4<
            DeviceVec<KeyA::Runtime, ValueA::Item>,
            DeviceVec<KeyA::Runtime, ValueB::Item>,
            DeviceVec<KeyA::Runtime, ValueC::Item>,
            DeviceVec<KeyA::Runtime, ValueD::Item>,
        >,
    );

    fn sort_by_key_input(
        self,
        values: SoA4<ValueA, ValueB, ValueC, ValueD>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.left)?;
        let key_b = super::device_expr_collect(&self.right)?;
        let value_a = super::device_expr_collect(&values.a)?;
        let value_b = super::device_expr_collect(&values.b)?;
        let value_c = super::device_expr_collect(&values.c)?;
        let value_d = super::device_expr_collect(&values.d)?;
        let (left, right, value_a) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &value_a, GpuOp::<Less>::new())?;
        let (_, _, value_b) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &value_b, GpuOp::<Less>::new())?;
        let (_, _, value_c) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &value_c, GpuOp::<Less>::new())?;
        let (_, _, value_d) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &value_d, GpuOp::<Less>::new())?;
        Ok((
            SoA2 { left, right },
            SoA4 {
                a: value_a,
                b: value_b,
                c: value_c,
                d: value_d,
            },
        ))
    }
}

impl<KeyA, KeyB, KeyC, ValueSource, Less> SortByKeyInput<ValueSource, Less>
    for SoAView3<KeyA, KeyB, KeyC>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item, KeyC::Item), Scalar = KeyA::Item>,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyC: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueSource: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    KeyC::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    KeyC::Expr: DeviceGpuExpr<KeyC::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Less: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
{
    type Output = (
        SoA3<
            DeviceVec<KeyA::Runtime, KeyA::Item>,
            DeviceVec<KeyA::Runtime, KeyB::Item>,
            DeviceVec<KeyA::Runtime, KeyC::Item>,
        >,
        SoA1<DeviceVec<KeyA::Runtime, ValueSource::Item>>,
    );

    fn sort_by_key_input(
        self,
        values: ValueSource,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        values.validate()?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let values = super::device_expr_collect(&values)?;
        let (first, second, third, source) =
            ordering::sort_tuple3_by_key(&key_a, &key_b, &key_c, &values, GpuOp::<Less>::new())?;
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

impl<KeyA, KeyB, KeyC, ValueA, ValueB, Less> SortByKeyInput<SoA2<ValueA, ValueB>, Less>
    for SoAView3<KeyA, KeyB, KeyC>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item, KeyC::Item), Scalar = KeyA::Item>,
    SoA2<ValueA, ValueB>: SoA,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyC: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
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
    Less: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
{
    type Output = (
        SoA3<
            DeviceVec<KeyA::Runtime, KeyA::Item>,
            DeviceVec<KeyA::Runtime, KeyB::Item>,
            DeviceVec<KeyA::Runtime, KeyC::Item>,
        >,
        SoA2<DeviceVec<KeyA::Runtime, ValueA::Item>, DeviceVec<KeyA::Runtime, ValueB::Item>>,
    );

    fn sort_by_key_input(
        self,
        values: SoA2<ValueA, ValueB>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let value_a = super::device_expr_collect(&values.left)?;
        let value_b = super::device_expr_collect(&values.right)?;
        let (first, second, third, value_a) =
            ordering::sort_tuple3_by_key(&key_a, &key_b, &key_c, &value_a, GpuOp::<Less>::new())?;
        let (_, _, _, value_b) =
            ordering::sort_tuple3_by_key(&key_a, &key_b, &key_c, &value_b, GpuOp::<Less>::new())?;
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

impl<KeyA, KeyB, KeyC, ValueA, ValueB, ValueC, Less>
    SortByKeyInput<SoA3<ValueA, ValueB, ValueC>, Less> for SoAView3<KeyA, KeyB, KeyC>
where
    Self: ReadOnlySoA<Item = (KeyA::Item, KeyB::Item, KeyC::Item), Scalar = KeyA::Item>,
    SoA3<ValueA, ValueB, ValueC>: SoA,
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyC: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueA: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueB: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueC: ReadOnlyKernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
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
    Less: BinaryPredicateOp<(KeyA::Item, KeyB::Item, KeyC::Item)>,
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

    fn sort_by_key_input(
        self,
        values: SoA3<ValueA, ValueB, ValueC>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let key_a = super::device_expr_collect(&self.first)?;
        let key_b = super::device_expr_collect(&self.second)?;
        let key_c = super::device_expr_collect(&self.third)?;
        let value_a = super::device_expr_collect(&values.first)?;
        let value_b = super::device_expr_collect(&values.second)?;
        let value_c = super::device_expr_collect(&values.third)?;
        let (first, second, third, value_a) =
            ordering::sort_tuple3_by_key(&key_a, &key_b, &key_c, &value_a, GpuOp::<Less>::new())?;
        let (_, _, _, value_b) =
            ordering::sort_tuple3_by_key(&key_a, &key_b, &key_c, &value_b, GpuOp::<Less>::new())?;
        let (_, _, _, value_c) =
            ordering::sort_tuple3_by_key(&key_a, &key_b, &key_c, &value_c, GpuOp::<Less>::new())?;
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

macro_rules! impl_sort_by_tuple_key_scalar_value {
    (
        $storage:ident,
        $input:ident -> $output:ident,
        $sort_fn:ident,
        ( $first:ident: $first_field:ident: $first_out:ident, $( $key:ident: $field:ident: $out_field:ident ),+ )
    ) => {
        impl<$first, $( $key ),+, ValueSource, Less> SortByKeyInput<ValueSource, Less>
            for $input<$first, $( $key ),+>
        where
            Self: $storage<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $key: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            ValueSource: ReadOnlyKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            ValueSource::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
            Less: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
        {
            type Output = (
                $output<
                    DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<<$first as KernelColumn>::Runtime, <$key as KernelColumn>::Item> ),+
                >,
                SoA1<DeviceVec<<$first as KernelColumn>::Runtime, ValueSource::Item>>,
            );

            fn sort_by_key_input(
                self,
                values: ValueSource,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                $storage::validate(&self)?;
                values.validate()?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let values = super::device_expr_collect(&values)?;
                let ($first_out, $( $out_field, )+ source) =
                    ordering::$sort_fn(&$first_field, $( &$field, )+ &values, GpuOp::<Less>::new())?;
                Ok(($output { $first_field: $first_out, $( $field: $out_field ),+ }, SoA1 { source }))
            }
        }
    };
}

impl_sort_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView4 -> SoA4, sort_tuple4_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d));
impl_sort_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView5 -> SoA5, sort_tuple5_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e));
impl_sort_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView6 -> SoA6, sort_tuple6_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f));
impl_sort_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView7 -> SoA7, sort_tuple7_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g));
impl_sort_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView8 -> SoA8, sort_tuple8_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h));
impl_sort_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView9 -> SoA9, sort_tuple9_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i));
impl_sort_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView10 -> SoA10, sort_tuple10_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i, J: j: out_j));
impl_sort_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView11 -> SoA11, sort_tuple11_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i, J: j: out_j, K: k: out_k));
impl_sort_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView12 -> SoA12, sort_tuple12_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i, J: j: out_j, K: k: out_k, L: l: out_l));
impl_sort_by_tuple_key_scalar_value!(SoA, SoA2 -> SoA2, sort_tuple2_by_key, (A: left: out_left, B: right: out_right));
impl_sort_by_tuple_key_scalar_value!(SoA, SoA3 -> SoA3, sort_tuple3_by_key, (A: first: out_first, B: second: out_second, C: third: out_third));
impl_sort_by_tuple_key_scalar_value!(SoA, SoA4 -> SoA4, sort_tuple4_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d));
impl_sort_by_tuple_key_scalar_value!(SoA, SoA5 -> SoA5, sort_tuple5_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e));
impl_sort_by_tuple_key_scalar_value!(SoA, SoA6 -> SoA6, sort_tuple6_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f));
impl_sort_by_tuple_key_scalar_value!(SoA, SoA7 -> SoA7, sort_tuple7_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g));
impl_sort_by_tuple_key_scalar_value!(SoA, SoA8 -> SoA8, sort_tuple8_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h));
impl_sort_by_tuple_key_scalar_value!(SoA, SoA9 -> SoA9, sort_tuple9_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i));
impl_sort_by_tuple_key_scalar_value!(SoA, SoA10 -> SoA10, sort_tuple10_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i, J: j: out_j));
impl_sort_by_tuple_key_scalar_value!(SoA, SoA11 -> SoA11, sort_tuple11_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i, J: j: out_j, K: k: out_k));
impl_sort_by_tuple_key_scalar_value!(SoA, SoA12 -> SoA12, sort_tuple12_by_key, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i, J: j: out_j, K: k: out_k, L: l: out_l));

macro_rules! impl_sort_by_tuple_key_soa2_values {
    (
        $storage:ident,
        $input:ident -> $output:ident,
        $sort_fn:ident,
        $value_index:tt,
        ( $first:ident: $first_field:ident: $first_out:ident, $( $key:ident: $field:ident: $out_field:ident ),+ )
    ) => {
        impl<$first, $( $key ),+, ValueA, ValueB, Less> SortByKeyInput<SoA2<ValueA, ValueB>, Less>
            for $input<$first, $( $key ),+>
        where
            Self: $storage<Scalar = <$first as KernelColumn>::Item>,
            SoA2<ValueA, ValueB>: SoA,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $key: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            ValueA: ReadOnlyKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            ValueB: ReadOnlyKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            ValueA::Item: CubePrimitive + CubeElement,
            ValueB::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
            ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
            Less: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
        {
            type Output = (
                $output<
                    DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<<$first as KernelColumn>::Runtime, <$key as KernelColumn>::Item> ),+
                >,
                SoA2<
                    DeviceVec<<$first as KernelColumn>::Runtime, ValueA::Item>,
                    DeviceVec<<$first as KernelColumn>::Runtime, ValueB::Item>,
                >,
            );

            fn sort_by_key_input(
                self,
                values: SoA2<ValueA, ValueB>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                $storage::validate(&self)?;
                SoA::validate(&values)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let value_a = super::device_expr_collect(&values.left)?;
                let value_b = super::device_expr_collect(&values.right)?;
                let ($first_out, $( $out_field, )+ left) =
                    ordering::$sort_fn(&$first_field, $( &$field, )+ &value_a, GpuOp::<Less>::new())?;
                let sorted_b =
                    ordering::$sort_fn(&$first_field, $( &$field, )+ &value_b, GpuOp::<Less>::new())?;
                let right = sorted_b.$value_index;
                Ok((
                    $output { $first_field: $first_out, $( $field: $out_field ),+ },
                    SoA2 { left, right },
                ))
            }
        }
    };
}

macro_rules! impl_sort_by_tuple_key_soa2_values_for_storage {
    ($storage:ident, $input:ident -> $output:ident, $sort_fn:ident, $value_index:tt, ( $first:ident: $first_field:ident: $first_out:ident, $( $key:ident: $field:ident: $out_field:ident ),+ )) => {
        impl_sort_by_tuple_key_soa2_values!($storage, $input -> $output, $sort_fn, $value_index, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
    };
}

impl_sort_by_tuple_key_soa2_values_for_storage!(ReadOnlySoA, SoAView4 -> SoA4, sort_tuple4_by_key, 4, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d));
impl_sort_by_tuple_key_soa2_values_for_storage!(ReadOnlySoA, SoAView5 -> SoA5, sort_tuple5_by_key, 5, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e));
impl_sort_by_tuple_key_soa2_values_for_storage!(ReadOnlySoA, SoAView6 -> SoA6, sort_tuple6_by_key, 6, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f));
impl_sort_by_tuple_key_soa2_values_for_storage!(ReadOnlySoA, SoAView7 -> SoA7, sort_tuple7_by_key, 7, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g));
impl_sort_by_tuple_key_soa2_values_for_storage!(ReadOnlySoA, SoAView8 -> SoA8, sort_tuple8_by_key, 8, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h));
impl_sort_by_tuple_key_soa2_values_for_storage!(ReadOnlySoA, SoAView9 -> SoA9, sort_tuple9_by_key, 9, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i));
impl_sort_by_tuple_key_soa2_values_for_storage!(ReadOnlySoA, SoAView10 -> SoA10, sort_tuple10_by_key, 10, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i, J: j: out_j));
impl_sort_by_tuple_key_soa2_values_for_storage!(ReadOnlySoA, SoAView11 -> SoA11, sort_tuple11_by_key, 11, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i, J: j: out_j, K: k: out_k));
impl_sort_by_tuple_key_soa2_values_for_storage!(ReadOnlySoA, SoAView12 -> SoA12, sort_tuple12_by_key, 12, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i, J: j: out_j, K: k: out_k, L: l: out_l));
impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA2 -> SoA2, sort_tuple2_by_key, 2, (A: left: out_left, B: right: out_right));
impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA3 -> SoA3, sort_tuple3_by_key, 3, (A: first: out_first, B: second: out_second, C: third: out_third));
impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA4 -> SoA4, sort_tuple4_by_key, 4, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d));
impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA5 -> SoA5, sort_tuple5_by_key, 5, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e));
impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA6 -> SoA6, sort_tuple6_by_key, 6, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f));
impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA7 -> SoA7, sort_tuple7_by_key, 7, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g));
impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA8 -> SoA8, sort_tuple8_by_key, 8, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h));
impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA9 -> SoA9, sort_tuple9_by_key, 9, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i));
impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA10 -> SoA10, sort_tuple10_by_key, 10, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i, J: j: out_j));
impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA11 -> SoA11, sort_tuple11_by_key, 11, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i, J: j: out_j, K: k: out_k));
impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA12 -> SoA12, sort_tuple12_by_key, 12, (A: a: out_a, B: b: out_b, C: c: out_c, D: d: out_d, E: e: out_e, F: f: out_f, G: g: out_g, H: h: out_h, I: i: out_i, J: j: out_j, K: k: out_k, L: l: out_l));

macro_rules! impl_sort_by_tuple_key_soa_view_values {
    (
        $storage:ident,
        $values:ident -> $out_values:ident < $first_value:ident, $( $value:ident ),+ > { $first_value_field:ident, $( $value_field:ident ),+ },
        $keys:ident -> $out_keys:ident,
        $sort_fn:ident,
        ( $first:ident: $first_field:ident: $first_out:ident, $( $key:ident: $field:ident: $out_field:ident ),+ )
    ) => {
        impl<$first, $( $key ),+, $first_value, $( $value ),+, Less>
            SortByKeyInput<$values<$first_value, $( $value ),+>, Less> for $keys<$first, $( $key ),+>
        where
            Self: $storage<Scalar = <$first as KernelColumn>::Item>,
            $values<$first_value, $( $value ),+>: SoA,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $key: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            $first_value: ReadOnlyKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            $( $value: ReadOnlyKernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$key as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            <$first_value as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$value as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$key as KernelColumn>::Expr: DeviceGpuExpr<<$key as KernelColumn>::Item>, )+
            <$first_value as KernelColumn>::Expr: DeviceGpuExpr<<$first_value as KernelColumn>::Item>,
            $( <$value as KernelColumn>::Expr: DeviceGpuExpr<<$value as KernelColumn>::Item>, )+
            Less: BinaryPredicateOp<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
        {
            type Output = (
                $out_keys<
                    DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<<$first as KernelColumn>::Runtime, <$key as KernelColumn>::Item> ),+
                >,
                $out_values<
                    DeviceVec<<$first as KernelColumn>::Runtime, <$first_value as KernelColumn>::Item>,
                    $( DeviceVec<<$first as KernelColumn>::Runtime, <$value as KernelColumn>::Item> ),+
                >,
            );

            fn sort_by_key_input(
                self,
                values: $values<$first_value, $( $value ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                $storage::validate(&self)?;
                SoA::validate(&values)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $( let $field = super::device_expr_collect(&self.$field)?; )+
                let indices = primitive_range::indices_u32($first_field.policy(), $first_field.len)?;
                let ($first_out, $( $out_field, )+ sorted_indices) =
                    ordering::$sort_fn(&$first_field, $( &$field, )+ &indices, GpuOp::<Less>::new())?;
                let $first_value_field = super::device_expr_collect(&values.$first_value_field)?;
                let $first_value_field = primitive_range::gather_device(&$first_value_field, &sorted_indices)?;
                $(
                    let $value_field = super::device_expr_collect(&values.$value_field)?;
                    let $value_field = primitive_range::gather_device(&$value_field, &sorted_indices)?;
                )+
                Ok((
                    $out_keys { $first_field: $first_out, $( $field: $out_field ),+ },
                    $out_values { $first_value_field, $( $value_field ),+ },
                ))
            }
        }
    };
}

macro_rules! impl_sort_by_tuple_key_soa_view_values_for_key {
    ($storage:ident, $keys:ident -> $out_keys:ident, $sort_fn:ident, ( $first:ident: $first_field:ident: $first_out:ident, $( $key:ident: $field:ident: $out_field:ident ),+ )) => {
        impl_sort_by_tuple_key_soa_view_values!($storage, SoA3 -> SoA3<A, B, C> { first, second, third }, $keys -> $out_keys, $sort_fn, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
        impl_sort_by_tuple_key_soa_view_values!($storage, SoA4 -> SoA4<A, B, C, D> { a, b, c, d }, $keys -> $out_keys, $sort_fn, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
        impl_sort_by_tuple_key_soa_view_values!($storage, SoA5 -> SoA5<A, B, C, D, E> { a, b, c, d, e }, $keys -> $out_keys, $sort_fn, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
        impl_sort_by_tuple_key_soa_view_values!($storage, SoA6 -> SoA6<A, B, C, D, E, F> { a, b, c, d, e, f }, $keys -> $out_keys, $sort_fn, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
        impl_sort_by_tuple_key_soa_view_values!($storage, SoA7 -> SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g }, $keys -> $out_keys, $sort_fn, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
        impl_sort_by_tuple_key_soa_view_values!($storage, SoA8 -> SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h }, $keys -> $out_keys, $sort_fn, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
        impl_sort_by_tuple_key_soa_view_values!($storage, SoA9 -> SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i }, $keys -> $out_keys, $sort_fn, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
        impl_sort_by_tuple_key_soa_view_values!($storage, SoA10 -> SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j }, $keys -> $out_keys, $sort_fn, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
        impl_sort_by_tuple_key_soa_view_values!($storage, SoA11 -> SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k }, $keys -> $out_keys, $sort_fn, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
        impl_sort_by_tuple_key_soa_view_values!($storage, SoA12 -> SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l }, $keys -> $out_keys, $sort_fn, ( $first: $first_field: $first_out, $( $key: $field: $out_field ),+ ));
    };
}

impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA2 -> SoA2, sort_tuple2_by_key, (KA: left: out_left, KB: right: out_right));
impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA3 -> SoA3, sort_tuple3_by_key, (KA: first: out_first, KB: second: out_second, KC: third: out_third));
impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA4 -> SoA4, sort_tuple4_by_key, (KA: a: out_a, KB: b: out_b, KC: c: out_c, KD: d: out_d));
impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA5 -> SoA5, sort_tuple5_by_key, (KA: a: out_a, KB: b: out_b, KC: c: out_c, KD: d: out_d, KE: e: out_e));
impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA6 -> SoA6, sort_tuple6_by_key, (KA: a: out_a, KB: b: out_b, KC: c: out_c, KD: d: out_d, KE: e: out_e, KF: f: out_f));
impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA7 -> SoA7, sort_tuple7_by_key, (KA: a: out_a, KB: b: out_b, KC: c: out_c, KD: d: out_d, KE: e: out_e, KF: f: out_f, KG: g: out_g));
impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA8 -> SoA8, sort_tuple8_by_key, (KA: a: out_a, KB: b: out_b, KC: c: out_c, KD: d: out_d, KE: e: out_e, KF: f: out_f, KG: g: out_g, KH: h: out_h));
impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA9 -> SoA9, sort_tuple9_by_key, (KA: a: out_a, KB: b: out_b, KC: c: out_c, KD: d: out_d, KE: e: out_e, KF: f: out_f, KG: g: out_g, KH: h: out_h, KI: i: out_i));
impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA10 -> SoA10, sort_tuple10_by_key, (KA: a: out_a, KB: b: out_b, KC: c: out_c, KD: d: out_d, KE: e: out_e, KF: f: out_f, KG: g: out_g, KH: h: out_h, KI: i: out_i, KJ: j: out_j));
impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA11 -> SoA11, sort_tuple11_by_key, (KA: a: out_a, KB: b: out_b, KC: c: out_c, KD: d: out_d, KE: e: out_e, KF: f: out_f, KG: g: out_g, KH: h: out_h, KI: i: out_i, KJ: j: out_j, KK: k: out_k));
impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA12 -> SoA12, sort_tuple12_by_key, (KA: a: out_a, KB: b: out_b, KC: c: out_c, KD: d: out_d, KE: e: out_e, KF: f: out_f, KG: g: out_g, KH: h: out_h, KI: i: out_i, KJ: j: out_j, KK: k: out_k, KL: l: out_l));

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
    MergeByKeyInput<SoAView1<LeftValue>, SoAView1<RightKey>, SoAView1<RightValue>, Less>
    for SoAView1<LeftKey>
where
    Self: ReadOnlySoA<Item = LeftKey::Item, Scalar = LeftKey::Item>,
    SoAView1<LeftValue>: ReadOnlySoA<Item = LeftValue::Item, Scalar = LeftValue::Item>,
    SoAView1<RightKey>: ReadOnlySoA<Item = RightKey::Item, Scalar = RightKey::Item>,
    SoAView1<RightValue>: ReadOnlySoA<Item = RightValue::Item, Scalar = RightValue::Item>,
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
        left_values: SoAView1<LeftValue>,
        right_keys: SoAView1<RightKey>,
        right_values: SoAView1<RightValue>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left_keys = materialize_soa_view_one(self)?;
        let left_values = materialize_soa_view_one(left_values)?;
        let right_keys = materialize_soa_view_one(right_keys)?;
        let right_values = materialize_soa_view_one(right_values)?;
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
            Self: ReadOnlySoA<Item = LeftKey::Item, Scalar = LeftKey::Item>,
            SoAView1<RightKey>: ReadOnlySoA<Item = RightKey::Item, Scalar = RightKey::Item>,
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
            type Output = (
                SoA1<DeviceVec<LeftKey::Runtime, LeftKey::Item>>,
                $output<
                    DeviceVec<LeftKey::Runtime, <$first_left as KernelColumn>::Item>,
                    $( DeviceVec<LeftKey::Runtime, <$left as KernelColumn>::Item> ),+
                >,
            );

            fn merge_by_key_input(
                self,
                left_values: $name<$first_left, $( $left ),+>,
                right_keys: SoAView1<RightKey>,
                right_values: $right_name<$first_right, $( $right ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                let left_keys = materialize_soa_view_one(self)?;
                let right_keys = materialize_soa_view_one(right_keys)?;
                left_values.$first_field.validate()?;
                right_values.$first_field.validate()?;
                $(
                    left_values.$field.validate()?;
                    right_values.$field.validate()?;
                )+

                // Compute merge-path control once and apply the same source
                // side/index stream to every value column.
                let (keys, control) =
                    ordering::merge_by_key_control::<LeftKey::Runtime, LeftKey::Item, Less>(
                        &left_keys,
                        &right_keys,
                    )?;
                let left_first = super::device_expr_collect(&left_values.$first_field)?;
                let right_first = super::device_expr_collect(&right_values.$first_field)?;
                let $first_field =
                    ordering::merge_by_key_values_with_control(&left_first, &right_first, &control)?;
                $(
                    let left_value = super::device_expr_collect(&left_values.$field)?;
                    let right_value = super::device_expr_collect(&right_values.$field)?;
                    let $field =
                        ordering::merge_by_key_values_with_control(&left_value, &right_value, &control)?;
                )+

                Ok((SoA1 { source: keys }, $output { $first_field, $( $field ),+ }))
            }
        }
    };
}

impl_merge_by_key_input!(SoAView2<A, B>, SoAView2<C, D>, SoA2 { left, right });
impl_merge_by_key_input!(SoAView3<A, B, C>, SoAView3<D, E, F>, SoA3 { first, second, third });
impl_merge_by_key_input!(SoAView4<A, B, C, D>, SoAView4<E, F, G, H>, SoA4 { a, b, c, d });
impl_merge_by_key_input!(SoAView5<A, B, C, D, E>, SoAView5<F, G, H, I, J>, SoA5 { a, b, c, d, e });
impl_merge_by_key_input!(SoAView6<A, B, C, D, E, F>, SoAView6<G, H, I, J, K, L>, SoA6 { a, b, c, d, e, f });
impl_merge_by_key_input!(SoAView7<A, B, C, D, E, F, G>, SoAView7<H, I, J, K, L, M, N>, SoA7 { a, b, c, d, e, f, g });
impl_merge_by_key_input!(SoAView8<A, B, C, D, E, F, G, H>, SoAView8<I, J, K, L, M, N, O, P>, SoA8 { a, b, c, d, e, f, g, h });
impl_merge_by_key_input!(SoAView9<A, B, C, D, E, F, G, H, I>, SoAView9<J, K, L, M, N, O, P, Q, R>, SoA9 { a, b, c, d, e, f, g, h, i });
impl_merge_by_key_input!(SoAView10<A, B, C, D, E, F, G, H, I, J>, SoAView10<K, L, M, N, O, P, Q, R, S, T>, SoA10 { a, b, c, d, e, f, g, h, i, j });
impl_merge_by_key_input!(SoAView11<A, B, C, D, E, F, G, H, I, J, K>, SoAView11<L, M, N, O, P, Q, R, S, T, U, V>, SoA11 { a, b, c, d, e, f, g, h, i, j, k });
impl_merge_by_key_input!(SoAView12<A, B, C, D, E, F, G, H, I, J, K, L>, SoAView12<M, N, O, P, Q, R, S, T, U, V, W, X>, SoA12 { a, b, c, d, e, f, g, h, i, j, k, l });
impl_merge_by_key_input!(SoA2<A, B>, SoA2<C, D>, SoA2 { left, right });
impl_merge_by_key_input!(SoA3<A, B, C>, SoA3<D, E, F>, SoA3 { first, second, third });
impl_merge_by_key_input!(SoA4<A, B, C, D>, SoA4<E, F, G, H>, SoA4 { a, b, c, d });
impl_merge_by_key_input!(SoA5<A, B, C, D, E>, SoA5<F, G, H, I, J>, SoA5 { a, b, c, d, e });
impl_merge_by_key_input!(SoA6<A, B, C, D, E, F>, SoA6<G, H, I, J, K, L>, SoA6 { a, b, c, d, e, f });
impl_merge_by_key_input!(SoA7<A, B, C, D, E, F, G>, SoA7<H, I, J, K, L, M, N>, SoA7 { a, b, c, d, e, f, g });
impl_merge_by_key_input!(SoA8<A, B, C, D, E, F, G, H>, SoA8<I, J, K, L, M, N, O, P>, SoA8 { a, b, c, d, e, f, g, h });
impl_merge_by_key_input!(SoA9<A, B, C, D, E, F, G, H, I>, SoA9<J, K, L, M, N, O, P, Q, R>, SoA9 { a, b, c, d, e, f, g, h, i });
impl_merge_by_key_input!(SoA10<A, B, C, D, E, F, G, H, I, J>, SoA10<K, L, M, N, O, P, Q, R, S, T>, SoA10 { a, b, c, d, e, f, g, h, i, j });
impl_merge_by_key_input!(SoA11<A, B, C, D, E, F, G, H, I, J, K>, SoA11<L, M, N, O, P, Q, R, S, T, U, V>, SoA11 { a, b, c, d, e, f, g, h, i, j, k });
impl_merge_by_key_input!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>, SoA12<M, N, O, P, Q, R, S, T, U, V, W, X>, SoA12 { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_merge_by_key_key_forward {
    ($left_values:ident < $( $left:ident ),+ >, $right_values:ident < $( $right:ident ),+ >) => {
        impl<LeftKey, RightKey, $( $left ),+, $( $right ),+, Less>
            MergeByKeyInput<$left_values<$( $left ),+>, RightKey, $right_values<$( $right ),+>, Less>
            for LeftKey
        where
            LeftKey: KernelColumn + KernelColumnAt<S0>,
            RightKey: KernelColumn<Runtime = LeftKey::Runtime, Item = LeftKey::Item> + KernelColumnAt<S0>,
            LeftKey::Item: CubePrimitive + CubeElement,
            LeftKey::Expr: DeviceGpuExpr<LeftKey::Item>,
            RightKey::Expr: DeviceGpuExpr<RightKey::Item>,
            SoAView1<LeftKey>: MergeByKeyInput<
                $left_values<$( $left ),+>,
                SoAView1<RightKey>,
                $right_values<$( $right ),+>,
                Less,
            >,
        {
            type Output = <SoAView1<LeftKey> as MergeByKeyInput<
                $left_values<$( $left ),+>,
                SoAView1<RightKey>,
                $right_values<$( $right ),+>,
                Less,
            >>::Output;

            fn merge_by_key_input(
                self,
                left_values: $left_values<$( $left ),+>,
                right_keys: RightKey,
                right_values: $right_values<$( $right ),+>,
                less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                <SoAView1<LeftKey> as MergeByKeyInput<
                    $left_values<$( $left ),+>,
                    SoAView1<RightKey>,
                    $right_values<$( $right ),+>,
                    Less,
                >>::merge_by_key_input(
                    SoAView1 { source: self },
                    left_values,
                    SoAView1 { source: right_keys },
                    right_values,
                    less,
                )
            }
        }
    };
}

impl_merge_by_key_key_forward!(SoAView2<A, B>, SoAView2<C, D>);
impl_merge_by_key_key_forward!(SoAView3<A, B, C>, SoAView3<D, E, F>);
impl_merge_by_key_key_forward!(SoAView4<A, B, C, D>, SoAView4<E, F, G, H>);
impl_merge_by_key_key_forward!(SoAView5<A, B, C, D, E>, SoAView5<F, G, H, I, J>);
impl_merge_by_key_key_forward!(SoAView6<A, B, C, D, E, F>, SoAView6<G, H, I, J, K, L>);
impl_merge_by_key_key_forward!(SoAView7<A, B, C, D, E, F, G>, SoAView7<H, I, J, K, L, M, N>);
impl_merge_by_key_key_forward!(SoAView8<A, B, C, D, E, F, G, H>, SoAView8<I, J, K, L, M, N, O, P>);
impl_merge_by_key_key_forward!(SoAView9<A, B, C, D, E, F, G, H, I>, SoAView9<J, K, L, M, N, O, P, Q, R>);
impl_merge_by_key_key_forward!(SoAView10<A, B, C, D, E, F, G, H, I, J>, SoAView10<K, L, M, N, O, P, Q, R, S, T>);
impl_merge_by_key_key_forward!(SoAView11<A, B, C, D, E, F, G, H, I, J, K>, SoAView11<L, M, N, O, P, Q, R, S, T, U, V>);
impl_merge_by_key_key_forward!(SoAView12<A, B, C, D, E, F, G, H, I, J, K, L>, SoAView12<M, N, O, P, Q, R, S, T, U, V, W, X>);
impl_merge_by_key_key_forward!(SoA2<A, B>, SoA2<C, D>);
impl_merge_by_key_key_forward!(SoA3<A, B, C>, SoA3<D, E, F>);
impl_merge_by_key_key_forward!(SoA4<A, B, C, D>, SoA4<E, F, G, H>);
impl_merge_by_key_key_forward!(SoA5<A, B, C, D, E>, SoA5<F, G, H, I, J>);
impl_merge_by_key_key_forward!(SoA6<A, B, C, D, E, F>, SoA6<G, H, I, J, K, L>);
impl_merge_by_key_key_forward!(SoA7<A, B, C, D, E, F, G>, SoA7<H, I, J, K, L, M, N>);
impl_merge_by_key_key_forward!(SoA8<A, B, C, D, E, F, G, H>, SoA8<I, J, K, L, M, N, O, P>);
impl_merge_by_key_key_forward!(SoA9<A, B, C, D, E, F, G, H, I>, SoA9<J, K, L, M, N, O, P, Q, R>);
impl_merge_by_key_key_forward!(SoA10<A, B, C, D, E, F, G, H, I, J>, SoA10<K, L, M, N, O, P, Q, R, S, T>);
impl_merge_by_key_key_forward!(SoA11<A, B, C, D, E, F, G, H, I, J, K>, SoA11<L, M, N, O, P, Q, R, S, T, U, V>);
impl_merge_by_key_key_forward!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>, SoA12<M, N, O, P, Q, R, S, T, U, V, W, X>);

macro_rules! impl_merge_by_tuple_key_scalar_value {
    (
        $storage:ident,
        $left_keys:ident,
        $right_keys:ident,
        $out_keys:ident,
        $sort_fn:ident,
        ( $first_left:ident: $first_right:ident: $first_field:ident: $first_concat:ident: $first_out:ident,
          $( $left:ident: $right:ident: $field:ident: $concat:ident: $out:ident ),+ )
    ) => {
        impl<$first_left, $( $left ),+, LeftValue, $first_right, $( $right ),+, RightValue, Less>
            MergeByKeyInput<LeftValue, $right_keys<$first_right, $( $right ),+>, RightValue, Less>
            for $left_keys<$first_left, $( $left ),+>
        where
            Self: $storage<Scalar = <$first_left as KernelColumn>::Item>,
            $right_keys<$first_right, $( $right ),+>: $storage<Scalar = <$first_right as KernelColumn>::Item>,
            $first_left: KernelColumn + KernelColumnAt<S0>,
            $( $left: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            $first_right:
                KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$first_left as KernelColumn>::Item>
                + KernelColumnAt<S0>,
            $(
                $right:
                    KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$left as KernelColumn>::Item>
                    + KernelColumnAt<S0>,
            )+
            LeftValue: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            RightValue:
                KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = LeftValue::Item>
                + KernelColumnAt<S0>,
            <$first_left as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$left as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            LeftValue::Item: CubePrimitive + CubeElement,
            <$first_left as KernelColumn>::Expr: DeviceGpuExpr<<$first_left as KernelColumn>::Item>,
            $( <$left as KernelColumn>::Expr: DeviceGpuExpr<<$left as KernelColumn>::Item>, )+
            <$first_right as KernelColumn>::Expr: DeviceGpuExpr<<$first_right as KernelColumn>::Item>,
            $( <$right as KernelColumn>::Expr: DeviceGpuExpr<<$right as KernelColumn>::Item>, )+
            LeftValue::Expr: DeviceGpuExpr<LeftValue::Item>,
            RightValue::Expr: DeviceGpuExpr<RightValue::Item>,
            Less: BinaryPredicateOp<(<$first_left as KernelColumn>::Item, $( <$left as KernelColumn>::Item ),+)>,
        {
            type Output = (
                $out_keys<
                    DeviceVec<<$first_left as KernelColumn>::Runtime, <$first_left as KernelColumn>::Item>,
                    $( DeviceVec<<$first_left as KernelColumn>::Runtime, <$left as KernelColumn>::Item> ),+
                >,
                SoA1<DeviceVec<<$first_left as KernelColumn>::Runtime, LeftValue::Item>>,
            );

            fn merge_by_key_input(
                self,
                left_values: LeftValue,
                right_keys: $right_keys<$first_right, $( $right ),+>,
                right_values: RightValue,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                $storage::validate(&self)?;
                $storage::validate(&right_keys)?;
                left_values.validate()?;
                right_values.validate()?;
                let left_first = super::device_expr_collect(&self.$first_field)?;
                let right_first = super::device_expr_collect(&right_keys.$first_field)?;
                let $first_concat = primitive_range::concat_device(&left_first, &right_first)?;
                $(
                    let left_key = super::device_expr_collect(&self.$field)?;
                    let right_key = super::device_expr_collect(&right_keys.$field)?;
                    let $concat = primitive_range::concat_device(&left_key, &right_key)?;
                )+
                let left_values = super::device_expr_collect(&left_values)?;
                let right_values = super::device_expr_collect(&right_values)?;
                super::ensure_same_len(left_values.len, left_first.len)?;
                super::ensure_same_len(right_values.len, right_first.len)?;
                let values = primitive_range::concat_device(&left_values, &right_values)?;
                let ($first_out, $( $out, )+ source) =
                    ordering::$sort_fn(&$first_concat, $( &$concat, )+ &values, GpuOp::<Less>::new())?;
                Ok((
                    $out_keys { $first_field: $first_out, $( $field: $out ),+ },
                    SoA1 { source },
                ))
            }
        }
    };
}

impl_merge_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView4, SoAView4, SoA4, sort_tuple4_by_key, (A: E: a: key_a: out_a, B: F: b: key_b: out_b, C: G: c: key_c: out_c, D: H: d: key_d: out_d));
impl_merge_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView5, SoAView5, SoA5, sort_tuple5_by_key, (A: F: a: key_a: out_a, B: G: b: key_b: out_b, C: H: c: key_c: out_c, D: I: d: key_d: out_d, E: J: e: key_e: out_e));
impl_merge_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView6, SoAView6, SoA6, sort_tuple6_by_key, (A: G: a: key_a: out_a, B: H: b: key_b: out_b, C: I: c: key_c: out_c, D: J: d: key_d: out_d, E: K: e: key_e: out_e, F: L: f: key_f: out_f));
impl_merge_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView7, SoAView7, SoA7, sort_tuple7_by_key, (A: H: a: key_a: out_a, B: I: b: key_b: out_b, C: J: c: key_c: out_c, D: K: d: key_d: out_d, E: L: e: key_e: out_e, F: M: f: key_f: out_f, G: N: g: key_g: out_g));
impl_merge_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView8, SoAView8, SoA8, sort_tuple8_by_key, (A: I: a: key_a: out_a, B: J: b: key_b: out_b, C: K: c: key_c: out_c, D: L: d: key_d: out_d, E: M: e: key_e: out_e, F: N: f: key_f: out_f, G: O: g: key_g: out_g, H: P: h: key_h: out_h));
impl_merge_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView9, SoAView9, SoA9, sort_tuple9_by_key, (A: J: a: key_a: out_a, B: K: b: key_b: out_b, C: L: c: key_c: out_c, D: M: d: key_d: out_d, E: N: e: key_e: out_e, F: O: f: key_f: out_f, G: P: g: key_g: out_g, H: Q: h: key_h: out_h, I: R: i: key_i: out_i));
impl_merge_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView10, SoAView10, SoA10, sort_tuple10_by_key, (A: K: a: key_a: out_a, B: L: b: key_b: out_b, C: M: c: key_c: out_c, D: N: d: key_d: out_d, E: O: e: key_e: out_e, F: P: f: key_f: out_f, G: Q: g: key_g: out_g, H: R: h: key_h: out_h, I: S: i: key_i: out_i, J: T: j: key_j: out_j));
impl_merge_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView11, SoAView11, SoA11, sort_tuple11_by_key, (A: L: a: key_a: out_a, B: M: b: key_b: out_b, C: N: c: key_c: out_c, D: O: d: key_d: out_d, E: P: e: key_e: out_e, F: Q: f: key_f: out_f, G: R: g: key_g: out_g, H: S: h: key_h: out_h, I: T: i: key_i: out_i, J: U: j: key_j: out_j, K: V: k: key_k: out_k));
impl_merge_by_tuple_key_scalar_value!(ReadOnlySoA, SoAView12, SoAView12, SoA12, sort_tuple12_by_key, (A: M: a: key_a: out_a, B: N: b: key_b: out_b, C: O: c: key_c: out_c, D: P: d: key_d: out_d, E: Q: e: key_e: out_e, F: R: f: key_f: out_f, G: S: g: key_g: out_g, H: T: h: key_h: out_h, I: U: i: key_i: out_i, J: V: j: key_j: out_j, K: W: k: key_k: out_k, L: X: l: key_l: out_l));
impl_merge_by_tuple_key_scalar_value!(SoA, SoA2, SoA2, SoA2, sort_tuple2_by_key, (A: C: left: key_left: out_left, B: D: right: key_right: out_right));
impl_merge_by_tuple_key_scalar_value!(SoA, SoA3, SoA3, SoA3, sort_tuple3_by_key, (A: D: first: key_first: out_first, B: E: second: key_second: out_second, C: F: third: key_third: out_third));
impl_merge_by_tuple_key_scalar_value!(SoA, SoA4, SoA4, SoA4, sort_tuple4_by_key, (A: E: a: key_a: out_a, B: F: b: key_b: out_b, C: G: c: key_c: out_c, D: H: d: key_d: out_d));
impl_merge_by_tuple_key_scalar_value!(SoA, SoA5, SoA5, SoA5, sort_tuple5_by_key, (A: F: a: key_a: out_a, B: G: b: key_b: out_b, C: H: c: key_c: out_c, D: I: d: key_d: out_d, E: J: e: key_e: out_e));
impl_merge_by_tuple_key_scalar_value!(SoA, SoA6, SoA6, SoA6, sort_tuple6_by_key, (A: G: a: key_a: out_a, B: H: b: key_b: out_b, C: I: c: key_c: out_c, D: J: d: key_d: out_d, E: K: e: key_e: out_e, F: L: f: key_f: out_f));
impl_merge_by_tuple_key_scalar_value!(SoA, SoA7, SoA7, SoA7, sort_tuple7_by_key, (A: H: a: key_a: out_a, B: I: b: key_b: out_b, C: J: c: key_c: out_c, D: K: d: key_d: out_d, E: L: e: key_e: out_e, F: M: f: key_f: out_f, G: N: g: key_g: out_g));
impl_merge_by_tuple_key_scalar_value!(SoA, SoA8, SoA8, SoA8, sort_tuple8_by_key, (A: I: a: key_a: out_a, B: J: b: key_b: out_b, C: K: c: key_c: out_c, D: L: d: key_d: out_d, E: M: e: key_e: out_e, F: N: f: key_f: out_f, G: O: g: key_g: out_g, H: P: h: key_h: out_h));
impl_merge_by_tuple_key_scalar_value!(SoA, SoA9, SoA9, SoA9, sort_tuple9_by_key, (A: J: a: key_a: out_a, B: K: b: key_b: out_b, C: L: c: key_c: out_c, D: M: d: key_d: out_d, E: N: e: key_e: out_e, F: O: f: key_f: out_f, G: P: g: key_g: out_g, H: Q: h: key_h: out_h, I: R: i: key_i: out_i));
impl_merge_by_tuple_key_scalar_value!(SoA, SoA10, SoA10, SoA10, sort_tuple10_by_key, (A: K: a: key_a: out_a, B: L: b: key_b: out_b, C: M: c: key_c: out_c, D: N: d: key_d: out_d, E: O: e: key_e: out_e, F: P: f: key_f: out_f, G: Q: g: key_g: out_g, H: R: h: key_h: out_h, I: S: i: key_i: out_i, J: T: j: key_j: out_j));
impl_merge_by_tuple_key_scalar_value!(SoA, SoA11, SoA11, SoA11, sort_tuple11_by_key, (A: L: a: key_a: out_a, B: M: b: key_b: out_b, C: N: c: key_c: out_c, D: O: d: key_d: out_d, E: P: e: key_e: out_e, F: Q: f: key_f: out_f, G: R: g: key_g: out_g, H: S: h: key_h: out_h, I: T: i: key_i: out_i, J: U: j: key_j: out_j, K: V: k: key_k: out_k));
impl_merge_by_tuple_key_scalar_value!(SoA, SoA12, SoA12, SoA12, sort_tuple12_by_key, (A: M: a: key_a: out_a, B: N: b: key_b: out_b, C: O: c: key_c: out_c, D: P: d: key_d: out_d, E: Q: e: key_e: out_e, F: R: f: key_f: out_f, G: S: g: key_g: out_g, H: T: h: key_h: out_h, I: U: i: key_i: out_i, J: V: j: key_j: out_j, K: W: k: key_k: out_k, L: X: l: key_l: out_l));

macro_rules! impl_merge_by_tuple_key_soa_view2_values {
    (
        $left_keys:ident,
        $right_keys:ident,
        $out_keys:ident,
        $sort_fn:ident,
        $value_index:tt,
        ( $first_left:ident: $first_right:ident: $first_field:ident: $first_concat:ident: $first_out:ident,
          $( $left:ident: $right:ident: $field:ident: $concat:ident: $out:ident ),+ )
    ) => {
        impl<
            $first_left,
            $( $left ),+,
            LeftValueA,
            LeftValueB,
            $first_right,
            $( $right ),+,
            RightValueA,
            RightValueB,
            Less,
        >
            MergeByKeyInput<
                SoAView2<LeftValueA, LeftValueB>,
                $right_keys<$first_right, $( $right ),+>,
                SoAView2<RightValueA, RightValueB>,
                Less,
            > for $left_keys<$first_left, $( $left ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first_left as KernelColumn>::Item>,
            $right_keys<$first_right, $( $right ),+>: ReadOnlySoA<Scalar = <$first_right as KernelColumn>::Item>,
            SoAView2<LeftValueA, LeftValueB>:
                ReadOnlySoA<Item = (LeftValueA::Item, LeftValueB::Item), Scalar = LeftValueA::Item>,
            SoAView2<RightValueA, RightValueB>:
                ReadOnlySoA<Item = (RightValueA::Item, RightValueB::Item), Scalar = RightValueA::Item>,
            $first_left: KernelColumn + KernelColumnAt<S0>,
            $( $left: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            $first_right:
                KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$first_left as KernelColumn>::Item>
                + KernelColumnAt<S0>,
            $(
                $right:
                    KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$left as KernelColumn>::Item>
                    + KernelColumnAt<S0>,
            )+
            LeftValueA: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            LeftValueB: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            RightValueA:
                KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = LeftValueA::Item>
                + KernelColumnAt<S0>,
            RightValueB:
                KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = LeftValueB::Item>
                + KernelColumnAt<S0>,
            <$first_left as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$left as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            LeftValueA::Item: CubePrimitive + CubeElement,
            LeftValueB::Item: CubePrimitive + CubeElement,
            <$first_left as KernelColumn>::Expr: DeviceGpuExpr<<$first_left as KernelColumn>::Item>,
            $( <$left as KernelColumn>::Expr: DeviceGpuExpr<<$left as KernelColumn>::Item>, )+
            <$first_right as KernelColumn>::Expr: DeviceGpuExpr<<$first_right as KernelColumn>::Item>,
            $( <$right as KernelColumn>::Expr: DeviceGpuExpr<<$right as KernelColumn>::Item>, )+
            LeftValueA::Expr: DeviceGpuExpr<LeftValueA::Item>,
            LeftValueB::Expr: DeviceGpuExpr<LeftValueB::Item>,
            RightValueA::Expr: DeviceGpuExpr<RightValueA::Item>,
            RightValueB::Expr: DeviceGpuExpr<RightValueB::Item>,
            Less: BinaryPredicateOp<(<$first_left as KernelColumn>::Item, $( <$left as KernelColumn>::Item ),+)>,
        {
            type Output = (
                $out_keys<
                    DeviceVec<<$first_left as KernelColumn>::Runtime, <$first_left as KernelColumn>::Item>,
                    $( DeviceVec<<$first_left as KernelColumn>::Runtime, <$left as KernelColumn>::Item> ),+
                >,
                SoA2<
                    DeviceVec<<$first_left as KernelColumn>::Runtime, LeftValueA::Item>,
                    DeviceVec<<$first_left as KernelColumn>::Runtime, LeftValueB::Item>,
                >,
            );

            fn merge_by_key_input(
                self,
                left_values: SoAView2<LeftValueA, LeftValueB>,
                right_keys: $right_keys<$first_right, $( $right ),+>,
                right_values: SoAView2<RightValueA, RightValueB>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&right_keys)?;
                ReadOnlySoA::validate(&left_values)?;
                ReadOnlySoA::validate(&right_values)?;
                let left_first = super::device_expr_collect(&self.$first_field)?;
                let right_first = super::device_expr_collect(&right_keys.$first_field)?;
                let $first_concat = primitive_range::concat_device(&left_first, &right_first)?;
                $(
                    let left_key = super::device_expr_collect(&self.$field)?;
                    let right_key = super::device_expr_collect(&right_keys.$field)?;
                    let $concat = primitive_range::concat_device(&left_key, &right_key)?;
                )+

                let left_value_a = super::device_expr_collect(&left_values.left)?;
                let right_value_a = super::device_expr_collect(&right_values.left)?;
                super::ensure_same_len(left_value_a.len, left_first.len)?;
                super::ensure_same_len(right_value_a.len, right_first.len)?;
                let values_a = primitive_range::concat_device(&left_value_a, &right_value_a)?;
                let ($first_out, $( $out, )+ left) =
                    ordering::$sort_fn(&$first_concat, $( &$concat, )+ &values_a, GpuOp::<Less>::new())?;

                let left_value_b = super::device_expr_collect(&left_values.right)?;
                let right_value_b = super::device_expr_collect(&right_values.right)?;
                super::ensure_same_len(left_value_b.len, left_first.len)?;
                super::ensure_same_len(right_value_b.len, right_first.len)?;
                let values_b = primitive_range::concat_device(&left_value_b, &right_value_b)?;
                let sorted_b =
                    ordering::$sort_fn(&$first_concat, $( &$concat, )+ &values_b, GpuOp::<Less>::new())?;
                let right = sorted_b.$value_index;

                Ok((
                    $out_keys { $first_field: $first_out, $( $field: $out ),+ },
                    SoA2 { left, right },
                ))
            }
        }
    };
}

impl_merge_by_tuple_key_soa_view2_values!(SoAView4, SoAView4, SoA4, sort_tuple4_by_key, 4, (A: E: a: key_a: out_a, B: F: b: key_b: out_b, C: G: c: key_c: out_c, D: H: d: key_d: out_d));

macro_rules! impl_merge_by_tuple_key_soa_view_values {
    (
        $values:ident -> $out_values:ident < $( $value:ident: $right_value:ident: $value_field:ident ),+ >,
        $left_keys:ident,
        $right_keys:ident,
        $out_keys:ident,
        $sort_fn:ident,
        ( $first_left:ident: $first_right:ident: $first_field:ident: $first_concat:ident: $first_out:ident,
          $( $left:ident: $right:ident: $field:ident: $concat:ident: $out:ident ),+ )
    ) => {
        impl<$first_left, $( $left ),+, $first_right, $( $right ),+, $( $value, )+ $( $right_value, )+ Less>
            MergeByKeyInput<$values<$( $value ),+>, $right_keys<$first_right, $( $right ),+>, $values<$( $right_value ),+>, Less>
            for $left_keys<$first_left, $( $left ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first_left as KernelColumn>::Item>,
            $right_keys<$first_right, $( $right ),+>: ReadOnlySoA<Scalar = <$first_right as KernelColumn>::Item>,
            $values<$( $value ),+>: ReadOnlySoA,
            $values<$( $right_value ),+>: ReadOnlySoA,
            $first_left: KernelColumn + KernelColumnAt<S0>,
            $( $left: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            $first_right: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$first_left as KernelColumn>::Item> + KernelColumnAt<S0>,
            $( $right: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$left as KernelColumn>::Item> + KernelColumnAt<S0>, )+
            $( $value: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            $( $right_value: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$value as KernelColumn>::Item> + KernelColumnAt<S0>, )+
            <$first_left as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$left as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            $( <$value as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            <$first_left as KernelColumn>::Expr: DeviceGpuExpr<<$first_left as KernelColumn>::Item>,
            $( <$left as KernelColumn>::Expr: DeviceGpuExpr<<$left as KernelColumn>::Item>, )+
            <$first_right as KernelColumn>::Expr: DeviceGpuExpr<<$first_right as KernelColumn>::Item>,
            $( <$right as KernelColumn>::Expr: DeviceGpuExpr<<$right as KernelColumn>::Item>, )+
            $( <$value as KernelColumn>::Expr: DeviceGpuExpr<<$value as KernelColumn>::Item>, )+
            $( <$right_value as KernelColumn>::Expr: DeviceGpuExpr<<$right_value as KernelColumn>::Item>, )+
            Less: BinaryPredicateOp<(<$first_left as KernelColumn>::Item, $( <$left as KernelColumn>::Item ),+)>,
        {
            type Output = (
                $out_keys<
                    DeviceVec<<$first_left as KernelColumn>::Runtime, <$first_left as KernelColumn>::Item>,
                    $( DeviceVec<<$first_left as KernelColumn>::Runtime, <$left as KernelColumn>::Item> ),+
                >,
                $out_values<$( DeviceVec<<$first_left as KernelColumn>::Runtime, <$value as KernelColumn>::Item> ),+>,
            );

            fn merge_by_key_input(
                self,
                left_values: $values<$( $value ),+>,
                right_keys: $right_keys<$first_right, $( $right ),+>,
                right_values: $values<$( $right_value ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&right_keys)?;
                ReadOnlySoA::validate(&left_values)?;
                ReadOnlySoA::validate(&right_values)?;
                let left_first = super::device_expr_collect(&self.$first_field)?;
                let right_first = super::device_expr_collect(&right_keys.$first_field)?;
                let $first_concat = primitive_range::concat_device(&left_first, &right_first)?;
                $(
                    let left_key = super::device_expr_collect(&self.$field)?;
                    let right_key = super::device_expr_collect(&right_keys.$field)?;
                    let $concat = primitive_range::concat_device(&left_key, &right_key)?;
                )+
                let indices = primitive_range::indices_u32($first_concat.policy(), $first_concat.len)?;
                let ($first_out, $( $out, )+ sorted_indices) =
                    ordering::$sort_fn(&$first_concat, $( &$concat, )+ &indices, GpuOp::<Less>::new())?;
                $(
                    let left_value = super::device_expr_collect(&left_values.$value_field)?;
                    let right_value = super::device_expr_collect(&right_values.$value_field)?;
                    super::ensure_same_len(left_value.len, left_first.len)?;
                    super::ensure_same_len(right_value.len, right_first.len)?;
                    let value = primitive_range::concat_device(&left_value, &right_value)?;
                    let $value_field = primitive_range::gather_device(&value, &sorted_indices)?;
                )+
                Ok((
                    $out_keys { $first_field: $first_out, $( $field: $out ),+ },
                    $out_values { $( $value_field ),+ },
                ))
            }
        }
    };
}

impl_merge_by_tuple_key_soa_view_values!(SoAView3 -> SoA3 < VA: RVA: first, VB: RVB: second, VC: RVC: third >, SoAView2, SoAView2, SoA2, sort_tuple2_by_key, (KA: RA: left: key_left: out_left, KB: RB: right: key_right: out_right));
impl_merge_by_tuple_key_soa_view_values!(SoAView4 -> SoA4 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d >, SoAView2, SoAView2, SoA2, sort_tuple2_by_key, (KA: RA: left: key_left: out_left, KB: RB: right: key_right: out_right));
impl_merge_by_tuple_key_soa_view_values!(SoAView5 -> SoA5 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e >, SoAView2, SoAView2, SoA2, sort_tuple2_by_key, (KA: RA: left: key_left: out_left, KB: RB: right: key_right: out_right));
impl_merge_by_tuple_key_soa_view_values!(SoAView6 -> SoA6 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f >, SoAView2, SoAView2, SoA2, sort_tuple2_by_key, (KA: RA: left: key_left: out_left, KB: RB: right: key_right: out_right));
impl_merge_by_tuple_key_soa_view_values!(SoAView7 -> SoA7 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g >, SoAView2, SoAView2, SoA2, sort_tuple2_by_key, (KA: RA: left: key_left: out_left, KB: RB: right: key_right: out_right));
impl_merge_by_tuple_key_soa_view_values!(SoAView8 -> SoA8 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h >, SoAView2, SoAView2, SoA2, sort_tuple2_by_key, (KA: RA: left: key_left: out_left, KB: RB: right: key_right: out_right));
impl_merge_by_tuple_key_soa_view_values!(SoAView9 -> SoA9 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i >, SoAView2, SoAView2, SoA2, sort_tuple2_by_key, (KA: RA: left: key_left: out_left, KB: RB: right: key_right: out_right));
impl_merge_by_tuple_key_soa_view_values!(SoAView10 -> SoA10 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j >, SoAView2, SoAView2, SoA2, sort_tuple2_by_key, (KA: RA: left: key_left: out_left, KB: RB: right: key_right: out_right));
impl_merge_by_tuple_key_soa_view_values!(SoAView11 -> SoA11 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k >, SoAView2, SoAView2, SoA2, sort_tuple2_by_key, (KA: RA: left: key_left: out_left, KB: RB: right: key_right: out_right));
impl_merge_by_tuple_key_soa_view_values!(SoAView12 -> SoA12 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k, VL: RVL: l >, SoAView2, SoAView2, SoA2, sort_tuple2_by_key, (KA: RA: left: key_left: out_left, KB: RB: right: key_right: out_right));
impl_merge_by_tuple_key_soa_view_values!(SoAView4 -> SoA4 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d >, SoAView3, SoAView3, SoA3, sort_tuple3_by_key, (KA: RA: first: key_first: out_first, KB: RB: second: key_second: out_second, KC: RC: third: key_third: out_third));
impl_merge_by_tuple_key_soa_view_values!(SoAView5 -> SoA5 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e >, SoAView3, SoAView3, SoA3, sort_tuple3_by_key, (KA: RA: first: key_first: out_first, KB: RB: second: key_second: out_second, KC: RC: third: key_third: out_third));
impl_merge_by_tuple_key_soa_view_values!(SoAView6 -> SoA6 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f >, SoAView3, SoAView3, SoA3, sort_tuple3_by_key, (KA: RA: first: key_first: out_first, KB: RB: second: key_second: out_second, KC: RC: third: key_third: out_third));
impl_merge_by_tuple_key_soa_view_values!(SoAView7 -> SoA7 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g >, SoAView3, SoAView3, SoA3, sort_tuple3_by_key, (KA: RA: first: key_first: out_first, KB: RB: second: key_second: out_second, KC: RC: third: key_third: out_third));
impl_merge_by_tuple_key_soa_view_values!(SoAView8 -> SoA8 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h >, SoAView3, SoAView3, SoA3, sort_tuple3_by_key, (KA: RA: first: key_first: out_first, KB: RB: second: key_second: out_second, KC: RC: third: key_third: out_third));
impl_merge_by_tuple_key_soa_view_values!(SoAView9 -> SoA9 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i >, SoAView3, SoAView3, SoA3, sort_tuple3_by_key, (KA: RA: first: key_first: out_first, KB: RB: second: key_second: out_second, KC: RC: third: key_third: out_third));
impl_merge_by_tuple_key_soa_view_values!(SoAView10 -> SoA10 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j >, SoAView3, SoAView3, SoA3, sort_tuple3_by_key, (KA: RA: first: key_first: out_first, KB: RB: second: key_second: out_second, KC: RC: third: key_third: out_third));
impl_merge_by_tuple_key_soa_view_values!(SoAView11 -> SoA11 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k >, SoAView3, SoAView3, SoA3, sort_tuple3_by_key, (KA: RA: first: key_first: out_first, KB: RB: second: key_second: out_second, KC: RC: third: key_third: out_third));
impl_merge_by_tuple_key_soa_view_values!(SoAView12 -> SoA12 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k, VL: RVL: l >, SoAView3, SoAView3, SoA3, sort_tuple3_by_key, (KA: RA: first: key_first: out_first, KB: RB: second: key_second: out_second, KC: RC: third: key_third: out_third));
impl_merge_by_tuple_key_soa_view_values!(SoAView3 -> SoA3 < VA: RVA: first, VB: RVB: second, VC: RVC: third >, SoAView4, SoAView4, SoA4, sort_tuple4_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d));
impl_merge_by_tuple_key_soa_view_values!(SoAView4 -> SoA4 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d >, SoAView4, SoAView4, SoA4, sort_tuple4_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d));
impl_merge_by_tuple_key_soa_view_values!(SoAView5 -> SoA5 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e >, SoAView4, SoAView4, SoA4, sort_tuple4_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d));
impl_merge_by_tuple_key_soa_view_values!(SoAView6 -> SoA6 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f >, SoAView4, SoAView4, SoA4, sort_tuple4_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d));
impl_merge_by_tuple_key_soa_view_values!(SoAView7 -> SoA7 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g >, SoAView4, SoAView4, SoA4, sort_tuple4_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d));
impl_merge_by_tuple_key_soa_view_values!(SoAView8 -> SoA8 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h >, SoAView4, SoAView4, SoA4, sort_tuple4_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d));
impl_merge_by_tuple_key_soa_view_values!(SoAView9 -> SoA9 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i >, SoAView4, SoAView4, SoA4, sort_tuple4_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d));
impl_merge_by_tuple_key_soa_view_values!(SoAView10 -> SoA10 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j >, SoAView4, SoAView4, SoA4, sort_tuple4_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d));
impl_merge_by_tuple_key_soa_view_values!(SoAView11 -> SoA11 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k >, SoAView4, SoAView4, SoA4, sort_tuple4_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d));
impl_merge_by_tuple_key_soa_view_values!(SoAView12 -> SoA12 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k, VL: RVL: l >, SoAView4, SoAView4, SoA4, sort_tuple4_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d));
impl_merge_by_tuple_key_soa_view_values!(SoAView3 -> SoA3 < VA: RVA: first, VB: RVB: second, VC: RVC: third >, SoAView5, SoAView5, SoA5, sort_tuple5_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e));
impl_merge_by_tuple_key_soa_view_values!(SoAView4 -> SoA4 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d >, SoAView5, SoAView5, SoA5, sort_tuple5_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e));
impl_merge_by_tuple_key_soa_view_values!(SoAView5 -> SoA5 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e >, SoAView5, SoAView5, SoA5, sort_tuple5_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e));
impl_merge_by_tuple_key_soa_view_values!(SoAView6 -> SoA6 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f >, SoAView5, SoAView5, SoA5, sort_tuple5_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e));
impl_merge_by_tuple_key_soa_view_values!(SoAView7 -> SoA7 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g >, SoAView5, SoAView5, SoA5, sort_tuple5_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e));
impl_merge_by_tuple_key_soa_view_values!(SoAView8 -> SoA8 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h >, SoAView5, SoAView5, SoA5, sort_tuple5_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e));
impl_merge_by_tuple_key_soa_view_values!(SoAView9 -> SoA9 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i >, SoAView5, SoAView5, SoA5, sort_tuple5_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e));
impl_merge_by_tuple_key_soa_view_values!(SoAView10 -> SoA10 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j >, SoAView5, SoAView5, SoA5, sort_tuple5_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e));
impl_merge_by_tuple_key_soa_view_values!(SoAView11 -> SoA11 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k >, SoAView5, SoAView5, SoA5, sort_tuple5_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e));
impl_merge_by_tuple_key_soa_view_values!(SoAView12 -> SoA12 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k, VL: RVL: l >, SoAView5, SoAView5, SoA5, sort_tuple5_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e));
impl_merge_by_tuple_key_soa_view_values!(SoAView3 -> SoA3 < VA: RVA: first, VB: RVB: second, VC: RVC: third >, SoAView6, SoAView6, SoA6, sort_tuple6_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f));
impl_merge_by_tuple_key_soa_view_values!(SoAView4 -> SoA4 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d >, SoAView6, SoAView6, SoA6, sort_tuple6_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f));
impl_merge_by_tuple_key_soa_view_values!(SoAView5 -> SoA5 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e >, SoAView6, SoAView6, SoA6, sort_tuple6_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f));
impl_merge_by_tuple_key_soa_view_values!(SoAView6 -> SoA6 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f >, SoAView6, SoAView6, SoA6, sort_tuple6_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f));
impl_merge_by_tuple_key_soa_view_values!(SoAView7 -> SoA7 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g >, SoAView6, SoAView6, SoA6, sort_tuple6_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f));
impl_merge_by_tuple_key_soa_view_values!(SoAView8 -> SoA8 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h >, SoAView6, SoAView6, SoA6, sort_tuple6_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f));
impl_merge_by_tuple_key_soa_view_values!(SoAView9 -> SoA9 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i >, SoAView6, SoAView6, SoA6, sort_tuple6_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f));
impl_merge_by_tuple_key_soa_view_values!(SoAView10 -> SoA10 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j >, SoAView6, SoAView6, SoA6, sort_tuple6_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f));
impl_merge_by_tuple_key_soa_view_values!(SoAView11 -> SoA11 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k >, SoAView6, SoAView6, SoA6, sort_tuple6_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f));
impl_merge_by_tuple_key_soa_view_values!(SoAView12 -> SoA12 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k, VL: RVL: l >, SoAView6, SoAView6, SoA6, sort_tuple6_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f));
impl_merge_by_tuple_key_soa_view_values!(SoAView3 -> SoA3 < VA: RVA: first, VB: RVB: second, VC: RVC: third >, SoAView7, SoAView7, SoA7, sort_tuple7_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g));
impl_merge_by_tuple_key_soa_view_values!(SoAView4 -> SoA4 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d >, SoAView7, SoAView7, SoA7, sort_tuple7_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g));
impl_merge_by_tuple_key_soa_view_values!(SoAView5 -> SoA5 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e >, SoAView7, SoAView7, SoA7, sort_tuple7_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g));
impl_merge_by_tuple_key_soa_view_values!(SoAView6 -> SoA6 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f >, SoAView7, SoAView7, SoA7, sort_tuple7_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g));
impl_merge_by_tuple_key_soa_view_values!(SoAView7 -> SoA7 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g >, SoAView7, SoAView7, SoA7, sort_tuple7_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g));
impl_merge_by_tuple_key_soa_view_values!(SoAView8 -> SoA8 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h >, SoAView7, SoAView7, SoA7, sort_tuple7_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g));
impl_merge_by_tuple_key_soa_view_values!(SoAView9 -> SoA9 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i >, SoAView7, SoAView7, SoA7, sort_tuple7_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g));
impl_merge_by_tuple_key_soa_view_values!(SoAView10 -> SoA10 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j >, SoAView7, SoAView7, SoA7, sort_tuple7_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g));
impl_merge_by_tuple_key_soa_view_values!(SoAView11 -> SoA11 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k >, SoAView7, SoAView7, SoA7, sort_tuple7_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g));
impl_merge_by_tuple_key_soa_view_values!(SoAView12 -> SoA12 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k, VL: RVL: l >, SoAView7, SoAView7, SoA7, sort_tuple7_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g));
impl_merge_by_tuple_key_soa_view_values!(SoAView3 -> SoA3 < VA: RVA: first, VB: RVB: second, VC: RVC: third >, SoAView8, SoAView8, SoA8, sort_tuple8_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h));
impl_merge_by_tuple_key_soa_view_values!(SoAView4 -> SoA4 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d >, SoAView8, SoAView8, SoA8, sort_tuple8_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h));
impl_merge_by_tuple_key_soa_view_values!(SoAView5 -> SoA5 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e >, SoAView8, SoAView8, SoA8, sort_tuple8_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h));
impl_merge_by_tuple_key_soa_view_values!(SoAView6 -> SoA6 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f >, SoAView8, SoAView8, SoA8, sort_tuple8_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h));
impl_merge_by_tuple_key_soa_view_values!(SoAView7 -> SoA7 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g >, SoAView8, SoAView8, SoA8, sort_tuple8_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h));
impl_merge_by_tuple_key_soa_view_values!(SoAView8 -> SoA8 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h >, SoAView8, SoAView8, SoA8, sort_tuple8_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h));
impl_merge_by_tuple_key_soa_view_values!(SoAView9 -> SoA9 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i >, SoAView8, SoAView8, SoA8, sort_tuple8_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h));
impl_merge_by_tuple_key_soa_view_values!(SoAView10 -> SoA10 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j >, SoAView8, SoAView8, SoA8, sort_tuple8_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h));
impl_merge_by_tuple_key_soa_view_values!(SoAView11 -> SoA11 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k >, SoAView8, SoAView8, SoA8, sort_tuple8_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h));
impl_merge_by_tuple_key_soa_view_values!(SoAView12 -> SoA12 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k, VL: RVL: l >, SoAView8, SoAView8, SoA8, sort_tuple8_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h));
impl_merge_by_tuple_key_soa_view_values!(SoAView3 -> SoA3 < VA: RVA: first, VB: RVB: second, VC: RVC: third >, SoAView9, SoAView9, SoA9, sort_tuple9_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i));
impl_merge_by_tuple_key_soa_view_values!(SoAView4 -> SoA4 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d >, SoAView9, SoAView9, SoA9, sort_tuple9_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i));
impl_merge_by_tuple_key_soa_view_values!(SoAView5 -> SoA5 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e >, SoAView9, SoAView9, SoA9, sort_tuple9_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i));
impl_merge_by_tuple_key_soa_view_values!(SoAView6 -> SoA6 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f >, SoAView9, SoAView9, SoA9, sort_tuple9_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i));
impl_merge_by_tuple_key_soa_view_values!(SoAView7 -> SoA7 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g >, SoAView9, SoAView9, SoA9, sort_tuple9_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i));
impl_merge_by_tuple_key_soa_view_values!(SoAView8 -> SoA8 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h >, SoAView9, SoAView9, SoA9, sort_tuple9_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i));
impl_merge_by_tuple_key_soa_view_values!(SoAView9 -> SoA9 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i >, SoAView9, SoAView9, SoA9, sort_tuple9_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i));
impl_merge_by_tuple_key_soa_view_values!(SoAView10 -> SoA10 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j >, SoAView9, SoAView9, SoA9, sort_tuple9_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i));
impl_merge_by_tuple_key_soa_view_values!(SoAView11 -> SoA11 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k >, SoAView9, SoAView9, SoA9, sort_tuple9_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i));
impl_merge_by_tuple_key_soa_view_values!(SoAView12 -> SoA12 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k, VL: RVL: l >, SoAView9, SoAView9, SoA9, sort_tuple9_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i));
impl_merge_by_tuple_key_soa_view_values!(SoAView3 -> SoA3 < VA: RVA: first, VB: RVB: second, VC: RVC: third >, SoAView10, SoAView10, SoA10, sort_tuple10_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j));
impl_merge_by_tuple_key_soa_view_values!(SoAView4 -> SoA4 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d >, SoAView10, SoAView10, SoA10, sort_tuple10_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j));
impl_merge_by_tuple_key_soa_view_values!(SoAView5 -> SoA5 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e >, SoAView10, SoAView10, SoA10, sort_tuple10_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j));
impl_merge_by_tuple_key_soa_view_values!(SoAView6 -> SoA6 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f >, SoAView10, SoAView10, SoA10, sort_tuple10_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j));
impl_merge_by_tuple_key_soa_view_values!(SoAView7 -> SoA7 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g >, SoAView10, SoAView10, SoA10, sort_tuple10_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j));
impl_merge_by_tuple_key_soa_view_values!(SoAView8 -> SoA8 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h >, SoAView10, SoAView10, SoA10, sort_tuple10_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j));
impl_merge_by_tuple_key_soa_view_values!(SoAView9 -> SoA9 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i >, SoAView10, SoAView10, SoA10, sort_tuple10_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j));
impl_merge_by_tuple_key_soa_view_values!(SoAView10 -> SoA10 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j >, SoAView10, SoAView10, SoA10, sort_tuple10_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j));
impl_merge_by_tuple_key_soa_view_values!(SoAView11 -> SoA11 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k >, SoAView10, SoAView10, SoA10, sort_tuple10_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j));
impl_merge_by_tuple_key_soa_view_values!(SoAView12 -> SoA12 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k, VL: RVL: l >, SoAView10, SoAView10, SoA10, sort_tuple10_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j));
impl_merge_by_tuple_key_soa_view_values!(SoAView3 -> SoA3 < VA: RVA: first, VB: RVB: second, VC: RVC: third >, SoAView11, SoAView11, SoA11, sort_tuple11_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k));
impl_merge_by_tuple_key_soa_view_values!(SoAView4 -> SoA4 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d >, SoAView11, SoAView11, SoA11, sort_tuple11_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k));
impl_merge_by_tuple_key_soa_view_values!(SoAView5 -> SoA5 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e >, SoAView11, SoAView11, SoA11, sort_tuple11_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k));
impl_merge_by_tuple_key_soa_view_values!(SoAView6 -> SoA6 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f >, SoAView11, SoAView11, SoA11, sort_tuple11_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k));
impl_merge_by_tuple_key_soa_view_values!(SoAView7 -> SoA7 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g >, SoAView11, SoAView11, SoA11, sort_tuple11_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k));
impl_merge_by_tuple_key_soa_view_values!(SoAView8 -> SoA8 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h >, SoAView11, SoAView11, SoA11, sort_tuple11_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k));
impl_merge_by_tuple_key_soa_view_values!(SoAView9 -> SoA9 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i >, SoAView11, SoAView11, SoA11, sort_tuple11_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k));
impl_merge_by_tuple_key_soa_view_values!(SoAView10 -> SoA10 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j >, SoAView11, SoAView11, SoA11, sort_tuple11_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k));
impl_merge_by_tuple_key_soa_view_values!(SoAView11 -> SoA11 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k >, SoAView11, SoAView11, SoA11, sort_tuple11_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k));
impl_merge_by_tuple_key_soa_view_values!(SoAView12 -> SoA12 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k, VL: RVL: l >, SoAView11, SoAView11, SoA11, sort_tuple11_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k));
impl_merge_by_tuple_key_soa_view_values!(SoAView3 -> SoA3 < VA: RVA: first, VB: RVB: second, VC: RVC: third >, SoAView12, SoAView12, SoA12, sort_tuple12_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k, KL: RL: l: key_l: out_l));
impl_merge_by_tuple_key_soa_view_values!(SoAView4 -> SoA4 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d >, SoAView12, SoAView12, SoA12, sort_tuple12_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k, KL: RL: l: key_l: out_l));
impl_merge_by_tuple_key_soa_view_values!(SoAView5 -> SoA5 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e >, SoAView12, SoAView12, SoA12, sort_tuple12_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k, KL: RL: l: key_l: out_l));
impl_merge_by_tuple_key_soa_view_values!(SoAView6 -> SoA6 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f >, SoAView12, SoAView12, SoA12, sort_tuple12_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k, KL: RL: l: key_l: out_l));
impl_merge_by_tuple_key_soa_view_values!(SoAView7 -> SoA7 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g >, SoAView12, SoAView12, SoA12, sort_tuple12_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k, KL: RL: l: key_l: out_l));
impl_merge_by_tuple_key_soa_view_values!(SoAView8 -> SoA8 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h >, SoAView12, SoAView12, SoA12, sort_tuple12_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k, KL: RL: l: key_l: out_l));
impl_merge_by_tuple_key_soa_view_values!(SoAView9 -> SoA9 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i >, SoAView12, SoAView12, SoA12, sort_tuple12_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k, KL: RL: l: key_l: out_l));
impl_merge_by_tuple_key_soa_view_values!(SoAView10 -> SoA10 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j >, SoAView12, SoAView12, SoA12, sort_tuple12_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k, KL: RL: l: key_l: out_l));
impl_merge_by_tuple_key_soa_view_values!(SoAView11 -> SoA11 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k >, SoAView12, SoAView12, SoA12, sort_tuple12_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k, KL: RL: l: key_l: out_l));
impl_merge_by_tuple_key_soa_view_values!(SoAView12 -> SoA12 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k, VL: RVL: l >, SoAView12, SoAView12, SoA12, sort_tuple12_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k, KL: RL: l: key_l: out_l));

impl_merge_by_tuple_key_soa_view2_values!(SoAView5, SoAView5, SoA5, sort_tuple5_by_key, 5, (A: F: a: key_a: out_a, B: G: b: key_b: out_b, C: H: c: key_c: out_c, D: I: d: key_d: out_d, E: J: e: key_e: out_e));
impl_merge_by_tuple_key_soa_view2_values!(SoAView6, SoAView6, SoA6, sort_tuple6_by_key, 6, (A: G: a: key_a: out_a, B: H: b: key_b: out_b, C: I: c: key_c: out_c, D: J: d: key_d: out_d, E: K: e: key_e: out_e, F: L: f: key_f: out_f));
impl_merge_by_tuple_key_soa_view2_values!(SoAView7, SoAView7, SoA7, sort_tuple7_by_key, 7, (A: H: a: key_a: out_a, B: I: b: key_b: out_b, C: J: c: key_c: out_c, D: K: d: key_d: out_d, E: L: e: key_e: out_e, F: M: f: key_f: out_f, G: N: g: key_g: out_g));
impl_merge_by_tuple_key_soa_view2_values!(SoAView8, SoAView8, SoA8, sort_tuple8_by_key, 8, (A: I: a: key_a: out_a, B: J: b: key_b: out_b, C: K: c: key_c: out_c, D: L: d: key_d: out_d, E: M: e: key_e: out_e, F: N: f: key_f: out_f, G: O: g: key_g: out_g, H: P: h: key_h: out_h));
impl_merge_by_tuple_key_soa_view2_values!(SoAView9, SoAView9, SoA9, sort_tuple9_by_key, 9, (A: J: a: key_a: out_a, B: K: b: key_b: out_b, C: L: c: key_c: out_c, D: M: d: key_d: out_d, E: N: e: key_e: out_e, F: O: f: key_f: out_f, G: P: g: key_g: out_g, H: Q: h: key_h: out_h, I: R: i: key_i: out_i));
impl_merge_by_tuple_key_soa_view2_values!(SoAView10, SoAView10, SoA10, sort_tuple10_by_key, 10, (A: K: a: key_a: out_a, B: L: b: key_b: out_b, C: M: c: key_c: out_c, D: N: d: key_d: out_d, E: O: e: key_e: out_e, F: P: f: key_f: out_f, G: Q: g: key_g: out_g, H: R: h: key_h: out_h, I: S: i: key_i: out_i, J: T: j: key_j: out_j));
impl_merge_by_tuple_key_soa_view2_values!(SoAView11, SoAView11, SoA11, sort_tuple11_by_key, 11, (A: L: a: key_a: out_a, B: M: b: key_b: out_b, C: N: c: key_c: out_c, D: O: d: key_d: out_d, E: P: e: key_e: out_e, F: Q: f: key_f: out_f, G: R: g: key_g: out_g, H: S: h: key_h: out_h, I: T: i: key_i: out_i, J: U: j: key_j: out_j, K: V: k: key_k: out_k));
impl_merge_by_tuple_key_soa_view2_values!(SoAView12, SoAView12, SoA12, sort_tuple12_by_key, 12, (A: M: a: key_a: out_a, B: N: b: key_b: out_b, C: O: c: key_c: out_c, D: P: d: key_d: out_d, E: Q: e: key_e: out_e, F: R: f: key_f: out_f, G: S: g: key_g: out_g, H: T: h: key_h: out_h, I: U: i: key_i: out_i, J: V: j: key_j: out_j, K: W: k: key_k: out_k, L: X: l: key_l: out_l));

macro_rules! impl_merge_by_tuple_key_soa_values {
    (
        $values:ident -> $out_values:ident < $( $value:ident: $right_value:ident: $value_field:ident ),+ >,
        $left_keys:ident,
        $right_keys:ident,
        $out_keys:ident,
        $sort_fn:ident,
        ( $first_left:ident: $first_right:ident: $first_field:ident: $first_concat:ident: $first_out:ident,
          $( $left:ident: $right:ident: $field:ident: $concat:ident: $out:ident ),+ )
    ) => {
        impl<$first_left, $( $left ),+, $first_right, $( $right ),+, $( $value, )+ $( $right_value, )+ Less>
            MergeByKeyInput<$values<$( $value ),+>, $right_keys<$first_right, $( $right ),+>, $values<$( $right_value ),+>, Less>
            for $left_keys<$first_left, $( $left ),+>
        where
            Self: SoA<Scalar = <$first_left as KernelColumn>::Item>,
            $right_keys<$first_right, $( $right ),+>: SoA<Scalar = <$first_right as KernelColumn>::Item>,
            $values<$( $value ),+>: SoA,
            $values<$( $right_value ),+>: SoA,
            $first_left: KernelColumn + KernelColumnAt<S0>,
            $( $left: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            $first_right: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$first_left as KernelColumn>::Item> + KernelColumnAt<S0>,
            $( $right: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$left as KernelColumn>::Item> + KernelColumnAt<S0>, )+
            $( $value: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            $( $right_value: KernelColumn<Runtime = <$first_left as KernelColumn>::Runtime, Item = <$value as KernelColumn>::Item> + KernelColumnAt<S0>, )+
            <$first_left as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$left as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            $( <$value as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            <$first_left as KernelColumn>::Expr: DeviceGpuExpr<<$first_left as KernelColumn>::Item>,
            $( <$left as KernelColumn>::Expr: DeviceGpuExpr<<$left as KernelColumn>::Item>, )+
            <$first_right as KernelColumn>::Expr: DeviceGpuExpr<<$first_right as KernelColumn>::Item>,
            $( <$right as KernelColumn>::Expr: DeviceGpuExpr<<$right as KernelColumn>::Item>, )+
            $( <$value as KernelColumn>::Expr: DeviceGpuExpr<<$value as KernelColumn>::Item>, )+
            $( <$right_value as KernelColumn>::Expr: DeviceGpuExpr<<$right_value as KernelColumn>::Item>, )+
            Less: BinaryPredicateOp<(<$first_left as KernelColumn>::Item, $( <$left as KernelColumn>::Item ),+)>,
        {
            type Output = (
                $out_keys<
                    DeviceVec<<$first_left as KernelColumn>::Runtime, <$first_left as KernelColumn>::Item>,
                    $( DeviceVec<<$first_left as KernelColumn>::Runtime, <$left as KernelColumn>::Item> ),+
                >,
                $out_values<$( DeviceVec<<$first_left as KernelColumn>::Runtime, <$value as KernelColumn>::Item> ),+>,
            );

            fn merge_by_key_input(
                self,
                left_values: $values<$( $value ),+>,
                right_keys: $right_keys<$first_right, $( $right ),+>,
                right_values: $values<$( $right_value ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                SoA::validate(&right_keys)?;
                SoA::validate(&left_values)?;
                SoA::validate(&right_values)?;
                let left_first = super::device_expr_collect(&self.$first_field)?;
                let right_first = super::device_expr_collect(&right_keys.$first_field)?;
                let $first_concat = primitive_range::concat_device(&left_first, &right_first)?;
                $(
                    let left_key = super::device_expr_collect(&self.$field)?;
                    let right_key = super::device_expr_collect(&right_keys.$field)?;
                    let $concat = primitive_range::concat_device(&left_key, &right_key)?;
                )+
                let indices = primitive_range::indices_u32($first_concat.policy(), $first_concat.len)?;
                let ($first_out, $( $out, )+ sorted_indices) =
                    ordering::$sort_fn(&$first_concat, $( &$concat, )+ &indices, GpuOp::<Less>::new())?;
                $(
                    let left_value = super::device_expr_collect(&left_values.$value_field)?;
                    let right_value = super::device_expr_collect(&right_values.$value_field)?;
                    super::ensure_same_len(left_value.len, left_first.len)?;
                    super::ensure_same_len(right_value.len, right_first.len)?;
                    let value = primitive_range::concat_device(&left_value, &right_value)?;
                    let $value_field = primitive_range::gather_device(&value, &sorted_indices)?;
                )+
                Ok((
                    $out_keys { $first_field: $first_out, $( $field: $out ),+ },
                    $out_values { $( $value_field ),+ },
                ))
            }
        }
    };
}

macro_rules! impl_merge_by_tuple_key_soa_values_for_key {
    ($keys:ident, $out_keys:ident, $sort_fn:ident, ( $first_left:ident: $first_right:ident: $first_field:ident: $first_concat:ident: $first_out:ident, $( $left:ident: $right:ident: $field:ident: $concat:ident: $out:ident ),+ )) => {
        impl_merge_by_tuple_key_soa_values!(SoA2 -> SoA2 < VA: RVA: left, VB: RVB: right >, $keys, $keys, $out_keys, $sort_fn, ( $first_left: $first_right: $first_field: $first_concat: $first_out, $( $left: $right: $field: $concat: $out ),+ ));
        impl_merge_by_tuple_key_soa_values!(SoA3 -> SoA3 < VA: RVA: first, VB: RVB: second, VC: RVC: third >, $keys, $keys, $out_keys, $sort_fn, ( $first_left: $first_right: $first_field: $first_concat: $first_out, $( $left: $right: $field: $concat: $out ),+ ));
        impl_merge_by_tuple_key_soa_values!(SoA4 -> SoA4 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d >, $keys, $keys, $out_keys, $sort_fn, ( $first_left: $first_right: $first_field: $first_concat: $first_out, $( $left: $right: $field: $concat: $out ),+ ));
        impl_merge_by_tuple_key_soa_values!(SoA5 -> SoA5 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e >, $keys, $keys, $out_keys, $sort_fn, ( $first_left: $first_right: $first_field: $first_concat: $first_out, $( $left: $right: $field: $concat: $out ),+ ));
        impl_merge_by_tuple_key_soa_values!(SoA6 -> SoA6 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f >, $keys, $keys, $out_keys, $sort_fn, ( $first_left: $first_right: $first_field: $first_concat: $first_out, $( $left: $right: $field: $concat: $out ),+ ));
        impl_merge_by_tuple_key_soa_values!(SoA7 -> SoA7 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g >, $keys, $keys, $out_keys, $sort_fn, ( $first_left: $first_right: $first_field: $first_concat: $first_out, $( $left: $right: $field: $concat: $out ),+ ));
        impl_merge_by_tuple_key_soa_values!(SoA8 -> SoA8 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h >, $keys, $keys, $out_keys, $sort_fn, ( $first_left: $first_right: $first_field: $first_concat: $first_out, $( $left: $right: $field: $concat: $out ),+ ));
        impl_merge_by_tuple_key_soa_values!(SoA9 -> SoA9 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i >, $keys, $keys, $out_keys, $sort_fn, ( $first_left: $first_right: $first_field: $first_concat: $first_out, $( $left: $right: $field: $concat: $out ),+ ));
        impl_merge_by_tuple_key_soa_values!(SoA10 -> SoA10 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j >, $keys, $keys, $out_keys, $sort_fn, ( $first_left: $first_right: $first_field: $first_concat: $first_out, $( $left: $right: $field: $concat: $out ),+ ));
        impl_merge_by_tuple_key_soa_values!(SoA11 -> SoA11 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k >, $keys, $keys, $out_keys, $sort_fn, ( $first_left: $first_right: $first_field: $first_concat: $first_out, $( $left: $right: $field: $concat: $out ),+ ));
        impl_merge_by_tuple_key_soa_values!(SoA12 -> SoA12 < VA: RVA: a, VB: RVB: b, VC: RVC: c, VD: RVD: d, VE: RVE: e, VF: RVF: f, VG: RVG: g, VH: RVH: h, VI: RVI: i, VJ: RVJ: j, VK: RVK: k, VL: RVL: l >, $keys, $keys, $out_keys, $sort_fn, ( $first_left: $first_right: $first_field: $first_concat: $first_out, $( $left: $right: $field: $concat: $out ),+ ));
    };
}

impl_merge_by_tuple_key_soa_values_for_key!(SoA2, SoA2, sort_tuple2_by_key, (KA: RA: left: key_left: out_left, KB: RB: right: key_right: out_right));
impl_merge_by_tuple_key_soa_values_for_key!(SoA3, SoA3, sort_tuple3_by_key, (KA: RA: first: key_first: out_first, KB: RB: second: key_second: out_second, KC: RC: third: key_third: out_third));
impl_merge_by_tuple_key_soa_values_for_key!(SoA4, SoA4, sort_tuple4_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d));
impl_merge_by_tuple_key_soa_values_for_key!(SoA5, SoA5, sort_tuple5_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e));
impl_merge_by_tuple_key_soa_values_for_key!(SoA6, SoA6, sort_tuple6_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f));
impl_merge_by_tuple_key_soa_values_for_key!(SoA7, SoA7, sort_tuple7_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g));
impl_merge_by_tuple_key_soa_values_for_key!(SoA8, SoA8, sort_tuple8_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h));
impl_merge_by_tuple_key_soa_values_for_key!(SoA9, SoA9, sort_tuple9_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i));
impl_merge_by_tuple_key_soa_values_for_key!(SoA10, SoA10, sort_tuple10_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j));
impl_merge_by_tuple_key_soa_values_for_key!(SoA11, SoA11, sort_tuple11_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k));
impl_merge_by_tuple_key_soa_values_for_key!(SoA12, SoA12, sort_tuple12_by_key, (KA: RA: a: key_a: out_a, KB: RB: b: key_b: out_b, KC: RC: c: key_c: out_c, KD: RD: d: key_d: out_d, KE: RE: e: key_e: out_e, KF: RF: f: key_f: out_f, KG: RG: g: key_g: out_g, KH: RH: h: key_h: out_h, KI: RI: i: key_i: out_i, KJ: RJ: j: key_j: out_j, KK: RK: k: key_k: out_k, KL: RL: l: key_l: out_l));

impl<LeftA, LeftB, LeftValue, RightA, RightB, RightValue, Less>
    MergeByKeyInput<LeftValue, SoAView2<RightA, RightB>, RightValue, Less>
    for SoAView2<LeftA, LeftB>
where
    Self: ReadOnlySoA<Item = (LeftA::Item, LeftB::Item), Scalar = LeftA::Item>,
    SoAView2<RightA, RightB>:
        ReadOnlySoA<Item = (RightA::Item, RightB::Item), Scalar = RightA::Item>,
    LeftA: KernelColumn + KernelColumnAt<S0>,
    LeftB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    RightA: KernelColumn<Runtime = LeftA::Runtime, Item = LeftA::Item> + KernelColumnAt<S0>,
    RightB: KernelColumn<Runtime = LeftA::Runtime, Item = LeftB::Item> + KernelColumnAt<S0>,
    LeftValue: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    RightValue: KernelColumn<Runtime = LeftA::Runtime, Item = LeftValue::Item> + KernelColumnAt<S0>,
    LeftA::Item: CubePrimitive + CubeElement,
    LeftB::Item: CubePrimitive + CubeElement,
    LeftValue::Item: CubePrimitive + CubeElement,
    LeftA::Expr: DeviceGpuExpr<LeftA::Item>,
    LeftB::Expr: DeviceGpuExpr<LeftB::Item>,
    RightA::Expr: DeviceGpuExpr<RightA::Item>,
    RightB::Expr: DeviceGpuExpr<RightB::Item>,
    LeftValue::Expr: DeviceGpuExpr<LeftValue::Item>,
    RightValue::Expr: DeviceGpuExpr<RightValue::Item>,
    Less: BinaryPredicateOp<(LeftA::Item, LeftB::Item)>,
{
    type Output = (
        SoA2<DeviceVec<LeftA::Runtime, LeftA::Item>, DeviceVec<LeftA::Runtime, LeftB::Item>>,
        SoA1<DeviceVec<LeftA::Runtime, LeftValue::Item>>,
    );

    fn merge_by_key_input(
        self,
        left_values: LeftValue,
        right_keys: SoAView2<RightA, RightB>,
        right_values: RightValue,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&right_keys)?;
        left_values.validate()?;
        right_values.validate()?;
        let left_a = super::device_expr_collect(&self.left)?;
        let left_b = super::device_expr_collect(&self.right)?;
        let left_values = super::device_expr_collect(&left_values)?;
        let right_a = super::device_expr_collect(&right_keys.left)?;
        let right_b = super::device_expr_collect(&right_keys.right)?;
        let right_values = super::device_expr_collect(&right_values)?;
        let key_a = primitive_range::concat_device(&left_a, &right_a)?;
        let key_b = primitive_range::concat_device(&left_b, &right_b)?;
        let values = primitive_range::concat_device(&left_values, &right_values)?;
        let (left, right, source) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &values, GpuOp::<Less>::new())?;
        Ok((SoA2 { left, right }, SoA1 { source }))
    }
}

impl<LeftA, LeftB, LeftValueA, LeftValueB, RightA, RightB, RightValueA, RightValueB, Less>
    MergeByKeyInput<
        SoAView2<LeftValueA, LeftValueB>,
        SoAView2<RightA, RightB>,
        SoAView2<RightValueA, RightValueB>,
        Less,
    > for SoAView2<LeftA, LeftB>
where
    Self: ReadOnlySoA<Item = (LeftA::Item, LeftB::Item), Scalar = LeftA::Item>,
    SoAView2<RightA, RightB>:
        ReadOnlySoA<Item = (RightA::Item, RightB::Item), Scalar = RightA::Item>,
    SoAView2<LeftValueA, LeftValueB>:
        ReadOnlySoA<Item = (LeftValueA::Item, LeftValueB::Item), Scalar = LeftValueA::Item>,
    SoAView2<RightValueA, RightValueB>:
        ReadOnlySoA<Item = (RightValueA::Item, RightValueB::Item), Scalar = RightValueA::Item>,
    LeftA: KernelColumn + KernelColumnAt<S0>,
    LeftB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    RightA: KernelColumn<Runtime = LeftA::Runtime, Item = LeftA::Item> + KernelColumnAt<S0>,
    RightB: KernelColumn<Runtime = LeftA::Runtime, Item = LeftB::Item> + KernelColumnAt<S0>,
    LeftValueA: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    LeftValueB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    RightValueA:
        KernelColumn<Runtime = LeftA::Runtime, Item = LeftValueA::Item> + KernelColumnAt<S0>,
    RightValueB:
        KernelColumn<Runtime = LeftA::Runtime, Item = LeftValueB::Item> + KernelColumnAt<S0>,
    LeftA::Item: CubePrimitive + CubeElement,
    LeftB::Item: CubePrimitive + CubeElement,
    LeftValueA::Item: CubePrimitive + CubeElement,
    LeftValueB::Item: CubePrimitive + CubeElement,
    LeftA::Expr: DeviceGpuExpr<LeftA::Item>,
    LeftB::Expr: DeviceGpuExpr<LeftB::Item>,
    RightA::Expr: DeviceGpuExpr<RightA::Item>,
    RightB::Expr: DeviceGpuExpr<RightB::Item>,
    LeftValueA::Expr: DeviceGpuExpr<LeftValueA::Item>,
    LeftValueB::Expr: DeviceGpuExpr<LeftValueB::Item>,
    RightValueA::Expr: DeviceGpuExpr<RightValueA::Item>,
    RightValueB::Expr: DeviceGpuExpr<RightValueB::Item>,
    Less: BinaryPredicateOp<(LeftA::Item, LeftB::Item)>,
{
    type Output = (
        SoA2<DeviceVec<LeftA::Runtime, LeftA::Item>, DeviceVec<LeftA::Runtime, LeftB::Item>>,
        SoA2<
            DeviceVec<LeftA::Runtime, LeftValueA::Item>,
            DeviceVec<LeftA::Runtime, LeftValueB::Item>,
        >,
    );

    fn merge_by_key_input(
        self,
        left_values: SoAView2<LeftValueA, LeftValueB>,
        right_keys: SoAView2<RightA, RightB>,
        right_values: SoAView2<RightValueA, RightValueB>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&right_keys)?;
        ReadOnlySoA::validate(&left_values)?;
        ReadOnlySoA::validate(&right_values)?;
        let left_a = super::device_expr_collect(&self.left)?;
        let left_b = super::device_expr_collect(&self.right)?;
        let right_a = super::device_expr_collect(&right_keys.left)?;
        let right_b = super::device_expr_collect(&right_keys.right)?;
        let key_a = primitive_range::concat_device(&left_a, &right_a)?;
        let key_b = primitive_range::concat_device(&left_b, &right_b)?;

        let left_value_a = super::device_expr_collect(&left_values.left)?;
        let right_value_a = super::device_expr_collect(&right_values.left)?;
        let values_a = primitive_range::concat_device(&left_value_a, &right_value_a)?;
        let (left, right, value_a) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &values_a, GpuOp::<Less>::new())?;

        let left_value_b = super::device_expr_collect(&left_values.right)?;
        let right_value_b = super::device_expr_collect(&right_values.right)?;
        let values_b = primitive_range::concat_device(&left_value_b, &right_value_b)?;
        let (_, _, value_b) =
            ordering::sort_tuple2_by_key(&key_a, &key_b, &values_b, GpuOp::<Less>::new())?;

        Ok((
            SoA2 { left, right },
            SoA2 {
                left: value_a,
                right: value_b,
            },
        ))
    }
}

impl<LeftA, LeftB, LeftC, LeftValue, RightA, RightB, RightC, RightValue, Less>
    MergeByKeyInput<LeftValue, SoAView3<RightA, RightB, RightC>, RightValue, Less>
    for SoAView3<LeftA, LeftB, LeftC>
where
    Self: ReadOnlySoA<Item = (LeftA::Item, LeftB::Item, LeftC::Item), Scalar = LeftA::Item>,
    SoAView3<RightA, RightB, RightC>:
        ReadOnlySoA<Item = (RightA::Item, RightB::Item, RightC::Item), Scalar = RightA::Item>,
    LeftA: KernelColumn + KernelColumnAt<S0>,
    LeftB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    LeftC: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    RightA: KernelColumn<Runtime = LeftA::Runtime, Item = LeftA::Item> + KernelColumnAt<S0>,
    RightB: KernelColumn<Runtime = LeftA::Runtime, Item = LeftB::Item> + KernelColumnAt<S0>,
    RightC: KernelColumn<Runtime = LeftA::Runtime, Item = LeftC::Item> + KernelColumnAt<S0>,
    LeftValue: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    RightValue: KernelColumn<Runtime = LeftA::Runtime, Item = LeftValue::Item> + KernelColumnAt<S0>,
    LeftA::Item: CubePrimitive + CubeElement,
    LeftB::Item: CubePrimitive + CubeElement,
    LeftC::Item: CubePrimitive + CubeElement,
    LeftValue::Item: CubePrimitive + CubeElement,
    LeftA::Expr: DeviceGpuExpr<LeftA::Item>,
    LeftB::Expr: DeviceGpuExpr<LeftB::Item>,
    LeftC::Expr: DeviceGpuExpr<LeftC::Item>,
    RightA::Expr: DeviceGpuExpr<RightA::Item>,
    RightB::Expr: DeviceGpuExpr<RightB::Item>,
    RightC::Expr: DeviceGpuExpr<RightC::Item>,
    LeftValue::Expr: DeviceGpuExpr<LeftValue::Item>,
    RightValue::Expr: DeviceGpuExpr<RightValue::Item>,
    Less: BinaryPredicateOp<(LeftA::Item, LeftB::Item, LeftC::Item)>,
{
    type Output = (
        SoA3<
            DeviceVec<LeftA::Runtime, LeftA::Item>,
            DeviceVec<LeftA::Runtime, LeftB::Item>,
            DeviceVec<LeftA::Runtime, LeftC::Item>,
        >,
        SoA1<DeviceVec<LeftA::Runtime, LeftValue::Item>>,
    );

    fn merge_by_key_input(
        self,
        left_values: LeftValue,
        right_keys: SoAView3<RightA, RightB, RightC>,
        right_values: RightValue,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&right_keys)?;
        left_values.validate()?;
        right_values.validate()?;
        let left_a = super::device_expr_collect(&self.first)?;
        let left_b = super::device_expr_collect(&self.second)?;
        let left_c = super::device_expr_collect(&self.third)?;
        let left_values = super::device_expr_collect(&left_values)?;
        let right_a = super::device_expr_collect(&right_keys.first)?;
        let right_b = super::device_expr_collect(&right_keys.second)?;
        let right_c = super::device_expr_collect(&right_keys.third)?;
        let right_values = super::device_expr_collect(&right_values)?;
        let key_a = primitive_range::concat_device(&left_a, &right_a)?;
        let key_b = primitive_range::concat_device(&left_b, &right_b)?;
        let key_c = primitive_range::concat_device(&left_c, &right_c)?;
        let values = primitive_range::concat_device(&left_values, &right_values)?;
        let (first, second, third, source) =
            ordering::sort_tuple3_by_key(&key_a, &key_b, &key_c, &values, GpuOp::<Less>::new())?;
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

impl<
    LeftA,
    LeftB,
    LeftC,
    LeftValueA,
    LeftValueB,
    RightA,
    RightB,
    RightC,
    RightValueA,
    RightValueB,
    Less,
>
    MergeByKeyInput<
        SoAView2<LeftValueA, LeftValueB>,
        SoAView3<RightA, RightB, RightC>,
        SoAView2<RightValueA, RightValueB>,
        Less,
    > for SoAView3<LeftA, LeftB, LeftC>
where
    Self: ReadOnlySoA<Item = (LeftA::Item, LeftB::Item, LeftC::Item), Scalar = LeftA::Item>,
    SoAView3<RightA, RightB, RightC>:
        ReadOnlySoA<Item = (RightA::Item, RightB::Item, RightC::Item), Scalar = RightA::Item>,
    SoAView2<LeftValueA, LeftValueB>:
        ReadOnlySoA<Item = (LeftValueA::Item, LeftValueB::Item), Scalar = LeftValueA::Item>,
    SoAView2<RightValueA, RightValueB>:
        ReadOnlySoA<Item = (RightValueA::Item, RightValueB::Item), Scalar = RightValueA::Item>,
    LeftA: KernelColumn + KernelColumnAt<S0>,
    LeftB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    LeftC: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    RightA: KernelColumn<Runtime = LeftA::Runtime, Item = LeftA::Item> + KernelColumnAt<S0>,
    RightB: KernelColumn<Runtime = LeftA::Runtime, Item = LeftB::Item> + KernelColumnAt<S0>,
    RightC: KernelColumn<Runtime = LeftA::Runtime, Item = LeftC::Item> + KernelColumnAt<S0>,
    LeftValueA: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    LeftValueB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    RightValueA:
        KernelColumn<Runtime = LeftA::Runtime, Item = LeftValueA::Item> + KernelColumnAt<S0>,
    RightValueB:
        KernelColumn<Runtime = LeftA::Runtime, Item = LeftValueB::Item> + KernelColumnAt<S0>,
    LeftA::Item: CubePrimitive + CubeElement,
    LeftB::Item: CubePrimitive + CubeElement,
    LeftC::Item: CubePrimitive + CubeElement,
    LeftValueA::Item: CubePrimitive + CubeElement,
    LeftValueB::Item: CubePrimitive + CubeElement,
    LeftA::Expr: DeviceGpuExpr<LeftA::Item>,
    LeftB::Expr: DeviceGpuExpr<LeftB::Item>,
    LeftC::Expr: DeviceGpuExpr<LeftC::Item>,
    RightA::Expr: DeviceGpuExpr<RightA::Item>,
    RightB::Expr: DeviceGpuExpr<RightB::Item>,
    RightC::Expr: DeviceGpuExpr<RightC::Item>,
    LeftValueA::Expr: DeviceGpuExpr<LeftValueA::Item>,
    LeftValueB::Expr: DeviceGpuExpr<LeftValueB::Item>,
    RightValueA::Expr: DeviceGpuExpr<RightValueA::Item>,
    RightValueB::Expr: DeviceGpuExpr<RightValueB::Item>,
    Less: BinaryPredicateOp<(LeftA::Item, LeftB::Item, LeftC::Item)>,
{
    type Output = (
        SoA3<
            DeviceVec<LeftA::Runtime, LeftA::Item>,
            DeviceVec<LeftA::Runtime, LeftB::Item>,
            DeviceVec<LeftA::Runtime, LeftC::Item>,
        >,
        SoA2<
            DeviceVec<LeftA::Runtime, LeftValueA::Item>,
            DeviceVec<LeftA::Runtime, LeftValueB::Item>,
        >,
    );

    fn merge_by_key_input(
        self,
        left_values: SoAView2<LeftValueA, LeftValueB>,
        right_keys: SoAView3<RightA, RightB, RightC>,
        right_values: SoAView2<RightValueA, RightValueB>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&right_keys)?;
        ReadOnlySoA::validate(&left_values)?;
        ReadOnlySoA::validate(&right_values)?;
        let left_a = super::device_expr_collect(&self.first)?;
        let left_b = super::device_expr_collect(&self.second)?;
        let left_c = super::device_expr_collect(&self.third)?;
        let right_a = super::device_expr_collect(&right_keys.first)?;
        let right_b = super::device_expr_collect(&right_keys.second)?;
        let right_c = super::device_expr_collect(&right_keys.third)?;
        let key_a = primitive_range::concat_device(&left_a, &right_a)?;
        let key_b = primitive_range::concat_device(&left_b, &right_b)?;
        let key_c = primitive_range::concat_device(&left_c, &right_c)?;

        let left_value_a = super::device_expr_collect(&left_values.left)?;
        let right_value_a = super::device_expr_collect(&right_values.left)?;
        let values_a = primitive_range::concat_device(&left_value_a, &right_value_a)?;
        let (first, second, third, value_a) =
            ordering::sort_tuple3_by_key(&key_a, &key_b, &key_c, &values_a, GpuOp::<Less>::new())?;

        let left_value_b = super::device_expr_collect(&left_values.right)?;
        let right_value_b = super::device_expr_collect(&right_values.right)?;
        let values_b = primitive_range::concat_device(&left_value_b, &right_value_b)?;
        let (_, _, _, value_b) =
            ordering::sort_tuple3_by_key(&key_a, &key_b, &key_c, &values_b, GpuOp::<Less>::new())?;

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

impl<
    LeftA,
    LeftB,
    LeftC,
    LeftValueA,
    LeftValueB,
    LeftValueC,
    RightA,
    RightB,
    RightC,
    RightValueA,
    RightValueB,
    RightValueC,
    Less,
>
    MergeByKeyInput<
        SoAView3<LeftValueA, LeftValueB, LeftValueC>,
        SoAView3<RightA, RightB, RightC>,
        SoAView3<RightValueA, RightValueB, RightValueC>,
        Less,
    > for SoAView3<LeftA, LeftB, LeftC>
where
    Self: ReadOnlySoA<Item = (LeftA::Item, LeftB::Item, LeftC::Item), Scalar = LeftA::Item>,
    SoAView3<RightA, RightB, RightC>:
        ReadOnlySoA<Item = (RightA::Item, RightB::Item, RightC::Item), Scalar = RightA::Item>,
    SoAView3<LeftValueA, LeftValueB, LeftValueC>: ReadOnlySoA<
            Item = (LeftValueA::Item, LeftValueB::Item, LeftValueC::Item),
            Scalar = LeftValueA::Item,
        >,
    SoAView3<RightValueA, RightValueB, RightValueC>: ReadOnlySoA<
            Item = (RightValueA::Item, RightValueB::Item, RightValueC::Item),
            Scalar = RightValueA::Item,
        >,
    LeftA: KernelColumn + KernelColumnAt<S0>,
    LeftB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    LeftC: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    RightA: KernelColumn<Runtime = LeftA::Runtime, Item = LeftA::Item> + KernelColumnAt<S0>,
    RightB: KernelColumn<Runtime = LeftA::Runtime, Item = LeftB::Item> + KernelColumnAt<S0>,
    RightC: KernelColumn<Runtime = LeftA::Runtime, Item = LeftC::Item> + KernelColumnAt<S0>,
    LeftValueA: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    LeftValueB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    LeftValueC: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    RightValueA:
        KernelColumn<Runtime = LeftA::Runtime, Item = LeftValueA::Item> + KernelColumnAt<S0>,
    RightValueB:
        KernelColumn<Runtime = LeftA::Runtime, Item = LeftValueB::Item> + KernelColumnAt<S0>,
    RightValueC:
        KernelColumn<Runtime = LeftA::Runtime, Item = LeftValueC::Item> + KernelColumnAt<S0>,
    LeftA::Item: CubePrimitive + CubeElement,
    LeftB::Item: CubePrimitive + CubeElement,
    LeftC::Item: CubePrimitive + CubeElement,
    LeftValueA::Item: CubePrimitive + CubeElement,
    LeftValueB::Item: CubePrimitive + CubeElement,
    LeftValueC::Item: CubePrimitive + CubeElement,
    LeftA::Expr: DeviceGpuExpr<LeftA::Item>,
    LeftB::Expr: DeviceGpuExpr<LeftB::Item>,
    LeftC::Expr: DeviceGpuExpr<LeftC::Item>,
    RightA::Expr: DeviceGpuExpr<RightA::Item>,
    RightB::Expr: DeviceGpuExpr<RightB::Item>,
    RightC::Expr: DeviceGpuExpr<RightC::Item>,
    LeftValueA::Expr: DeviceGpuExpr<LeftValueA::Item>,
    LeftValueB::Expr: DeviceGpuExpr<LeftValueB::Item>,
    LeftValueC::Expr: DeviceGpuExpr<LeftValueC::Item>,
    RightValueA::Expr: DeviceGpuExpr<RightValueA::Item>,
    RightValueB::Expr: DeviceGpuExpr<RightValueB::Item>,
    RightValueC::Expr: DeviceGpuExpr<RightValueC::Item>,
    Less: BinaryPredicateOp<(LeftA::Item, LeftB::Item, LeftC::Item)>,
{
    type Output = (
        SoA3<
            DeviceVec<LeftA::Runtime, LeftA::Item>,
            DeviceVec<LeftA::Runtime, LeftB::Item>,
            DeviceVec<LeftA::Runtime, LeftC::Item>,
        >,
        SoA3<
            DeviceVec<LeftA::Runtime, LeftValueA::Item>,
            DeviceVec<LeftA::Runtime, LeftValueB::Item>,
            DeviceVec<LeftA::Runtime, LeftValueC::Item>,
        >,
    );

    fn merge_by_key_input(
        self,
        left_values: SoAView3<LeftValueA, LeftValueB, LeftValueC>,
        right_keys: SoAView3<RightA, RightB, RightC>,
        right_values: SoAView3<RightValueA, RightValueB, RightValueC>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&right_keys)?;
        ReadOnlySoA::validate(&left_values)?;
        ReadOnlySoA::validate(&right_values)?;
        let left_a = super::device_expr_collect(&self.first)?;
        let left_b = super::device_expr_collect(&self.second)?;
        let left_c = super::device_expr_collect(&self.third)?;
        let right_a = super::device_expr_collect(&right_keys.first)?;
        let right_b = super::device_expr_collect(&right_keys.second)?;
        let right_c = super::device_expr_collect(&right_keys.third)?;
        let key_a = primitive_range::concat_device(&left_a, &right_a)?;
        let key_b = primitive_range::concat_device(&left_b, &right_b)?;
        let key_c = primitive_range::concat_device(&left_c, &right_c)?;

        let left_value_a = super::device_expr_collect(&left_values.first)?;
        let right_value_a = super::device_expr_collect(&right_values.first)?;
        let values_a = primitive_range::concat_device(&left_value_a, &right_value_a)?;
        let (first, second, third, value_a) =
            ordering::sort_tuple3_by_key(&key_a, &key_b, &key_c, &values_a, GpuOp::<Less>::new())?;

        let left_value_b = super::device_expr_collect(&left_values.second)?;
        let right_value_b = super::device_expr_collect(&right_values.second)?;
        let values_b = primitive_range::concat_device(&left_value_b, &right_value_b)?;
        let (_, _, _, value_b) =
            ordering::sort_tuple3_by_key(&key_a, &key_b, &key_c, &values_b, GpuOp::<Less>::new())?;

        let left_value_c = super::device_expr_collect(&left_values.third)?;
        let right_value_c = super::device_expr_collect(&right_values.third)?;
        let values_c = primitive_range::concat_device(&left_value_c, &right_value_c)?;
        let (_, _, _, value_c) =
            ordering::sort_tuple3_by_key(&key_a, &key_b, &key_c, &values_c, GpuOp::<Less>::new())?;

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
        <SoAView1<LeftKey> as MergeByKeyInput<
            SoAView1<LeftValue>,
            SoAView1<RightKey>,
            SoAView1<RightValue>,
            Less,
        >>::merge_by_key_input(
            SoAView1 { source: self },
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

impl<Source, Less> SortInput<Less> for SoA1<Source>
where
    Self: SoA<Item = Source::Item, Scalar = Source::Item>,
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
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
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
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
    Left: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Right: ReadOnlyKernelColumn<Runtime = Left::Runtime>
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
    First: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Second: ReadOnlyKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: ReadOnlyKernelColumn<Runtime = First::Runtime>
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
            $first: ReadOnlyKernelColumn + KernelColumnAt<S0>,
            $(
                $rest: ReadOnlyKernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
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

/// Sorts read-only SoA input and returns owned device storage.
pub fn sort<Input, Less>(
    input: Input,
    _less: Less,
) -> Result<<<Input as SortInput<Less>>::Output as MaterializeOutput>::Output, Error>
where
    Input: SortInput<Less>,
    <Input as SortInput<Less>>::Output: MaterializeOutput,
{
    materialize(input.sort_input(GpuOp::<Less>::new())?)
}

/// Merges two sorted read-only inputs into owned device storage.
///
/// This is a borrowing algorithm. Both inputs are read, and the merged output is
/// newly materialized.
pub fn merge<Left, Right, Less>(
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<<Left as PairOrderingInput<Right, Less>>::Output as MaterializeOutput>::Output, Error>
where
    Left: PairOrderingInput<Right, Less>,
    <Left as PairOrderingInput<Right, Less>>::Output: MaterializeOutput,
{
    materialize(left.merge_input(right, GpuOp::<Less>::new())?)
}

/// Sorts read-only key-value pairs by key and returns owned SoA outputs.
pub fn sort_by_key<Keys, Values, Less>(
    keys: Keys,
    values: Values,
    _less: Less,
) -> Result<<<Keys as SortByKeyInput<Values, Less>>::Output as MaterializeOutput>::Output, Error>
where
    Keys: SortByKeyInput<Values, Less>,
    <Keys as SortByKeyInput<Values, Less>>::Output: MaterializeOutput,
{
    materialize(keys.sort_by_key_input(values, GpuOp::<Less>::new())?)
}

/// Stable sort. The current device implementation is stable.
pub fn stable_sort<Input, Less>(
    input: Input,
    less: Less,
) -> Result<<<Input as SortInput<Less>>::Output as MaterializeOutput>::Output, Error>
where
    Input: SortInput<Less>,
    <Input as SortInput<Less>>::Output: MaterializeOutput,
{
    sort(input, less)
}

/// Stable key-value sort. The current device implementation is stable.
pub fn stable_sort_by_key<Keys, Values, Less>(
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<<<Keys as SortByKeyInput<Values, Less>>::Output as MaterializeOutput>::Output, Error>
where
    Keys: SortByKeyInput<Values, Less>,
    <Keys as SortByKeyInput<Values, Less>>::Output: MaterializeOutput,
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
) -> Result<
    <<LeftKeys as MergeByKeyInput<LeftValues, RightKeys, RightValues, Less>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    LeftKeys: MergeByKeyInput<LeftValues, RightKeys, RightValues, Less>,
    <LeftKeys as MergeByKeyInput<LeftValues, RightKeys, RightValues, Less>>::Output:
        MaterializeOutput,
{
    materialize(left_keys.merge_by_key_input(
        left_values,
        right_keys,
        right_values,
        GpuOp::<Less>::new(),
    )?)
}

/// Computes the sorted set union of two sorted device vectors.
pub fn set_union<Left, Right, Less>(
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<<Left as PairOrderingInput<Right, Less>>::Output as MaterializeOutput>::Output, Error>
where
    Left: PairOrderingInput<Right, Less>,
    <Left as PairOrderingInput<Right, Less>>::Output: MaterializeOutput,
{
    materialize(left.set_union_input(right, GpuOp::<Less>::new())?)
}

/// Computes the sorted set intersection of two sorted device vectors.
pub fn set_intersection<Left, Right, Less>(
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<<Left as PairOrderingInput<Right, Less>>::Output as MaterializeOutput>::Output, Error>
where
    Left: PairOrderingInput<Right, Less>,
    <Left as PairOrderingInput<Right, Less>>::Output: MaterializeOutput,
{
    materialize(left.set_intersection_input(right, GpuOp::<Less>::new())?)
}

/// Computes the sorted set difference `left - right`.
pub fn set_difference<Left, Right, Less>(
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<<Left as PairOrderingInput<Right, Less>>::Output as MaterializeOutput>::Output, Error>
where
    Left: PairOrderingInput<Right, Less>,
    <Left as PairOrderingInput<Right, Less>>::Output: MaterializeOutput,
{
    materialize(left.set_difference_input(right, GpuOp::<Less>::new())?)
}
