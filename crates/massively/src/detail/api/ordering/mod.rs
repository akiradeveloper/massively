use super::memory::{MaterializeOutput, materialize};
use crate::{
    detail::op::kernel::PredicateOp2,
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlyKernelColumn, ReadOnlySoA, S0, SoA, SoA1,
        SoA2, SoA3, SoAView1, SoAView2, SoAView3,
    },
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr},
    kernels::*,
    op::GpuOp,
    policy::CubePolicy,
    primitives::{ordering, range as primitive_range, select},
};
use cubecl::prelude::*;

const BLOCK_ORDERING_SIZE: u32 = 256;

fn materialize_soa_view_one_with_policy<Source>(
    policy: &CubePolicy<Source::Runtime>,
    input: SoAView1<Source>,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    SoAView1<Source>: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    ReadOnlySoA::validate(&input)?;
    let bindings = input.source.stage(policy)?;
    if let Some(handle) = input.source.staged_value_handle(&bindings) {
        return Ok(DeviceVec::from_handle(
            policy.id(),
            handle,
            input.source.len(),
        ));
    }
    super::device_expr_collect_with_policy(policy, &input.source)
}

/// Pair input accepted by sorted binary ordering algorithms.
#[doc(hidden)]
pub trait PairOrderingInput<Other, Less> {
    /// Runtime used by this input.
    type Runtime: Runtime;
    /// Output produced by this algorithm.
    type Output;

    /// Merges two sorted inputs.
    fn merge_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;
    /// Computes the sorted set union.
    fn set_union_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;
    /// Computes the sorted set intersection.
    fn set_intersection_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;
    /// Computes the sorted set difference.
    fn set_difference_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;
}

impl<Left, Right, Less> PairOrderingInput<SoAView1<Right>, Less> for SoAView1<Left>
where
    Self: ReadOnlySoA<Item = (Left::Item,), Scalar = Left::Item>,
    SoAView1<Right>: ReadOnlySoA<Item = (Right::Item,), Scalar = Right::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: PredicateOp2<Left::Item>,
{
    type Runtime = Left::Runtime;
    type Output = SoA1<DeviceVec<Left::Runtime, Left::Item>>;

    fn merge_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: SoAView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_soa_view_one_with_policy(policy, self)?;
        let right = materialize_soa_view_one_with_policy(policy, other)?;
        Ok(SoA1 {
            source: ordering::merge_with_policy(policy, &left, &right, GpuOp::<Less>::new())?,
        })
    }

    fn set_union_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: SoAView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_soa_view_one_with_policy(policy, self)?;
        let right = materialize_soa_view_one_with_policy(policy, other)?;
        Ok(SoA1 {
            source: ordering::set_union_with_policy(policy, &left, &right, GpuOp::<Less>::new())?,
        })
    }

    fn set_intersection_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: SoAView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_soa_view_one_with_policy(policy, self)?;
        let right = materialize_soa_view_one_with_policy(policy, other)?;
        Ok(SoA1 {
            source: ordering::set_intersection_with_policy(
                policy,
                &left,
                &right,
                GpuOp::<Less>::new(),
            )?,
        })
    }

    fn set_difference_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: SoAView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        let left = materialize_soa_view_one_with_policy(policy, self)?;
        let right = materialize_soa_view_one_with_policy(policy, other)?;
        Ok(SoA1 {
            source: ordering::set_difference_with_policy(
                policy,
                &left,
                &right,
                GpuOp::<Less>::new(),
            )?,
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
    Less: PredicateOp2<Left::Item>,
{
    type Runtime = Left::Runtime;
    type Output = SoA1<DeviceVec<Left::Runtime, Left::Item>>;

    fn merge_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: Right,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Left> as PairOrderingInput<SoAView1<Right>, Less>>::merge_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: other },
            less,
        )
    }

    fn set_union_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: Right,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Left> as PairOrderingInput<SoAView1<Right>, Less>>::set_union_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: other },
            less,
        )
    }

    fn set_intersection_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: Right,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Left> as PairOrderingInput<SoAView1<Right>, Less>>::set_intersection_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: other },
            less,
        )
    }

    fn set_difference_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: Right,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Left> as PairOrderingInput<SoAView1<Right>, Less>>::set_difference_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: other },
            less,
        )
    }
}

impl<Left, Right, Less> PairOrderingInput<(Right,), Less> for (Left,)
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: PredicateOp2<(Left::Item,)>,
{
    type Runtime = Left::Runtime;
    type Output = SoA1<DeviceVec<Left::Runtime, Left::Item>>;

    fn merge_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: (Right,),
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <Left as PairOrderingInput<Right, super::Tuple1Less<Less>>>::merge_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn set_union_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: (Right,),
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <Left as PairOrderingInput<Right, super::Tuple1Less<Less>>>::set_union_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn set_intersection_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: (Right,),
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <Left as PairOrderingInput<Right, super::Tuple1Less<Less>>>::set_intersection_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn set_difference_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: (Right,),
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <Left as PairOrderingInput<Right, super::Tuple1Less<Less>>>::set_difference_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }
}

macro_rules! tuple_membership_handles {
    (
        $kernel_name:ident,
        ($first_item_ty:ty, $( $item_ty:ty ),+),
        $runtime_ty:ty,
        $less_ty:ty,
        $policy:expr,
        ($first_candidate:ident, $( $candidate:ident ),+),
        ($first_sorted:ident, $( $sorted:ident ),+),
        $keep_present:expr
    ) => {{
        let len = $first_candidate.len();
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let client = $policy.client();
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
                    unsafe { BufferArg::from_raw_parts($first_candidate.handle.clone(), len) },
                    $(
                        unsafe { BufferArg::from_raw_parts($candidate.handle.clone(), len) },
                    )+
                    unsafe { BufferArg::from_raw_parts($first_sorted.handle.clone(), $first_sorted.len()) },
                    $(
                        unsafe { BufferArg::from_raw_parts($sorted.handle.clone(), $sorted.len()) },
                    )+
                    unsafe { BufferArg::from_raw_parts(keep_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(flag.clone(), len) },
                );
            }
        }

        select::handles_from_flags(
            $policy,
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
        $policy:expr,
        $handles:ident,
        $count:ident,
        ($first_item_ty:ty, $( $item_ty:ty ),+),
        { $first_output_field:ident : $first_source:ident, $( $output_field:ident : $source:ident ),+ }
    ) => {{
        let $first_source = select::compact_with_count::<$runtime_ty, $first_item_ty>(
            $policy,
            $handles.clone(),
            $count,
        )?;
        $(
            let $source = select::compact_with_count::<$runtime_ty, $item_ty>(
                $policy,
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
            Less: PredicateOp2<(
                impl_tuple_pair_ordering!(@item_ty $first),
                $( impl_tuple_pair_ordering!(@item_ty $rest) ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = $output<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$first as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn merge_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                other: $input<$right_first_ty, $( $right_rest_ty ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = super::device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let $right_first_var = super::device_expr_collect_with_policy(policy, &other.$first_field)?;
                $(
                    let $right_var = super::device_expr_collect_with_policy(policy, &other.$field)?;
                )+
                let $first_field = primitive_range::concat_device_with_policy(policy, &$first_field, &$right_first_var)?;
                $(
                    let $field = primitive_range::concat_device_with_policy(policy, &$field, &$right_var)?;
                )+
                let ($first_field, $( $field ),+) = ordering::$sort_fn(
                    policy,
                    &$first_field,
                    $( &$field, )+
                    GpuOp::<Less>::new(),
                )?;
                Ok($output { $first_field, $( $field ),+ })
            }

            fn set_union_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                other: $input<$right_first_ty, $( $right_rest_ty ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = super::device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let $right_first_var = super::device_expr_collect_with_policy(policy, &other.$first_field)?;
                $(
                    let $right_var = super::device_expr_collect_with_policy(policy, &other.$field)?;
                )+
                let handles = tuple_membership_handles!(
                    $membership_kernel,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    <$first as KernelColumn>::Runtime,
                    Less,
                    policy,
                    ($right_first_var, $( $right_var ),+),
                    ($first_field, $( $field ),+),
                    false
                )?;
                let count = select::selected_count(policy, &handles)?;
                let right_only = compact_tuple_from_handles!(
                    $output,
                    <$first as KernelColumn>::Runtime,
                    policy,
                    handles,
                    count,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    { $first_field: $right_first_var, $( $field: $right_var ),+ }
                );
                let $output { $first_field: $right_first_var, $( $field: $right_var ),+ } = right_only;
                let $first_field = primitive_range::concat_device_with_policy(policy, &$first_field, &$right_first_var)?;
                $(
                    let $field = primitive_range::concat_device_with_policy(policy, &$field, &$right_var)?;
                )+
                let ($first_field, $( $field ),+) = ordering::$sort_fn(
                    policy,
                    &$first_field,
                    $( &$field, )+
                    GpuOp::<Less>::new(),
                )?;
                Ok($output { $first_field, $( $field ),+ })
            }

            fn set_intersection_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                other: $input<$right_first_ty, $( $right_rest_ty ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = super::device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let $right_first_var = super::device_expr_collect_with_policy(policy, &other.$first_field)?;
                $(
                    let $right_var = super::device_expr_collect_with_policy(policy, &other.$field)?;
                )+
                let handles = tuple_membership_handles!(
                    $membership_kernel,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    <$first as KernelColumn>::Runtime,
                    Less,
                    policy,
                    ($first_field, $( $field ),+),
                    ($right_first_var, $( $right_var ),+),
                    true
                )?;
                let count = select::selected_count(policy, &handles)?;
                Ok(compact_tuple_from_handles!(
                    $output,
                    <$first as KernelColumn>::Runtime,
                    policy,
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
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                other: $input<$right_first_ty, $( $right_rest_ty ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = super::device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let $right_first_var = super::device_expr_collect_with_policy(policy, &other.$first_field)?;
                $(
                    let $right_var = super::device_expr_collect_with_policy(policy, &other.$field)?;
                )+
                let handles = tuple_membership_handles!(
                    $membership_kernel,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    <$first as KernelColumn>::Runtime,
                    Less,
                    policy,
                    ($first_field, $( $field ),+),
                    ($right_first_var, $( $right_var ),+),
                    false
                )?;
                let count = select::selected_count(policy, &handles)?;
                Ok(compact_tuple_from_handles!(
                    $output,
                    <$first as KernelColumn>::Runtime,
                    policy,
                    handles,
                    count,
                    (
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item ),+
                    ),
                    { $first_field: $first_field, $( $field: $field ),+ }
                ))
            }

        }
    };
}

impl_tuple_pair_ordering!(SoAView2 -> SoA2<A, B; RA, RB> { left / right_left, right / right_right }, sort_tuple2, tuple2_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA2 -> SoA2<A, B; RA, RB> { left / right_left, right / right_right }, sort_tuple2, tuple2_membership_flags_kernel);
impl_tuple_pair_ordering!(SoAView3 -> SoA3<A, B, C; RA, RB, RC> { first / right_first, second / right_second, third / right_third }, sort_tuple3, tuple3_membership_flags_kernel);
impl_tuple_pair_ordering!(SoA3 -> SoA3<A, B, C; RA, RB, RC> { first / right_first, second / right_second, third / right_third }, sort_tuple3, tuple3_membership_flags_kernel);

mod reverse;
pub use reverse::reverse;

/// Input accepted by [`sort`].
#[doc(hidden)]
pub trait SortInput<Less> {
    /// Runtime used by this input.
    type Runtime: Runtime;
    /// Output produced by sorting this input.
    type Output;

    /// Sorts this input.
    fn sort_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;
}

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
    Less: PredicateOp2<KeySource::Item>,
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
    Less: PredicateOp2<KeySource::Item>,
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
    KeySource: SortByKeyInput<ValueSource, super::Tuple1Less<Less>>,
{
    type Runtime = <KeySource as SortByKeyInput<ValueSource, super::Tuple1Less<Less>>>::Runtime;
    type Output = <KeySource as SortByKeyInput<ValueSource, super::Tuple1Less<Less>>>::Output;

    fn sort_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: (ValueSource,),
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <KeySource as SortByKeyInput<ValueSource, super::Tuple1Less<Less>>>::sort_by_key_input(
            self.0,
            policy,
            values.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
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
            Less: PredicateOp2<KeySource::Item>,
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
                let $first_field = super::device_expr_gather_with_policy(policy, &values.$first_field, &sorted_indices)?;
                $(
                    let $field = super::device_expr_gather_with_policy(policy, &values.$field, &sorted_indices)?;
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
            Less: PredicateOp2<KeySource::Item>,
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
                    let $field = super::device_expr_gather_with_policy(policy, &values.$field, &sorted_indices)?;
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
    Less: PredicateOp2<(KeyA::Item, KeyB::Item)>,
{
    type Runtime = KeyA::Runtime;
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA1<DeviceVec<KeyA::Runtime, ValueSource::Item>>,
    );

    fn sort_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: ValueSource,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        values.validate()?;
        let (left, right, source) = ordering::sort_tuple2_by_key_input(
            policy,
            &self.left,
            &self.right,
            &values,
            GpuOp::<Less>::new(),
        )?;
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
    ValueA::Expr: GpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    ValueB::Expr: GpuExpr<ValueB::Item>,
    Less: PredicateOp2<(KeyA::Item, KeyB::Item)>,
{
    type Runtime = KeyA::Runtime;
    type Output = (
        SoA2<DeviceVec<KeyA::Runtime, KeyA::Item>, DeviceVec<KeyA::Runtime, KeyB::Item>>,
        SoA2<DeviceVec<KeyA::Runtime, ValueA::Item>, DeviceVec<KeyA::Runtime, ValueB::Item>>,
    );

    fn sort_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: SoA2<ValueA, ValueB>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let indices = primitive_range::indices_u32(policy, self.left.len())?;
        let (left, right, sorted_indices) = ordering::sort_tuple2_by_key_input(
            policy,
            &self.left,
            &self.right,
            &indices,
            GpuOp::<Less>::new(),
        )?;
        let value_a = super::device_expr_gather_with_policy(policy, &values.left, &sorted_indices)?;
        let value_b =
            super::device_expr_gather_with_policy(policy, &values.right, &sorted_indices)?;
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
    ValueA::Expr: GpuExpr<ValueA::Item>,
    ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
    ValueB::Expr: GpuExpr<ValueB::Item>,
    ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
    ValueC::Expr: GpuExpr<ValueC::Item>,
    Less: PredicateOp2<(KeyA::Item, KeyB::Item)>,
{
    type Runtime = KeyA::Runtime;
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
        policy: &CubePolicy<Self::Runtime>,
        values: SoA3<ValueA, ValueB, ValueC>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let indices = primitive_range::indices_u32(policy, self.left.len())?;
        let (left, right, sorted_indices) = ordering::sort_tuple2_by_key_input(
            policy,
            &self.left,
            &self.right,
            &indices,
            GpuOp::<Less>::new(),
        )?;
        let value_a =
            super::device_expr_gather_with_policy(policy, &values.first, &sorted_indices)?;
        let value_b =
            super::device_expr_gather_with_policy(policy, &values.second, &sorted_indices)?;
        let value_c =
            super::device_expr_gather_with_policy(policy, &values.third, &sorted_indices)?;
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
    Less: PredicateOp2<(KeyA::Item, KeyB::Item, KeyC::Item)>,
{
    type Runtime = KeyA::Runtime;
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
        policy: &CubePolicy<Self::Runtime>,
        values: ValueSource,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        values.validate()?;
        let key_a = super::device_expr_collect_with_policy(policy, &self.first)?;
        let key_b = super::device_expr_collect_with_policy(policy, &self.second)?;
        let key_c = super::device_expr_collect_with_policy(policy, &self.third)?;
        let values = super::device_expr_collect_with_policy(policy, &values)?;
        let (first, second, third, source) = ordering::sort_tuple3_by_key(
            policy,
            &key_a,
            &key_b,
            &key_c,
            &values,
            GpuOp::<Less>::new(),
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
    Less: PredicateOp2<(KeyA::Item, KeyB::Item, KeyC::Item)>,
{
    type Runtime = KeyA::Runtime;
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
        policy: &CubePolicy<Self::Runtime>,
        values: SoA2<ValueA, ValueB>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let key_a = super::device_expr_collect_with_policy(policy, &self.first)?;
        let key_b = super::device_expr_collect_with_policy(policy, &self.second)?;
        let key_c = super::device_expr_collect_with_policy(policy, &self.third)?;
        let value_a = super::device_expr_collect_with_policy(policy, &values.left)?;
        let value_b = super::device_expr_collect_with_policy(policy, &values.right)?;
        let (first, second, third, value_a) = ordering::sort_tuple3_by_key(
            policy,
            &key_a,
            &key_b,
            &key_c,
            &value_a,
            GpuOp::<Less>::new(),
        )?;
        let (_, _, _, value_b) = ordering::sort_tuple3_by_key(
            policy,
            &key_a,
            &key_b,
            &key_c,
            &value_b,
            GpuOp::<Less>::new(),
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
    Less: PredicateOp2<(KeyA::Item, KeyB::Item, KeyC::Item)>,
{
    type Runtime = KeyA::Runtime;
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
        policy: &CubePolicy<Self::Runtime>,
        values: SoA3<ValueA, ValueB, ValueC>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        SoA::validate(&values)?;
        let key_a = super::device_expr_collect_with_policy(policy, &self.first)?;
        let key_b = super::device_expr_collect_with_policy(policy, &self.second)?;
        let key_c = super::device_expr_collect_with_policy(policy, &self.third)?;
        let value_a = super::device_expr_collect_with_policy(policy, &values.first)?;
        let value_b = super::device_expr_collect_with_policy(policy, &values.second)?;
        let value_c = super::device_expr_collect_with_policy(policy, &values.third)?;
        let (first, second, third, value_a) = ordering::sort_tuple3_by_key(
            policy,
            &key_a,
            &key_b,
            &key_c,
            &value_a,
            GpuOp::<Less>::new(),
        )?;
        let (_, _, _, value_b) = ordering::sort_tuple3_by_key(
            policy,
            &key_a,
            &key_b,
            &key_c,
            &value_b,
            GpuOp::<Less>::new(),
        )?;
        let (_, _, _, value_c) = ordering::sort_tuple3_by_key(
            policy,
            &key_a,
            &key_b,
            &key_c,
            &value_c,
            GpuOp::<Less>::new(),
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
            Less: PredicateOp2<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Output = (
                $output<
                    DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<<$first as KernelColumn>::Runtime, <$key as KernelColumn>::Item> ),+
                >,
                SoA1<DeviceVec<<$first as KernelColumn>::Runtime, ValueSource::Item>>,
            );

            fn sort_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                values: ValueSource,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                $storage::validate(&self)?;
                values.validate()?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $( let $field = super::device_expr_collect_with_policy(policy, &self.$field)?; )+
                let values = super::device_expr_collect_with_policy(policy, &values)?;
                let ($first_out, $( $out_field, )+ source) =
                    ordering::$sort_fn(policy, &$first_field, $( &$field, )+ &values, GpuOp::<Less>::new())?;
                Ok(($output { $first_field: $first_out, $( $field: $out_field ),+ }, SoA1 { source }))
            }
        }
    };
}

impl_sort_by_tuple_key_scalar_value!(SoA, SoA2 -> SoA2, sort_tuple2_by_key, (A: left: out_left, B: right: out_right));
impl_sort_by_tuple_key_scalar_value!(SoA, SoA3 -> SoA3, sort_tuple3_by_key, (A: first: out_first, B: second: out_second, C: third: out_third));

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
            Less: PredicateOp2<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;
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
                policy: &CubePolicy<Self::Runtime>,
                values: SoA2<ValueA, ValueB>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                $storage::validate(&self)?;
                SoA::validate(&values)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $( let $field = super::device_expr_collect_with_policy(policy, &self.$field)?; )+
                let value_a = super::device_expr_collect_with_policy(policy, &values.left)?;
                let value_b = super::device_expr_collect_with_policy(policy, &values.right)?;
                let ($first_out, $( $out_field, )+ left) =
                    ordering::$sort_fn(policy, &$first_field, $( &$field, )+ &value_a, GpuOp::<Less>::new())?;
                let sorted_b =
                    ordering::$sort_fn(policy, &$first_field, $( &$field, )+ &value_b, GpuOp::<Less>::new())?;
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

impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA2 -> SoA2, sort_tuple2_by_key, 2, (A: left: out_left, B: right: out_right));
impl_sort_by_tuple_key_soa2_values_for_storage!(SoA, SoA3 -> SoA3, sort_tuple3_by_key, 3, (A: first: out_first, B: second: out_second, C: third: out_third));

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
            Less: PredicateOp2<(<$first as KernelColumn>::Item, $( <$key as KernelColumn>::Item ),+)>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;
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
                policy: &CubePolicy<Self::Runtime>,
                values: $values<$first_value, $( $value ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                $storage::validate(&self)?;
                SoA::validate(&values)?;
                let $first_field = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                $( let $field = super::device_expr_collect_with_policy(policy, &self.$field)?; )+
                let indices = primitive_range::indices_u32(policy, $first_field.len)?;
                let ($first_out, $( $out_field, )+ sorted_indices) =
                    ordering::$sort_fn(policy, &$first_field, $( &$field, )+ &indices, GpuOp::<Less>::new())?;
                let $first_value_field = super::device_expr_collect_with_policy(policy, &values.$first_value_field)?;
                let $first_value_field = primitive_range::gather_device_with_policy(policy, &$first_value_field, &sorted_indices)?;
                $(
                    let $value_field = super::device_expr_collect_with_policy(policy, &values.$value_field)?;
                    let $value_field = primitive_range::gather_device_with_policy(policy, &$value_field, &sorted_indices)?;
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
    };
}

impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA2 -> SoA2, sort_tuple2_by_key, (KA: left: out_left, KB: right: out_right));
impl_sort_by_tuple_key_soa_view_values_for_key!(SoA, SoA3 -> SoA3, sort_tuple3_by_key, (KA: first: out_first, KB: second: out_second, KC: third: out_third));

macro_rules! impl_sort_by_key_tuple_keys {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Values, Less> SortByKeyInput<Values, Less> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: SortByKeyInput<Values, Less>,
        {
            type Runtime = <$view<$( $ty ),+> as SortByKeyInput<Values, Less>>::Runtime;
            type Output = <$view<$( $ty ),+> as SortByKeyInput<Values, Less>>::Output;

            fn sort_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                values: Values,
                less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                <$view<$( $ty ),+> as SortByKeyInput<Values, Less>>::sort_by_key_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    values,
                    less,
                )
            }
        }
    };
}

impl_sort_by_key_tuple_keys!(SoAView2<A, B> { left: 0, right: 1 });
impl_sort_by_key_tuple_keys!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

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
    Less: PredicateOp2<LeftKey::Item>,
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
        let left_keys = materialize_soa_view_one_with_policy(policy, self)?;
        let left_values = materialize_soa_view_one_with_policy(policy, left_values)?;
        let right_keys = materialize_soa_view_one_with_policy(policy, right_keys)?;
        let right_values = materialize_soa_view_one_with_policy(policy, right_values)?;
        let (keys, values) = ordering::merge_by_key_with_policy(
            policy,
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
            Less: PredicateOp2<LeftKey::Item>,
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
                let left_keys = materialize_soa_view_one_with_policy(policy, self)?;
                let right_keys = materialize_soa_view_one_with_policy(policy, right_keys)?;
                left_values.$first_field.validate()?;
                right_values.$first_field.validate()?;
                $(
                    left_values.$field.validate()?;
                    right_values.$field.validate()?;
                )+

                // Compute merge-path control once and apply the same source
                // side/index stream to every value column.
                let (keys, control) =
                    ordering::merge_by_key_control_with_policy::<LeftKey::Runtime, LeftKey::Item, Less>(
                        policy,
                        &left_keys,
                        &right_keys,
                    )?;
                let left_first = super::device_expr_collect_with_policy(policy, &left_values.$first_field)?;
                let right_first = super::device_expr_collect_with_policy(policy, &right_values.$first_field)?;
                let $first_field =
                    ordering::merge_by_key_values_with_control_with_policy(policy, &left_first, &right_first, &control)?;
                $(
                    let left_value = super::device_expr_collect_with_policy(policy, &left_values.$field)?;
                    let right_value = super::device_expr_collect_with_policy(policy, &right_values.$field)?;
                    let $field =
                        ordering::merge_by_key_values_with_control_with_policy(policy, &left_value, &right_value, &control)?;
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
            type Runtime = <SoAView1<LeftKey> as MergeByKeyInput<
                $left_values<$( $left ),+>,
                SoAView1<RightKey>,
                $right_values<$( $right ),+>,
                Less,
            >>::Runtime;
            type Output = <SoAView1<LeftKey> as MergeByKeyInput<
                $left_values<$( $left ),+>,
                SoAView1<RightKey>,
                $right_values<$( $right ),+>,
                Less,
            >>::Output;

            fn merge_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
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
                    policy,
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
impl_merge_by_key_key_forward!(SoA2<A, B>, SoA2<C, D>);
impl_merge_by_key_key_forward!(SoA3<A, B, C>, SoA3<D, E, F>);

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
            Less: PredicateOp2<(<$first_left as KernelColumn>::Item, $( <$left as KernelColumn>::Item ),+)>,
        {
            type Runtime = <$first_left as KernelColumn>::Runtime;
            type Output = (
                $out_keys<
                    DeviceVec<<$first_left as KernelColumn>::Runtime, <$first_left as KernelColumn>::Item>,
                    $( DeviceVec<<$first_left as KernelColumn>::Runtime, <$left as KernelColumn>::Item> ),+
                >,
                SoA1<DeviceVec<<$first_left as KernelColumn>::Runtime, LeftValue::Item>>,
            );

            fn merge_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                left_values: LeftValue,
                right_keys: $right_keys<$first_right, $( $right ),+>,
                right_values: RightValue,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                $storage::validate(&self)?;
                $storage::validate(&right_keys)?;
                left_values.validate()?;
                right_values.validate()?;
                let left_first = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                let right_first = super::device_expr_collect_with_policy(policy, &right_keys.$first_field)?;
                let $first_concat = primitive_range::concat_device_with_policy(policy, &left_first, &right_first)?;
                $(
                    let left_key = super::device_expr_collect_with_policy(policy, &self.$field)?;
                    let right_key = super::device_expr_collect_with_policy(policy, &right_keys.$field)?;
                    let $concat = primitive_range::concat_device_with_policy(policy, &left_key, &right_key)?;
                )+
                let left_values = super::device_expr_collect_with_policy(policy, &left_values)?;
                let right_values = super::device_expr_collect_with_policy(policy, &right_values)?;
                super::ensure_same_len(left_values.len, left_first.len)?;
                super::ensure_same_len(right_values.len, right_first.len)?;
                let values = primitive_range::concat_device_with_policy(policy, &left_values, &right_values)?;
                let ($first_out, $( $out, )+ source) =
                    ordering::$sort_fn(policy, &$first_concat, $( &$concat, )+ &values, GpuOp::<Less>::new())?;
                Ok((
                    $out_keys { $first_field: $first_out, $( $field: $out ),+ },
                    SoA1 { source },
                ))
            }
        }
    };
}

impl_merge_by_tuple_key_scalar_value!(SoA, SoA2, SoA2, SoA2, sort_tuple2_by_key, (A: C: left: key_left: out_left, B: D: right: key_right: out_right));
impl_merge_by_tuple_key_scalar_value!(SoA, SoA3, SoA3, SoA3, sort_tuple3_by_key, (A: D: first: key_first: out_first, B: E: second: key_second: out_second, C: F: third: key_third: out_third));

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
            Less: PredicateOp2<(<$first_left as KernelColumn>::Item, $( <$left as KernelColumn>::Item ),+)>,
        {
            type Runtime = <$first_left as KernelColumn>::Runtime;
            type Output = (
                $out_keys<
                    DeviceVec<<$first_left as KernelColumn>::Runtime, <$first_left as KernelColumn>::Item>,
                    $( DeviceVec<<$first_left as KernelColumn>::Runtime, <$left as KernelColumn>::Item> ),+
                >,
                $out_values<$( DeviceVec<<$first_left as KernelColumn>::Runtime, <$value as KernelColumn>::Item> ),+>,
            );

            fn merge_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                left_values: $values<$( $value ),+>,
                right_keys: $right_keys<$first_right, $( $right ),+>,
                right_values: $values<$( $right_value ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&right_keys)?;
                ReadOnlySoA::validate(&left_values)?;
                ReadOnlySoA::validate(&right_values)?;
                let left_first = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                let right_first = super::device_expr_collect_with_policy(policy, &right_keys.$first_field)?;
                let $first_concat = primitive_range::concat_device_with_policy(policy, &left_first, &right_first)?;
                $(
                    let left_key = super::device_expr_collect_with_policy(policy, &self.$field)?;
                    let right_key = super::device_expr_collect_with_policy(policy, &right_keys.$field)?;
                    let $concat = primitive_range::concat_device_with_policy(policy, &left_key, &right_key)?;
                )+
                let indices = primitive_range::indices_u32(policy, $first_concat.len)?;
                let ($first_out, $( $out, )+ sorted_indices) =
                    ordering::$sort_fn(policy, &$first_concat, $( &$concat, )+ &indices, GpuOp::<Less>::new())?;
                $(
                    let left_value = super::device_expr_collect_with_policy(policy, &left_values.$value_field)?;
                    let right_value = super::device_expr_collect_with_policy(policy, &right_values.$value_field)?;
                    super::ensure_same_len(left_value.len, left_first.len)?;
                    super::ensure_same_len(right_value.len, right_first.len)?;
                    let value = primitive_range::concat_device_with_policy(policy, &left_value, &right_value)?;
                    let $value_field = primitive_range::gather_device_with_policy(policy, &value, &sorted_indices)?;
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
            Less: PredicateOp2<(<$first_left as KernelColumn>::Item, $( <$left as KernelColumn>::Item ),+)>,
        {
            type Runtime = <$first_left as KernelColumn>::Runtime;
            type Output = (
                $out_keys<
                    DeviceVec<<$first_left as KernelColumn>::Runtime, <$first_left as KernelColumn>::Item>,
                    $( DeviceVec<<$first_left as KernelColumn>::Runtime, <$left as KernelColumn>::Item> ),+
                >,
                $out_values<$( DeviceVec<<$first_left as KernelColumn>::Runtime, <$value as KernelColumn>::Item> ),+>,
            );

            fn merge_by_key_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                left_values: $values<$( $value ),+>,
                right_keys: $right_keys<$first_right, $( $right ),+>,
                right_values: $values<$( $right_value ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                SoA::validate(&right_keys)?;
                SoA::validate(&left_values)?;
                SoA::validate(&right_values)?;
                let left_first = super::device_expr_collect_with_policy(policy, &self.$first_field)?;
                let right_first = super::device_expr_collect_with_policy(policy, &right_keys.$first_field)?;
                let $first_concat = primitive_range::concat_device_with_policy(policy, &left_first, &right_first)?;
                $(
                    let left_key = super::device_expr_collect_with_policy(policy, &self.$field)?;
                    let right_key = super::device_expr_collect_with_policy(policy, &right_keys.$field)?;
                    let $concat = primitive_range::concat_device_with_policy(policy, &left_key, &right_key)?;
                )+
                let indices = primitive_range::indices_u32(policy, $first_concat.len)?;
                let ($first_out, $( $out, )+ sorted_indices) =
                    ordering::$sort_fn(policy, &$first_concat, $( &$concat, )+ &indices, GpuOp::<Less>::new())?;
                $(
                    let left_value = super::device_expr_collect_with_policy(policy, &left_values.$value_field)?;
                    let right_value = super::device_expr_collect_with_policy(policy, &right_values.$value_field)?;
                    super::ensure_same_len(left_value.len, left_first.len)?;
                    super::ensure_same_len(right_value.len, right_first.len)?;
                    let value = primitive_range::concat_device_with_policy(policy, &left_value, &right_value)?;
                    let $value_field = primitive_range::gather_device_with_policy(policy, &value, &sorted_indices)?;
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
    };
}

impl_merge_by_tuple_key_soa_values_for_key!(SoA2, SoA2, sort_tuple2_by_key, (KA: RA: left: key_left: out_left, KB: RB: right: key_right: out_right));
impl_merge_by_tuple_key_soa_values_for_key!(SoA3, SoA3, sort_tuple3_by_key, (KA: RA: first: key_first: out_first, KB: RB: second: key_second: out_second, KC: RC: third: key_third: out_third));

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
    Less: PredicateOp2<(LeftA::Item, LeftB::Item)>,
{
    type Runtime = LeftA::Runtime;
    type Output = (
        SoA2<DeviceVec<LeftA::Runtime, LeftA::Item>, DeviceVec<LeftA::Runtime, LeftB::Item>>,
        SoA1<DeviceVec<LeftA::Runtime, LeftValue::Item>>,
    );

    fn merge_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        left_values: LeftValue,
        right_keys: SoAView2<RightA, RightB>,
        right_values: RightValue,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&right_keys)?;
        left_values.validate()?;
        right_values.validate()?;
        let left_a = super::device_expr_collect_with_policy(policy, &self.left)?;
        let left_b = super::device_expr_collect_with_policy(policy, &self.right)?;
        let left_values = super::device_expr_collect_with_policy(policy, &left_values)?;
        let right_a = super::device_expr_collect_with_policy(policy, &right_keys.left)?;
        let right_b = super::device_expr_collect_with_policy(policy, &right_keys.right)?;
        let right_values = super::device_expr_collect_with_policy(policy, &right_values)?;
        let key_a = primitive_range::concat_device_with_policy(policy, &left_a, &right_a)?;
        let key_b = primitive_range::concat_device_with_policy(policy, &left_b, &right_b)?;
        let values =
            primitive_range::concat_device_with_policy(policy, &left_values, &right_values)?;
        let (left, right, source) =
            ordering::sort_tuple2_by_key(policy, &key_a, &key_b, &values, GpuOp::<Less>::new())?;
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
    Less: PredicateOp2<(LeftA::Item, LeftB::Item)>,
{
    type Runtime = LeftA::Runtime;
    type Output = (
        SoA2<DeviceVec<LeftA::Runtime, LeftA::Item>, DeviceVec<LeftA::Runtime, LeftB::Item>>,
        SoA2<
            DeviceVec<LeftA::Runtime, LeftValueA::Item>,
            DeviceVec<LeftA::Runtime, LeftValueB::Item>,
        >,
    );

    fn merge_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        left_values: SoAView2<LeftValueA, LeftValueB>,
        right_keys: SoAView2<RightA, RightB>,
        right_values: SoAView2<RightValueA, RightValueB>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&right_keys)?;
        ReadOnlySoA::validate(&left_values)?;
        ReadOnlySoA::validate(&right_values)?;
        let left_a = super::device_expr_collect_with_policy(policy, &self.left)?;
        let left_b = super::device_expr_collect_with_policy(policy, &self.right)?;
        let right_a = super::device_expr_collect_with_policy(policy, &right_keys.left)?;
        let right_b = super::device_expr_collect_with_policy(policy, &right_keys.right)?;
        let key_a = primitive_range::concat_device_with_policy(policy, &left_a, &right_a)?;
        let key_b = primitive_range::concat_device_with_policy(policy, &left_b, &right_b)?;

        let left_value_a = super::device_expr_collect_with_policy(policy, &left_values.left)?;
        let right_value_a = super::device_expr_collect_with_policy(policy, &right_values.left)?;
        let values_a =
            primitive_range::concat_device_with_policy(policy, &left_value_a, &right_value_a)?;
        let (left, right, value_a) =
            ordering::sort_tuple2_by_key(policy, &key_a, &key_b, &values_a, GpuOp::<Less>::new())?;

        let left_value_b = super::device_expr_collect_with_policy(policy, &left_values.right)?;
        let right_value_b = super::device_expr_collect_with_policy(policy, &right_values.right)?;
        let values_b =
            primitive_range::concat_device_with_policy(policy, &left_value_b, &right_value_b)?;
        let (_, _, value_b) =
            ordering::sort_tuple2_by_key(policy, &key_a, &key_b, &values_b, GpuOp::<Less>::new())?;

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
    Less: PredicateOp2<(LeftA::Item, LeftB::Item, LeftC::Item)>,
{
    type Runtime = LeftA::Runtime;
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
        policy: &CubePolicy<Self::Runtime>,
        left_values: LeftValue,
        right_keys: SoAView3<RightA, RightB, RightC>,
        right_values: RightValue,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&right_keys)?;
        left_values.validate()?;
        right_values.validate()?;
        let left_a = super::device_expr_collect_with_policy(policy, &self.first)?;
        let left_b = super::device_expr_collect_with_policy(policy, &self.second)?;
        let left_c = super::device_expr_collect_with_policy(policy, &self.third)?;
        let left_values = super::device_expr_collect_with_policy(policy, &left_values)?;
        let right_a = super::device_expr_collect_with_policy(policy, &right_keys.first)?;
        let right_b = super::device_expr_collect_with_policy(policy, &right_keys.second)?;
        let right_c = super::device_expr_collect_with_policy(policy, &right_keys.third)?;
        let right_values = super::device_expr_collect_with_policy(policy, &right_values)?;
        let key_a = primitive_range::concat_device_with_policy(policy, &left_a, &right_a)?;
        let key_b = primitive_range::concat_device_with_policy(policy, &left_b, &right_b)?;
        let key_c = primitive_range::concat_device_with_policy(policy, &left_c, &right_c)?;
        let values =
            primitive_range::concat_device_with_policy(policy, &left_values, &right_values)?;
        let (first, second, third, source) = ordering::sort_tuple3_by_key(
            policy,
            &key_a,
            &key_b,
            &key_c,
            &values,
            GpuOp::<Less>::new(),
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
    Less: PredicateOp2<(LeftA::Item, LeftB::Item, LeftC::Item)>,
{
    type Runtime = LeftA::Runtime;
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
        policy: &CubePolicy<Self::Runtime>,
        left_values: SoAView2<LeftValueA, LeftValueB>,
        right_keys: SoAView3<RightA, RightB, RightC>,
        right_values: SoAView2<RightValueA, RightValueB>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&right_keys)?;
        ReadOnlySoA::validate(&left_values)?;
        ReadOnlySoA::validate(&right_values)?;
        let left_a = super::device_expr_collect_with_policy(policy, &self.first)?;
        let left_b = super::device_expr_collect_with_policy(policy, &self.second)?;
        let left_c = super::device_expr_collect_with_policy(policy, &self.third)?;
        let right_a = super::device_expr_collect_with_policy(policy, &right_keys.first)?;
        let right_b = super::device_expr_collect_with_policy(policy, &right_keys.second)?;
        let right_c = super::device_expr_collect_with_policy(policy, &right_keys.third)?;
        let key_a = primitive_range::concat_device_with_policy(policy, &left_a, &right_a)?;
        let key_b = primitive_range::concat_device_with_policy(policy, &left_b, &right_b)?;
        let key_c = primitive_range::concat_device_with_policy(policy, &left_c, &right_c)?;

        let left_value_a = super::device_expr_collect_with_policy(policy, &left_values.left)?;
        let right_value_a = super::device_expr_collect_with_policy(policy, &right_values.left)?;
        let values_a =
            primitive_range::concat_device_with_policy(policy, &left_value_a, &right_value_a)?;
        let (first, second, third, value_a) = ordering::sort_tuple3_by_key(
            policy,
            &key_a,
            &key_b,
            &key_c,
            &values_a,
            GpuOp::<Less>::new(),
        )?;

        let left_value_b = super::device_expr_collect_with_policy(policy, &left_values.right)?;
        let right_value_b = super::device_expr_collect_with_policy(policy, &right_values.right)?;
        let values_b =
            primitive_range::concat_device_with_policy(policy, &left_value_b, &right_value_b)?;
        let (_, _, _, value_b) = ordering::sort_tuple3_by_key(
            policy,
            &key_a,
            &key_b,
            &key_c,
            &values_b,
            GpuOp::<Less>::new(),
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
    Less: PredicateOp2<(LeftA::Item, LeftB::Item, LeftC::Item)>,
{
    type Runtime = LeftA::Runtime;
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
        policy: &CubePolicy<Self::Runtime>,
        left_values: SoAView3<LeftValueA, LeftValueB, LeftValueC>,
        right_keys: SoAView3<RightA, RightB, RightC>,
        right_values: SoAView3<RightValueA, RightValueB, RightValueC>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&right_keys)?;
        ReadOnlySoA::validate(&left_values)?;
        ReadOnlySoA::validate(&right_values)?;
        let left_a = super::device_expr_collect_with_policy(policy, &self.first)?;
        let left_b = super::device_expr_collect_with_policy(policy, &self.second)?;
        let left_c = super::device_expr_collect_with_policy(policy, &self.third)?;
        let right_a = super::device_expr_collect_with_policy(policy, &right_keys.first)?;
        let right_b = super::device_expr_collect_with_policy(policy, &right_keys.second)?;
        let right_c = super::device_expr_collect_with_policy(policy, &right_keys.third)?;
        let key_a = primitive_range::concat_device_with_policy(policy, &left_a, &right_a)?;
        let key_b = primitive_range::concat_device_with_policy(policy, &left_b, &right_b)?;
        let key_c = primitive_range::concat_device_with_policy(policy, &left_c, &right_c)?;

        let left_value_a = super::device_expr_collect_with_policy(policy, &left_values.first)?;
        let right_value_a = super::device_expr_collect_with_policy(policy, &right_values.first)?;
        let values_a =
            primitive_range::concat_device_with_policy(policy, &left_value_a, &right_value_a)?;
        let (first, second, third, value_a) = ordering::sort_tuple3_by_key(
            policy,
            &key_a,
            &key_b,
            &key_c,
            &values_a,
            GpuOp::<Less>::new(),
        )?;

        let left_value_b = super::device_expr_collect_with_policy(policy, &left_values.second)?;
        let right_value_b = super::device_expr_collect_with_policy(policy, &right_values.second)?;
        let values_b =
            primitive_range::concat_device_with_policy(policy, &left_value_b, &right_value_b)?;
        let (_, _, _, value_b) = ordering::sort_tuple3_by_key(
            policy,
            &key_a,
            &key_b,
            &key_c,
            &values_b,
            GpuOp::<Less>::new(),
        )?;

        let left_value_c = super::device_expr_collect_with_policy(policy, &left_values.third)?;
        let right_value_c = super::device_expr_collect_with_policy(policy, &right_values.third)?;
        let values_c =
            primitive_range::concat_device_with_policy(policy, &left_value_c, &right_value_c)?;
        let (_, _, _, value_c) = ordering::sort_tuple3_by_key(
            policy,
            &key_a,
            &key_b,
            &key_c,
            &values_c,
            GpuOp::<Less>::new(),
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
    Less: PredicateOp2<LeftKey::Item>,
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
    LeftKey: MergeByKeyInput<LeftValue, RightKey, RightValue, super::Tuple1Less<Less>>,
{
    type Runtime = <LeftKey as MergeByKeyInput<
        LeftValue,
        RightKey,
        RightValue,
        super::Tuple1Less<Less>,
    >>::Runtime;
    type Output = <LeftKey as MergeByKeyInput<
        LeftValue,
        RightKey,
        RightValue,
        super::Tuple1Less<Less>,
    >>::Output;

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
            super::Tuple1Less<Less>,
        >>::merge_by_key_input(
            self.0,
            policy,
            left_values.0,
            right_keys.0,
            right_values.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }
}

impl<Source, Less> SortInput<Less> for SoA1<Source>
where
    Self: SoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: PredicateOp2<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn sort_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        Ok(SoA1 {
            source: ordering::sort_input_with_policy(policy, &self.source, GpuOp::<Less>::new())?,
        })
    }
}

impl<Source, Less> SortInput<Less> for Source
where
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: PredicateOp2<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn sort_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoA1<Source> as SortInput<Less>>::sort_input(SoA1 { source: self }, policy, less)
    }
}

impl<Source, Less> SortInput<Less> for (Source,)
where
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: PredicateOp2<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn sort_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoA1<Source> as SortInput<super::Tuple1Less<Less>>>::sort_input(
            SoA1 { source: self.0 },
            policy,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }
}

impl<Left, Right, Less> SortInput<Less> for (Left, Right)
where
    SoAView2<Left, Right>: SortInput<Less>,
    Left: KernelColumnAt<S0>,
    Right: KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
{
    type Runtime = <SoAView2<Left, Right> as SortInput<Less>>::Runtime;
    type Output = <SoAView2<Left, Right> as SortInput<Less>>::Output;

    fn sort_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as SortInput<Less>>::sort_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            less,
        )
    }
}

impl<First, Second, Third, Less> SortInput<Less> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: SortInput<Less>,
    First: KernelColumnAt<S0>,
    Second: KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
{
    type Runtime = <SoAView3<First, Second, Third> as SortInput<Less>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as SortInput<Less>>::Output;

    fn sort_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as SortInput<Less>>::sort_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            less,
        )
    }
}

impl<Left, Right, Less> SortInput<Less> for SoA2<Left, Right>
where
    Self: ReadOnlySoA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Right: ReadOnlyKernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: PredicateOp2<(Left::Item, Right::Item)>,
{
    type Runtime = Left::Runtime;
    type Output = SoA2<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

    fn sort_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let (first, second) =
            ordering::sort_tuple2_input(policy, &self.left, &self.right, GpuOp::<Less>::new())?;
        Ok(SoA2 {
            left: first,
            right: second,
        })
    }
}

impl<First, Second, Third, Less> SortInput<Less> for SoA3<First, Second, Third>
where
    Self: ReadOnlySoA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
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
    Less: PredicateOp2<(First::Item, Second::Item, Third::Item)>,
{
    type Runtime = First::Runtime;
    type Output = SoA3<
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    >;

    fn sort_input(
        self,
        policy: &CubePolicy<First::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let (first, second, third) = ordering::sort_tuple3_input(
            policy,
            &self.first,
            &self.second,
            &self.third,
            GpuOp::<Less>::new(),
        )?;
        Ok(SoA3 {
            first,
            second,
            third,
        })
    }
}

impl<Left, Right, Less> SortInput<Less> for SoAView2<Left, Right>
where
    Self: ReadOnlySoA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Right: ReadOnlyKernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: PredicateOp2<(Left::Item, Right::Item)>,
{
    type Runtime = Left::Runtime;
    type Output = SoA2<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

    fn sort_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let (left, right) =
            ordering::sort_tuple2_input(policy, &self.left, &self.right, GpuOp::<Less>::new())?;
        Ok(SoA2 { left, right })
    }
}

impl<First, Second, Third, Less> SortInput<Less> for SoAView3<First, Second, Third>
where
    Self: ReadOnlySoA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
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
    Less: PredicateOp2<(First::Item, Second::Item, Third::Item)>,
{
    type Runtime = First::Runtime;
    type Output = SoA3<
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    >;

    fn sort_input(
        self,
        policy: &CubePolicy<First::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let (first, second, third) = ordering::sort_tuple3_input(
            policy,
            &self.first,
            &self.second,
            &self.third,
            GpuOp::<Less>::new(),
        )?;
        Ok(SoA3 {
            first,
            second,
            third,
        })
    }
}

/// Sorts read-only SoA input and returns owned device storage.
pub fn sort<R, Input, Less>(
    policy: &CubePolicy<R>,
    input: Input,
    _less: Less,
) -> Result<<<Input as SortInput<Less>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Input: SortInput<Less, Runtime = R>,
    <Input as SortInput<Less>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(policy, input.sort_input(policy, GpuOp::<Less>::new())?)
}

/// Merges two sorted read-only inputs into owned device storage.
///
/// This is a borrowing algorithm. Both inputs are read, and the merged output is
/// newly materialized.
pub fn merge<R, Left, Right, Less>(
    policy: &CubePolicy<R>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<<Left as PairOrderingInput<Right, Less>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Left: PairOrderingInput<Right, Less, Runtime = R>,
    <Left as PairOrderingInput<Right, Less>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        left.merge_input(policy, right, GpuOp::<Less>::new())?,
    )
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

impl<LeftA, LeftB, LeftValue, RightA, RightB, RightValue, Less>
    MergeByKeyInput<LeftValue, (RightA, RightB), RightValue, Less> for (LeftA, LeftB)
where
    SoAView2<LeftA, LeftB>: MergeByKeyInput<LeftValue, SoAView2<RightA, RightB>, RightValue, Less>,
{
    type Runtime = <SoAView2<LeftA, LeftB> as MergeByKeyInput<
        LeftValue,
        SoAView2<RightA, RightB>,
        RightValue,
        Less,
    >>::Runtime;
    type Output = <SoAView2<LeftA, LeftB> as MergeByKeyInput<
        LeftValue,
        SoAView2<RightA, RightB>,
        RightValue,
        Less,
    >>::Output;

    fn merge_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        left_values: LeftValue,
        right_keys: (RightA, RightB),
        right_values: RightValue,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<LeftA, LeftB> as MergeByKeyInput<
            LeftValue,
            SoAView2<RightA, RightB>,
            RightValue,
            Less,
        >>::merge_by_key_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            left_values,
            SoAView2 {
                left: right_keys.0,
                right: right_keys.1,
            },
            right_values,
            less,
        )
    }
}

impl<LeftA, LeftB, LeftValueA, LeftValueB, RightA, RightB, RightValueA, RightValueB, Less>
    MergeByKeyInput<
        (LeftValueA, LeftValueB),
        SoAView2<RightA, RightB>,
        (RightValueA, RightValueB),
        Less,
    > for SoAView2<LeftA, LeftB>
where
    SoAView2<LeftA, LeftB>: MergeByKeyInput<
            SoAView2<LeftValueA, LeftValueB>,
            SoAView2<RightA, RightB>,
            SoAView2<RightValueA, RightValueB>,
            Less,
        >,
{
    type Runtime = <SoAView2<LeftA, LeftB> as MergeByKeyInput<
        SoAView2<LeftValueA, LeftValueB>,
        SoAView2<RightA, RightB>,
        SoAView2<RightValueA, RightValueB>,
        Less,
    >>::Runtime;
    type Output = <SoAView2<LeftA, LeftB> as MergeByKeyInput<
        SoAView2<LeftValueA, LeftValueB>,
        SoAView2<RightA, RightB>,
        SoAView2<RightValueA, RightValueB>,
        Less,
    >>::Output;

    fn merge_by_key_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        left_values: (LeftValueA, LeftValueB),
        right_keys: SoAView2<RightA, RightB>,
        right_values: (RightValueA, RightValueB),
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<LeftA, LeftB> as MergeByKeyInput<
            SoAView2<LeftValueA, LeftValueB>,
            SoAView2<RightA, RightB>,
            SoAView2<RightValueA, RightValueB>,
            Less,
        >>::merge_by_key_input(
            self,
            policy,
            SoAView2 {
                left: left_values.0,
                right: left_values.1,
            },
            right_keys,
            SoAView2 {
                left: right_values.0,
                right: right_values.1,
            },
            less,
        )
    }
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

/// Computes the sorted set union of two sorted device vectors.
pub fn set_union<R, Left, Right, Less>(
    policy: &CubePolicy<R>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<<Left as PairOrderingInput<Right, Less>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Left: PairOrderingInput<Right, Less, Runtime = R>,
    <Left as PairOrderingInput<Right, Less>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        left.set_union_input(policy, right, GpuOp::<Less>::new())?,
    )
}

/// Computes the sorted set intersection of two sorted device vectors.
pub fn set_intersection<R, Left, Right, Less>(
    policy: &CubePolicy<R>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<<Left as PairOrderingInput<Right, Less>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Left: PairOrderingInput<Right, Less, Runtime = R>,
    <Left as PairOrderingInput<Right, Less>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        left.set_intersection_input(policy, right, GpuOp::<Less>::new())?,
    )
}

/// Computes the sorted set difference `left - right`.
pub fn set_difference<R, Left, Right, Less>(
    policy: &CubePolicy<R>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<<Left as PairOrderingInput<Right, Less>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Left: PairOrderingInput<Right, Less, Runtime = R>,
    <Left as PairOrderingInput<Right, Less>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        left.set_difference_input(policy, right, GpuOp::<Less>::new())?,
    )
}
