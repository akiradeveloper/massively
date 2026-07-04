use crate::{
    detail::{
        api::search::{
            BLOCK_SEARCH_SIZE, device_expr_adjacent_find, device_expr_find_first_of,
            device_expr_is_sorted_until, device_expr_lexicographical_compare,
            device_expr_lower_bound, device_expr_mismatch, device_expr_upper_bound,
            search_block_count, stage_search_column,
        },
        device::{
            DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA2, SoA3, SoAView2,
            SoAView3, SoAView4, SoAView5, SoAView6, SoAView7,
        },
        op::kernel::BinaryPredicateOp,
    },
    error::Error,
    expr::DeviceGpuExpr,
    index::MIndex,
    kernels::*,
    policy::CubePolicy,
};
use cubecl::prelude::*;

pub(in crate::detail) struct SearchControlApply;

impl SearchControlApply {
    pub(in crate::detail) fn adjacent_find_expr<Source, Pred>(
        policy: &CubePolicy<Source::Runtime>,
        input: &Source,
    ) -> Result<Option<MIndex>, Error>
    where
        Source: KernelColumn + KernelColumnAt<S0>,
        Source::Item: CubePrimitive + CubeElement,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Pred: BinaryPredicateOp<Source::Item>,
    {
        device_expr_adjacent_find::<Source, Pred>(policy, input)
    }

    pub(in crate::detail) fn lower_bound_expr<Source, Less>(
        policy: &CubePolicy<Source::Runtime>,
        input: &Source,
        value: Source::Item,
    ) -> Result<MIndex, Error>
    where
        Source: KernelColumn + KernelColumnAt<S0>,
        Source::Item: CubePrimitive + CubeElement,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Less: BinaryPredicateOp<Source::Item>,
    {
        device_expr_lower_bound::<Source, Less>(policy, input, value)
    }

    pub(in crate::detail) fn upper_bound_expr<Source, Less>(
        policy: &CubePolicy<Source::Runtime>,
        input: &Source,
        value: Source::Item,
    ) -> Result<MIndex, Error>
    where
        Source: KernelColumn + KernelColumnAt<S0>,
        Source::Item: CubePrimitive + CubeElement,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Less: BinaryPredicateOp<Source::Item>,
    {
        device_expr_upper_bound::<Source, Less>(policy, input, value)
    }

    pub(in crate::detail) fn is_sorted_until_expr<Source, Less>(
        policy: &CubePolicy<Source::Runtime>,
        input: &Source,
    ) -> Result<MIndex, Error>
    where
        Source: KernelColumn + KernelColumnAt<S0>,
        Source::Item: CubePrimitive + CubeElement,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Less: BinaryPredicateOp<Source::Item>,
    {
        device_expr_is_sorted_until::<Source, Less>(policy, input)
    }

    pub(in crate::detail) fn mismatch_expr<Left, Right, Op>(
        policy: &CubePolicy<Left::Runtime>,
        left: &Left,
        right: &Right,
    ) -> Result<Option<MIndex>, Error>
    where
        Left: KernelColumn + KernelColumnAt<S0>,
        Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
        Left::Item: CubePrimitive + CubeElement,
        Left::Expr: DeviceGpuExpr<Left::Item>,
        Right::Expr: DeviceGpuExpr<Right::Item>,
        Op: BinaryPredicateOp<Left::Item>,
    {
        device_expr_mismatch::<Left, Right, Op>(policy, left, right)
    }

    pub(in crate::detail) fn find_first_of_expr<Left, Right, Op>(
        policy: &CubePolicy<Left::Runtime>,
        input: &Left,
        needles: &Right,
    ) -> Result<Option<MIndex>, Error>
    where
        Left: KernelColumn + KernelColumnAt<S0>,
        Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
        Left::Item: CubePrimitive + CubeElement,
        Left::Expr: DeviceGpuExpr<Left::Item>,
        Right::Expr: DeviceGpuExpr<Right::Item>,
        Op: BinaryPredicateOp<Left::Item>,
    {
        device_expr_find_first_of::<Left, Right, Op>(policy, input, needles)
    }

    pub(in crate::detail) fn lexicographical_compare_expr<Left, Right, Less>(
        policy: &CubePolicy<Left::Runtime>,
        left: &Left,
        right: &Right,
    ) -> Result<bool, Error>
    where
        Left: KernelColumn + KernelColumnAt<S0>,
        Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
        Left::Item: CubePrimitive + CubeElement,
        Left::Expr: DeviceGpuExpr<Left::Item>,
        Right::Expr: DeviceGpuExpr<Right::Item>,
        Less: BinaryPredicateOp<Left::Item>,
    {
        device_expr_lexicographical_compare::<Left, Right, Less>(policy, left, right)
    }
}

pub(in crate::detail) struct SearchPayloadLaunch {
    pub(in crate::detail) source_len_handle: cubecl::server::Handle,
    pub(in crate::detail) value_len_handle: cubecl::server::Handle,
    pub(in crate::detail) output_handle: cubecl::server::Handle,
    pub(in crate::detail) block_count_u32: u32,
    pub(in crate::detail) value_len: usize,
}

pub(in crate::detail) struct SearchPayloadApply;

impl SearchPayloadApply {
    pub(in crate::detail) fn empty_or_zero<R: Runtime>(
        policy: &CubePolicy<R>,
        source_len: usize,
        value_len: usize,
    ) -> Option<Result<DeviceVec<R, MIndex>, Error>> {
        if value_len == 0 {
            return Some(Ok(policy.empty_device_vec()));
        }
        if source_len == 0 {
            return Some(policy.device_filled(value_len, 0 as MIndex));
        }
        None
    }

    pub(in crate::detail) fn prepare<R: Runtime>(
        policy: &CubePolicy<R>,
        source_len: usize,
        value_len: usize,
    ) -> Result<SearchPayloadLaunch, Error> {
        let source_len_u32 =
            u32::try_from(source_len).map_err(|_| Error::LengthTooLarge { len: source_len })?;
        let value_len_u32 =
            u32::try_from(value_len).map_err(|_| Error::LengthTooLarge { len: value_len })?;
        let client = policy.client();
        Ok(SearchPayloadLaunch {
            source_len_handle: client.create_from_slice(u32::as_bytes(&[source_len_u32])),
            value_len_handle: client.create_from_slice(u32::as_bytes(&[value_len_u32])),
            output_handle: client.empty(value_len * std::mem::size_of::<MIndex>()),
            block_count_u32: search_block_count(value_len)?,
            value_len,
        })
    }

    pub(in crate::detail) fn finish<R: Runtime>(
        policy: &CubePolicy<R>,
        launch: SearchPayloadLaunch,
    ) -> DeviceVec<R, MIndex> {
        DeviceVec::from_handle(policy.id(), launch.output_handle, launch.value_len)
    }

    pub(in crate::detail) fn lower_bound_many_expr<Source, Values, Less>(
        policy: &CubePolicy<Source::Runtime>,
        input: &Source,
        values: &Values,
    ) -> Result<DeviceVec<Source::Runtime, MIndex>, Error>
    where
        Source: KernelColumn + KernelColumnAt<S0>,
        Values: KernelColumn<Runtime = Source::Runtime, Item = Source::Item> + KernelColumnAt<S0>,
        Source::Item: CubePrimitive + CubeElement,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Values::Expr: DeviceGpuExpr<Values::Item>,
        Less: BinaryPredicateOp<Source::Item>,
    {
        input.validate()?;
        values.validate()?;
        let source_len = input.len();
        let value_len = values.len();
        if let Some(output) = Self::empty_or_zero(policy, source_len, value_len) {
            return output;
        }

        let launch = Self::prepare(policy, source_len, value_len)?;
        let input = stage_search_column(policy, input)?;
        let values = stage_search_column(policy, values)?;

        unsafe {
            lower_bound_device_expr_many_kernel::launch_unchecked::<
                Source::Item,
                Source::Expr,
                Values::Expr,
                Less,
                Source::Runtime,
            >(
                policy.client(),
                CubeCount::Static(launch.block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                unsafe { BufferArg::from_raw_parts(input.slot0.0.clone(), input.slot0.1) },
                unsafe { BufferArg::from_raw_parts(input.slot1.0.clone(), input.slot1.1) },
                unsafe { BufferArg::from_raw_parts(input.slot2.0.clone(), input.slot2.1) },
                unsafe { BufferArg::from_raw_parts(input.slot3.0.clone(), input.slot3.1) },
                unsafe { BufferArg::from_raw_parts(input.slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(values.slot0.0.clone(), values.slot0.1) },
                unsafe { BufferArg::from_raw_parts(values.slot1.0.clone(), values.slot1.1) },
                unsafe { BufferArg::from_raw_parts(values.slot2.0.clone(), values.slot2.1) },
                unsafe { BufferArg::from_raw_parts(values.slot3.0.clone(), values.slot3.1) },
                unsafe { BufferArg::from_raw_parts(values.slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(launch.source_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(launch.value_len_handle.clone(), 1) },
                unsafe {
                    BufferArg::from_raw_parts(launch.output_handle.clone(), launch.value_len)
                },
            );
        }

        Ok(Self::finish(policy, launch))
    }

    pub(in crate::detail) fn upper_bound_many_expr<Source, Values, Less>(
        policy: &CubePolicy<Source::Runtime>,
        input: &Source,
        values: &Values,
    ) -> Result<DeviceVec<Source::Runtime, MIndex>, Error>
    where
        Source: KernelColumn + KernelColumnAt<S0>,
        Values: KernelColumn<Runtime = Source::Runtime, Item = Source::Item> + KernelColumnAt<S0>,
        Source::Item: CubePrimitive + CubeElement,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Values::Expr: DeviceGpuExpr<Values::Item>,
        Less: BinaryPredicateOp<Source::Item>,
    {
        input.validate()?;
        values.validate()?;
        let source_len = input.len();
        let value_len = values.len();
        if let Some(output) = Self::empty_or_zero(policy, source_len, value_len) {
            return output;
        }

        let launch = Self::prepare(policy, source_len, value_len)?;
        let input = stage_search_column(policy, input)?;
        let values = stage_search_column(policy, values)?;

        unsafe {
            upper_bound_device_expr_many_kernel::launch_unchecked::<
                Source::Item,
                Source::Expr,
                Values::Expr,
                Less,
                Source::Runtime,
            >(
                policy.client(),
                CubeCount::Static(launch.block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                unsafe { BufferArg::from_raw_parts(input.slot0.0.clone(), input.slot0.1) },
                unsafe { BufferArg::from_raw_parts(input.slot1.0.clone(), input.slot1.1) },
                unsafe { BufferArg::from_raw_parts(input.slot2.0.clone(), input.slot2.1) },
                unsafe { BufferArg::from_raw_parts(input.slot3.0.clone(), input.slot3.1) },
                unsafe { BufferArg::from_raw_parts(input.slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(values.slot0.0.clone(), values.slot0.1) },
                unsafe { BufferArg::from_raw_parts(values.slot1.0.clone(), values.slot1.1) },
                unsafe { BufferArg::from_raw_parts(values.slot2.0.clone(), values.slot2.1) },
                unsafe { BufferArg::from_raw_parts(values.slot3.0.clone(), values.slot3.1) },
                unsafe { BufferArg::from_raw_parts(values.slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(launch.source_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(launch.value_len_handle.clone(), 1) },
                unsafe {
                    BufferArg::from_raw_parts(launch.output_handle.clone(), launch.value_len)
                },
            );
        }

        Ok(Self::finish(policy, launch))
    }
}

pub(in crate::detail) trait TupleSearchPayloadApply<Values, Less>: Sized {
    type Runtime: Runtime;

    fn lower_bound_many_payload(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
    ) -> Result<DeviceVec<Self::Runtime, MIndex>, Error>;

    fn upper_bound_many_payload(
        self,
        policy: &CubePolicy<Self::Runtime>,
        values: Values,
    ) -> Result<DeviceVec<Self::Runtime, MIndex>, Error>;
}

macro_rules! impl_tuple_search_payload_apply {
    (
        $name:ident < $first:ident, $( $rest:ident ),+ > {
            $first_field:ident,
            $( $field:ident ),+
        },
        $lower_bound_many_kernel:ident,
        $upper_bound_many_kernel:ident
    ) => {
        impl<$first, $( $rest ),+, Less>
            TupleSearchPayloadApply<$name<$first, $( $rest ),+>, Less>
            for $name<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Less: BinaryPredicateOp<(
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            fn lower_bound_many_payload(
                self,
                policy: &CubePolicy<Self::Runtime>,
                values: $name<$first, $( $rest ),+>,
            ) -> Result<DeviceVec<Self::Runtime, MIndex>, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&values)?;
                let source_len = self.$first_field.len();
                let value_len = values.$first_field.len();
                if let Some(output) =
                    SearchPayloadApply::empty_or_zero(policy, source_len, value_len)
                {
                    return output;
                }
                let launch = SearchPayloadApply::prepare(policy, source_len, value_len)?;
                let $first_field = (
                    stage_search_column(policy, &self.$first_field)?,
                    stage_search_column(policy, &values.$first_field)?,
                );
                $(
                    let $field = (
                        stage_search_column(policy, &self.$field)?,
                        stage_search_column(policy, &values.$field)?,
                    );
                )+
                unsafe {
                    $lower_bound_many_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        <$first as KernelColumn>::Expr,
                        <$first as KernelColumn>::Expr,
                        $( <$rest as KernelColumn>::Expr, )+
                        $( <$rest as KernelColumn>::Expr, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        policy.client(),
                        CubeCount::Static(launch.block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        unsafe { BufferArg::from_raw_parts($first_field.0.slot0.0.clone(), $first_field.0.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.0.slot1.0.clone(), $first_field.0.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.0.slot2.0.clone(), $first_field.0.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.0.slot3.0.clone(), $first_field.0.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.0.slot_offsets.clone(), 4) },
                        unsafe { BufferArg::from_raw_parts($first_field.1.slot0.0.clone(), $first_field.1.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.1.slot1.0.clone(), $first_field.1.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.1.slot2.0.clone(), $first_field.1.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.1.slot3.0.clone(), $first_field.1.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.1.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.0.slot0.0.clone(), $field.0.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot1.0.clone(), $field.0.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot2.0.clone(), $field.0.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot3.0.clone(), $field.0.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot_offsets.clone(), 4) },
                            unsafe { BufferArg::from_raw_parts($field.1.slot0.0.clone(), $field.1.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($field.1.slot1.0.clone(), $field.1.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($field.1.slot2.0.clone(), $field.1.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($field.1.slot3.0.clone(), $field.1.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($field.1.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(launch.source_len_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(launch.value_len_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(launch.output_handle.clone(), launch.value_len) },
                    );
                }
                Ok(SearchPayloadApply::finish(policy, launch))
            }

            fn upper_bound_many_payload(
                self,
                policy: &CubePolicy<Self::Runtime>,
                values: $name<$first, $( $rest ),+>,
            ) -> Result<DeviceVec<Self::Runtime, MIndex>, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&values)?;
                let source_len = self.$first_field.len();
                let value_len = values.$first_field.len();
                if let Some(output) =
                    SearchPayloadApply::empty_or_zero(policy, source_len, value_len)
                {
                    return output;
                }
                let launch = SearchPayloadApply::prepare(policy, source_len, value_len)?;
                let $first_field = (
                    stage_search_column(policy, &self.$first_field)?,
                    stage_search_column(policy, &values.$first_field)?,
                );
                $(
                    let $field = (
                        stage_search_column(policy, &self.$field)?,
                        stage_search_column(policy, &values.$field)?,
                    );
                )+
                unsafe {
                    $upper_bound_many_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        <$first as KernelColumn>::Expr,
                        <$first as KernelColumn>::Expr,
                        $( <$rest as KernelColumn>::Expr, )+
                        $( <$rest as KernelColumn>::Expr, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        policy.client(),
                        CubeCount::Static(launch.block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        unsafe { BufferArg::from_raw_parts($first_field.0.slot0.0.clone(), $first_field.0.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.0.slot1.0.clone(), $first_field.0.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.0.slot2.0.clone(), $first_field.0.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.0.slot3.0.clone(), $first_field.0.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.0.slot_offsets.clone(), 4) },
                        unsafe { BufferArg::from_raw_parts($first_field.1.slot0.0.clone(), $first_field.1.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.1.slot1.0.clone(), $first_field.1.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.1.slot2.0.clone(), $first_field.1.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.1.slot3.0.clone(), $first_field.1.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.1.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.0.slot0.0.clone(), $field.0.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot1.0.clone(), $field.0.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot2.0.clone(), $field.0.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot3.0.clone(), $field.0.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot_offsets.clone(), 4) },
                            unsafe { BufferArg::from_raw_parts($field.1.slot0.0.clone(), $field.1.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($field.1.slot1.0.clone(), $field.1.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($field.1.slot2.0.clone(), $field.1.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($field.1.slot3.0.clone(), $field.1.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($field.1.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(launch.source_len_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(launch.value_len_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(launch.output_handle.clone(), launch.value_len) },
                    );
                }
                Ok(SearchPayloadApply::finish(policy, launch))
            }
        }
    };
}

impl_tuple_search_payload_apply!(
    SoAView2<A, B> { left, right },
    tuple2_lower_bound_device_expr_many_kernel,
    tuple2_upper_bound_device_expr_many_kernel
);
impl_tuple_search_payload_apply!(
    SoAView3<A, B, C> { first, second, third },
    tuple3_lower_bound_device_expr_many_kernel,
    tuple3_upper_bound_device_expr_many_kernel
);
impl_tuple_search_payload_apply!(
    SoAView4<A, B, C, D> { a, b, c, d },
    tuple4_lower_bound_device_expr_many_kernel,
    tuple4_upper_bound_device_expr_many_kernel
);
impl_tuple_search_payload_apply!(
    SoAView5<A, B, C, D, E> { a, b, c, d, e },
    tuple5_lower_bound_device_expr_many_kernel,
    tuple5_upper_bound_device_expr_many_kernel
);
impl_tuple_search_payload_apply!(
    SoAView6<A, B, C, D, E, F> { a, b, c, d, e, f },
    tuple6_lower_bound_device_expr_many_kernel,
    tuple6_upper_bound_device_expr_many_kernel
);
impl_tuple_search_payload_apply!(
    SoAView7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g },
    tuple7_lower_bound_device_expr_many_kernel,
    tuple7_upper_bound_device_expr_many_kernel
);
impl_tuple_search_payload_apply!(
    SoA2<A, B> { left, right },
    tuple2_lower_bound_device_expr_many_kernel,
    tuple2_upper_bound_device_expr_many_kernel
);
impl_tuple_search_payload_apply!(
    SoA3<A, B, C> { first, second, third },
    tuple3_lower_bound_device_expr_many_kernel,
    tuple3_upper_bound_device_expr_many_kernel
);
