use crate::{
    detail::op::kernel::BinaryPredicateOp,
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, ReadOnlySoA, S0, SoA2, SoA3, SoAView1, SoAView2,
        SoAView3, SoAView4, SoAView5, SoAView6, SoAView7,
    },
    error::Error,
    expr::DeviceGpuExpr,
    index::{MIndex, mindex_from_usize},
    kernels::*,
    op::GpuOp,
    policy::CubePolicy,
    primitives::scan,
};
use cubecl::prelude::*;

const BLOCK_SEARCH_SIZE: u32 = 256;

fn search_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_SEARCH_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

struct StagedSearchColumn {
    slot0: (cubecl::server::Handle, usize),
    slot1: (cubecl::server::Handle, usize),
    slot2: (cubecl::server::Handle, usize),
    slot3: (cubecl::server::Handle, usize),
    slot_offsets: cubecl::server::Handle,
}

struct SearchPayloadLaunch {
    source_len_handle: cubecl::server::Handle,
    value_len_handle: cubecl::server::Handle,
    output_handle: cubecl::server::Handle,
    block_count_u32: u32,
    value_len: usize,
}

struct SearchPayloadApply;

impl SearchPayloadApply {
    fn empty_or_zero<R: Runtime>(
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

    fn prepare<R: Runtime>(
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

    fn finish<R: Runtime>(
        policy: &CubePolicy<R>,
        launch: SearchPayloadLaunch,
    ) -> DeviceVec<R, MIndex> {
        DeviceVec::from_handle(policy.id(), launch.output_handle, launch.value_len)
    }

    fn lower_bound_many_expr<Source, Values, Less>(
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

    fn upper_bound_many_expr<Source, Values, Less>(
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

fn stage_search_column<Source>(
    policy: &CubePolicy<Source::Runtime>,
    source: &Source,
) -> Result<StagedSearchColumn, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
{
    let bindings = source.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(policy.client())?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    Ok(StagedSearchColumn {
        slot0: (slot0.0.clone(), slot0.1),
        slot1: (slot1.0.clone(), slot1.1),
        slot2: (slot2.0.clone(), slot2.1),
        slot3: (slot3.0.clone(), slot3.1),
        slot_offsets,
    })
}

fn device_expr_mismatch<Left, Right, Op>(
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
    left.validate()?;
    right.validate()?;
    let min_len = left.len().min(right.len());
    if min_len == 0 {
        return if left.len() == right.len() {
            Ok(None)
        } else {
            Ok(Some(0))
        };
    }

    let client = policy.client();
    let block_count_u32 = search_block_count(min_len)?;
    let left_bindings = left.stage(policy)?;
    let right_bindings = right.stage(policy)?;
    let left_slot_offsets = left_bindings.slot_offsets_handle(client)?;
    let right_slot_offsets = right_bindings.slot_offsets_handle(client)?;
    let left_slot0 = left_bindings.slots.first().unwrap();
    let left_slot1 = left_bindings.slots.get(1).unwrap_or(left_slot0);
    let left_slot2 = left_bindings.slots.get(2).unwrap_or(left_slot0);
    let left_slot3 = left_bindings.slots.get(3).unwrap_or(left_slot0);
    let right_slot0 = right_bindings.slots.first().unwrap();
    let right_slot1 = right_bindings.slots.get(1).unwrap_or(right_slot0);
    let right_slot2 = right_bindings.slots.get(2).unwrap_or(right_slot0);
    let right_slot3 = right_bindings.slots.get(3).unwrap_or(right_slot0);
    let flag_handle = client.empty(min_len * std::mem::size_of::<u32>());

    unsafe {
        mismatch_device_expr_flags_kernel::launch_unchecked::<
            Left::Item,
            Left::Expr,
            Right::Expr,
            Op,
            Left::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SEARCH_SIZE),
            unsafe { BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1) },
            unsafe { BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1) },
            unsafe { BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1) },
            unsafe { BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1) },
            unsafe { BufferArg::from_raw_parts(left_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1) },
            unsafe { BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1) },
            unsafe { BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1) },
            unsafe { BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1) },
            unsafe { BufferArg::from_raw_parts(right_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), min_len) },
        );
    }

    let control = crate::detail::control::SearchControl::from_flags(flag_handle, min_len, min_len);
    if let Some(index) = super::QueryApply::first_flag(policy, control)? {
        return Ok(Some(index));
    }

    if left.len() == right.len() {
        Ok(None)
    } else {
        Ok(Some(mindex_from_usize(min_len)?))
    }
}

fn device_expr_adjacent_find<Source, Pred>(
    policy: &CubePolicy<Source::Runtime>,
    input: &Source,
) -> Result<Option<MIndex>, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<Source::Item>,
{
    input.validate()?;
    let len = input.len();
    if len < 2 {
        return Ok(None);
    }

    let client = policy.client();
    let block_count_u32 = search_block_count(len)?;
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());
    let bindings = input.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);

    unsafe {
        adjacent_find_device_expr_flags_kernel::launch_unchecked::<
            Source::Item,
            Source::Expr,
            Pred,
            Source::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SEARCH_SIZE),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
        );
    }

    let control = crate::detail::control::SearchControl::from_flags(flag_handle, len, len - 1);
    super::QueryApply::first_flag(policy, control)
}

fn device_expr_is_sorted_until<Source, Less>(
    policy: &CubePolicy<Source::Runtime>,
    input: &Source,
) -> Result<MIndex, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    input.validate()?;
    let len = input.len();
    if len <= 1 {
        return mindex_from_usize(len);
    }

    let client = policy.client();
    let block_count_u32 = search_block_count(len)?;
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());
    let bindings = input.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);

    unsafe {
        sorted_break_device_expr_flags_kernel::launch_unchecked::<
            Source::Item,
            Source::Expr,
            Less,
            Source::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SEARCH_SIZE),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
        );
    }

    let control = crate::detail::control::SearchControl::from_flags(flag_handle, len, len);
    super::QueryApply::first_flag_or(policy, control, mindex_from_usize(len)?)
}

fn device_expr_lower_bound<Source, Less>(
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
    input.validate()?;
    let len = input.len();
    if len == 0 {
        return Ok(0);
    }

    let client = policy.client();
    let block_count_u32 = search_block_count(len)?;
    let value_handle = client.create_from_slice(Source::Item::as_bytes(&[value]));
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());
    let bindings = input.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);

    unsafe {
        lower_bound_device_expr_flags_kernel::launch_unchecked::<
            Source::Item,
            Source::Expr,
            Less,
            Source::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SEARCH_SIZE),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(value_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
        );
    }

    let control = crate::detail::control::SearchControl::from_flags(flag_handle, len, len);
    super::QueryApply::first_flag_or(policy, control, mindex_from_usize(len)?)
}

fn device_expr_upper_bound<Source, Less>(
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
    input.validate()?;
    let len = input.len();
    if len == 0 {
        return Ok(0);
    }

    let client = policy.client();
    let block_count_u32 = search_block_count(len)?;
    let value_handle = client.create_from_slice(Source::Item::as_bytes(&[value]));
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());
    let bindings = input.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);

    unsafe {
        upper_bound_device_expr_flags_kernel::launch_unchecked::<
            Source::Item,
            Source::Expr,
            Less,
            Source::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SEARCH_SIZE),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(value_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
        );
    }

    let control = crate::detail::control::SearchControl::from_flags(flag_handle, len, len);
    super::QueryApply::first_flag_or(policy, control, mindex_from_usize(len)?)
}

fn device_expr_find_first_of<Left, Right, Op>(
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
    input.validate()?;
    needles.validate()?;
    if input.len() == 0 || needles.len() == 0 {
        return Ok(None);
    }

    let client = policy.client();
    let len = input.len();
    let needle_len_u32 =
        u32::try_from(needles.len()).map_err(|_| Error::LengthTooLarge { len: needles.len() })?;
    let needle_len_handle = client.create_from_slice(u32::as_bytes(&[needle_len_u32]));
    let block_count_u32 = search_block_count(len)?;
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());
    let input_bindings = input.stage(policy)?;
    let needle_bindings = needles.stage(policy)?;
    let input_slot_offsets = input_bindings.slot_offsets_handle(client)?;
    let needle_slot_offsets = needle_bindings.slot_offsets_handle(client)?;
    let input_slot0 = input_bindings.slots.first().unwrap();
    let input_slot1 = input_bindings.slots.get(1).unwrap_or(input_slot0);
    let input_slot2 = input_bindings.slots.get(2).unwrap_or(input_slot0);
    let input_slot3 = input_bindings.slots.get(3).unwrap_or(input_slot0);
    let needle_slot0 = needle_bindings.slots.first().unwrap();
    let needle_slot1 = needle_bindings.slots.get(1).unwrap_or(needle_slot0);
    let needle_slot2 = needle_bindings.slots.get(2).unwrap_or(needle_slot0);
    let needle_slot3 = needle_bindings.slots.get(3).unwrap_or(needle_slot0);

    unsafe {
        find_first_of_device_expr_flags_kernel::launch_unchecked::<
            Left::Item,
            Left::Expr,
            Right::Expr,
            Op,
            Left::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SEARCH_SIZE),
            unsafe { BufferArg::from_raw_parts(input_slot0.0.clone(), input_slot0.1) },
            unsafe { BufferArg::from_raw_parts(input_slot1.0.clone(), input_slot1.1) },
            unsafe { BufferArg::from_raw_parts(input_slot2.0.clone(), input_slot2.1) },
            unsafe { BufferArg::from_raw_parts(input_slot3.0.clone(), input_slot3.1) },
            unsafe { BufferArg::from_raw_parts(input_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(needle_slot0.0.clone(), needle_slot0.1) },
            unsafe { BufferArg::from_raw_parts(needle_slot1.0.clone(), needle_slot1.1) },
            unsafe { BufferArg::from_raw_parts(needle_slot2.0.clone(), needle_slot2.1) },
            unsafe { BufferArg::from_raw_parts(needle_slot3.0.clone(), needle_slot3.1) },
            unsafe { BufferArg::from_raw_parts(needle_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(needle_len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
        );
    }

    let control = crate::detail::control::SearchControl::from_flags(flag_handle, len, len);
    super::QueryApply::first_flag(policy, control)
}

fn device_expr_lexicographical_compare<Left, Right, Less>(
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
    left.validate()?;
    right.validate()?;
    let min_len = left.len().min(right.len());
    if min_len == 0 {
        return Ok(left.len() < right.len());
    }

    let client = policy.client();
    let block_count_u32 = search_block_count(min_len)?;
    let flag_handle = client.empty(min_len * std::mem::size_of::<u32>());
    let left_bindings = left.stage(policy)?;
    let right_bindings = right.stage(policy)?;
    let left_slot_offsets = left_bindings.slot_offsets_handle(client)?;
    let right_slot_offsets = right_bindings.slot_offsets_handle(client)?;
    let left_slot0 = left_bindings.slots.first().unwrap();
    let left_slot1 = left_bindings.slots.get(1).unwrap_or(left_slot0);
    let left_slot2 = left_bindings.slots.get(2).unwrap_or(left_slot0);
    let left_slot3 = left_bindings.slots.get(3).unwrap_or(left_slot0);
    let right_slot0 = right_bindings.slots.first().unwrap();
    let right_slot1 = right_bindings.slots.get(1).unwrap_or(right_slot0);
    let right_slot2 = right_bindings.slots.get(2).unwrap_or(right_slot0);
    let right_slot3 = right_bindings.slots.get(3).unwrap_or(right_slot0);

    unsafe {
        lexicographical_diff_device_expr_flags_kernel::launch_unchecked::<
            Left::Item,
            Left::Expr,
            Right::Expr,
            Less,
            Left::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SEARCH_SIZE),
            unsafe { BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1) },
            unsafe { BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1) },
            unsafe { BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1) },
            unsafe { BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1) },
            unsafe { BufferArg::from_raw_parts(left_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1) },
            unsafe { BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1) },
            unsafe { BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1) },
            unsafe { BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1) },
            unsafe { BufferArg::from_raw_parts(right_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), min_len) },
        );
    }

    let control = crate::detail::control::SearchControl::from_flags(flag_handle, min_len, min_len);
    let Some(index) = super::QueryApply::first_flag(policy, control)? else {
        return Ok(left.len() < right.len());
    };

    let index_handle = client.create_from_slice(u32::as_bytes(&[index as u32]));
    let output_handle = client.empty(std::mem::size_of::<u32>());
    unsafe {
        lexicographical_compare_at_device_expr_kernel::launch_unchecked::<
            Left::Item,
            Left::Expr,
            Right::Expr,
            Less,
            Left::Runtime,
        >(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1) },
            unsafe { BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1) },
            unsafe { BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1) },
            unsafe { BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1) },
            unsafe { BufferArg::from_raw_parts(left_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1) },
            unsafe { BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1) },
            unsafe { BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1) },
            unsafe { BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1) },
            unsafe { BufferArg::from_raw_parts(right_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(index_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), 1) },
        );
    }

    Ok(scan::read_u32_scalar::<Left::Runtime>(client, output_handle)? != 0)
}

impl<Source, Less> crate::detail::read::KernelMinMaxInput<Less> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;

    fn min_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Option<MIndex>, Error> {
        ReadOnlySoA::validate(&self)?;
        Ok(
            super::QueryApply::minmax_expr::<Source, Less>(policy, &self.source)?
                .map(|(min, _)| min),
        )
    }

    fn max_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Option<MIndex>, Error> {
        ReadOnlySoA::validate(&self)?;
        Ok(
            super::QueryApply::minmax_expr::<Source, Less>(policy, &self.source)?
                .map(|(_, max)| max),
        )
    }

    fn minmax_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Option<(MIndex, MIndex)>, Error> {
        ReadOnlySoA::validate(&self)?;
        super::QueryApply::minmax_expr::<Source, Less>(policy, &self.source)
    }
}

impl<Source, Less> crate::detail::read::KernelMinMaxInput<Less> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;

    fn min_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Option<MIndex>, Error> {
        <SoAView1<Source> as crate::detail::read::KernelMinMaxInput<Less>>::min_element_input(
            SoAView1 { source: self },
            policy,
            less,
        )
    }

    fn max_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Option<MIndex>, Error> {
        <SoAView1<Source> as crate::detail::read::KernelMinMaxInput<Less>>::max_element_input(
            SoAView1 { source: self },
            policy,
            less,
        )
    }

    fn minmax_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Option<(MIndex, MIndex)>, Error> {
        <SoAView1<Source> as crate::detail::read::KernelMinMaxInput<Less>>::minmax_element_input(
            SoAView1 { source: self },
            policy,
            less,
        )
    }
}

impl<Source, Less> crate::detail::read::KernelMinMaxInput<Less> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;

    fn min_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Option<MIndex>, Error> {
        <Source as crate::detail::read::KernelMinMaxInput<super::Tuple1Less<Less>>>::min_element_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn max_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Option<MIndex>, Error> {
        <Source as crate::detail::read::KernelMinMaxInput<super::Tuple1Less<Less>>>::max_element_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn minmax_element_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Option<(MIndex, MIndex)>, Error> {
        <Source as crate::detail::read::KernelMinMaxInput<super::Tuple1Less<Less>>>::minmax_element_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }
}

impl<Source, Pred> crate::detail::read::KernelAdjacentFindInput<Pred> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;

    fn adjacent_find_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _pred: GpuOp<Pred>,
    ) -> Result<Option<MIndex>, Error> {
        ReadOnlySoA::validate(&self)?;
        device_expr_adjacent_find::<Source, Pred>(policy, &self.source)
    }
}

impl<Source, Pred> crate::detail::read::KernelAdjacentFindInput<Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;

    fn adjacent_find_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        pred: GpuOp<Pred>,
    ) -> Result<Option<MIndex>, Error> {
        <SoAView1<Source> as crate::detail::read::KernelAdjacentFindInput<Pred>>::adjacent_find_input(
            SoAView1 { source: self },
            policy,
            pred,
        )
    }
}

impl<Source, Pred> crate::detail::read::KernelAdjacentFindInput<Pred> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;

    fn adjacent_find_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _pred: GpuOp<Pred>,
    ) -> Result<Option<MIndex>, Error> {
        <Source as crate::detail::read::KernelAdjacentFindInput<super::Tuple1Less<Pred>>>::adjacent_find_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Pred>>::new(),
        )
    }
}

impl<Source, Less> crate::detail::read::KernelSortedSearchInput<Less> for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Item = Source::Item;

    fn lower_bound_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        _less: GpuOp<Less>,
    ) -> Result<MIndex, Error> {
        ReadOnlySoA::validate(&self)?;
        device_expr_lower_bound::<Source, Less>(policy, &self.source, value)
    }

    fn upper_bound_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        _less: GpuOp<Less>,
    ) -> Result<MIndex, Error> {
        ReadOnlySoA::validate(&self)?;
        device_expr_upper_bound::<Source, Less>(policy, &self.source, value)
    }

    fn is_sorted_until_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<MIndex, Error> {
        ReadOnlySoA::validate(&self)?;
        device_expr_is_sorted_until::<Source, Less>(policy, &self.source)
    }

    fn is_sorted_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<bool, Error> {
        let len = ReadOnlySoA::len(&self);
        Ok(
            <Self as crate::detail::read::KernelSortedSearchInput<Less>>::is_sorted_until_input(
                self, policy, less,
            )? == mindex_from_usize(len)?,
        )
    }
}

impl<Source, Values, Less> crate::detail::read::KernelSortedSearchManyInput<SoAView1<Values>, Less>
    for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    SoAView1<Values>: ReadOnlySoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Values: KernelColumn<Runtime = Source::Runtime, Item = Source::Item> + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Values::Expr: DeviceGpuExpr<Values::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;

    fn lower_bound_many_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        values: SoAView1<Values>,
        _less: GpuOp<Less>,
    ) -> Result<DeviceVec<Source::Runtime, MIndex>, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        SearchPayloadApply::lower_bound_many_expr::<Source, Values, Less>(
            policy,
            &self.source,
            &values.source,
        )
    }

    fn upper_bound_many_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        values: SoAView1<Values>,
        _less: GpuOp<Less>,
    ) -> Result<DeviceVec<Source::Runtime, MIndex>, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&values)?;
        SearchPayloadApply::upper_bound_many_expr::<Source, Values, Less>(
            policy,
            &self.source,
            &values.source,
        )
    }
}

impl<Source, Less> crate::detail::read::KernelSortedSearchInput<Less> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Item = Source::Item;

    fn lower_bound_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<MIndex, Error> {
        <SoAView1<Source> as crate::detail::read::KernelSortedSearchInput<Less>>::lower_bound_input(
            SoAView1 { source: self },
            policy,
            value,
            less,
        )
    }

    fn upper_bound_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        less: GpuOp<Less>,
    ) -> Result<MIndex, Error> {
        <SoAView1<Source> as crate::detail::read::KernelSortedSearchInput<Less>>::upper_bound_input(
            SoAView1 { source: self },
            policy,
            value,
            less,
        )
    }

    fn is_sorted_until_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<MIndex, Error> {
        <SoAView1<Source> as crate::detail::read::KernelSortedSearchInput<Less>>::is_sorted_until_input(
            SoAView1 { source: self },
            policy,
            less,
        )
    }

    fn is_sorted_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<bool, Error> {
        <SoAView1<Source> as crate::detail::read::KernelSortedSearchInput<Less>>::is_sorted_input(
            SoAView1 { source: self },
            policy,
            less,
        )
    }
}

impl<Source, Values, Less> crate::detail::read::KernelSortedSearchManyInput<Values, Less> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Values: KernelColumn<Runtime = Source::Runtime, Item = Source::Item> + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Values::Expr: DeviceGpuExpr<Values::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;

    fn lower_bound_many_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        values: Values,
        less: GpuOp<Less>,
    ) -> Result<DeviceVec<Source::Runtime, MIndex>, Error> {
        <SoAView1<Source> as crate::detail::read::KernelSortedSearchManyInput<
            SoAView1<Values>,
            Less,
        >>::lower_bound_many_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: values },
            less,
        )
    }

    fn upper_bound_many_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        values: Values,
        less: GpuOp<Less>,
    ) -> Result<DeviceVec<Source::Runtime, MIndex>, Error> {
        <SoAView1<Source> as crate::detail::read::KernelSortedSearchManyInput<
            SoAView1<Values>,
            Less,
        >>::upper_bound_many_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: values },
            less,
        )
    }
}

impl<Source, Less> crate::detail::read::KernelSortedSearchInput<Less> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Item = (Source::Item,);

    fn lower_bound_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        _less: GpuOp<Less>,
    ) -> Result<MIndex, Error> {
        <Source as crate::detail::read::KernelSortedSearchInput<super::Tuple1Less<Less>>>::lower_bound_input(
            self.0,
            policy,
            value.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn upper_bound_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        value: Self::Item,
        _less: GpuOp<Less>,
    ) -> Result<MIndex, Error> {
        <Source as crate::detail::read::KernelSortedSearchInput<super::Tuple1Less<Less>>>::upper_bound_input(
            self.0,
            policy,
            value.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn is_sorted_until_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<MIndex, Error> {
        <Source as crate::detail::read::KernelSortedSearchInput<super::Tuple1Less<Less>>>::is_sorted_until_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn is_sorted_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<bool, Error> {
        <Source as crate::detail::read::KernelSortedSearchInput<super::Tuple1Less<Less>>>::is_sorted_input(
            self.0,
            policy,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }
}

impl<Source, Values, Less> crate::detail::read::KernelSortedSearchManyInput<(Values,), Less>
    for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Values: KernelColumn<Runtime = Source::Runtime, Item = Source::Item> + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Values::Expr: DeviceGpuExpr<Values::Item>,
    Less: BinaryPredicateOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;

    fn lower_bound_many_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        values: (Values,),
        _less: GpuOp<Less>,
    ) -> Result<DeviceVec<Source::Runtime, MIndex>, Error> {
        <Source as crate::detail::read::KernelSortedSearchManyInput<
            Values,
            super::Tuple1Less<Less>,
        >>::lower_bound_many_input(
            self.0,
            policy,
            values.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }

    fn upper_bound_many_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        values: (Values,),
        _less: GpuOp<Less>,
    ) -> Result<DeviceVec<Source::Runtime, MIndex>, Error> {
        <Source as crate::detail::read::KernelSortedSearchManyInput<
            Values,
            super::Tuple1Less<Less>,
        >>::upper_bound_many_input(
            self.0,
            policy,
            values.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }
}

macro_rules! impl_sorted_search_tuple_input {
    ($view:ident < $( $ty:ident ),+ > { $( $field:ident: $index:tt ),+ }) => {
        impl<$( $ty ),+, Less> crate::detail::read::KernelSortedSearchInput<Less> for ($( $ty ),+)
        where
            $view<$( $ty ),+>: crate::detail::read::KernelSortedSearchInput<Less>,
        {
            type Runtime = <$view<$( $ty ),+> as crate::detail::read::KernelSortedSearchInput<Less>>::Runtime;
            type Item = <$view<$( $ty ),+> as crate::detail::read::KernelSortedSearchInput<Less>>::Item;

            fn lower_bound_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                value: Self::Item,
                less: GpuOp<Less>,
            ) -> Result<MIndex, Error> {
                <$view<$( $ty ),+> as crate::detail::read::KernelSortedSearchInput<Less>>::lower_bound_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    value,
                    less,
                )
            }

            fn upper_bound_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                value: Self::Item,
                less: GpuOp<Less>,
            ) -> Result<MIndex, Error> {
                <$view<$( $ty ),+> as crate::detail::read::KernelSortedSearchInput<Less>>::upper_bound_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    value,
                    less,
                )
            }

            fn is_sorted_until_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                less: GpuOp<Less>,
            ) -> Result<MIndex, Error> {
                <$view<$( $ty ),+> as crate::detail::read::KernelSortedSearchInput<Less>>::is_sorted_until_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    less,
                )
            }

            fn is_sorted_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                less: GpuOp<Less>,
            ) -> Result<bool, Error> {
                <$view<$( $ty ),+> as crate::detail::read::KernelSortedSearchInput<Less>>::is_sorted_input(
                    $view { $( $field: self.$index ),+ },
                    policy,
                    less,
                )
            }
        }
    };
}

impl_sorted_search_tuple_input!(SoAView2<A, B> { left: 0, right: 1 });
impl_sorted_search_tuple_input!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 });

impl<Left, Right, Op> crate::detail::read::KernelPairSearchInput<SoAView1<Right>, Op>
    for SoAView1<Left>
where
    Self: ReadOnlySoA<Item = (Left::Item,), Scalar = Left::Item>,
    SoAView1<Right>: ReadOnlySoA<Item = (Right::Item,), Scalar = Right::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: BinaryPredicateOp<Left::Item>,
{
    type Runtime = Left::Runtime;

    fn equal_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: SoAView1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        if self.source.len() != other.source.len() {
            return Ok(false);
        }
        Ok(device_expr_mismatch::<Left, Right, Op>(policy, &self.source, &other.source)?.is_none())
    }

    fn mismatch_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: SoAView1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<Option<MIndex>, Error> {
        device_expr_mismatch::<Left, Right, Op>(policy, &self.source, &other.source)
    }

    fn find_first_of_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: SoAView1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<Option<MIndex>, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&other)?;
        device_expr_find_first_of::<Left, Right, Op>(policy, &self.source, &other.source)
    }

    fn lexicographical_compare_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: SoAView1<Right>,
        _op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        ReadOnlySoA::validate(&self)?;
        ReadOnlySoA::validate(&other)?;
        device_expr_lexicographical_compare::<Left, Right, Op>(policy, &self.source, &other.source)
    }
}

impl<Left, Right, Op> crate::detail::read::KernelPairSearchInput<Right, Op> for Left
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: BinaryPredicateOp<Left::Item>,
{
    type Runtime = Left::Runtime;

    fn equal_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Right,
        op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        <SoAView1<Left> as crate::detail::read::KernelPairSearchInput<SoAView1<Right>, Op>>::equal_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: other },
            op,
        )
    }

    fn mismatch_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Right,
        op: GpuOp<Op>,
    ) -> Result<Option<MIndex>, Error> {
        <SoAView1<Left> as crate::detail::read::KernelPairSearchInput<SoAView1<Right>, Op>>::mismatch_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: other },
            op,
        )
    }

    fn find_first_of_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Right,
        op: GpuOp<Op>,
    ) -> Result<Option<MIndex>, Error> {
        <SoAView1<Left> as crate::detail::read::KernelPairSearchInput<SoAView1<Right>, Op>>::find_first_of_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: other },
            op,
        )
    }

    fn lexicographical_compare_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Right,
        op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        <SoAView1<Left> as crate::detail::read::KernelPairSearchInput<SoAView1<Right>, Op>>::lexicographical_compare_input(
            SoAView1 { source: self },
            policy,
            SoAView1 { source: other },
            op,
        )
    }
}

impl<Left, Right, Op> crate::detail::read::KernelPairSearchInput<(Right,), Op> for (Left,)
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: BinaryPredicateOp<(Left::Item,)>,
{
    type Runtime = Left::Runtime;

    fn equal_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: (Right,),
        _op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        <Left as crate::detail::read::KernelPairSearchInput<Right, super::Tuple1Less<Op>>>::equal_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Op>>::new(),
        )
    }

    fn mismatch_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: (Right,),
        _op: GpuOp<Op>,
    ) -> Result<Option<MIndex>, Error> {
        <Left as crate::detail::read::KernelPairSearchInput<Right, super::Tuple1Less<Op>>>::mismatch_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Op>>::new(),
        )
    }

    fn find_first_of_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: (Right,),
        _op: GpuOp<Op>,
    ) -> Result<Option<MIndex>, Error> {
        <Left as crate::detail::read::KernelPairSearchInput<Right, super::Tuple1Less<Op>>>::find_first_of_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Op>>::new(),
        )
    }

    fn lexicographical_compare_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: (Right,),
        _op: GpuOp<Op>,
    ) -> Result<bool, Error> {
        <Left as crate::detail::read::KernelPairSearchInput<Right, super::Tuple1Less<Op>>>::lexicographical_compare_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Op>>::new(),
        )
    }
}

macro_rules! impl_pair_search_tuple_input {
    (
        $view:ident < $( $left_ty:ident ),+ ; $( $right_ty:ident ),+ > {
            $( $field:ident: $left_index:tt / $right_index:tt ),+
        }
    ) => {
        impl<$( $left_ty ),+, $( $right_ty ),+, Op>
            crate::detail::read::KernelPairSearchInput<($( $right_ty ),+), Op>
            for ($( $left_ty ),+)
        where
            $view<$( $left_ty ),+>: crate::detail::read::KernelPairSearchInput<$view<$( $right_ty ),+>, Op>,
        {
            type Runtime =
                <$view<$( $left_ty ),+> as crate::detail::read::KernelPairSearchInput<$view<$( $right_ty ),+>, Op>>::Runtime;

            fn equal_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: ($( $right_ty ),+),
                op: GpuOp<Op>,
            ) -> Result<bool, Error> {
                <$view<$( $left_ty ),+> as crate::detail::read::KernelPairSearchInput<$view<$( $right_ty ),+>, Op>>::equal_input(
                    $view { $( $field: self.$left_index ),+ },
                    policy,
                    $view { $( $field: other.$right_index ),+ },
                    op,
                )
            }

            fn mismatch_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: ($( $right_ty ),+),
                op: GpuOp<Op>,
            ) -> Result<Option<MIndex>, Error> {
                <$view<$( $left_ty ),+> as crate::detail::read::KernelPairSearchInput<$view<$( $right_ty ),+>, Op>>::mismatch_input(
                    $view { $( $field: self.$left_index ),+ },
                    policy,
                    $view { $( $field: other.$right_index ),+ },
                    op,
                )
            }

            fn find_first_of_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: ($( $right_ty ),+),
                op: GpuOp<Op>,
            ) -> Result<Option<MIndex>, Error> {
                <$view<$( $left_ty ),+> as crate::detail::read::KernelPairSearchInput<$view<$( $right_ty ),+>, Op>>::find_first_of_input(
                    $view { $( $field: self.$left_index ),+ },
                    policy,
                    $view { $( $field: other.$right_index ),+ },
                    op,
                )
            }

            fn lexicographical_compare_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: ($( $right_ty ),+),
                op: GpuOp<Op>,
            ) -> Result<bool, Error> {
                <$view<$( $left_ty ),+> as crate::detail::read::KernelPairSearchInput<$view<$( $right_ty ),+>, Op>>::lexicographical_compare_input(
                    $view { $( $field: self.$left_index ),+ },
                    policy,
                    $view { $( $field: other.$right_index ),+ },
                    op,
                )
            }
        }
    };
}

impl_pair_search_tuple_input!(SoAView2<A, B; RA, RB> { left: 0 / 0, right: 1 / 1 });
impl_pair_search_tuple_input!(SoAView3<A, B, C; RA, RB, RC> { first: 0 / 0, second: 1 / 1, third: 2 / 2 });

macro_rules! impl_tuple_search {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $name:ident < $first:ident, $( $rest:ident ),+ > {
            $first_field:ident: $first_index:tt,
            $( $field:ident: $index:tt ),+
        },
        $adjacent_kernel:ident,
        $sorted_break_kernel:ident,
        $lower_bound_kernel:ident,
        $upper_bound_kernel:ident,
        $lower_bound_many_kernel:ident,
        $upper_bound_many_kernel:ident,
        $minmax_element_kernel:ident,
        $minmax_index_kernel:ident
    ) => {
        impl<$first, $( $rest ),+, Less> crate::detail::read::KernelMinMaxInput<Less> for $name<$first, $( $rest ),+>
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
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            fn min_element_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                less: GpuOp<Less>,
            ) -> Result<Option<MIndex>, Error> {
                Ok(
                    <Self as crate::detail::read::KernelMinMaxInput<Less>>::minmax_element_input(
                        self, policy, less,
                    )?
                    .map(|(min, _)| min),
                )
            }

            fn max_element_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                less: GpuOp<Less>,
            ) -> Result<Option<MIndex>, Error> {
                Ok(
                    <Self as crate::detail::read::KernelMinMaxInput<Less>>::minmax_element_input(
                        self, policy, less,
                    )?
                    .map(|(_, max)| max),
                )
            }

            fn minmax_element_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                _less: GpuOp<Less>,
            ) -> Result<Option<(MIndex, MIndex)>, Error> {
                ReadOnlySoA::validate(&self)?;
                let len = self.$first_field.len();
                let $first_field = stage_search_column(policy, &self.$first_field)?;
                $(
                    let $field = stage_search_column(policy, &self.$field)?;
                )+
                if len == 0 {
                    return Ok(None);
                }

                let client = policy.client();
                let mut current_count = len.div_ceil(BLOCK_SEARCH_SIZE as usize);
                let mut current_count_u32 = u32::try_from(current_count)
                    .map_err(|_| Error::LengthTooLarge { len: current_count })?;
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                let mut current_handle =
                    client.empty(current_count * 2 * std::mem::size_of::<MIndex>());

                unsafe {
                    $minmax_element_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        <$first as KernelColumn>::Expr,
                        $( <$rest as KernelColumn>::Expr, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(current_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        unsafe { BufferArg::from_raw_parts($first_field.slot0.0.clone(), $first_field.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot1.0.clone(), $first_field.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot2.0.clone(), $first_field.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot3.0.clone(), $first_field.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.slot0.0.clone(), $field.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($field.slot1.0.clone(), $field.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($field.slot2.0.clone(), $field.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($field.slot3.0.clone(), $field.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($field.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count * 2) },
                    );
                }

                while current_count > 1 {
                    let next_count = current_count.div_ceil(BLOCK_SEARCH_SIZE as usize);
                    let next_count_u32 = u32::try_from(next_count)
                        .map_err(|_| Error::LengthTooLarge { len: next_count })?;
                    let candidate_len_handle =
                        client.create_from_slice(u32::as_bytes(&[current_count_u32]));
                    let next_handle =
                        client.empty(next_count * 2 * std::mem::size_of::<MIndex>());

                    unsafe {
                        $minmax_index_kernel::launch_unchecked::<
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item, )+
                            <$first as KernelColumn>::Expr,
                            $( <$rest as KernelColumn>::Expr, )+
                            Less,
                            <$first as KernelColumn>::Runtime,
                        >(
                            client,
                            CubeCount::Static(next_count_u32, 1, 1),
                            CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                            unsafe { BufferArg::from_raw_parts($first_field.slot0.0.clone(), $first_field.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($first_field.slot1.0.clone(), $first_field.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($first_field.slot2.0.clone(), $first_field.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($first_field.slot3.0.clone(), $first_field.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($first_field.slot_offsets.clone(), 4) },
                            $(
                                unsafe { BufferArg::from_raw_parts($field.slot0.0.clone(), $field.slot0.1) },
                                unsafe { BufferArg::from_raw_parts($field.slot1.0.clone(), $field.slot1.1) },
                                unsafe { BufferArg::from_raw_parts($field.slot2.0.clone(), $field.slot2.1) },
                                unsafe { BufferArg::from_raw_parts($field.slot3.0.clone(), $field.slot3.1) },
                                unsafe { BufferArg::from_raw_parts($field.slot_offsets.clone(), 4) },
                            )+
                            unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count * 2) },
                            unsafe { BufferArg::from_raw_parts(candidate_len_handle.clone(), 1) },
                            unsafe { BufferArg::from_raw_parts(next_handle.clone(), next_count * 2) },
                        );
                    }

                    current_handle = next_handle;
                    current_count = next_count;
                    current_count_u32 = next_count_u32;
                }

                let bytes = client.read_one(current_handle).map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
                let indices = u32::from_bytes(&bytes);
                Ok(Some((indices[0], indices[1])))
            }
        }

        impl<$first, $( $rest ),+, Pred> crate::detail::read::KernelAdjacentFindInput<Pred> for $name<$first, $( $rest ),+>
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
            Pred: BinaryPredicateOp<(
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            fn adjacent_find_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                _pred: GpuOp<Pred>,
            ) -> Result<Option<MIndex>, Error> {
                ReadOnlySoA::validate(&self)?;
                let len = self.$first_field.len();
                let $first_field = stage_search_column(policy, &self.$first_field)?;
                $(
                    let $field = stage_search_column(policy, &self.$field)?;
                )+
                if len < 2 {
                    return Ok(None);
                }
                let block_count_u32 = search_block_count(len)?;
                let client = policy.client();
                let flag_handle = client.empty(len * std::mem::size_of::<u32>());
                unsafe {
                    $adjacent_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        <$first as KernelColumn>::Expr,
                        $( <$rest as KernelColumn>::Expr, )+
                        Pred,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        unsafe { BufferArg::from_raw_parts($first_field.slot0.0.clone(), $first_field.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot1.0.clone(), $first_field.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot2.0.clone(), $first_field.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot3.0.clone(), $first_field.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.slot0.0.clone(), $field.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($field.slot1.0.clone(), $field.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($field.slot2.0.clone(), $field.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($field.slot3.0.clone(), $field.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($field.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                    );
                }
                let control = crate::detail::control::SearchControl::from_flags(
                    flag_handle,
                    len,
                    len - 1,
                );
                super::QueryApply::first_flag(policy, control)
            }
        }

        impl<$first, $( $rest ),+, Less> crate::detail::read::KernelSortedSearchInput<Less> for $name<$first, $( $rest ),+>
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
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            type Item = (
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            );

            fn lower_bound_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                value: Self::Item,
                _less: GpuOp<Less>,
            ) -> Result<MIndex, Error> {
                ReadOnlySoA::validate(&self)?;
                let len = self.$first_field.len();
                let $first_field = stage_search_column(policy, &self.$first_field)?;
                $(
                    let $field = stage_search_column(policy, &self.$field)?;
                )+
                if len == 0 {
                    return Ok(0);
                }
                let client = policy.client();
                let flag_handle = client.empty(len * std::mem::size_of::<u32>());
                let first_value_handle = client.create_from_slice(
                    <<$first as KernelColumn>::Item as CubeElement>::as_bytes(&[value.$first_index])
                );
                $(
                    let $field = (
                        $field,
                        client.create_from_slice(
                            <<$rest as KernelColumn>::Item as CubeElement>::as_bytes(&[value.$index])
                        ),
                    );
                )+
                let block_count_u32 = search_block_count(len)?;
                unsafe {
                    $lower_bound_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        <$first as KernelColumn>::Expr,
                        $( <$rest as KernelColumn>::Expr, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        unsafe { BufferArg::from_raw_parts($first_field.slot0.0.clone(), $first_field.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot1.0.clone(), $first_field.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot2.0.clone(), $first_field.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot3.0.clone(), $first_field.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.0.slot0.0.clone(), $field.0.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot1.0.clone(), $field.0.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot2.0.clone(), $field.0.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot3.0.clone(), $field.0.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(first_value_handle.clone(), 1) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.1.clone(), 1) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                    );
                }
                let control = crate::detail::control::SearchControl::from_flags(
                    flag_handle,
                    len,
                    len,
                );
                super::QueryApply::first_flag_or(policy, control, mindex_from_usize(len)?)
            }

            fn upper_bound_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                value: Self::Item,
                _less: GpuOp<Less>,
            ) -> Result<MIndex, Error> {
                ReadOnlySoA::validate(&self)?;
                let len = self.$first_field.len();
                let $first_field = stage_search_column(policy, &self.$first_field)?;
                $(
                    let $field = stage_search_column(policy, &self.$field)?;
                )+
                if len == 0 {
                    return Ok(0);
                }
                let client = policy.client();
                let flag_handle = client.empty(len * std::mem::size_of::<u32>());
                let first_value_handle = client.create_from_slice(
                    <<$first as KernelColumn>::Item as CubeElement>::as_bytes(&[value.$first_index])
                );
                $(
                    let $field = (
                        $field,
                        client.create_from_slice(
                            <<$rest as KernelColumn>::Item as CubeElement>::as_bytes(&[value.$index])
                        ),
                    );
                )+
                let block_count_u32 = search_block_count(len)?;
                unsafe {
                    $upper_bound_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        <$first as KernelColumn>::Expr,
                        $( <$rest as KernelColumn>::Expr, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        unsafe { BufferArg::from_raw_parts($first_field.slot0.0.clone(), $first_field.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot1.0.clone(), $first_field.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot2.0.clone(), $first_field.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot3.0.clone(), $first_field.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.0.slot0.0.clone(), $field.0.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot1.0.clone(), $field.0.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot2.0.clone(), $field.0.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot3.0.clone(), $field.0.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($field.0.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(first_value_handle.clone(), 1) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.1.clone(), 1) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                    );
                }
                let control = crate::detail::control::SearchControl::from_flags(
                    flag_handle,
                    len,
                    len,
                );
                super::QueryApply::first_flag_or(policy, control, mindex_from_usize(len)?)
            }

            fn is_sorted_until_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                _less: GpuOp<Less>,
            ) -> Result<MIndex, Error> {
                ReadOnlySoA::validate(&self)?;
                let len = self.$first_field.len();
                let $first_field = stage_search_column(policy, &self.$first_field)?;
                $(
                    let $field = stage_search_column(policy, &self.$field)?;
                )+
                if len <= 1 {
                    return mindex_from_usize(len);
                }
                let block_count_u32 = search_block_count(len)?;
                let client = policy.client();
                let flag_handle = client.empty(len * std::mem::size_of::<u32>());
                unsafe {
                    $sorted_break_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        <$first as KernelColumn>::Expr,
                        $( <$rest as KernelColumn>::Expr, )+
                        Less,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        unsafe { BufferArg::from_raw_parts($first_field.slot0.0.clone(), $first_field.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot1.0.clone(), $first_field.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot2.0.clone(), $first_field.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot3.0.clone(), $first_field.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($first_field.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($field.slot0.0.clone(), $field.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($field.slot1.0.clone(), $field.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($field.slot2.0.clone(), $field.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($field.slot3.0.clone(), $field.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($field.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                    );
                }
                let control = crate::detail::control::SearchControl::from_flags(
                    flag_handle,
                    len,
                    len,
                );
                super::QueryApply::first_flag_or(policy, control, mindex_from_usize(len)?)
            }

            fn is_sorted_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                less: GpuOp<Less>,
            ) -> Result<bool, Error> {
                let len = ReadOnlySoA::len(&self);
                Ok(
                    <Self as crate::detail::read::KernelSortedSearchInput<Less>>::is_sorted_until_input(
                        self, policy, less,
                    )?
                    == mindex_from_usize(len)?,
                )
            }
        }

        impl<$first, $( $rest ),+, Less>
            crate::detail::read::KernelSortedSearchManyInput<$name<$first, $( $rest ),+>, Less>
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
                impl_tuple_search!(@item_ty $first),
                $( impl_tuple_search!(@item_ty $rest) ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            fn lower_bound_many_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                values: $name<$first, $( $rest ),+>,
                _less: GpuOp<Less>,
            ) -> Result<DeviceVec<<$first as KernelColumn>::Runtime, MIndex>, Error> {
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

            fn upper_bound_many_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                values: $name<$first, $( $rest ),+>,
                _less: GpuOp<Less>,
            ) -> Result<DeviceVec<<$first as KernelColumn>::Runtime, MIndex>, Error> {
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

macro_rules! impl_tuple_pair_search {
    (
        $name:ident <
            $first:ident, $( $rest:ident ),+ ;
            $right_first:ident, $( $right_rest:ident ),+
        > {
            $first_field:ident: $left_first:ident / $right_first_value:ident,
            $( $field:ident: $left_value:ident / $right_value:ident ),+
        },
        $mismatch_kernel:ident,
        $find_first_of_kernel:ident,
        $lexicographical_diff_kernel:ident,
        $lexicographical_compare_at_kernel:ident
    ) => {
        impl<$first, $( $rest ),+, $right_first, $( $right_rest ),+, Op>
            crate::detail::read::KernelPairSearchInput<$name<$right_first, $( $right_rest ),+>, Op>
            for $name<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $name<$right_first, $( $right_rest ),+>: ReadOnlySoA<Scalar = <$right_first as KernelColumn>::Item>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $right_first: KernelColumn<
                    Runtime = <$first as KernelColumn>::Runtime,
                    Item = <$first as KernelColumn>::Item,
                > + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
                $right_rest: KernelColumn<
                        Runtime = <$first as KernelColumn>::Runtime,
                        Item = <$rest as KernelColumn>::Item,
                    > + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            <$right_first as KernelColumn>::Expr: DeviceGpuExpr<<$right_first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
                <$right_rest as KernelColumn>::Expr:
                    DeviceGpuExpr<<$right_rest as KernelColumn>::Item>,
            )+
            Op: BinaryPredicateOp<(
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            fn equal_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: $name<$right_first, $( $right_rest ),+>,
                op: GpuOp<Op>,
            ) -> Result<bool, Error> {
                if ReadOnlySoA::len(&self) != ReadOnlySoA::len(&other) {
                    return Ok(false);
                }
                Ok(
                    <Self as crate::detail::read::KernelPairSearchInput<
                        $name<$right_first, $( $right_rest ),+>,
                        Op,
                    >>::mismatch_input(self, policy, other, op)?
                    .is_none(),
                )
            }

            fn mismatch_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: $name<$right_first, $( $right_rest ),+>,
                _op: GpuOp<Op>,
            ) -> Result<Option<MIndex>, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let left_len = self.$first_field.len();
                let right_len = other.$first_field.len();
                let min_len = left_len.min(right_len);
                if min_len == 0 {
                    return if left_len == right_len {
                        Ok(None)
                    } else {
                        Ok(Some(0))
                    };
                }

                let block_count_u32 = search_block_count(min_len)?;
                let client = policy.client();
                let $left_first = stage_search_column(policy, &self.$first_field)?;
                let $right_first_value = stage_search_column(policy, &other.$first_field)?;
                $(
                    let $left_value = stage_search_column(policy, &self.$field)?;
                    let $right_value = stage_search_column(policy, &other.$field)?;
                )+
                let flag_handle = client.empty(min_len * std::mem::size_of::<u32>());
                unsafe {
                    $mismatch_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        <$first as KernelColumn>::Expr,
                        <$right_first as KernelColumn>::Expr,
                        $(
                            <$rest as KernelColumn>::Expr,
                            <$right_rest as KernelColumn>::Expr,
                        )+
                        Op,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        unsafe { BufferArg::from_raw_parts($left_first.slot0.0.clone(), $left_first.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot1.0.clone(), $left_first.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot2.0.clone(), $left_first.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot3.0.clone(), $left_first.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($left_value.slot0.0.clone(), $left_value.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot1.0.clone(), $left_value.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot2.0.clone(), $left_value.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot3.0.clone(), $left_value.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot0.0.clone(), $right_first_value.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot1.0.clone(), $right_first_value.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot2.0.clone(), $right_first_value.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot3.0.clone(), $right_first_value.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($right_value.slot0.0.clone(), $right_value.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot1.0.clone(), $right_value.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot2.0.clone(), $right_value.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot3.0.clone(), $right_value.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), min_len) },
                    );
                }

                let control = crate::detail::control::SearchControl::from_flags(
                    flag_handle,
                    min_len,
                    min_len,
                );
                if let Some(index) = super::QueryApply::first_flag(policy, control)? {
                    return Ok(Some(index));
                }
                if left_len == right_len {
                    Ok(None)
                } else {
                    Ok(Some(mindex_from_usize(min_len)?))
                }
            }

            fn find_first_of_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: $name<$right_first, $( $right_rest ),+>,
                _op: GpuOp<Op>,
            ) -> Result<Option<MIndex>, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let input_len = self.$first_field.len();
                let needle_len = other.$first_field.len();
                if input_len == 0 || needle_len == 0 {
                    return Ok(None);
                }

                let block_count_u32 = search_block_count(input_len)?;
                let client = policy.client();
                let needle_len_u32 =
                    u32::try_from(needle_len).map_err(|_| Error::LengthTooLarge { len: needle_len })?;
                let needle_len_handle = client.create_from_slice(u32::as_bytes(&[needle_len_u32]));
                let $left_first = stage_search_column(policy, &self.$first_field)?;
                let $right_first_value = stage_search_column(policy, &other.$first_field)?;
                $(
                    let $left_value = stage_search_column(policy, &self.$field)?;
                    let $right_value = stage_search_column(policy, &other.$field)?;
                )+
                let flag_handle = client.empty(input_len * std::mem::size_of::<u32>());
                unsafe {
                    $find_first_of_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        <$first as KernelColumn>::Expr,
                        <$right_first as KernelColumn>::Expr,
                        $(
                            <$rest as KernelColumn>::Expr,
                            <$right_rest as KernelColumn>::Expr,
                        )+
                        Op,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        unsafe { BufferArg::from_raw_parts($left_first.slot0.0.clone(), $left_first.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot1.0.clone(), $left_first.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot2.0.clone(), $left_first.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot3.0.clone(), $left_first.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($left_value.slot0.0.clone(), $left_value.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot1.0.clone(), $left_value.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot2.0.clone(), $left_value.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot3.0.clone(), $left_value.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot0.0.clone(), $right_first_value.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot1.0.clone(), $right_first_value.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot2.0.clone(), $right_first_value.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot3.0.clone(), $right_first_value.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($right_value.slot0.0.clone(), $right_value.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot1.0.clone(), $right_value.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot2.0.clone(), $right_value.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot3.0.clone(), $right_value.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(needle_len_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), input_len) },
                    );
                }
                let control = crate::detail::control::SearchControl::from_flags(
                    flag_handle,
                    input_len,
                    input_len,
                );
                super::QueryApply::first_flag(policy, control)
            }

            fn lexicographical_compare_input(
                self,
                policy: &CubePolicy<Self::Runtime>,
                other: $name<$right_first, $( $right_rest ),+>,
                _op: GpuOp<Op>,
            ) -> Result<bool, Error> {
                ReadOnlySoA::validate(&self)?;
                ReadOnlySoA::validate(&other)?;
                let left_len = self.$first_field.len();
                let right_len = other.$first_field.len();
                let min_len = left_len.min(right_len);
                if min_len == 0 {
                    return Ok(left_len < right_len);
                }

                let block_count_u32 = search_block_count(min_len)?;
                let client = policy.client();
                let $left_first = stage_search_column(policy, &self.$first_field)?;
                let $right_first_value = stage_search_column(policy, &other.$first_field)?;
                $(
                    let $left_value = stage_search_column(policy, &self.$field)?;
                    let $right_value = stage_search_column(policy, &other.$field)?;
                )+
                let flag_handle = client.empty(min_len * std::mem::size_of::<u32>());
                unsafe {
                    $lexicographical_diff_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        <$first as KernelColumn>::Expr,
                        <$right_first as KernelColumn>::Expr,
                        $(
                            <$rest as KernelColumn>::Expr,
                            <$right_rest as KernelColumn>::Expr,
                        )+
                        Op,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SEARCH_SIZE),
                        unsafe { BufferArg::from_raw_parts($left_first.slot0.0.clone(), $left_first.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot1.0.clone(), $left_first.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot2.0.clone(), $left_first.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot3.0.clone(), $left_first.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($left_value.slot0.0.clone(), $left_value.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot1.0.clone(), $left_value.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot2.0.clone(), $left_value.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot3.0.clone(), $left_value.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot0.0.clone(), $right_first_value.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot1.0.clone(), $right_first_value.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot2.0.clone(), $right_first_value.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot3.0.clone(), $right_first_value.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($right_value.slot0.0.clone(), $right_value.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot1.0.clone(), $right_value.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot2.0.clone(), $right_value.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot3.0.clone(), $right_value.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(flag_handle.clone(), min_len) },
                    );
                }

                let control = crate::detail::control::SearchControl::from_flags(
                    flag_handle,
                    min_len,
                    min_len,
                );
                let Some(index) = super::QueryApply::first_flag(policy, control)? else {
                    return Ok(left_len < right_len);
                };

                let index_handle = client.create_from_slice(u32::as_bytes(&[index as u32]));
                let output_handle = client.empty(std::mem::size_of::<u32>());
                unsafe {
                    $lexicographical_compare_at_kernel::launch_unchecked::<
                        <$first as KernelColumn>::Item,
                        $( <$rest as KernelColumn>::Item, )+
                        <$first as KernelColumn>::Expr,
                        <$right_first as KernelColumn>::Expr,
                        $(
                            <$rest as KernelColumn>::Expr,
                            <$right_rest as KernelColumn>::Expr,
                        )+
                        Op,
                        <$first as KernelColumn>::Runtime,
                    >(
                        client,
                        CubeCount::new_single(),
                        CubeDim::new_1d(1),
                        unsafe { BufferArg::from_raw_parts($left_first.slot0.0.clone(), $left_first.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot1.0.clone(), $left_first.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot2.0.clone(), $left_first.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot3.0.clone(), $left_first.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($left_first.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($left_value.slot0.0.clone(), $left_value.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot1.0.clone(), $left_value.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot2.0.clone(), $left_value.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot3.0.clone(), $left_value.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($left_value.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot0.0.clone(), $right_first_value.slot0.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot1.0.clone(), $right_first_value.slot1.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot2.0.clone(), $right_first_value.slot2.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot3.0.clone(), $right_first_value.slot3.1) },
                        unsafe { BufferArg::from_raw_parts($right_first_value.slot_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($right_value.slot0.0.clone(), $right_value.slot0.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot1.0.clone(), $right_value.slot1.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot2.0.clone(), $right_value.slot2.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot3.0.clone(), $right_value.slot3.1) },
                            unsafe { BufferArg::from_raw_parts($right_value.slot_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(index_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(output_handle.clone(), 1) },
                    );
                }
                Ok(scan::read_u32_scalar::<<$first as KernelColumn>::Runtime>(
                    client,
                    output_handle,
                )? != 0)
            }
        }
    };
}

impl_tuple_search!(SoAView2<A, B> { left: 0, right: 1 }, tuple2_adjacent_device_expr_flags_kernel, tuple2_sorted_break_device_expr_flags_kernel, tuple2_lower_bound_device_expr_flags_kernel, tuple2_upper_bound_device_expr_flags_kernel, tuple2_lower_bound_device_expr_many_kernel, tuple2_upper_bound_device_expr_many_kernel, tuple2_minmax_element_device_expr_partials_kernel, tuple2_minmax_index_device_expr_partials_kernel);
impl_tuple_search!(SoAView3<A, B, C> { first: 0, second: 1, third: 2 }, tuple3_adjacent_device_expr_flags_kernel, tuple3_sorted_break_device_expr_flags_kernel, tuple3_lower_bound_device_expr_flags_kernel, tuple3_upper_bound_device_expr_flags_kernel, tuple3_lower_bound_device_expr_many_kernel, tuple3_upper_bound_device_expr_many_kernel, tuple3_minmax_element_device_expr_partials_kernel, tuple3_minmax_index_device_expr_partials_kernel);
impl_tuple_search!(SoAView4<A, B, C, D> { a: 0, b: 1, c: 2, d: 3 }, tuple4_adjacent_device_expr_flags_kernel, tuple4_sorted_break_device_expr_flags_kernel, tuple4_lower_bound_device_expr_flags_kernel, tuple4_upper_bound_device_expr_flags_kernel, tuple4_lower_bound_device_expr_many_kernel, tuple4_upper_bound_device_expr_many_kernel, tuple4_minmax_element_device_expr_partials_kernel, tuple4_minmax_index_device_expr_partials_kernel);
impl_tuple_search!(SoAView5<A, B, C, D, E> { a: 0, b: 1, c: 2, d: 3, e: 4 }, tuple5_adjacent_device_expr_flags_kernel, tuple5_sorted_break_device_expr_flags_kernel, tuple5_lower_bound_device_expr_flags_kernel, tuple5_upper_bound_device_expr_flags_kernel, tuple5_lower_bound_device_expr_many_kernel, tuple5_upper_bound_device_expr_many_kernel, tuple5_minmax_element_device_expr_partials_kernel, tuple5_minmax_index_device_expr_partials_kernel);
impl_tuple_search!(SoAView6<A, B, C, D, E, F> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5 }, tuple6_adjacent_device_expr_flags_kernel, tuple6_sorted_break_device_expr_flags_kernel, tuple6_lower_bound_device_expr_flags_kernel, tuple6_upper_bound_device_expr_flags_kernel, tuple6_lower_bound_device_expr_many_kernel, tuple6_upper_bound_device_expr_many_kernel, tuple6_minmax_element_device_expr_partials_kernel, tuple6_minmax_index_device_expr_partials_kernel);
impl_tuple_search!(SoAView7<A, B, C, D, E, F, G> { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6 }, tuple7_adjacent_device_expr_flags_kernel, tuple7_sorted_break_device_expr_flags_kernel, tuple7_lower_bound_device_expr_flags_kernel, tuple7_upper_bound_device_expr_flags_kernel, tuple7_lower_bound_device_expr_many_kernel, tuple7_upper_bound_device_expr_many_kernel, tuple7_minmax_element_device_expr_partials_kernel, tuple7_minmax_index_device_expr_partials_kernel);

impl_tuple_pair_search!(SoAView2<A, B; RA, RB> { left: left_a / right_a, right: left_b / right_b }, tuple2_mismatch_device_expr_flags_kernel, tuple2_find_first_of_device_expr_flags_kernel, tuple2_lexicographical_diff_device_expr_flags_kernel, tuple2_lexicographical_compare_at_device_expr_kernel);
impl_tuple_pair_search!(SoAView3<A, B, C; RA, RB, RC> { first: left_a / right_a, second: left_b / right_b, third: left_c / right_c }, tuple3_mismatch_device_expr_flags_kernel, tuple3_find_first_of_device_expr_flags_kernel, tuple3_lexicographical_diff_device_expr_flags_kernel, tuple3_lexicographical_compare_at_device_expr_kernel);
impl_tuple_pair_search!(SoAView4<A, B, C, D; RA, RB, RC, RD> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d }, tuple4_mismatch_device_expr_flags_kernel, tuple4_find_first_of_device_expr_flags_kernel, tuple4_lexicographical_diff_device_expr_flags_kernel, tuple4_lexicographical_compare_at_device_expr_kernel);
impl_tuple_pair_search!(SoAView5<A, B, C, D, E; RA, RB, RC, RD, RE> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e }, tuple5_mismatch_device_expr_flags_kernel, tuple5_find_first_of_device_expr_flags_kernel, tuple5_lexicographical_diff_device_expr_flags_kernel, tuple5_lexicographical_compare_at_device_expr_kernel);
impl_tuple_pair_search!(SoAView6<A, B, C, D, E, F; RA, RB, RC, RD, RE, RF> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f }, tuple6_mismatch_device_expr_flags_kernel, tuple6_find_first_of_device_expr_flags_kernel, tuple6_lexicographical_diff_device_expr_flags_kernel, tuple6_lexicographical_compare_at_device_expr_kernel);
impl_tuple_pair_search!(SoAView7<A, B, C, D, E, F, G; RA, RB, RC, RD, RE, RF, RG> { a: left_a / right_a, b: left_b / right_b, c: left_c / right_c, d: left_d / right_d, e: left_e / right_e, f: left_f / right_f, g: left_g / right_g }, tuple7_mismatch_device_expr_flags_kernel, tuple7_find_first_of_device_expr_flags_kernel, tuple7_lexicographical_diff_device_expr_flags_kernel, tuple7_lexicographical_compare_at_device_expr_kernel);
impl_tuple_search!(SoA2<A, B> { left: 0, right: 1 }, tuple2_adjacent_device_expr_flags_kernel, tuple2_sorted_break_device_expr_flags_kernel, tuple2_lower_bound_device_expr_flags_kernel, tuple2_upper_bound_device_expr_flags_kernel, tuple2_lower_bound_device_expr_many_kernel, tuple2_upper_bound_device_expr_many_kernel, tuple2_minmax_element_device_expr_partials_kernel, tuple2_minmax_index_device_expr_partials_kernel);
impl_tuple_search!(SoA3<A, B, C> { first: 0, second: 1, third: 2 }, tuple3_adjacent_device_expr_flags_kernel, tuple3_sorted_break_device_expr_flags_kernel, tuple3_lower_bound_device_expr_flags_kernel, tuple3_upper_bound_device_expr_flags_kernel, tuple3_lower_bound_device_expr_many_kernel, tuple3_upper_bound_device_expr_many_kernel, tuple3_minmax_element_device_expr_partials_kernel, tuple3_minmax_index_device_expr_partials_kernel);
impl_tuple_pair_search!(SoA2<A, B; RA, RB> { left: left_a / right_a, right: left_b / right_b }, tuple2_mismatch_device_expr_flags_kernel, tuple2_find_first_of_device_expr_flags_kernel, tuple2_lexicographical_diff_device_expr_flags_kernel, tuple2_lexicographical_compare_at_device_expr_kernel);
impl_tuple_pair_search!(SoA3<A, B, C; RA, RB, RC> { first: left_a / right_a, second: left_b / right_b, third: left_c / right_c }, tuple3_mismatch_device_expr_flags_kernel, tuple3_find_first_of_device_expr_flags_kernel, tuple3_lexicographical_diff_device_expr_flags_kernel, tuple3_lexicographical_compare_at_device_expr_kernel);

/// Finds the minimum element index according to `Less`.
pub fn min_element<Input, Less>(
    policy: &CubePolicy<<Input as crate::detail::read::KernelMinMaxInput<Less>>::Runtime>,
    input: Input,
    _less: Less,
) -> Result<Option<MIndex>, Error>
where
    Input: crate::detail::read::KernelMinMaxInput<Less>,
{
    input.min_element_input(policy, GpuOp::<Less>::new())
}

/// Finds the maximum element index according to `Less`.
pub fn max_element<Input, Less>(
    policy: &CubePolicy<<Input as crate::detail::read::KernelMinMaxInput<Less>>::Runtime>,
    input: Input,
    _less: Less,
) -> Result<Option<MIndex>, Error>
where
    Input: crate::detail::read::KernelMinMaxInput<Less>,
{
    input.max_element_input(policy, GpuOp::<Less>::new())
}

/// Finds both minimum and maximum element indices according to `Less`.
pub fn minmax_element<Input, Less>(
    policy: &CubePolicy<<Input as crate::detail::read::KernelMinMaxInput<Less>>::Runtime>,
    input: Input,
    _less: Less,
) -> Result<Option<(MIndex, MIndex)>, Error>
where
    Input: crate::detail::read::KernelMinMaxInput<Less>,
{
    input.minmax_element_input(policy, GpuOp::<Less>::new())
}

/// Finds the first adjacent pair that satisfies `Pred`.
pub fn adjacent_find<Input, Pred>(
    policy: &CubePolicy<<Input as crate::detail::read::KernelAdjacentFindInput<Pred>>::Runtime>,
    input: Input,
    _pred: Pred,
) -> Result<Option<MIndex>, Error>
where
    Input: crate::detail::read::KernelAdjacentFindInput<Pred>,
{
    input.adjacent_find_input(policy, GpuOp::<Pred>::new())
}

/// Returns whether two inputs are equal under `Eq`.
pub fn equal<Left, Right, Eq>(
    policy: &CubePolicy<<Left as crate::detail::read::KernelPairSearchInput<Right, Eq>>::Runtime>,
    left: Left,
    right: Right,
    _eq: Eq,
) -> Result<bool, Error>
where
    Left: crate::detail::read::KernelPairSearchInput<Right, Eq>,
{
    left.equal_input(policy, right, GpuOp::<Eq>::new())
}

/// Finds the first mismatch between two inputs.
pub fn mismatch<Left, Right, Eq>(
    policy: &CubePolicy<<Left as crate::detail::read::KernelPairSearchInput<Right, Eq>>::Runtime>,
    left: Left,
    right: Right,
    _eq: Eq,
) -> Result<Option<MIndex>, Error>
where
    Left: crate::detail::read::KernelPairSearchInput<Right, Eq>,
{
    left.mismatch_input(policy, right, GpuOp::<Eq>::new())
}

/// Finds the first input element equal to any value in `needles`.
pub fn find_first_of<Input, Needles, Eq>(
    policy: &CubePolicy<
        <Input as crate::detail::read::KernelPairSearchInput<Needles, Eq>>::Runtime,
    >,
    input: Input,
    needles: Needles,
    _eq: Eq,
) -> Result<Option<MIndex>, Error>
where
    Input: crate::detail::read::KernelPairSearchInput<Needles, Eq>,
{
    input.find_first_of_input(policy, needles, GpuOp::<Eq>::new())
}

/// Finds the first sorted insertion point for each value.
pub fn lower_bound_many<Input, Values, Less>(
    policy: &CubePolicy<
        <Input as crate::detail::read::KernelSortedSearchManyInput<Values, Less>>::Runtime,
    >,
    input: Input,
    values: Values,
    _less: Less,
) -> Result<
    DeviceVec<
        <Input as crate::detail::read::KernelSortedSearchManyInput<Values, Less>>::Runtime,
        u32,
    >,
    Error,
>
where
    Input: crate::detail::read::KernelSortedSearchManyInput<Values, Less>,
{
    input.lower_bound_many_input(policy, values, GpuOp::<Less>::new())
}

/// Finds the last sorted insertion point for each value.
pub fn upper_bound_many<Input, Values, Less>(
    policy: &CubePolicy<
        <Input as crate::detail::read::KernelSortedSearchManyInput<Values, Less>>::Runtime,
    >,
    input: Input,
    values: Values,
    _less: Less,
) -> Result<
    DeviceVec<
        <Input as crate::detail::read::KernelSortedSearchManyInput<Values, Less>>::Runtime,
        u32,
    >,
    Error,
>
where
    Input: crate::detail::read::KernelSortedSearchManyInput<Values, Less>,
{
    input.upper_bound_many_input(policy, values, GpuOp::<Less>::new())
}

/// Returns the first position where the sorted order is broken.
pub fn is_sorted_until<Input, Less>(
    policy: &CubePolicy<<Input as crate::detail::read::KernelSortedSearchInput<Less>>::Runtime>,
    input: Input,
    _less: Less,
) -> Result<MIndex, Error>
where
    Input: crate::detail::read::KernelSortedSearchInput<Less>,
{
    input.is_sorted_until_input(policy, GpuOp::<Less>::new())
}

/// Returns whether an input is sorted.
pub fn is_sorted<Input, Less>(
    policy: &CubePolicy<<Input as crate::detail::read::KernelSortedSearchInput<Less>>::Runtime>,
    input: Input,
    _less: Less,
) -> Result<bool, Error>
where
    Input: crate::detail::read::KernelSortedSearchInput<Less>,
{
    input.is_sorted_input(policy, GpuOp::<Less>::new())
}

/// Lexicographically compares two inputs.
pub fn lexicographical_compare<Left, Right, Less>(
    policy: &CubePolicy<<Left as crate::detail::read::KernelPairSearchInput<Right, Less>>::Runtime>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<bool, Error>
where
    Left: crate::detail::read::KernelPairSearchInput<Right, Less>,
{
    left.lexicographical_compare_input(policy, right, GpuOp::<Less>::new())
}
