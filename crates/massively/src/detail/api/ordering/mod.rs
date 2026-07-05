use super::memory::{MaterializeOutput, materialize};
use crate::{
    detail::op::kernel::BinaryPredicateOp,
    device::{
        DeviceColumnMutView, DeviceVec, KernelColumn, KernelColumnAt, ReadOnlyZip, S0, Zip1, Zip2,
        Zip3, ZipView1, ZipView2, ZipView3,
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::GpuOp,
    policy::CubePolicy,
    primitives::{ordering, select},
};
use cubecl::prelude::*;

const BLOCK_ORDERING_SIZE: u32 = 256;

pub(in crate::detail) fn device_expr_merge_control_with_policy<Left, Right, Less>(
    policy: &CubePolicy<Left::Runtime>,
    left: &Left,
    right: &Right,
) -> Result<crate::detail::control::MergeControl<Left::Runtime>, Error>
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
    let len = left.len() + right.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    let source_sides = client.empty(len * std::mem::size_of::<u32>());
    let source_indices = client.empty(len * std::mem::size_of::<u32>());

    if len != 0 {
        let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
        let left_len_u32 =
            u32::try_from(left.len()).map_err(|_| Error::LengthTooLarge { len: left.len() })?;
        let right_len_u32 =
            u32::try_from(right.len()).map_err(|_| Error::LengthTooLarge { len: right.len() })?;
        let left_len_handle = client.create_from_slice(u32::as_bytes(&[left_len_u32]));
        let right_len_handle = client.create_from_slice(u32::as_bytes(&[right_len_u32]));
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
            merge_path_control_device_expr_kernel::launch_unchecked::<
                Left::Item,
                Left::Expr,
                Right::Expr,
                Less,
                Left::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1) },
                unsafe { BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1) },
                unsafe { BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1) },
                unsafe { BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1) },
                unsafe { BufferArg::from_raw_parts(left_slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(left_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1) },
                unsafe { BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1) },
                unsafe { BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1) },
                unsafe { BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1) },
                unsafe { BufferArg::from_raw_parts(right_slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(right_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(source_sides.clone(), len) },
                unsafe { BufferArg::from_raw_parts(source_indices.clone(), len) },
            );
        }
    }

    Ok(crate::detail::control::MergeControl {
        source_side: source_sides,
        source_index: source_indices,
        left_len: left.len(),
        right_len: right.len(),
        len,
        len_u32,
        _runtime: std::marker::PhantomData,
    })
}

fn device_expr_membership_compact_with_policy<Candidate, Sorted, Less>(
    policy: &CubePolicy<Candidate::Runtime>,
    candidates: &Candidate,
    sorted: &Sorted,
    keep_present: bool,
) -> Result<DeviceVec<Candidate::Runtime, Candidate::Item>, Error>
where
    Candidate: KernelColumn + KernelColumnAt<S0>,
    Sorted: KernelColumn<Runtime = Candidate::Runtime, Item = Candidate::Item> + KernelColumnAt<S0>,
    Candidate::Item: CubePrimitive + CubeElement,
    Candidate::Expr: DeviceGpuExpr<Candidate::Item>,
    Sorted::Expr: DeviceGpuExpr<Sorted::Item>,
    Less: BinaryPredicateOp<Candidate::Item>,
{
    candidates.validate()?;
    sorted.validate()?;
    let len = candidates.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let sorted_len_u32 =
        u32::try_from(sorted.len()).map_err(|_| Error::LengthTooLarge { len: sorted.len() })?;
    let candidate_len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let sorted_len_handle = client.create_from_slice(u32::as_bytes(&[sorted_len_u32]));
    let keep_values = [if keep_present { 1_u32 } else { 0_u32 }];
    let keep_handle = client.create_from_slice(u32::as_bytes(&keep_values));
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());
    let candidate_bindings = candidates.stage(policy)?;
    let sorted_bindings = sorted.stage(policy)?;
    let candidate_slot_offsets = candidate_bindings.slot_offsets_handle(client)?;
    let sorted_slot_offsets = sorted_bindings.slot_offsets_handle(client)?;
    let candidate_slot0 = candidate_bindings.slots.first().unwrap();
    let candidate_slot1 = candidate_bindings.slots.get(1).unwrap_or(candidate_slot0);
    let candidate_slot2 = candidate_bindings.slots.get(2).unwrap_or(candidate_slot0);
    let candidate_slot3 = candidate_bindings.slots.get(3).unwrap_or(candidate_slot0);
    let sorted_slot0 = sorted_bindings.slots.first().unwrap();
    let sorted_slot1 = sorted_bindings.slots.get(1).unwrap_or(sorted_slot0);
    let sorted_slot2 = sorted_bindings.slots.get(2).unwrap_or(sorted_slot0);
    let sorted_slot3 = sorted_bindings.slots.get(3).unwrap_or(sorted_slot0);

    unsafe {
        sorted_membership_device_expr_flags_kernel::launch_unchecked::<
            Candidate::Item,
            Candidate::Expr,
            Sorted::Expr,
            Less,
            Candidate::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_ORDERING_SIZE),
            unsafe { BufferArg::from_raw_parts(candidate_slot0.0.clone(), candidate_slot0.1) },
            unsafe { BufferArg::from_raw_parts(candidate_slot1.0.clone(), candidate_slot1.1) },
            unsafe { BufferArg::from_raw_parts(candidate_slot2.0.clone(), candidate_slot2.1) },
            unsafe { BufferArg::from_raw_parts(candidate_slot3.0.clone(), candidate_slot3.1) },
            unsafe { BufferArg::from_raw_parts(candidate_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(candidate_len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(sorted_slot0.0.clone(), sorted_slot0.1) },
            unsafe { BufferArg::from_raw_parts(sorted_slot1.0.clone(), sorted_slot1.1) },
            unsafe { BufferArg::from_raw_parts(sorted_slot2.0.clone(), sorted_slot2.1) },
            unsafe { BufferArg::from_raw_parts(sorted_slot3.0.clone(), sorted_slot3.1) },
            unsafe { BufferArg::from_raw_parts(sorted_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(sorted_len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(keep_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
        );
    }

    let selected_rank = select::selected_rank_from_flags(policy, len, len_u32, flag_handle)?;
    let count = select::selected_count(policy, &selected_rank)?;
    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count)
        .apply_expr(policy, candidates)
}

fn selected_rank_from_flags_with_policy<R>(
    policy: &CubePolicy<R>,
    len: usize,
    flag_handle: cubecl::server::Handle,
) -> Result<(select::SelectedRankControl, usize), Error>
where
    R: Runtime,
{
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let selected_rank = select::selected_rank_from_flags(policy, len, len_u32, flag_handle)?;
    let count = select::selected_count(policy, &selected_rank)?;
    Ok((selected_rank, count))
}

pub(in crate::detail) fn tuple2_membership_expr_flags_with_policy<
    CandidateA,
    CandidateB,
    SortedA,
    SortedB,
    Less,
>(
    policy: &CubePolicy<CandidateA::Runtime>,
    candidate_a: &CandidateA,
    candidate_b: &CandidateB,
    sorted_a: &SortedA,
    sorted_b: &SortedB,
    keep_present: bool,
) -> Result<cubecl::server::Handle, Error>
where
    CandidateA: KernelColumn + KernelColumnAt<S0>,
    CandidateB: KernelColumn<Runtime = CandidateA::Runtime> + KernelColumnAt<S0>,
    SortedA:
        KernelColumn<Runtime = CandidateA::Runtime, Item = CandidateA::Item> + KernelColumnAt<S0>,
    SortedB:
        KernelColumn<Runtime = CandidateA::Runtime, Item = CandidateB::Item> + KernelColumnAt<S0>,
    CandidateA::Item: CubePrimitive + CubeElement,
    CandidateB::Item: CubePrimitive + CubeElement,
    CandidateA::Expr: DeviceGpuExpr<CandidateA::Item>,
    CandidateB::Expr: DeviceGpuExpr<CandidateB::Item>,
    SortedA::Expr: DeviceGpuExpr<SortedA::Item>,
    SortedB::Expr: DeviceGpuExpr<SortedB::Item>,
    Less: BinaryPredicateOp<(CandidateA::Item, CandidateB::Item)>,
{
    candidate_a.validate()?;
    candidate_b.validate()?;
    sorted_a.validate()?;
    sorted_b.validate()?;
    super::ensure_same_len(candidate_a.len(), candidate_b.len())?;
    super::ensure_same_len(sorted_a.len(), sorted_b.len())?;
    let len = candidate_a.len();
    let client = policy.client();
    let flag = client.empty(len * std::mem::size_of::<u32>());
    if len != 0 {
        let block_count = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
        let block_count_u32 =
            u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
        let candidate_len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let sorted_len_u32 = u32::try_from(sorted_a.len()).map_err(|_| Error::LengthTooLarge {
            len: sorted_a.len(),
        })?;
        let candidate_len_handle = client.create_from_slice(u32::as_bytes(&[candidate_len_u32]));
        let sorted_len_handle = client.create_from_slice(u32::as_bytes(&[sorted_len_u32]));
        let keep = [if keep_present { 1_u32 } else { 0_u32 }];
        let keep_handle = client.create_from_slice(u32::as_bytes(&keep));
        let candidate_a_bindings = candidate_a.stage(policy)?;
        let candidate_b_bindings = candidate_b.stage(policy)?;
        let sorted_a_bindings = sorted_a.stage(policy)?;
        let sorted_b_bindings = sorted_b.stage(policy)?;
        let candidate_a_offsets = candidate_a_bindings.slot_offsets_handle(client)?;
        let candidate_b_offsets = candidate_b_bindings.slot_offsets_handle(client)?;
        let sorted_a_offsets = sorted_a_bindings.slot_offsets_handle(client)?;
        let sorted_b_offsets = sorted_b_bindings.slot_offsets_handle(client)?;
        let ca0 = candidate_a_bindings.slot_or_first(0);
        let ca1 = candidate_a_bindings.slot_or_first(1);
        let ca2 = candidate_a_bindings.slot_or_first(2);
        let ca3 = candidate_a_bindings.slot_or_first(3);
        let cb0 = candidate_b_bindings.slot_or_first(0);
        let cb1 = candidate_b_bindings.slot_or_first(1);
        let cb2 = candidate_b_bindings.slot_or_first(2);
        let cb3 = candidate_b_bindings.slot_or_first(3);
        let sa0 = sorted_a_bindings.slot_or_first(0);
        let sa1 = sorted_a_bindings.slot_or_first(1);
        let sa2 = sorted_a_bindings.slot_or_first(2);
        let sa3 = sorted_a_bindings.slot_or_first(3);
        let sb0 = sorted_b_bindings.slot_or_first(0);
        let sb1 = sorted_b_bindings.slot_or_first(1);
        let sb2 = sorted_b_bindings.slot_or_first(2);
        let sb3 = sorted_b_bindings.slot_or_first(3);

        unsafe {
            tuple2_membership_device_expr_flags_kernel::launch_unchecked::<
                CandidateA::Item,
                CandidateB::Item,
                CandidateA::Expr,
                CandidateB::Expr,
                SortedA::Expr,
                SortedB::Expr,
                Less,
                CandidateA::Runtime,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(ca0.0.clone(), ca0.1) },
                unsafe { BufferArg::from_raw_parts(ca1.0.clone(), ca1.1) },
                unsafe { BufferArg::from_raw_parts(ca2.0.clone(), ca2.1) },
                unsafe { BufferArg::from_raw_parts(ca3.0.clone(), ca3.1) },
                unsafe { BufferArg::from_raw_parts(candidate_a_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(cb0.0.clone(), cb0.1) },
                unsafe { BufferArg::from_raw_parts(cb1.0.clone(), cb1.1) },
                unsafe { BufferArg::from_raw_parts(cb2.0.clone(), cb2.1) },
                unsafe { BufferArg::from_raw_parts(cb3.0.clone(), cb3.1) },
                unsafe { BufferArg::from_raw_parts(candidate_b_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(candidate_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(sa0.0.clone(), sa0.1) },
                unsafe { BufferArg::from_raw_parts(sa1.0.clone(), sa1.1) },
                unsafe { BufferArg::from_raw_parts(sa2.0.clone(), sa2.1) },
                unsafe { BufferArg::from_raw_parts(sa3.0.clone(), sa3.1) },
                unsafe { BufferArg::from_raw_parts(sorted_a_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(sb0.0.clone(), sb0.1) },
                unsafe { BufferArg::from_raw_parts(sb1.0.clone(), sb1.1) },
                unsafe { BufferArg::from_raw_parts(sb2.0.clone(), sb2.1) },
                unsafe { BufferArg::from_raw_parts(sb3.0.clone(), sb3.1) },
                unsafe { BufferArg::from_raw_parts(sorted_b_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(sorted_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(keep_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag.clone(), len) },
            );
        }
    }
    Ok(flag)
}

pub(in crate::detail) fn tuple3_membership_expr_flags_with_policy<
    CandidateA,
    CandidateB,
    CandidateC,
    SortedA,
    SortedB,
    SortedC,
    Less,
>(
    policy: &CubePolicy<CandidateA::Runtime>,
    candidate_a: &CandidateA,
    candidate_b: &CandidateB,
    candidate_c: &CandidateC,
    sorted_a: &SortedA,
    sorted_b: &SortedB,
    sorted_c: &SortedC,
    keep_present: bool,
) -> Result<cubecl::server::Handle, Error>
where
    CandidateA: KernelColumn + KernelColumnAt<S0>,
    CandidateB: KernelColumn<Runtime = CandidateA::Runtime> + KernelColumnAt<S0>,
    CandidateC: KernelColumn<Runtime = CandidateA::Runtime> + KernelColumnAt<S0>,
    SortedA:
        KernelColumn<Runtime = CandidateA::Runtime, Item = CandidateA::Item> + KernelColumnAt<S0>,
    SortedB:
        KernelColumn<Runtime = CandidateA::Runtime, Item = CandidateB::Item> + KernelColumnAt<S0>,
    SortedC:
        KernelColumn<Runtime = CandidateA::Runtime, Item = CandidateC::Item> + KernelColumnAt<S0>,
    CandidateA::Item: CubePrimitive + CubeElement,
    CandidateB::Item: CubePrimitive + CubeElement,
    CandidateC::Item: CubePrimitive + CubeElement,
    CandidateA::Expr: DeviceGpuExpr<CandidateA::Item>,
    CandidateB::Expr: DeviceGpuExpr<CandidateB::Item>,
    CandidateC::Expr: DeviceGpuExpr<CandidateC::Item>,
    SortedA::Expr: DeviceGpuExpr<SortedA::Item>,
    SortedB::Expr: DeviceGpuExpr<SortedB::Item>,
    SortedC::Expr: DeviceGpuExpr<SortedC::Item>,
    Less: BinaryPredicateOp<(CandidateA::Item, CandidateB::Item, CandidateC::Item)>,
{
    candidate_a.validate()?;
    candidate_b.validate()?;
    candidate_c.validate()?;
    sorted_a.validate()?;
    sorted_b.validate()?;
    sorted_c.validate()?;
    super::ensure_same_len(candidate_a.len(), candidate_b.len())?;
    super::ensure_same_len(candidate_a.len(), candidate_c.len())?;
    super::ensure_same_len(sorted_a.len(), sorted_b.len())?;
    super::ensure_same_len(sorted_a.len(), sorted_c.len())?;
    let len = candidate_a.len();
    let client = policy.client();
    let flag = client.empty(len * std::mem::size_of::<u32>());
    if len != 0 {
        let block_count = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
        let block_count_u32 =
            u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
        let candidate_len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let sorted_len_u32 = u32::try_from(sorted_a.len()).map_err(|_| Error::LengthTooLarge {
            len: sorted_a.len(),
        })?;
        let candidate_len_handle = client.create_from_slice(u32::as_bytes(&[candidate_len_u32]));
        let sorted_len_handle = client.create_from_slice(u32::as_bytes(&[sorted_len_u32]));
        let keep = [if keep_present { 1_u32 } else { 0_u32 }];
        let keep_handle = client.create_from_slice(u32::as_bytes(&keep));
        let candidate_a_bindings = candidate_a.stage(policy)?;
        let candidate_b_bindings = candidate_b.stage(policy)?;
        let candidate_c_bindings = candidate_c.stage(policy)?;
        let sorted_a_bindings = sorted_a.stage(policy)?;
        let sorted_b_bindings = sorted_b.stage(policy)?;
        let sorted_c_bindings = sorted_c.stage(policy)?;
        let candidate_a_offsets = candidate_a_bindings.slot_offsets_handle(client)?;
        let candidate_b_offsets = candidate_b_bindings.slot_offsets_handle(client)?;
        let candidate_c_offsets = candidate_c_bindings.slot_offsets_handle(client)?;
        let sorted_a_offsets = sorted_a_bindings.slot_offsets_handle(client)?;
        let sorted_b_offsets = sorted_b_bindings.slot_offsets_handle(client)?;
        let sorted_c_offsets = sorted_c_bindings.slot_offsets_handle(client)?;
        let ca0 = candidate_a_bindings.slot_or_first(0);
        let ca1 = candidate_a_bindings.slot_or_first(1);
        let ca2 = candidate_a_bindings.slot_or_first(2);
        let ca3 = candidate_a_bindings.slot_or_first(3);
        let cb0 = candidate_b_bindings.slot_or_first(0);
        let cb1 = candidate_b_bindings.slot_or_first(1);
        let cb2 = candidate_b_bindings.slot_or_first(2);
        let cb3 = candidate_b_bindings.slot_or_first(3);
        let cc0 = candidate_c_bindings.slot_or_first(0);
        let cc1 = candidate_c_bindings.slot_or_first(1);
        let cc2 = candidate_c_bindings.slot_or_first(2);
        let cc3 = candidate_c_bindings.slot_or_first(3);
        let sa0 = sorted_a_bindings.slot_or_first(0);
        let sa1 = sorted_a_bindings.slot_or_first(1);
        let sa2 = sorted_a_bindings.slot_or_first(2);
        let sa3 = sorted_a_bindings.slot_or_first(3);
        let sb0 = sorted_b_bindings.slot_or_first(0);
        let sb1 = sorted_b_bindings.slot_or_first(1);
        let sb2 = sorted_b_bindings.slot_or_first(2);
        let sb3 = sorted_b_bindings.slot_or_first(3);
        let sc0 = sorted_c_bindings.slot_or_first(0);
        let sc1 = sorted_c_bindings.slot_or_first(1);
        let sc2 = sorted_c_bindings.slot_or_first(2);
        let sc3 = sorted_c_bindings.slot_or_first(3);

        unsafe {
            tuple3_membership_device_expr_flags_kernel::launch_unchecked::<
                CandidateA::Item,
                CandidateB::Item,
                CandidateC::Item,
                CandidateA::Expr,
                CandidateB::Expr,
                CandidateC::Expr,
                SortedA::Expr,
                SortedB::Expr,
                SortedC::Expr,
                Less,
                CandidateA::Runtime,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(ca0.0.clone(), ca0.1) },
                unsafe { BufferArg::from_raw_parts(ca1.0.clone(), ca1.1) },
                unsafe { BufferArg::from_raw_parts(ca2.0.clone(), ca2.1) },
                unsafe { BufferArg::from_raw_parts(ca3.0.clone(), ca3.1) },
                unsafe { BufferArg::from_raw_parts(candidate_a_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(cb0.0.clone(), cb0.1) },
                unsafe { BufferArg::from_raw_parts(cb1.0.clone(), cb1.1) },
                unsafe { BufferArg::from_raw_parts(cb2.0.clone(), cb2.1) },
                unsafe { BufferArg::from_raw_parts(cb3.0.clone(), cb3.1) },
                unsafe { BufferArg::from_raw_parts(candidate_b_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(cc0.0.clone(), cc0.1) },
                unsafe { BufferArg::from_raw_parts(cc1.0.clone(), cc1.1) },
                unsafe { BufferArg::from_raw_parts(cc2.0.clone(), cc2.1) },
                unsafe { BufferArg::from_raw_parts(cc3.0.clone(), cc3.1) },
                unsafe { BufferArg::from_raw_parts(candidate_c_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(candidate_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(sa0.0.clone(), sa0.1) },
                unsafe { BufferArg::from_raw_parts(sa1.0.clone(), sa1.1) },
                unsafe { BufferArg::from_raw_parts(sa2.0.clone(), sa2.1) },
                unsafe { BufferArg::from_raw_parts(sa3.0.clone(), sa3.1) },
                unsafe { BufferArg::from_raw_parts(sorted_a_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(sb0.0.clone(), sb0.1) },
                unsafe { BufferArg::from_raw_parts(sb1.0.clone(), sb1.1) },
                unsafe { BufferArg::from_raw_parts(sb2.0.clone(), sb2.1) },
                unsafe { BufferArg::from_raw_parts(sb3.0.clone(), sb3.1) },
                unsafe { BufferArg::from_raw_parts(sorted_b_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(sc0.0.clone(), sc0.1) },
                unsafe { BufferArg::from_raw_parts(sc1.0.clone(), sc1.1) },
                unsafe { BufferArg::from_raw_parts(sc2.0.clone(), sc2.1) },
                unsafe { BufferArg::from_raw_parts(sc3.0.clone(), sc3.1) },
                unsafe { BufferArg::from_raw_parts(sorted_c_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(sorted_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(keep_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag.clone(), len) },
            );
        }
    }
    Ok(flag)
}

macro_rules! define_tuple_membership_expr_flags_with_policy {
    (
        $fn_name:ident,
        $kernel_name:ident,
        (
            $first_candidate_ty:ident: $first_candidate:ident,
            $first_candidate_bindings:ident, $first_candidate_offsets:ident,
            $first_candidate_slot0:ident, $first_candidate_slot1:ident,
            $first_candidate_slot2:ident, $first_candidate_slot3:ident /
            $first_sorted_ty:ident: $first_sorted:ident,
            $first_sorted_bindings:ident, $first_sorted_offsets:ident,
            $first_sorted_slot0:ident, $first_sorted_slot1:ident,
            $first_sorted_slot2:ident, $first_sorted_slot3:ident
        )
        $(,
        (
            $candidate_ty:ident: $candidate:ident,
            $candidate_bindings:ident, $candidate_offsets:ident,
            $candidate_slot0:ident, $candidate_slot1:ident,
            $candidate_slot2:ident, $candidate_slot3:ident /
            $sorted_ty:ident: $sorted:ident,
            $sorted_bindings:ident, $sorted_offsets:ident,
            $sorted_slot0:ident, $sorted_slot1:ident,
            $sorted_slot2:ident, $sorted_slot3:ident
        ))+
    ) => {
        #[allow(clippy::too_many_arguments)]
        pub(in crate::detail) fn $fn_name<
            $first_candidate_ty,
            $( $candidate_ty, )+
            $first_sorted_ty,
            $( $sorted_ty, )+
            Less,
        >(
            policy: &CubePolicy<$first_candidate_ty::Runtime>,
            $first_candidate: &$first_candidate_ty,
            $( $candidate: &$candidate_ty, )+
            $first_sorted: &$first_sorted_ty,
            $( $sorted: &$sorted_ty, )+
            keep_present: bool,
        ) -> Result<cubecl::server::Handle, Error>
        where
            $first_candidate_ty: KernelColumn + KernelColumnAt<S0>,
            $(
                $candidate_ty: KernelColumn<Runtime = $first_candidate_ty::Runtime>
                    + KernelColumnAt<S0>,
            )+
            $first_sorted_ty:
                KernelColumn<Runtime = $first_candidate_ty::Runtime, Item = $first_candidate_ty::Item>
                + KernelColumnAt<S0>,
            $(
                $sorted_ty:
                    KernelColumn<Runtime = $first_candidate_ty::Runtime, Item = $candidate_ty::Item>
                    + KernelColumnAt<S0>,
            )+
            $first_candidate_ty::Item: CubePrimitive + CubeElement,
            $first_candidate_ty::Expr: DeviceGpuExpr<$first_candidate_ty::Item>,
            $first_sorted_ty::Expr: DeviceGpuExpr<$first_sorted_ty::Item>,
            $(
                $candidate_ty::Item: CubePrimitive + CubeElement,
                $candidate_ty::Expr: DeviceGpuExpr<$candidate_ty::Item>,
                $sorted_ty::Expr: DeviceGpuExpr<$sorted_ty::Item>,
            )+
            Less: BinaryPredicateOp<(
                $first_candidate_ty::Item,
                $( $candidate_ty::Item, )+
            )>,
        {
            $first_candidate.validate()?;
            $( $candidate.validate()?; )+
            $first_sorted.validate()?;
            $( $sorted.validate()?; )+
            $( super::ensure_same_len($first_candidate.len(), $candidate.len())?; )+
            $( super::ensure_same_len($first_sorted.len(), $sorted.len())?; )+
            let len = $first_candidate.len();
            let client = policy.client();
            let flag = client.empty(len * std::mem::size_of::<u32>());
            if len != 0 {
                let block_count = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
                let block_count_u32 = u32::try_from(block_count)
                    .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                let candidate_len_u32 =
                    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let sorted_len_u32 =
                    u32::try_from($first_sorted.len()).map_err(|_| Error::LengthTooLarge {
                        len: $first_sorted.len(),
                    })?;
                let candidate_len_handle =
                    client.create_from_slice(u32::as_bytes(&[candidate_len_u32]));
                let sorted_len_handle =
                    client.create_from_slice(u32::as_bytes(&[sorted_len_u32]));
                let keep = [if keep_present { 1_u32 } else { 0_u32 }];
                let keep_handle = client.create_from_slice(u32::as_bytes(&keep));
                let $first_candidate_bindings = $first_candidate.stage(policy)?;
                $( let $candidate_bindings = $candidate.stage(policy)?; )+
                let $first_sorted_bindings = $first_sorted.stage(policy)?;
                $( let $sorted_bindings = $sorted.stage(policy)?; )+
                let $first_candidate_offsets =
                    $first_candidate_bindings.slot_offsets_handle(client)?;
                $( let $candidate_offsets = $candidate_bindings.slot_offsets_handle(client)?; )+
                let $first_sorted_offsets = $first_sorted_bindings.slot_offsets_handle(client)?;
                $( let $sorted_offsets = $sorted_bindings.slot_offsets_handle(client)?; )+
                let $first_candidate_slot0 = $first_candidate_bindings.slot_or_first(0);
                let $first_candidate_slot1 = $first_candidate_bindings.slot_or_first(1);
                let $first_candidate_slot2 = $first_candidate_bindings.slot_or_first(2);
                let $first_candidate_slot3 = $first_candidate_bindings.slot_or_first(3);
                $(
                    let $candidate_slot0 = $candidate_bindings.slot_or_first(0);
                    let $candidate_slot1 = $candidate_bindings.slot_or_first(1);
                    let $candidate_slot2 = $candidate_bindings.slot_or_first(2);
                    let $candidate_slot3 = $candidate_bindings.slot_or_first(3);
                )+
                let $first_sorted_slot0 = $first_sorted_bindings.slot_or_first(0);
                let $first_sorted_slot1 = $first_sorted_bindings.slot_or_first(1);
                let $first_sorted_slot2 = $first_sorted_bindings.slot_or_first(2);
                let $first_sorted_slot3 = $first_sorted_bindings.slot_or_first(3);
                $(
                    let $sorted_slot0 = $sorted_bindings.slot_or_first(0);
                    let $sorted_slot1 = $sorted_bindings.slot_or_first(1);
                    let $sorted_slot2 = $sorted_bindings.slot_or_first(2);
                    let $sorted_slot3 = $sorted_bindings.slot_or_first(3);
                )+

                unsafe {
                    $kernel_name::launch_unchecked::<
                        $first_candidate_ty::Item,
                        $( $candidate_ty::Item, )+
                        $first_candidate_ty::Expr,
                        $( $candidate_ty::Expr, )+
                        $first_sorted_ty::Expr,
                        $( $sorted_ty::Expr, )+
                        Less,
                        $first_candidate_ty::Runtime,
                    >(
                        client,
                        CubeCount::Static(block_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                        unsafe { BufferArg::from_raw_parts($first_candidate_slot0.0.clone(), $first_candidate_slot0.1) },
                        unsafe { BufferArg::from_raw_parts($first_candidate_slot1.0.clone(), $first_candidate_slot1.1) },
                        unsafe { BufferArg::from_raw_parts($first_candidate_slot2.0.clone(), $first_candidate_slot2.1) },
                        unsafe { BufferArg::from_raw_parts($first_candidate_slot3.0.clone(), $first_candidate_slot3.1) },
                        unsafe { BufferArg::from_raw_parts($first_candidate_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($candidate_slot0.0.clone(), $candidate_slot0.1) },
                            unsafe { BufferArg::from_raw_parts($candidate_slot1.0.clone(), $candidate_slot1.1) },
                            unsafe { BufferArg::from_raw_parts($candidate_slot2.0.clone(), $candidate_slot2.1) },
                            unsafe { BufferArg::from_raw_parts($candidate_slot3.0.clone(), $candidate_slot3.1) },
                            unsafe { BufferArg::from_raw_parts($candidate_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(candidate_len_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts($first_sorted_slot0.0.clone(), $first_sorted_slot0.1) },
                        unsafe { BufferArg::from_raw_parts($first_sorted_slot1.0.clone(), $first_sorted_slot1.1) },
                        unsafe { BufferArg::from_raw_parts($first_sorted_slot2.0.clone(), $first_sorted_slot2.1) },
                        unsafe { BufferArg::from_raw_parts($first_sorted_slot3.0.clone(), $first_sorted_slot3.1) },
                        unsafe { BufferArg::from_raw_parts($first_sorted_offsets.clone(), 4) },
                        $(
                            unsafe { BufferArg::from_raw_parts($sorted_slot0.0.clone(), $sorted_slot0.1) },
                            unsafe { BufferArg::from_raw_parts($sorted_slot1.0.clone(), $sorted_slot1.1) },
                            unsafe { BufferArg::from_raw_parts($sorted_slot2.0.clone(), $sorted_slot2.1) },
                            unsafe { BufferArg::from_raw_parts($sorted_slot3.0.clone(), $sorted_slot3.1) },
                            unsafe { BufferArg::from_raw_parts($sorted_offsets.clone(), 4) },
                        )+
                        unsafe { BufferArg::from_raw_parts(sorted_len_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(keep_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(flag.clone(), len) },
                    );
                }
            }
            Ok(flag)
        }
    };
}

define_tuple_membership_expr_flags_with_policy!(
    tuple4_membership_expr_flags_with_policy,
    tuple4_membership_device_expr_flags_kernel,
    (CandidateA: candidate_a, candidate_a_bindings, candidate_a_offsets, ca0, ca1, ca2, ca3 /
     SortedA: sorted_a, sorted_a_bindings, sorted_a_offsets, sa0, sa1, sa2, sa3),
    (CandidateB: candidate_b, candidate_b_bindings, candidate_b_offsets, cb0, cb1, cb2, cb3 /
     SortedB: sorted_b, sorted_b_bindings, sorted_b_offsets, sb0, sb1, sb2, sb3),
    (CandidateC: candidate_c, candidate_c_bindings, candidate_c_offsets, cc0, cc1, cc2, cc3 /
     SortedC: sorted_c, sorted_c_bindings, sorted_c_offsets, sc0, sc1, sc2, sc3),
    (CandidateD: candidate_d, candidate_d_bindings, candidate_d_offsets, cd0, cd1, cd2, cd3 /
     SortedD: sorted_d, sorted_d_bindings, sorted_d_offsets, sd0, sd1, sd2, sd3)
);

define_tuple_membership_expr_flags_with_policy!(
    tuple5_membership_expr_flags_with_policy,
    tuple5_membership_device_expr_flags_kernel,
    (CandidateA: candidate_a, candidate_a_bindings, candidate_a_offsets, ca0, ca1, ca2, ca3 /
     SortedA: sorted_a, sorted_a_bindings, sorted_a_offsets, sa0, sa1, sa2, sa3),
    (CandidateB: candidate_b, candidate_b_bindings, candidate_b_offsets, cb0, cb1, cb2, cb3 /
     SortedB: sorted_b, sorted_b_bindings, sorted_b_offsets, sb0, sb1, sb2, sb3),
    (CandidateC: candidate_c, candidate_c_bindings, candidate_c_offsets, cc0, cc1, cc2, cc3 /
     SortedC: sorted_c, sorted_c_bindings, sorted_c_offsets, sc0, sc1, sc2, sc3),
    (CandidateD: candidate_d, candidate_d_bindings, candidate_d_offsets, cd0, cd1, cd2, cd3 /
     SortedD: sorted_d, sorted_d_bindings, sorted_d_offsets, sd0, sd1, sd2, sd3),
    (CandidateE: candidate_e, candidate_e_bindings, candidate_e_offsets, ce0, ce1, ce2, ce3 /
     SortedE: sorted_e, sorted_e_bindings, sorted_e_offsets, se0, se1, se2, se3)
);

define_tuple_membership_expr_flags_with_policy!(
    tuple6_membership_expr_flags_with_policy,
    tuple6_membership_device_expr_flags_kernel,
    (CandidateA: candidate_a, candidate_a_bindings, candidate_a_offsets, ca0, ca1, ca2, ca3 /
     SortedA: sorted_a, sorted_a_bindings, sorted_a_offsets, sa0, sa1, sa2, sa3),
    (CandidateB: candidate_b, candidate_b_bindings, candidate_b_offsets, cb0, cb1, cb2, cb3 /
     SortedB: sorted_b, sorted_b_bindings, sorted_b_offsets, sb0, sb1, sb2, sb3),
    (CandidateC: candidate_c, candidate_c_bindings, candidate_c_offsets, cc0, cc1, cc2, cc3 /
     SortedC: sorted_c, sorted_c_bindings, sorted_c_offsets, sc0, sc1, sc2, sc3),
    (CandidateD: candidate_d, candidate_d_bindings, candidate_d_offsets, cd0, cd1, cd2, cd3 /
     SortedD: sorted_d, sorted_d_bindings, sorted_d_offsets, sd0, sd1, sd2, sd3),
    (CandidateE: candidate_e, candidate_e_bindings, candidate_e_offsets, ce0, ce1, ce2, ce3 /
     SortedE: sorted_e, sorted_e_bindings, sorted_e_offsets, se0, se1, se2, se3),
    (CandidateF: candidate_f, candidate_f_bindings, candidate_f_offsets, cf0, cf1, cf2, cf3 /
     SortedF: sorted_f, sorted_f_bindings, sorted_f_offsets, sf0, sf1, sf2, sf3)
);

define_tuple_membership_expr_flags_with_policy!(
    tuple7_membership_expr_flags_with_policy,
    tuple7_membership_device_expr_flags_kernel,
    (CandidateA: candidate_a, candidate_a_bindings, candidate_a_offsets, ca0, ca1, ca2, ca3 /
     SortedA: sorted_a, sorted_a_bindings, sorted_a_offsets, sa0, sa1, sa2, sa3),
    (CandidateB: candidate_b, candidate_b_bindings, candidate_b_offsets, cb0, cb1, cb2, cb3 /
     SortedB: sorted_b, sorted_b_bindings, sorted_b_offsets, sb0, sb1, sb2, sb3),
    (CandidateC: candidate_c, candidate_c_bindings, candidate_c_offsets, cc0, cc1, cc2, cc3 /
     SortedC: sorted_c, sorted_c_bindings, sorted_c_offsets, sc0, sc1, sc2, sc3),
    (CandidateD: candidate_d, candidate_d_bindings, candidate_d_offsets, cd0, cd1, cd2, cd3 /
     SortedD: sorted_d, sorted_d_bindings, sorted_d_offsets, sd0, sd1, sd2, sd3),
    (CandidateE: candidate_e, candidate_e_bindings, candidate_e_offsets, ce0, ce1, ce2, ce3 /
     SortedE: sorted_e, sorted_e_bindings, sorted_e_offsets, se0, se1, se2, se3),
    (CandidateF: candidate_f, candidate_f_bindings, candidate_f_offsets, cf0, cf1, cf2, cf3 /
     SortedF: sorted_f, sorted_f_bindings, sorted_f_offsets, sf0, sf1, sf2, sf3),
    (CandidateG: candidate_g, candidate_g_bindings, candidate_g_offsets, cg0, cg1, cg2, cg3 /
     SortedG: sorted_g, sorted_g_bindings, sorted_g_offsets, sg0, sg1, sg2, sg3)
);

pub(in crate::detail) fn device_expr_set_difference_with_policy<Left, Right, Less>(
    policy: &CubePolicy<Left::Runtime>,
    left: &Left,
    right: &Right,
) -> Result<DeviceVec<Left::Runtime, Left::Item>, Error>
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<Left::Item>,
{
    device_expr_membership_compact_with_policy::<Left, Right, Less>(policy, left, right, false)
}

pub(in crate::detail) fn device_expr_set_intersection_with_policy<Left, Right, Less>(
    policy: &CubePolicy<Left::Runtime>,
    left: &Left,
    right: &Right,
) -> Result<DeviceVec<Left::Runtime, Left::Item>, Error>
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<Left::Item>,
{
    device_expr_membership_compact_with_policy::<Left, Right, Less>(policy, left, right, true)
}

pub(in crate::detail) fn device_expr_set_union_with_policy<Left, Right, Less>(
    policy: &CubePolicy<Left::Runtime>,
    left: &Left,
    right: &Right,
) -> Result<DeviceVec<Left::Runtime, Left::Item>, Error>
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<Left::Item>,
{
    let right_only =
        device_expr_set_difference_with_policy::<Right, Left, Less>(policy, right, left)?;
    crate::detail::apply::MergeExprApply::apply_expr::<
        Left,
        DeviceVec<Left::Runtime, Left::Item>,
        Less,
    >(policy, left, &right_only)
}

pub(crate) fn device_expr_merge_by_key_control_with_policy<LeftKey, RightKey, Less>(
    policy: &CubePolicy<LeftKey::Runtime>,
    left_keys: &LeftKey,
    right_keys: &RightKey,
) -> Result<
    (
        DeviceVec<LeftKey::Runtime, LeftKey::Item>,
        ordering::MergeByKeyControl,
    ),
    Error,
>
where
    LeftKey: KernelColumn + KernelColumnAt<S0>,
    RightKey: KernelColumn<Runtime = LeftKey::Runtime, Item = LeftKey::Item> + KernelColumnAt<S0>,
    LeftKey::Item: CubePrimitive + CubeElement,
    LeftKey::Expr: DeviceGpuExpr<LeftKey::Item>,
    RightKey::Expr: DeviceGpuExpr<RightKey::Item>,
    Less: BinaryPredicateOp<LeftKey::Item>,
{
    left_keys.validate()?;
    right_keys.validate()?;
    let len = left_keys.len() + right_keys.len();
    let client = policy.client();
    let out_key_handle = client.empty(len * std::mem::size_of::<LeftKey::Item>());
    let source_sides = client.empty(len * std::mem::size_of::<u32>());
    let source_indices = client.empty(len * std::mem::size_of::<u32>());

    if len != 0 {
        let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
        let left_len_u32 = u32::try_from(left_keys.len()).map_err(|_| Error::LengthTooLarge {
            len: left_keys.len(),
        })?;
        let right_len_u32 = u32::try_from(right_keys.len()).map_err(|_| Error::LengthTooLarge {
            len: right_keys.len(),
        })?;
        let left_len_handle = client.create_from_slice(u32::as_bytes(&[left_len_u32]));
        let right_len_handle = client.create_from_slice(u32::as_bytes(&[right_len_u32]));
        let left_bindings = left_keys.stage(policy)?;
        let right_bindings = right_keys.stage(policy)?;
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
            merge_by_key_control_device_expr_kernel::launch_unchecked::<
                LeftKey::Item,
                LeftKey::Expr,
                RightKey::Expr,
                Less,
                LeftKey::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1) },
                unsafe { BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1) },
                unsafe { BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1) },
                unsafe { BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1) },
                unsafe { BufferArg::from_raw_parts(left_slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(left_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1) },
                unsafe { BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1) },
                unsafe { BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1) },
                unsafe { BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1) },
                unsafe { BufferArg::from_raw_parts(right_slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(right_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(out_key_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(source_sides.clone(), len) },
                unsafe { BufferArg::from_raw_parts(source_indices.clone(), len) },
            );
        }
    }

    Ok((
        DeviceVec::from_handle(policy.id(), out_key_handle, len),
        ordering::MergeByKeyControl {
            source_sides,
            source_indices,
            left_len: left_keys.len(),
            right_len: right_keys.len(),
            len,
        },
    ))
}

pub(crate) fn device_expr_merge_tuple2_by_key_control_with_policy<
    LeftA,
    LeftB,
    RightA,
    RightB,
    Less,
>(
    policy: &CubePolicy<LeftA::Runtime>,
    left_a: &LeftA,
    left_b: &LeftB,
    right_a: &RightA,
    right_b: &RightB,
) -> Result<
    (
        Zip2<DeviceVec<LeftA::Runtime, LeftA::Item>, DeviceVec<LeftA::Runtime, LeftB::Item>>,
        ordering::MergeByKeyControl,
    ),
    Error,
>
where
    LeftA: KernelColumn + KernelColumnAt<S0>,
    LeftB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    RightA: KernelColumn<Runtime = LeftA::Runtime, Item = LeftA::Item> + KernelColumnAt<S0>,
    RightB: KernelColumn<Runtime = LeftA::Runtime, Item = LeftB::Item> + KernelColumnAt<S0>,
    LeftA::Item: CubePrimitive + CubeElement,
    LeftB::Item: CubePrimitive + CubeElement,
    LeftA::Expr: DeviceGpuExpr<LeftA::Item>,
    LeftB::Expr: DeviceGpuExpr<LeftB::Item>,
    RightA::Expr: DeviceGpuExpr<RightA::Item>,
    RightB::Expr: DeviceGpuExpr<RightB::Item>,
    Less: BinaryPredicateOp<(LeftA::Item, LeftB::Item)>,
{
    left_a.validate()?;
    left_b.validate()?;
    right_a.validate()?;
    right_b.validate()?;
    super::ensure_same_len(left_a.len(), left_b.len())?;
    super::ensure_same_len(right_a.len(), right_b.len())?;
    let len = left_a.len() + right_a.len();
    let client = policy.client();
    let out_a_handle = client.empty(len * std::mem::size_of::<LeftA::Item>());
    let out_b_handle = client.empty(len * std::mem::size_of::<LeftB::Item>());
    let source_sides = client.empty(len * std::mem::size_of::<u32>());
    let source_indices = client.empty(len * std::mem::size_of::<u32>());

    if len != 0 {
        let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
        let left_len_u32 =
            u32::try_from(left_a.len()).map_err(|_| Error::LengthTooLarge { len: left_a.len() })?;
        let right_len_u32 = u32::try_from(right_a.len())
            .map_err(|_| Error::LengthTooLarge { len: right_a.len() })?;
        let left_len_handle = client.create_from_slice(u32::as_bytes(&[left_len_u32]));
        let right_len_handle = client.create_from_slice(u32::as_bytes(&[right_len_u32]));
        let left_a_bindings = left_a.stage(policy)?;
        let left_b_bindings = left_b.stage(policy)?;
        let right_a_bindings = right_a.stage(policy)?;
        let right_b_bindings = right_b.stage(policy)?;
        let left_a_offsets = left_a_bindings.slot_offsets_handle(client)?;
        let left_b_offsets = left_b_bindings.slot_offsets_handle(client)?;
        let right_a_offsets = right_a_bindings.slot_offsets_handle(client)?;
        let right_b_offsets = right_b_bindings.slot_offsets_handle(client)?;
        let la0 = left_a_bindings.slot_or_first(0);
        let la1 = left_a_bindings.slot_or_first(1);
        let la2 = left_a_bindings.slot_or_first(2);
        let la3 = left_a_bindings.slot_or_first(3);
        let lb0 = left_b_bindings.slot_or_first(0);
        let lb1 = left_b_bindings.slot_or_first(1);
        let lb2 = left_b_bindings.slot_or_first(2);
        let lb3 = left_b_bindings.slot_or_first(3);
        let ra0 = right_a_bindings.slot_or_first(0);
        let ra1 = right_a_bindings.slot_or_first(1);
        let ra2 = right_a_bindings.slot_or_first(2);
        let ra3 = right_a_bindings.slot_or_first(3);
        let rb0 = right_b_bindings.slot_or_first(0);
        let rb1 = right_b_bindings.slot_or_first(1);
        let rb2 = right_b_bindings.slot_or_first(2);
        let rb3 = right_b_bindings.slot_or_first(3);

        unsafe {
            merge_tuple2_by_key_control_device_expr_kernel::launch_unchecked::<
                LeftA::Item,
                LeftB::Item,
                LeftA::Expr,
                LeftB::Expr,
                RightA::Expr,
                RightB::Expr,
                Less,
                LeftA::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(la0.0.clone(), la0.1) },
                unsafe { BufferArg::from_raw_parts(la1.0.clone(), la1.1) },
                unsafe { BufferArg::from_raw_parts(la2.0.clone(), la2.1) },
                unsafe { BufferArg::from_raw_parts(la3.0.clone(), la3.1) },
                unsafe { BufferArg::from_raw_parts(left_a_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(lb0.0.clone(), lb0.1) },
                unsafe { BufferArg::from_raw_parts(lb1.0.clone(), lb1.1) },
                unsafe { BufferArg::from_raw_parts(lb2.0.clone(), lb2.1) },
                unsafe { BufferArg::from_raw_parts(lb3.0.clone(), lb3.1) },
                unsafe { BufferArg::from_raw_parts(left_b_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(left_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(ra0.0.clone(), ra0.1) },
                unsafe { BufferArg::from_raw_parts(ra1.0.clone(), ra1.1) },
                unsafe { BufferArg::from_raw_parts(ra2.0.clone(), ra2.1) },
                unsafe { BufferArg::from_raw_parts(ra3.0.clone(), ra3.1) },
                unsafe { BufferArg::from_raw_parts(right_a_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(rb0.0.clone(), rb0.1) },
                unsafe { BufferArg::from_raw_parts(rb1.0.clone(), rb1.1) },
                unsafe { BufferArg::from_raw_parts(rb2.0.clone(), rb2.1) },
                unsafe { BufferArg::from_raw_parts(rb3.0.clone(), rb3.1) },
                unsafe { BufferArg::from_raw_parts(right_b_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(right_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(out_a_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(out_b_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(source_sides.clone(), len) },
                unsafe { BufferArg::from_raw_parts(source_indices.clone(), len) },
            );
        }
    }

    Ok((
        Zip2 {
            left: DeviceVec::from_handle(policy.id(), out_a_handle, len),
            right: DeviceVec::from_handle(policy.id(), out_b_handle, len),
        },
        ordering::MergeByKeyControl {
            source_sides,
            source_indices,
            left_len: left_a.len(),
            right_len: right_a.len(),
            len,
        },
    ))
}

pub(crate) fn device_expr_merge_tuple3_by_key_control_with_policy<
    LeftA,
    LeftB,
    LeftC,
    RightA,
    RightB,
    RightC,
    Less,
>(
    policy: &CubePolicy<LeftA::Runtime>,
    left_a: &LeftA,
    left_b: &LeftB,
    left_c: &LeftC,
    right_a: &RightA,
    right_b: &RightB,
    right_c: &RightC,
) -> Result<
    (
        Zip3<
            DeviceVec<LeftA::Runtime, LeftA::Item>,
            DeviceVec<LeftA::Runtime, LeftB::Item>,
            DeviceVec<LeftA::Runtime, LeftC::Item>,
        >,
        ordering::MergeByKeyControl,
    ),
    Error,
>
where
    LeftA: KernelColumn + KernelColumnAt<S0>,
    LeftB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    LeftC: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
    RightA: KernelColumn<Runtime = LeftA::Runtime, Item = LeftA::Item> + KernelColumnAt<S0>,
    RightB: KernelColumn<Runtime = LeftA::Runtime, Item = LeftB::Item> + KernelColumnAt<S0>,
    RightC: KernelColumn<Runtime = LeftA::Runtime, Item = LeftC::Item> + KernelColumnAt<S0>,
    LeftA::Item: CubePrimitive + CubeElement,
    LeftB::Item: CubePrimitive + CubeElement,
    LeftC::Item: CubePrimitive + CubeElement,
    LeftA::Expr: DeviceGpuExpr<LeftA::Item>,
    LeftB::Expr: DeviceGpuExpr<LeftB::Item>,
    LeftC::Expr: DeviceGpuExpr<LeftC::Item>,
    RightA::Expr: DeviceGpuExpr<RightA::Item>,
    RightB::Expr: DeviceGpuExpr<RightB::Item>,
    RightC::Expr: DeviceGpuExpr<RightC::Item>,
    Less: BinaryPredicateOp<(LeftA::Item, LeftB::Item, LeftC::Item)>,
{
    left_a.validate()?;
    left_b.validate()?;
    left_c.validate()?;
    right_a.validate()?;
    right_b.validate()?;
    right_c.validate()?;
    super::ensure_same_len(left_a.len(), left_b.len())?;
    super::ensure_same_len(left_a.len(), left_c.len())?;
    super::ensure_same_len(right_a.len(), right_b.len())?;
    super::ensure_same_len(right_a.len(), right_c.len())?;
    let len = left_a.len() + right_a.len();
    let client = policy.client();
    let out_a_handle = client.empty(len * std::mem::size_of::<LeftA::Item>());
    let out_b_handle = client.empty(len * std::mem::size_of::<LeftB::Item>());
    let out_c_handle = client.empty(len * std::mem::size_of::<LeftC::Item>());
    let source_sides = client.empty(len * std::mem::size_of::<u32>());
    let source_indices = client.empty(len * std::mem::size_of::<u32>());

    if len != 0 {
        let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
        let left_len_u32 =
            u32::try_from(left_a.len()).map_err(|_| Error::LengthTooLarge { len: left_a.len() })?;
        let right_len_u32 = u32::try_from(right_a.len())
            .map_err(|_| Error::LengthTooLarge { len: right_a.len() })?;
        let left_len_handle = client.create_from_slice(u32::as_bytes(&[left_len_u32]));
        let right_len_handle = client.create_from_slice(u32::as_bytes(&[right_len_u32]));
        let left_a_bindings = left_a.stage(policy)?;
        let left_b_bindings = left_b.stage(policy)?;
        let left_c_bindings = left_c.stage(policy)?;
        let right_a_bindings = right_a.stage(policy)?;
        let right_b_bindings = right_b.stage(policy)?;
        let right_c_bindings = right_c.stage(policy)?;
        let left_a_offsets = left_a_bindings.slot_offsets_handle(client)?;
        let left_b_offsets = left_b_bindings.slot_offsets_handle(client)?;
        let left_c_offsets = left_c_bindings.slot_offsets_handle(client)?;
        let right_a_offsets = right_a_bindings.slot_offsets_handle(client)?;
        let right_b_offsets = right_b_bindings.slot_offsets_handle(client)?;
        let right_c_offsets = right_c_bindings.slot_offsets_handle(client)?;
        let la0 = left_a_bindings.slot_or_first(0);
        let la1 = left_a_bindings.slot_or_first(1);
        let la2 = left_a_bindings.slot_or_first(2);
        let la3 = left_a_bindings.slot_or_first(3);
        let lb0 = left_b_bindings.slot_or_first(0);
        let lb1 = left_b_bindings.slot_or_first(1);
        let lb2 = left_b_bindings.slot_or_first(2);
        let lb3 = left_b_bindings.slot_or_first(3);
        let lc0 = left_c_bindings.slot_or_first(0);
        let lc1 = left_c_bindings.slot_or_first(1);
        let lc2 = left_c_bindings.slot_or_first(2);
        let lc3 = left_c_bindings.slot_or_first(3);
        let ra0 = right_a_bindings.slot_or_first(0);
        let ra1 = right_a_bindings.slot_or_first(1);
        let ra2 = right_a_bindings.slot_or_first(2);
        let ra3 = right_a_bindings.slot_or_first(3);
        let rb0 = right_b_bindings.slot_or_first(0);
        let rb1 = right_b_bindings.slot_or_first(1);
        let rb2 = right_b_bindings.slot_or_first(2);
        let rb3 = right_b_bindings.slot_or_first(3);
        let rc0 = right_c_bindings.slot_or_first(0);
        let rc1 = right_c_bindings.slot_or_first(1);
        let rc2 = right_c_bindings.slot_or_first(2);
        let rc3 = right_c_bindings.slot_or_first(3);

        unsafe {
            merge_tuple3_by_key_control_device_expr_kernel::launch_unchecked::<
                LeftA::Item,
                LeftB::Item,
                LeftC::Item,
                LeftA::Expr,
                LeftB::Expr,
                LeftC::Expr,
                RightA::Expr,
                RightB::Expr,
                RightC::Expr,
                Less,
                LeftA::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(la0.0.clone(), la0.1) },
                unsafe { BufferArg::from_raw_parts(la1.0.clone(), la1.1) },
                unsafe { BufferArg::from_raw_parts(la2.0.clone(), la2.1) },
                unsafe { BufferArg::from_raw_parts(la3.0.clone(), la3.1) },
                unsafe { BufferArg::from_raw_parts(left_a_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(lb0.0.clone(), lb0.1) },
                unsafe { BufferArg::from_raw_parts(lb1.0.clone(), lb1.1) },
                unsafe { BufferArg::from_raw_parts(lb2.0.clone(), lb2.1) },
                unsafe { BufferArg::from_raw_parts(lb3.0.clone(), lb3.1) },
                unsafe { BufferArg::from_raw_parts(left_b_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(lc0.0.clone(), lc0.1) },
                unsafe { BufferArg::from_raw_parts(lc1.0.clone(), lc1.1) },
                unsafe { BufferArg::from_raw_parts(lc2.0.clone(), lc2.1) },
                unsafe { BufferArg::from_raw_parts(lc3.0.clone(), lc3.1) },
                unsafe { BufferArg::from_raw_parts(left_c_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(left_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(ra0.0.clone(), ra0.1) },
                unsafe { BufferArg::from_raw_parts(ra1.0.clone(), ra1.1) },
                unsafe { BufferArg::from_raw_parts(ra2.0.clone(), ra2.1) },
                unsafe { BufferArg::from_raw_parts(ra3.0.clone(), ra3.1) },
                unsafe { BufferArg::from_raw_parts(right_a_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(rb0.0.clone(), rb0.1) },
                unsafe { BufferArg::from_raw_parts(rb1.0.clone(), rb1.1) },
                unsafe { BufferArg::from_raw_parts(rb2.0.clone(), rb2.1) },
                unsafe { BufferArg::from_raw_parts(rb3.0.clone(), rb3.1) },
                unsafe { BufferArg::from_raw_parts(right_b_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(rc0.0.clone(), rc0.1) },
                unsafe { BufferArg::from_raw_parts(rc1.0.clone(), rc1.1) },
                unsafe { BufferArg::from_raw_parts(rc2.0.clone(), rc2.1) },
                unsafe { BufferArg::from_raw_parts(rc3.0.clone(), rc3.1) },
                unsafe { BufferArg::from_raw_parts(right_c_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(right_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(out_a_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(out_b_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(out_c_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(source_sides.clone(), len) },
                unsafe { BufferArg::from_raw_parts(source_indices.clone(), len) },
            );
        }
    }

    Ok((
        Zip3 {
            first: DeviceVec::from_handle(policy.id(), out_a_handle, len),
            second: DeviceVec::from_handle(policy.id(), out_b_handle, len),
            third: DeviceVec::from_handle(policy.id(), out_c_handle, len),
        },
        ordering::MergeByKeyControl {
            source_sides,
            source_indices,
            left_len: left_a.len(),
            right_len: right_a.len(),
            len,
        },
    ))
}

pub(crate) fn device_expr_merge_by_key_values_with_control_with_policy<LeftValue, RightValue>(
    policy: &CubePolicy<LeftValue::Runtime>,
    left_values: &LeftValue,
    right_values: &RightValue,
    control: &ordering::MergeByKeyControl,
) -> Result<DeviceVec<LeftValue::Runtime, LeftValue::Item>, Error>
where
    LeftValue: KernelColumn + KernelColumnAt<S0>,
    RightValue:
        KernelColumn<Runtime = LeftValue::Runtime, Item = LeftValue::Item> + KernelColumnAt<S0>,
    LeftValue::Item: CubePrimitive + CubeElement,
    LeftValue::Expr: DeviceGpuExpr<LeftValue::Item>,
    RightValue::Expr: DeviceGpuExpr<RightValue::Item>,
{
    left_values.validate()?;
    right_values.validate()?;
    super::ensure_same_len(left_values.len(), control.left_len)?;
    super::ensure_same_len(right_values.len(), control.right_len)?;

    let len = control.len;
    let client = policy.client();
    let out_value_handle = client.empty(len * std::mem::size_of::<LeftValue::Item>());

    if len != 0 {
        let output_offset = client.create_from_slice(u32::as_bytes(&[0]));
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
        let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
        let left_bindings = left_values.stage(policy)?;
        let right_bindings = right_values.stage(policy)?;
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
            merge_by_key_values_from_control_device_expr_kernel::launch_unchecked::<
                LeftValue::Item,
                LeftValue::Expr,
                RightValue::Expr,
                LeftValue::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
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
                unsafe { BufferArg::from_raw_parts(control.source_sides.clone(), len) },
                unsafe { BufferArg::from_raw_parts(control.source_indices.clone(), len) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(out_value_handle.clone(), len) },
            );
        }
    }

    Ok(DeviceVec::from_handle(policy.id(), out_value_handle, len))
}

pub(crate) fn device_expr_merge_by_key_values_into_with_control_with_policy<LeftValue, RightValue>(
    policy: &CubePolicy<LeftValue::Runtime>,
    left_values: &LeftValue,
    right_values: &RightValue,
    control: &ordering::MergeByKeyControl,
    output: &DeviceColumnMutView<LeftValue::Runtime, LeftValue::Item>,
) -> Result<(), Error>
where
    LeftValue: KernelColumn + KernelColumnAt<S0>,
    RightValue:
        KernelColumn<Runtime = LeftValue::Runtime, Item = LeftValue::Item> + KernelColumnAt<S0>,
    LeftValue::Item: CubePrimitive + CubeElement,
    LeftValue::Expr: DeviceGpuExpr<LeftValue::Item>,
    RightValue::Expr: DeviceGpuExpr<RightValue::Item>,
{
    left_values.validate()?;
    right_values.validate()?;
    super::ensure_same_len(left_values.len(), control.left_len)?;
    super::ensure_same_len(right_values.len(), control.right_len)?;
    super::ensure_same_len(output.len, control.len)?;

    let len = control.len;
    if len == 0 {
        return Ok(());
    }

    let client = policy.client();
    let output_offset_u32 =
        u32::try_from(output.offset).map_err(|_| Error::LengthTooLarge { len: output.offset })?;
    let output_offset = client.create_from_slice(u32::as_bytes(&[output_offset_u32]));
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let left_bindings = left_values.stage(policy)?;
    let right_bindings = right_values.stage(policy)?;
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
        merge_by_key_values_from_control_device_expr_kernel::launch_unchecked::<
            LeftValue::Item,
            LeftValue::Expr,
            RightValue::Expr,
            LeftValue::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_ORDERING_SIZE),
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
            unsafe { BufferArg::from_raw_parts(control.source_sides.clone(), len) },
            unsafe { BufferArg::from_raw_parts(control.source_indices.clone(), len) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()) },
        );
    }

    Ok(())
}

impl<Left, Right, Less> crate::detail::read::KernelPairOrderingInput<ZipView1<Right>, Less>
    for ZipView1<Left>
where
    Self: ReadOnlyZip<Item = (Left::Item,), Scalar = Left::Item>,
    ZipView1<Right>: ReadOnlyZip<Item = (Right::Item,), Scalar = Right::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<Left::Item>,
{
    type Runtime = Left::Runtime;
    type Output = Zip1<DeviceVec<Left::Runtime, Left::Item>>;

    fn merge_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: ZipView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlyZip::validate(&self)?;
        ReadOnlyZip::validate(&other)?;
        Ok(Zip1 {
            source: crate::detail::apply::MergeExprApply::apply_expr::<Left, Right, Less>(
                policy,
                &self.source,
                &other.source,
            )?,
        })
    }

    fn set_union_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: ZipView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlyZip::validate(&self)?;
        ReadOnlyZip::validate(&other)?;
        Ok(Zip1 {
            source: crate::detail::apply::SetMembershipControlApply::set_union_expr::<
                Left,
                Right,
                Less,
            >(policy, &self.source, &other.source)?,
        })
    }

    fn set_intersection_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: ZipView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlyZip::validate(&self)?;
        ReadOnlyZip::validate(&other)?;
        Ok(Zip1 {
            source: crate::detail::apply::SetMembershipControlApply::set_intersection_expr::<
                Left,
                Right,
                Less,
            >(policy, &self.source, &other.source)?,
        })
    }

    fn set_difference_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: ZipView1<Right>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlyZip::validate(&self)?;
        ReadOnlyZip::validate(&other)?;
        Ok(Zip1 {
            source: crate::detail::apply::SetMembershipControlApply::set_difference_expr::<
                Left,
                Right,
                Less,
            >(policy, &self.source, &other.source)?,
        })
    }
}

impl<Left, Right, Less> crate::detail::read::KernelPairOrderingInput<Right, Less> for Left
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<Left::Item>,
{
    type Runtime = Left::Runtime;
    type Output = Zip1<DeviceVec<Left::Runtime, Left::Item>>;

    fn merge_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: Right,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <ZipView1<Left> as crate::detail::read::KernelPairOrderingInput<ZipView1<Right>, Less>>::merge_input(
            ZipView1 { source: self },
            policy,
            ZipView1 { source: other },
            less,
        )
    }

    fn set_union_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: Right,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <ZipView1<Left> as crate::detail::read::KernelPairOrderingInput<ZipView1<Right>, Less>>::set_union_input(
            ZipView1 { source: self },
            policy,
            ZipView1 { source: other },
            less,
        )
    }

    fn set_intersection_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: Right,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <ZipView1<Left> as crate::detail::read::KernelPairOrderingInput<ZipView1<Right>, Less>>::set_intersection_input(
            ZipView1 { source: self },
            policy,
            ZipView1 { source: other },
            less,
        )
    }

    fn set_difference_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: Right,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <ZipView1<Left> as crate::detail::read::KernelPairOrderingInput<ZipView1<Right>, Less>>::set_difference_input(
            ZipView1 { source: self },
            policy,
            ZipView1 { source: other },
            less,
        )
    }
}

impl<Left, Right, Less> crate::detail::read::KernelPairOrderingInput<(Right,), Less> for (Left,)
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item> + KernelColumnAt<S0>,
    Left::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<(Left::Item,)>,
{
    type Runtime = Left::Runtime;
    type Output = Zip1<DeviceVec<Left::Runtime, Left::Item>>;

    fn merge_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        other: (Right,),
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <Left as crate::detail::read::KernelPairOrderingInput<Right, super::Tuple1Less<Less>>>::merge_input(
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
        <Left as crate::detail::read::KernelPairOrderingInput<Right, super::Tuple1Less<Less>>>::set_union_input(
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
        <Left as crate::detail::read::KernelPairOrderingInput<Right, super::Tuple1Less<Less>>>::set_intersection_input(
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
        <Left as crate::detail::read::KernelPairOrderingInput<Right, super::Tuple1Less<Less>>>::set_difference_input(
            self.0,
            policy,
            other.0,
            GpuOp::<super::Tuple1Less<Less>>::new(),
        )
    }
}

macro_rules! impl_tuple_pair_ordering {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $input:ident -> $output:ident < $first:ident, $( $rest:ident ),+ ; $right_first_ty:ident, $( $right_rest_ty:ident ),+ >
        { $first_field:ident / $right_first_var:ident, $( $field:ident / $right_var:ident ),+ },
        $merge_control_fn:ident,
        $membership_expr_fn:ident,
        $selected_apply:ident
    ) => {
        impl<$first, $( $rest ),+, $right_first_ty, $( $right_rest_ty ),+, Less>
            crate::detail::read::KernelPairOrderingInput<$input<$right_first_ty, $( $right_rest_ty ),+>, Less>
            for $input<$first, $( $rest ),+>
        where
            Self: ReadOnlyZip<Scalar = <$first as KernelColumn>::Item>,
            $input<$right_first_ty, $( $right_rest_ty ),+>: ReadOnlyZip<Scalar = <$first as KernelColumn>::Item>,
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
                ReadOnlyZip::validate(&self)?;
                ReadOnlyZip::validate(&other)?;
                let (output, _) = $merge_control_fn::<
                    $first,
                    $( $rest, )+
                    $right_first_ty,
                    $( $right_rest_ty, )+
                    Less,
                >(
                    policy,
                    &self.$first_field,
                    $( &self.$field, )+
                    &other.$first_field,
                    $( &other.$field, )+
                )?;
                Ok(output)
            }

            fn set_union_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                other: $input<$right_first_ty, $( $right_rest_ty ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlyZip::validate(&self)?;
                ReadOnlyZip::validate(&other)?;
                let flags = crate::detail::apply::SetMembershipControlApply::$membership_expr_fn::<
                    $right_first_ty,
                    $( $right_rest_ty, )+
                    $first,
                    $( $rest, )+
                    Less,
                >(
                    policy,
                    &other.$first_field,
                    $( &other.$field, )+
                    &self.$first_field,
                    $( &self.$field, )+
                    false,
                )?;
                let (selection, count) =
                    selected_rank_from_flags_with_policy(policy, other.$first_field.len(), flags)?;
                let selected_apply = crate::detail::apply::SelectedPayloadApply::new(&selection, count);
                let ($right_first_var, $( $right_var ),+) =
                    selected_apply.$selected_apply(policy, &other.$first_field, $( &other.$field ),+)?;
                let (output, _) = $merge_control_fn::<
                    $first,
                    $( $rest, )+
                    DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                    $( DeviceVec<<$first as KernelColumn>::Runtime, <$rest as KernelColumn>::Item>, )+
                    Less,
                >(
                    policy,
                    &self.$first_field,
                    $( &self.$field, )+
                    &$right_first_var,
                    $( &$right_var, )+
                )?;
                Ok(output)
            }

            fn set_intersection_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                other: $input<$right_first_ty, $( $right_rest_ty ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlyZip::validate(&self)?;
                ReadOnlyZip::validate(&other)?;
                let flags = crate::detail::apply::SetMembershipControlApply::$membership_expr_fn::<
                    $first,
                    $( $rest, )+
                    $right_first_ty,
                    $( $right_rest_ty, )+
                    Less,
                >(
                    policy,
                    &self.$first_field,
                    $( &self.$field, )+
                    &other.$first_field,
                    $( &other.$field, )+
                    true,
                )?;
                let (selection, count) =
                    selected_rank_from_flags_with_policy(policy, self.$first_field.len(), flags)?;
                let selected_apply = crate::detail::apply::SelectedPayloadApply::new(&selection, count);
                let ($first_field, $( $field ),+) =
                    selected_apply.$selected_apply(policy, &self.$first_field, $( &self.$field ),+)?;
                Ok($output { $first_field, $( $field ),+ })
            }

            fn set_difference_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
                other: $input<$right_first_ty, $( $right_rest_ty ),+>,
                _less: GpuOp<Less>,
            ) -> Result<Self::Output, Error> {
                ReadOnlyZip::validate(&self)?;
                ReadOnlyZip::validate(&other)?;
                let flags = crate::detail::apply::SetMembershipControlApply::$membership_expr_fn::<
                    $first,
                    $( $rest, )+
                    $right_first_ty,
                    $( $right_rest_ty, )+
                    Less,
                >(
                    policy,
                    &self.$first_field,
                    $( &self.$field, )+
                    &other.$first_field,
                    $( &other.$field, )+
                    false,
                )?;
                let (selection, count) =
                    selected_rank_from_flags_with_policy(policy, self.$first_field.len(), flags)?;
                let selected_apply = crate::detail::apply::SelectedPayloadApply::new(&selection, count);
                let ($first_field, $( $field ),+) =
                    selected_apply.$selected_apply(policy, &self.$first_field, $( &self.$field ),+)?;
                Ok($output { $first_field, $( $field ),+ })
            }

        }
    };
}

impl_tuple_pair_ordering!(ZipView2 -> Zip2<A, B; RA, RB> { left / right_left, right / right_right }, device_expr_merge_tuple2_by_key_control_with_policy, tuple2_membership_expr_flags_with_policy, apply_expr2);
impl_tuple_pair_ordering!(Zip2 -> Zip2<A, B; RA, RB> { left / right_left, right / right_right }, device_expr_merge_tuple2_by_key_control_with_policy, tuple2_membership_expr_flags_with_policy, apply_expr2);
impl_tuple_pair_ordering!(ZipView3 -> Zip3<A, B, C; RA, RB, RC> { first / right_first, second / right_second, third / right_third }, device_expr_merge_tuple3_by_key_control_with_policy, tuple3_membership_expr_flags_with_policy, apply_expr3);
impl_tuple_pair_ordering!(Zip3 -> Zip3<A, B, C; RA, RB, RC> { first / right_first, second / right_second, third / right_third }, device_expr_merge_tuple3_by_key_control_with_policy, tuple3_membership_expr_flags_with_policy, apply_expr3);

mod reverse;
pub use reverse::reverse;

mod by_key;
mod sort;
pub use sort::sort;

/// Merges two sorted read-only inputs into owned device storage.
///
/// This is a borrowing algorithm. Both inputs are read, and the merged output is
/// newly materialized.
pub fn merge<R, Left, Right, Less>(
    policy: &CubePolicy<R>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<<Left as crate::detail::read::KernelPairOrderingInput<Right, Less>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Left: crate::detail::read::KernelPairOrderingInput<Right, Less, Runtime = R>,
    <Left as crate::detail::read::KernelPairOrderingInput<Right, Less>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        left.merge_input(policy, right, GpuOp::<Less>::new())?,
    )
}

/// Computes the sorted set union of two sorted device vectors.
pub fn set_union<R, Left, Right, Less>(
    policy: &CubePolicy<R>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<<<Left as crate::detail::read::KernelPairOrderingInput<Right, Less>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Left: crate::detail::read::KernelPairOrderingInput<Right, Less, Runtime = R>,
    <Left as crate::detail::read::KernelPairOrderingInput<Right, Less>>::Output: MaterializeOutput<Runtime = R>,
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
) -> Result<<<Left as crate::detail::read::KernelPairOrderingInput<Right, Less>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Left: crate::detail::read::KernelPairOrderingInput<Right, Less, Runtime = R>,
    <Left as crate::detail::read::KernelPairOrderingInput<Right, Less>>::Output: MaterializeOutput<Runtime = R>,
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
) -> Result<<<Left as crate::detail::read::KernelPairOrderingInput<Right, Less>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Left: crate::detail::read::KernelPairOrderingInput<Right, Less, Runtime = R>,
    <Left as crate::detail::read::KernelPairOrderingInput<Right, Less>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(
        policy,
        left.set_difference_input(policy, right, GpuOp::<Less>::new())?,
    )
}
