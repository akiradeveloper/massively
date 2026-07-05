use crate::{
    detail::{
        api::{
            Tuple2AsTuple3Less,
            ordering::{
                device_expr_merge_by_key_control_with_policy,
                device_expr_merge_control_with_policy,
                device_expr_merge_tuple2_by_key_control_with_policy,
                device_expr_merge_tuple3_by_key_control_with_policy,
                device_expr_set_difference_with_policy, device_expr_set_intersection_with_policy,
                device_expr_set_union_with_policy, tuple2_membership_expr_flags_with_policy,
                tuple3_membership_expr_flags_with_policy,
            },
        },
        device::{DeviceVec, KernelColumn, KernelColumnAt, ReadOnlyKernelColumn, S0, SoA2, SoA3},
        primitives::{ordering as primitive_ordering, range as primitive_range},
    },
    error::Error,
    expr::DeviceGpuExpr,
    index::MIndex,
    op::kernel::BinaryPredicateOp,
    policy::CubePolicy,
    value::MStorageElement,
};
use cubecl::prelude::*;

pub(in crate::detail) struct SortApply;

impl SortApply {
    pub(in crate::detail) fn apply_expr<Source, Less>(
        policy: &CubePolicy<Source::Runtime>,
        source: &Source,
    ) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
    where
        Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
        Source::Item: MStorageElement + 'static,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Less: BinaryPredicateOp<Source::Item>,
    {
        primitive_ordering::sort_input_with_policy(policy, source, crate::op::GpuOp::<Less>::new())
    }

    pub(in crate::detail) fn apply_expr2<Left, Right, Less>(
        policy: &CubePolicy<Left::Runtime>,
        left: &Left,
        right: &Right,
    ) -> Result<
        (
            DeviceVec<Left::Runtime, Left::Item>,
            DeviceVec<Left::Runtime, Right::Item>,
        ),
        Error,
    >
    where
        Left: KernelColumn + KernelColumnAt<S0>,
        Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
        Left::Item: MStorageElement + 'static,
        Right::Item: MStorageElement + 'static,
        Left::Expr: DeviceGpuExpr<Left::Item>,
        Right::Expr: DeviceGpuExpr<Right::Item>,
        Less: BinaryPredicateOp<(Left::Item, Right::Item)>,
    {
        primitive_ordering::sort_tuple2_input(policy, left, right, crate::op::GpuOp::<Less>::new())
    }

    pub(in crate::detail) fn apply_expr3<First, Second, Third, Less>(
        policy: &CubePolicy<First::Runtime>,
        first: &First,
        second: &Second,
        third: &Third,
    ) -> Result<
        (
            DeviceVec<First::Runtime, First::Item>,
            DeviceVec<First::Runtime, Second::Item>,
            DeviceVec<First::Runtime, Third::Item>,
        ),
        Error,
    >
    where
        First: KernelColumn + KernelColumnAt<S0>,
        Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
        Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
        First::Item: MStorageElement + 'static,
        Second::Item: MStorageElement + 'static,
        Third::Item: MStorageElement + 'static,
        First::Expr: DeviceGpuExpr<First::Item>,
        Second::Expr: DeviceGpuExpr<Second::Item>,
        Third::Expr: DeviceGpuExpr<Third::Item>,
        Less: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
    {
        primitive_ordering::sort_tuple3_input(
            policy,
            first,
            second,
            third,
            crate::op::GpuOp::<Less>::new(),
        )
    }
}

pub(in crate::detail) struct SortByKeyApply;

impl SortByKeyApply {
    pub(in crate::detail) fn apply_keys1<KeySource, Less>(
        policy: &CubePolicy<KeySource::Runtime>,
        keys: &KeySource,
    ) -> Result<
        (
            DeviceVec<KeySource::Runtime, KeySource::Item>,
            DeviceVec<KeySource::Runtime, MIndex>,
        ),
        Error,
    >
    where
        KeySource: KernelColumn + KernelColumnAt<S0>,
        KeySource::Item: MStorageElement + 'static,
        KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
        Less: BinaryPredicateOp<KeySource::Item>,
    {
        let indices =
            primitive_range::indices_mindex(policy, <KeySource as KernelColumn>::len(keys))?;
        primitive_ordering::sort_by_key_input_with_policy(
            policy,
            keys,
            &indices,
            crate::op::GpuOp::<Less>::new(),
        )
    }

    pub(in crate::detail) fn apply_keys2<First, Second, Less>(
        policy: &CubePolicy<First::Runtime>,
        first: &First,
        second: &Second,
    ) -> Result<
        (
            DeviceVec<First::Runtime, First::Item>,
            DeviceVec<First::Runtime, Second::Item>,
            DeviceVec<First::Runtime, MIndex>,
        ),
        Error,
    >
    where
        First: KernelColumn + KernelColumnAt<S0>,
        Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
        First::Item: MStorageElement + 'static,
        Second::Item: MStorageElement + 'static,
        First::Expr: DeviceGpuExpr<First::Item>,
        Second::Expr: DeviceGpuExpr<Second::Item>,
        Less: BinaryPredicateOp<(First::Item, Second::Item)>,
        Tuple2AsTuple3Less<Less>: BinaryPredicateOp<(First::Item, Second::Item, u32)>,
    {
        let indices = primitive_range::indices_mindex(policy, <First as KernelColumn>::len(first))?;
        let (first, second, _stable_tie, indices) =
            primitive_ordering::sort_tuple3_by_key_input_with_policy(
                policy,
                first,
                second,
                &indices,
                &indices,
                crate::op::GpuOp::<Tuple2AsTuple3Less<Less>>::new(),
            )?;
        Ok((first, second, indices))
    }

    pub(in crate::detail) fn apply_keys3<First, Second, Third, Less>(
        policy: &CubePolicy<First::Runtime>,
        first: &First,
        second: &Second,
        third: &Third,
    ) -> Result<
        (
            DeviceVec<First::Runtime, First::Item>,
            DeviceVec<First::Runtime, Second::Item>,
            DeviceVec<First::Runtime, Third::Item>,
            DeviceVec<First::Runtime, MIndex>,
        ),
        Error,
    >
    where
        First: KernelColumn + KernelColumnAt<S0>,
        Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
        Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
        First::Item: MStorageElement + 'static,
        Second::Item: MStorageElement + 'static,
        Third::Item: MStorageElement + 'static,
        First::Expr: DeviceGpuExpr<First::Item>,
        Second::Expr: DeviceGpuExpr<Second::Item>,
        Third::Expr: DeviceGpuExpr<Third::Item>,
        Less: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
    {
        let indices = primitive_range::indices_mindex(policy, <First as KernelColumn>::len(first))?;
        primitive_ordering::sort_tuple3_by_key_input_with_policy(
            policy,
            first,
            second,
            third,
            &indices,
            crate::op::GpuOp::<Less>::new(),
        )
    }
}

pub(in crate::detail) struct MergeExprApply;

impl MergeExprApply {
    pub(in crate::detail) fn apply_expr<Left, Right, Less>(
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
        let control = MergeControlApply::apply_expr::<Left, Right, Less>(policy, left, right)?;
        let payload_control = control.as_merge_by_key_control();
        crate::detail::apply::MergePayloadApply::new(&payload_control)
            .apply_expr(policy, left, right)
    }
}

pub(in crate::detail) struct MergeControlApply;

impl MergeControlApply {
    pub(in crate::detail) fn apply_expr<Left, Right, Less>(
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
        device_expr_merge_control_with_policy::<Left, Right, Less>(policy, left, right)
    }
}

pub(in crate::detail) struct MergeByKeyControlApply;

impl MergeByKeyControlApply {
    pub(in crate::detail) fn apply_keys1<LeftKey, RightKey, Less>(
        policy: &CubePolicy<LeftKey::Runtime>,
        left_keys: &LeftKey,
        right_keys: &RightKey,
    ) -> Result<
        (
            DeviceVec<LeftKey::Runtime, LeftKey::Item>,
            primitive_ordering::MergeByKeyControl,
        ),
        Error,
    >
    where
        LeftKey: KernelColumn + KernelColumnAt<S0>,
        RightKey:
            KernelColumn<Runtime = LeftKey::Runtime, Item = LeftKey::Item> + KernelColumnAt<S0>,
        LeftKey::Item: CubePrimitive + CubeElement,
        LeftKey::Expr: DeviceGpuExpr<LeftKey::Item>,
        RightKey::Expr: DeviceGpuExpr<RightKey::Item>,
        Less: BinaryPredicateOp<LeftKey::Item>,
    {
        device_expr_merge_by_key_control_with_policy::<LeftKey, RightKey, Less>(
            policy, left_keys, right_keys,
        )
    }

    #[allow(clippy::type_complexity)]
    pub(in crate::detail) fn apply_keys2<LeftA, LeftB, RightA, RightB, Less>(
        policy: &CubePolicy<LeftA::Runtime>,
        left_a: &LeftA,
        left_b: &LeftB,
        right_a: &RightA,
        right_b: &RightB,
    ) -> Result<
        (
            SoA2<DeviceVec<LeftA::Runtime, LeftA::Item>, DeviceVec<LeftA::Runtime, LeftB::Item>>,
            primitive_ordering::MergeByKeyControl,
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
        device_expr_merge_tuple2_by_key_control_with_policy::<LeftA, LeftB, RightA, RightB, Less>(
            policy, left_a, left_b, right_a, right_b,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub(in crate::detail) fn apply_keys3<LeftA, LeftB, LeftC, RightA, RightB, RightC, Less>(
        policy: &CubePolicy<LeftA::Runtime>,
        left_a: &LeftA,
        left_b: &LeftB,
        left_c: &LeftC,
        right_a: &RightA,
        right_b: &RightB,
        right_c: &RightC,
    ) -> Result<
        (
            SoA3<
                DeviceVec<LeftA::Runtime, LeftA::Item>,
                DeviceVec<LeftA::Runtime, LeftB::Item>,
                DeviceVec<LeftA::Runtime, LeftC::Item>,
            >,
            primitive_ordering::MergeByKeyControl,
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
        device_expr_merge_tuple3_by_key_control_with_policy::<
            LeftA,
            LeftB,
            LeftC,
            RightA,
            RightB,
            RightC,
            Less,
        >(policy, left_a, left_b, left_c, right_a, right_b, right_c)
    }
}

pub(in crate::detail) struct SetMembershipControlApply;

impl SetMembershipControlApply {
    pub(in crate::detail) fn set_union_expr<Left, Right, Less>(
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
        device_expr_set_union_with_policy::<Left, Right, Less>(policy, left, right)
    }

    pub(in crate::detail) fn set_intersection_expr<Left, Right, Less>(
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
        device_expr_set_intersection_with_policy::<Left, Right, Less>(policy, left, right)
    }

    pub(in crate::detail) fn set_difference_expr<Left, Right, Less>(
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
        device_expr_set_difference_with_policy::<Left, Right, Less>(policy, left, right)
    }

    #[allow(clippy::too_many_arguments)]
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
        keep_members: bool,
    ) -> Result<cubecl::server::Handle, Error>
    where
        CandidateA: KernelColumn + KernelColumnAt<S0>,
        CandidateB: KernelColumn<Runtime = CandidateA::Runtime> + KernelColumnAt<S0>,
        SortedA: KernelColumn<Runtime = CandidateA::Runtime, Item = CandidateA::Item>
            + KernelColumnAt<S0>,
        SortedB: KernelColumn<Runtime = CandidateA::Runtime, Item = CandidateB::Item>
            + KernelColumnAt<S0>,
        CandidateA::Item: CubePrimitive + CubeElement,
        CandidateB::Item: CubePrimitive + CubeElement,
        CandidateA::Expr: DeviceGpuExpr<CandidateA::Item>,
        CandidateB::Expr: DeviceGpuExpr<CandidateB::Item>,
        SortedA::Expr: DeviceGpuExpr<SortedA::Item>,
        SortedB::Expr: DeviceGpuExpr<SortedB::Item>,
        Less: BinaryPredicateOp<(CandidateA::Item, CandidateB::Item)>,
    {
        tuple2_membership_expr_flags_with_policy::<CandidateA, CandidateB, SortedA, SortedB, Less>(
            policy,
            candidate_a,
            candidate_b,
            sorted_a,
            sorted_b,
            keep_members,
        )
    }

    #[allow(clippy::too_many_arguments)]
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
        keep_members: bool,
    ) -> Result<cubecl::server::Handle, Error>
    where
        CandidateA: KernelColumn + KernelColumnAt<S0>,
        CandidateB: KernelColumn<Runtime = CandidateA::Runtime> + KernelColumnAt<S0>,
        CandidateC: KernelColumn<Runtime = CandidateA::Runtime> + KernelColumnAt<S0>,
        SortedA: KernelColumn<Runtime = CandidateA::Runtime, Item = CandidateA::Item>
            + KernelColumnAt<S0>,
        SortedB: KernelColumn<Runtime = CandidateA::Runtime, Item = CandidateB::Item>
            + KernelColumnAt<S0>,
        SortedC: KernelColumn<Runtime = CandidateA::Runtime, Item = CandidateC::Item>
            + KernelColumnAt<S0>,
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
        tuple3_membership_expr_flags_with_policy::<
            CandidateA,
            CandidateB,
            CandidateC,
            SortedA,
            SortedB,
            SortedC,
            Less,
        >(
            policy,
            candidate_a,
            candidate_b,
            candidate_c,
            sorted_a,
            sorted_b,
            sorted_c,
            keep_members,
        )
    }
}
