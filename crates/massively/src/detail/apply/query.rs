use super::*;

pub(in crate::detail) struct QueryApply;

impl QueryApply {
    pub(in crate::detail) fn count_expr<Source, Pred>(
        policy: &crate::policy::CubePolicy<Source::Runtime>,
        source: &Source,
        invert: bool,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Source::Runtime>,
    ) -> Result<MIndex, Error>
    where
        Source: KernelColumn + KernelColumnAt<S0>,
        Source::Item: CubePrimitive + CubeElement,
        Source::Expr: GpuExpr<Source::Item>,
        Pred: PredicateOp<Source::Item>,
    {
        device_expr_count_if_with_policy::<Source, Pred>(policy, source, invert, env)
    }

    pub(in crate::detail) fn find_expr<Source, Pred>(
        policy: &crate::policy::CubePolicy<Source::Runtime>,
        source: &Source,
        invert: bool,
        env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Source::Runtime>,
    ) -> Result<Option<MIndex>, Error>
    where
        Source: KernelColumn + KernelColumnAt<S0>,
        Source::Item: CubePrimitive + CubeElement,
        Source::Expr: GpuExpr<Source::Item>,
        Pred: PredicateOp<Source::Item>,
    {
        device_expr_find_if_with_policy::<Source, Pred>(policy, source, invert, env)
    }

    pub(in crate::detail) fn minmax_expr<Source, Less>(
        policy: &crate::policy::CubePolicy<Source::Runtime>,
        source: &Source,
    ) -> Result<Option<(MIndex, MIndex)>, Error>
    where
        Source: KernelColumn + KernelColumnAt<S0>,
        Source::Item: CubePrimitive + CubeElement,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Less: BinaryPredicateOp<Source::Item>,
    {
        device_expr_minmax_element_with_policy::<Source, Less>(policy, source)
    }

    pub(in crate::detail) fn count_selected<R: Runtime>(
        policy: &crate::policy::CubePolicy<R>,
        selected_rank: &select::SelectedRankControl,
    ) -> Result<MIndex, Error> {
        mindex_from_usize(select::selected_count(policy, selected_rank)?)
    }

    pub(in crate::detail) fn first_selected<R: Runtime>(
        policy: &crate::policy::CubePolicy<R>,
        selected_rank: select::SelectedRankControl,
    ) -> Result<Option<MIndex>, Error> {
        let control = crate::detail::control::SearchControl::from_flags(
            selected_rank.flag,
            selected_rank.len,
            selected_rank.len,
        );
        Self::first_flag(policy, control)
    }

    pub(in crate::detail) fn first_flag<R: Runtime>(
        policy: &crate::policy::CubePolicy<R>,
        control: crate::detail::control::SearchControl,
    ) -> Result<Option<MIndex>, Error> {
        primitive_search::first_flag(
            policy,
            control.flag,
            control.storage_len,
            control.logical_len,
        )
    }

    pub(in crate::detail) fn first_flag_or<R: Runtime>(
        policy: &crate::policy::CubePolicy<R>,
        control: crate::detail::control::SearchControl,
        fallback: MIndex,
    ) -> Result<MIndex, Error> {
        Ok(Self::first_flag(policy, control)?.unwrap_or(fallback))
    }

    pub(in crate::detail) fn first_unset_flag<R: Runtime>(
        policy: &crate::policy::CubePolicy<R>,
        control: crate::detail::control::SearchControl,
    ) -> Result<Option<MIndex>, Error> {
        primitive_search::first_unset_flag(
            policy,
            control.flag,
            control.storage_len,
            control.logical_len,
        )
    }

    pub(in crate::detail) fn first_unset_flag_or<R: Runtime>(
        policy: &crate::policy::CubePolicy<R>,
        control: crate::detail::control::SearchControl,
        fallback: MIndex,
    ) -> Result<MIndex, Error> {
        Ok(Self::first_unset_flag(policy, control)?.unwrap_or(fallback))
    }
}
