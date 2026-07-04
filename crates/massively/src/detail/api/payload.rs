use super::*;

pub(in crate::detail) struct SelectedPayloadApply<'a> {
    control: &'a select::SelectedRankControl,
    count: usize,
}

impl<'a> SelectedPayloadApply<'a> {
    pub(in crate::detail) fn new(control: &'a select::SelectedRankControl, count: usize) -> Self {
        Self { control, count }
    }

    pub(in crate::detail) fn apply_expr<ExprSource>(
        &self,
        policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
        expr: &ExprSource,
    ) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
    where
        ExprSource: KernelColumn + KernelColumnAt<S0>,
        ExprSource::Runtime: Runtime,
        ExprSource::Item: CubePrimitive + CubeElement,
        ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    {
        expr::selection::device_expr_compact_with_selection_with_policy(
            policy,
            expr,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_expr2<Left, Right>(
        &self,
        policy: &crate::policy::CubePolicy<Left::Runtime>,
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
        Left::Runtime: Runtime,
        Left::Item: CubePrimitive + CubeElement,
        Right::Item: CubePrimitive + CubeElement,
        Left::Expr: DeviceGpuExpr<Left::Item>,
        Right::Expr: DeviceGpuExpr<Right::Item>,
    {
        expr::selection::device_expr_apply_selected2_with_policy(
            policy,
            left,
            right,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_expr3<First, Second, Third>(
        &self,
        policy: &crate::policy::CubePolicy<First::Runtime>,
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
        First::Runtime: Runtime,
        First::Item: CubePrimitive + CubeElement,
        Second::Item: CubePrimitive + CubeElement,
        Third::Item: CubePrimitive + CubeElement,
        First::Expr: DeviceGpuExpr<First::Item>,
        Second::Expr: DeviceGpuExpr<Second::Item>,
        Third::Expr: DeviceGpuExpr<Third::Item>,
    {
        expr::selection::device_expr_apply_selected3_with_policy(
            policy,
            first,
            second,
            third,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_expr4<A, B, C, D>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
    ) -> Result<
        (
            DeviceVec<A::Runtime, A::Item>,
            DeviceVec<A::Runtime, B::Item>,
            DeviceVec<A::Runtime, C::Item>,
            DeviceVec<A::Runtime, D::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
    {
        expr::selection::device_expr_apply_selected4_with_policy(
            policy,
            a,
            b,
            c,
            d,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_expr5<A, B, C, D, E>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
    ) -> Result<
        (
            DeviceVec<A::Runtime, A::Item>,
            DeviceVec<A::Runtime, B::Item>,
            DeviceVec<A::Runtime, C::Item>,
            DeviceVec<A::Runtime, D::Item>,
            DeviceVec<A::Runtime, E::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        E::Expr: DeviceGpuExpr<E::Item>,
    {
        expr::selection::device_expr_apply_selected5_with_policy(
            policy,
            a,
            b,
            c,
            d,
            e,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_expr6<A, B, C, D, E, F>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
        f: &F,
    ) -> Result<
        (
            DeviceVec<A::Runtime, A::Item>,
            DeviceVec<A::Runtime, B::Item>,
            DeviceVec<A::Runtime, C::Item>,
            DeviceVec<A::Runtime, D::Item>,
            DeviceVec<A::Runtime, E::Item>,
            DeviceVec<A::Runtime, F::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        F::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        E::Expr: DeviceGpuExpr<E::Item>,
        F::Expr: DeviceGpuExpr<F::Item>,
    {
        expr::selection::device_expr_apply_selected6_with_policy(
            policy,
            a,
            b,
            c,
            d,
            e,
            f,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_expr7<A, B, C, D, E, F, G>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
        f: &F,
        g: &G,
    ) -> Result<
        (
            DeviceVec<A::Runtime, A::Item>,
            DeviceVec<A::Runtime, B::Item>,
            DeviceVec<A::Runtime, C::Item>,
            DeviceVec<A::Runtime, D::Item>,
            DeviceVec<A::Runtime, E::Item>,
            DeviceVec<A::Runtime, F::Item>,
            DeviceVec<A::Runtime, G::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        G: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        F::Item: CubePrimitive + CubeElement,
        G::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        E::Expr: DeviceGpuExpr<E::Item>,
        F::Expr: DeviceGpuExpr<F::Item>,
        G::Expr: DeviceGpuExpr<G::Item>,
    {
        expr::selection::device_expr_apply_selected7_with_policy(
            policy,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_value<R, T>(
        &self,
        policy: &crate::detail::CubePolicy<R>,
        value_handle: cubecl::server::Handle,
    ) -> Result<DeviceVec<R, T>, Error>
    where
        R: Runtime,
        T: CubePrimitive + CubeElement,
    {
        select::compact_value_with_count(policy, self.control, value_handle, self.count)
    }

    #[allow(dead_code)]
    pub(in crate::detail) fn apply_value2<R, Left, Right>(
        &self,
        policy: &crate::detail::CubePolicy<R>,
        left_handle: cubecl::server::Handle,
        right_handle: cubecl::server::Handle,
    ) -> Result<(DeviceVec<R, Left>, DeviceVec<R, Right>), Error>
    where
        R: Runtime,
        Left: CubePrimitive + CubeElement,
        Right: CubePrimitive + CubeElement,
    {
        Ok((
            self.apply_value(policy, left_handle)?,
            self.apply_value(policy, right_handle)?,
        ))
    }

    #[allow(dead_code)]
    pub(in crate::detail) fn apply_value3<R, First, Second, Third>(
        &self,
        policy: &crate::detail::CubePolicy<R>,
        first_handle: cubecl::server::Handle,
        second_handle: cubecl::server::Handle,
        third_handle: cubecl::server::Handle,
    ) -> Result<
        (
            DeviceVec<R, First>,
            DeviceVec<R, Second>,
            DeviceVec<R, Third>,
        ),
        Error,
    >
    where
        R: Runtime,
        First: CubePrimitive + CubeElement,
        Second: CubePrimitive + CubeElement,
        Third: CubePrimitive + CubeElement,
    {
        Ok((
            self.apply_value(policy, first_handle)?,
            self.apply_value(policy, second_handle)?,
            self.apply_value(policy, third_handle)?,
        ))
    }
}

pub(in crate::detail) struct SplitPayloadApply<'a> {
    control: &'a select::SplitRankControl,
    selected_count: usize,
    rejected_count: usize,
}

impl<'a> SplitPayloadApply<'a> {
    pub(in crate::detail) fn new(
        control: &'a select::SplitRankControl,
        selected_count: usize,
        rejected_count: usize,
    ) -> Self {
        Self {
            control,
            selected_count,
            rejected_count,
        }
    }

    pub(in crate::detail) fn apply_expr<ExprSource>(
        &self,
        policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
        expr: &ExprSource,
    ) -> Result<
        (
            DeviceVec<ExprSource::Runtime, ExprSource::Item>,
            DeviceVec<ExprSource::Runtime, ExprSource::Item>,
        ),
        Error,
    >
    where
        ExprSource: KernelColumn + KernelColumnAt<S0>,
        ExprSource::Runtime: Runtime,
        ExprSource::Item: CubePrimitive + CubeElement,
        ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    {
        expr::selection::device_expr_compact_split_with_split_with_policy(
            policy,
            expr,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }

    pub(in crate::detail) fn apply_expr2<Left, Right>(
        &self,
        policy: &crate::policy::CubePolicy<Left::Runtime>,
        left: &Left,
        right: &Right,
    ) -> Result<
        (
            (
                DeviceVec<Left::Runtime, Left::Item>,
                DeviceVec<Left::Runtime, Right::Item>,
            ),
            (
                DeviceVec<Left::Runtime, Left::Item>,
                DeviceVec<Left::Runtime, Right::Item>,
            ),
        ),
        Error,
    >
    where
        Left: KernelColumn + KernelColumnAt<S0>,
        Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
        Left::Runtime: Runtime,
        Left::Item: CubePrimitive + CubeElement,
        Right::Item: CubePrimitive + CubeElement,
        Left::Expr: DeviceGpuExpr<Left::Item>,
        Right::Expr: DeviceGpuExpr<Right::Item>,
    {
        expr::selection::device_expr_apply_split2_with_policy(
            policy,
            left,
            right,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }

    pub(in crate::detail) fn apply_expr3<First, Second, Third>(
        &self,
        policy: &crate::policy::CubePolicy<First::Runtime>,
        first: &First,
        second: &Second,
        third: &Third,
    ) -> Result<
        (
            (
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            ),
            (
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            ),
        ),
        Error,
    >
    where
        First: KernelColumn + KernelColumnAt<S0>,
        Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
        Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
        First::Runtime: Runtime,
        First::Item: CubePrimitive + CubeElement,
        Second::Item: CubePrimitive + CubeElement,
        Third::Item: CubePrimitive + CubeElement,
        First::Expr: DeviceGpuExpr<First::Item>,
        Second::Expr: DeviceGpuExpr<Second::Item>,
        Third::Expr: DeviceGpuExpr<Third::Item>,
    {
        expr::selection::device_expr_apply_split3_with_policy(
            policy,
            first,
            second,
            third,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }

    pub(in crate::detail) fn apply_expr4<A, B, C, D>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
    ) -> Result<
        (
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
            ),
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
            ),
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
    {
        expr::selection::device_expr_apply_split4_with_policy(
            policy,
            a,
            b,
            c,
            d,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }

    pub(in crate::detail) fn apply_expr5<A, B, C, D, E>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
    ) -> Result<
        (
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
                DeviceVec<A::Runtime, E::Item>,
            ),
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
                DeviceVec<A::Runtime, E::Item>,
            ),
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        E::Expr: DeviceGpuExpr<E::Item>,
    {
        expr::selection::device_expr_apply_split5_with_policy(
            policy,
            a,
            b,
            c,
            d,
            e,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }

    pub(in crate::detail) fn apply_expr6<A, B, C, D, E, F>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
        f: &F,
    ) -> Result<
        (
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
                DeviceVec<A::Runtime, E::Item>,
                DeviceVec<A::Runtime, F::Item>,
            ),
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
                DeviceVec<A::Runtime, E::Item>,
                DeviceVec<A::Runtime, F::Item>,
            ),
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        F::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        E::Expr: DeviceGpuExpr<E::Item>,
        F::Expr: DeviceGpuExpr<F::Item>,
    {
        expr::selection::device_expr_apply_split6_with_policy(
            policy,
            a,
            b,
            c,
            d,
            e,
            f,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }

    pub(in crate::detail) fn apply_expr7<A, B, C, D, E, F, G>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
        f: &F,
        g: &G,
    ) -> Result<
        (
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
                DeviceVec<A::Runtime, E::Item>,
                DeviceVec<A::Runtime, F::Item>,
                DeviceVec<A::Runtime, G::Item>,
            ),
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
                DeviceVec<A::Runtime, E::Item>,
                DeviceVec<A::Runtime, F::Item>,
                DeviceVec<A::Runtime, G::Item>,
            ),
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        G: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        F::Item: CubePrimitive + CubeElement,
        G::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        E::Expr: DeviceGpuExpr<E::Item>,
        F::Expr: DeviceGpuExpr<F::Item>,
        G::Expr: DeviceGpuExpr<G::Item>,
    {
        expr::selection::device_expr_apply_split7_with_policy(
            policy,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }
}

pub(in crate::detail) struct PermutationPayloadApply<'a, R: Runtime> {
    control: &'a crate::detail::control::PermutationControl<R>,
}

impl<'a, R: Runtime> PermutationPayloadApply<'a, R> {
    pub(in crate::detail) fn new(
        control: &'a crate::detail::control::PermutationControl<R>,
    ) -> Self {
        Self { control }
    }

    pub(in crate::detail) fn apply_expr<ExprSource>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        expr: &ExprSource,
    ) -> Result<DeviceVec<R, ExprSource::Item>, Error>
    where
        ExprSource: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ExprSource::Item: CubePrimitive + CubeElement,
        ExprSource::Expr: GpuExpr<ExprSource::Item>,
    {
        ensure_same_len(expr.len(), self.control.len)?;
        let indices = self.control.indices(policy);
        expr::device_expr_gather_with_policy(policy, expr, &indices)
    }

    pub(in crate::detail) fn apply_expr2<Left, Right>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        left: &Left,
        right: &Right,
    ) -> Result<(DeviceVec<R, Left::Item>, DeviceVec<R, Right::Item>), Error>
    where
        Left: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        Right: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        Left::Item: CubePrimitive + CubeElement,
        Right::Item: CubePrimitive + CubeElement,
        Left::Expr: GpuExpr<Left::Item>,
        Right::Expr: GpuExpr<Right::Item>,
    {
        Ok((
            self.apply_expr(policy, left)?,
            self.apply_expr(policy, right)?,
        ))
    }

    pub(in crate::detail) fn apply_expr3<First, Second, Third>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        first: &First,
        second: &Second,
        third: &Third,
    ) -> Result<
        (
            DeviceVec<R, First::Item>,
            DeviceVec<R, Second::Item>,
            DeviceVec<R, Third::Item>,
        ),
        Error,
    >
    where
        First: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        Second: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        Third: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        First::Item: CubePrimitive + CubeElement,
        Second::Item: CubePrimitive + CubeElement,
        Third::Item: CubePrimitive + CubeElement,
        First::Expr: GpuExpr<First::Item>,
        Second::Expr: GpuExpr<Second::Item>,
        Third::Expr: GpuExpr<Third::Item>,
    {
        Ok((
            self.apply_expr(policy, first)?,
            self.apply_expr(policy, second)?,
            self.apply_expr(policy, third)?,
        ))
    }

    pub(in crate::detail) fn apply_expr4<A, B, C, D>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
    ) -> Result<
        (
            DeviceVec<R, A::Item>,
            DeviceVec<R, B::Item>,
            DeviceVec<R, C::Item>,
            DeviceVec<R, D::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        A::Expr: GpuExpr<A::Item>,
        B::Expr: GpuExpr<B::Item>,
        C::Expr: GpuExpr<C::Item>,
        D::Expr: GpuExpr<D::Item>,
    {
        Ok((
            self.apply_expr(policy, a)?,
            self.apply_expr(policy, b)?,
            self.apply_expr(policy, c)?,
            self.apply_expr(policy, d)?,
        ))
    }

    pub(in crate::detail) fn apply_expr5<A, B, C, D, E>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
    ) -> Result<
        (
            DeviceVec<R, A::Item>,
            DeviceVec<R, B::Item>,
            DeviceVec<R, C::Item>,
            DeviceVec<R, D::Item>,
            DeviceVec<R, E::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        A::Expr: GpuExpr<A::Item>,
        B::Expr: GpuExpr<B::Item>,
        C::Expr: GpuExpr<C::Item>,
        D::Expr: GpuExpr<D::Item>,
        E::Expr: GpuExpr<E::Item>,
    {
        Ok((
            self.apply_expr(policy, a)?,
            self.apply_expr(policy, b)?,
            self.apply_expr(policy, c)?,
            self.apply_expr(policy, d)?,
            self.apply_expr(policy, e)?,
        ))
    }

    pub(in crate::detail) fn apply_expr6<A, B, C, D, E, F>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
        f: &F,
    ) -> Result<
        (
            DeviceVec<R, A::Item>,
            DeviceVec<R, B::Item>,
            DeviceVec<R, C::Item>,
            DeviceVec<R, D::Item>,
            DeviceVec<R, E::Item>,
            DeviceVec<R, F::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        F: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        F::Item: CubePrimitive + CubeElement,
        A::Expr: GpuExpr<A::Item>,
        B::Expr: GpuExpr<B::Item>,
        C::Expr: GpuExpr<C::Item>,
        D::Expr: GpuExpr<D::Item>,
        E::Expr: GpuExpr<E::Item>,
        F::Expr: GpuExpr<F::Item>,
    {
        Ok((
            self.apply_expr(policy, a)?,
            self.apply_expr(policy, b)?,
            self.apply_expr(policy, c)?,
            self.apply_expr(policy, d)?,
            self.apply_expr(policy, e)?,
            self.apply_expr(policy, f)?,
        ))
    }

    pub(in crate::detail) fn apply_expr7<A, B, C, D, E, F, G>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
        f: &F,
        g: &G,
    ) -> Result<
        (
            DeviceVec<R, A::Item>,
            DeviceVec<R, B::Item>,
            DeviceVec<R, C::Item>,
            DeviceVec<R, D::Item>,
            DeviceVec<R, E::Item>,
            DeviceVec<R, F::Item>,
            DeviceVec<R, G::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        F: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        G: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        F::Item: CubePrimitive + CubeElement,
        G::Item: CubePrimitive + CubeElement,
        A::Expr: GpuExpr<A::Item>,
        B::Expr: GpuExpr<B::Item>,
        C::Expr: GpuExpr<C::Item>,
        D::Expr: GpuExpr<D::Item>,
        E::Expr: GpuExpr<E::Item>,
        F::Expr: GpuExpr<F::Item>,
        G::Expr: GpuExpr<G::Item>,
    {
        Ok((
            self.apply_expr(policy, a)?,
            self.apply_expr(policy, b)?,
            self.apply_expr(policy, c)?,
            self.apply_expr(policy, d)?,
            self.apply_expr(policy, e)?,
            self.apply_expr(policy, f)?,
            self.apply_expr(policy, g)?,
        ))
    }
}

pub(in crate::detail) struct MergePayloadApply<'a> {
    control: &'a crate::detail::control::MergeByKeyControl,
}

impl<'a> MergePayloadApply<'a> {
    pub(in crate::detail) fn new(control: &'a crate::detail::control::MergeByKeyControl) -> Self {
        Self { control }
    }

    pub(in crate::detail) fn apply_expr<LeftValue, RightValue>(
        &self,
        policy: &crate::policy::CubePolicy<LeftValue::Runtime>,
        left: &LeftValue,
        right: &RightValue,
    ) -> Result<DeviceVec<LeftValue::Runtime, LeftValue::Item>, Error>
    where
        LeftValue: KernelColumn + KernelColumnAt<S0>,
        RightValue:
            KernelColumn<Runtime = LeftValue::Runtime, Item = LeftValue::Item> + KernelColumnAt<S0>,
        LeftValue::Item: CubePrimitive + CubeElement,
        LeftValue::Expr: DeviceGpuExpr<LeftValue::Item>,
        RightValue::Expr: DeviceGpuExpr<RightValue::Item>,
    {
        device_expr_merge_by_key_values_with_control_with_policy(policy, left, right, self.control)
    }

    pub(in crate::detail) fn apply_expr2<LeftA, LeftB, RightA, RightB>(
        &self,
        policy: &crate::policy::CubePolicy<LeftA::Runtime>,
        left_a: &LeftA,
        left_b: &LeftB,
        right_a: &RightA,
        right_b: &RightB,
    ) -> Result<
        (
            DeviceVec<LeftA::Runtime, LeftA::Item>,
            DeviceVec<LeftA::Runtime, LeftB::Item>,
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
    {
        Ok((
            self.apply_expr(policy, left_a, right_a)?,
            self.apply_expr(policy, left_b, right_b)?,
        ))
    }

    pub(in crate::detail) fn apply_expr3<LeftA, LeftB, LeftC, RightA, RightB, RightC>(
        &self,
        policy: &crate::policy::CubePolicy<LeftA::Runtime>,
        left_a: &LeftA,
        left_b: &LeftB,
        left_c: &LeftC,
        right_a: &RightA,
        right_b: &RightB,
        right_c: &RightC,
    ) -> Result<
        (
            DeviceVec<LeftA::Runtime, LeftA::Item>,
            DeviceVec<LeftA::Runtime, LeftB::Item>,
            DeviceVec<LeftA::Runtime, LeftC::Item>,
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
    {
        Ok((
            self.apply_expr(policy, left_a, right_a)?,
            self.apply_expr(policy, left_b, right_b)?,
            self.apply_expr(policy, left_c, right_c)?,
        ))
    }

    pub(in crate::detail) fn apply_expr4<LA, LB, LC, LD, RA, RB, RC, RD>(
        &self,
        policy: &crate::policy::CubePolicy<LA::Runtime>,
        left: (&LA, &LB, &LC, &LD),
        right: (&RA, &RB, &RC, &RD),
    ) -> Result<
        (
            DeviceVec<LA::Runtime, LA::Item>,
            DeviceVec<LA::Runtime, LB::Item>,
            DeviceVec<LA::Runtime, LC::Item>,
            DeviceVec<LA::Runtime, LD::Item>,
        ),
        Error,
    >
    where
        LA: KernelColumn + KernelColumnAt<S0>,
        LB: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LC: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LD: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        RA: KernelColumn<Runtime = LA::Runtime, Item = LA::Item> + KernelColumnAt<S0>,
        RB: KernelColumn<Runtime = LA::Runtime, Item = LB::Item> + KernelColumnAt<S0>,
        RC: KernelColumn<Runtime = LA::Runtime, Item = LC::Item> + KernelColumnAt<S0>,
        RD: KernelColumn<Runtime = LA::Runtime, Item = LD::Item> + KernelColumnAt<S0>,
        LA::Item: CubePrimitive + CubeElement,
        LB::Item: CubePrimitive + CubeElement,
        LC::Item: CubePrimitive + CubeElement,
        LD::Item: CubePrimitive + CubeElement,
        LA::Expr: DeviceGpuExpr<LA::Item>,
        LB::Expr: DeviceGpuExpr<LB::Item>,
        LC::Expr: DeviceGpuExpr<LC::Item>,
        LD::Expr: DeviceGpuExpr<LD::Item>,
        RA::Expr: DeviceGpuExpr<RA::Item>,
        RB::Expr: DeviceGpuExpr<RB::Item>,
        RC::Expr: DeviceGpuExpr<RC::Item>,
        RD::Expr: DeviceGpuExpr<RD::Item>,
    {
        Ok((
            self.apply_expr(policy, left.0, right.0)?,
            self.apply_expr(policy, left.1, right.1)?,
            self.apply_expr(policy, left.2, right.2)?,
            self.apply_expr(policy, left.3, right.3)?,
        ))
    }

    pub(in crate::detail) fn apply_expr5<LA, LB, LC, LD, LE, RA, RB, RC, RD, RE>(
        &self,
        policy: &crate::policy::CubePolicy<LA::Runtime>,
        left: (&LA, &LB, &LC, &LD, &LE),
        right: (&RA, &RB, &RC, &RD, &RE),
    ) -> Result<
        (
            DeviceVec<LA::Runtime, LA::Item>,
            DeviceVec<LA::Runtime, LB::Item>,
            DeviceVec<LA::Runtime, LC::Item>,
            DeviceVec<LA::Runtime, LD::Item>,
            DeviceVec<LA::Runtime, LE::Item>,
        ),
        Error,
    >
    where
        LA: KernelColumn + KernelColumnAt<S0>,
        LB: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LC: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LD: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LE: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        RA: KernelColumn<Runtime = LA::Runtime, Item = LA::Item> + KernelColumnAt<S0>,
        RB: KernelColumn<Runtime = LA::Runtime, Item = LB::Item> + KernelColumnAt<S0>,
        RC: KernelColumn<Runtime = LA::Runtime, Item = LC::Item> + KernelColumnAt<S0>,
        RD: KernelColumn<Runtime = LA::Runtime, Item = LD::Item> + KernelColumnAt<S0>,
        RE: KernelColumn<Runtime = LA::Runtime, Item = LE::Item> + KernelColumnAt<S0>,
        LA::Item: CubePrimitive + CubeElement,
        LB::Item: CubePrimitive + CubeElement,
        LC::Item: CubePrimitive + CubeElement,
        LD::Item: CubePrimitive + CubeElement,
        LE::Item: CubePrimitive + CubeElement,
        LA::Expr: DeviceGpuExpr<LA::Item>,
        LB::Expr: DeviceGpuExpr<LB::Item>,
        LC::Expr: DeviceGpuExpr<LC::Item>,
        LD::Expr: DeviceGpuExpr<LD::Item>,
        LE::Expr: DeviceGpuExpr<LE::Item>,
        RA::Expr: DeviceGpuExpr<RA::Item>,
        RB::Expr: DeviceGpuExpr<RB::Item>,
        RC::Expr: DeviceGpuExpr<RC::Item>,
        RD::Expr: DeviceGpuExpr<RD::Item>,
        RE::Expr: DeviceGpuExpr<RE::Item>,
    {
        let (a, b, c, d) = self.apply_expr4(
            policy,
            (left.0, left.1, left.2, left.3),
            (right.0, right.1, right.2, right.3),
        )?;
        Ok((a, b, c, d, self.apply_expr(policy, left.4, right.4)?))
    }

    pub(in crate::detail) fn apply_expr6<LA, LB, LC, LD, LE, LF, RA, RB, RC, RD, RE, RF>(
        &self,
        policy: &crate::policy::CubePolicy<LA::Runtime>,
        left: (&LA, &LB, &LC, &LD, &LE, &LF),
        right: (&RA, &RB, &RC, &RD, &RE, &RF),
    ) -> Result<
        (
            DeviceVec<LA::Runtime, LA::Item>,
            DeviceVec<LA::Runtime, LB::Item>,
            DeviceVec<LA::Runtime, LC::Item>,
            DeviceVec<LA::Runtime, LD::Item>,
            DeviceVec<LA::Runtime, LE::Item>,
            DeviceVec<LA::Runtime, LF::Item>,
        ),
        Error,
    >
    where
        LA: KernelColumn + KernelColumnAt<S0>,
        LB: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LC: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LD: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LE: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LF: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        RA: KernelColumn<Runtime = LA::Runtime, Item = LA::Item> + KernelColumnAt<S0>,
        RB: KernelColumn<Runtime = LA::Runtime, Item = LB::Item> + KernelColumnAt<S0>,
        RC: KernelColumn<Runtime = LA::Runtime, Item = LC::Item> + KernelColumnAt<S0>,
        RD: KernelColumn<Runtime = LA::Runtime, Item = LD::Item> + KernelColumnAt<S0>,
        RE: KernelColumn<Runtime = LA::Runtime, Item = LE::Item> + KernelColumnAt<S0>,
        RF: KernelColumn<Runtime = LA::Runtime, Item = LF::Item> + KernelColumnAt<S0>,
        LA::Item: CubePrimitive + CubeElement,
        LB::Item: CubePrimitive + CubeElement,
        LC::Item: CubePrimitive + CubeElement,
        LD::Item: CubePrimitive + CubeElement,
        LE::Item: CubePrimitive + CubeElement,
        LF::Item: CubePrimitive + CubeElement,
        LA::Expr: DeviceGpuExpr<LA::Item>,
        LB::Expr: DeviceGpuExpr<LB::Item>,
        LC::Expr: DeviceGpuExpr<LC::Item>,
        LD::Expr: DeviceGpuExpr<LD::Item>,
        LE::Expr: DeviceGpuExpr<LE::Item>,
        LF::Expr: DeviceGpuExpr<LF::Item>,
        RA::Expr: DeviceGpuExpr<RA::Item>,
        RB::Expr: DeviceGpuExpr<RB::Item>,
        RC::Expr: DeviceGpuExpr<RC::Item>,
        RD::Expr: DeviceGpuExpr<RD::Item>,
        RE::Expr: DeviceGpuExpr<RE::Item>,
        RF::Expr: DeviceGpuExpr<RF::Item>,
    {
        let (a, b, c, d, e) = self.apply_expr5(
            policy,
            (left.0, left.1, left.2, left.3, left.4),
            (right.0, right.1, right.2, right.3, right.4),
        )?;
        Ok((a, b, c, d, e, self.apply_expr(policy, left.5, right.5)?))
    }

    pub(in crate::detail) fn apply_expr7<LA, LB, LC, LD, LE, LF, LG, RA, RB, RC, RD, RE, RF, RG>(
        &self,
        policy: &crate::policy::CubePolicy<LA::Runtime>,
        left: (&LA, &LB, &LC, &LD, &LE, &LF, &LG),
        right: (&RA, &RB, &RC, &RD, &RE, &RF, &RG),
    ) -> Result<
        (
            DeviceVec<LA::Runtime, LA::Item>,
            DeviceVec<LA::Runtime, LB::Item>,
            DeviceVec<LA::Runtime, LC::Item>,
            DeviceVec<LA::Runtime, LD::Item>,
            DeviceVec<LA::Runtime, LE::Item>,
            DeviceVec<LA::Runtime, LF::Item>,
            DeviceVec<LA::Runtime, LG::Item>,
        ),
        Error,
    >
    where
        LA: KernelColumn + KernelColumnAt<S0>,
        LB: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LC: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LD: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LE: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LF: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LG: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        RA: KernelColumn<Runtime = LA::Runtime, Item = LA::Item> + KernelColumnAt<S0>,
        RB: KernelColumn<Runtime = LA::Runtime, Item = LB::Item> + KernelColumnAt<S0>,
        RC: KernelColumn<Runtime = LA::Runtime, Item = LC::Item> + KernelColumnAt<S0>,
        RD: KernelColumn<Runtime = LA::Runtime, Item = LD::Item> + KernelColumnAt<S0>,
        RE: KernelColumn<Runtime = LA::Runtime, Item = LE::Item> + KernelColumnAt<S0>,
        RF: KernelColumn<Runtime = LA::Runtime, Item = LF::Item> + KernelColumnAt<S0>,
        RG: KernelColumn<Runtime = LA::Runtime, Item = LG::Item> + KernelColumnAt<S0>,
        LA::Item: CubePrimitive + CubeElement,
        LB::Item: CubePrimitive + CubeElement,
        LC::Item: CubePrimitive + CubeElement,
        LD::Item: CubePrimitive + CubeElement,
        LE::Item: CubePrimitive + CubeElement,
        LF::Item: CubePrimitive + CubeElement,
        LG::Item: CubePrimitive + CubeElement,
        LA::Expr: DeviceGpuExpr<LA::Item>,
        LB::Expr: DeviceGpuExpr<LB::Item>,
        LC::Expr: DeviceGpuExpr<LC::Item>,
        LD::Expr: DeviceGpuExpr<LD::Item>,
        LE::Expr: DeviceGpuExpr<LE::Item>,
        LF::Expr: DeviceGpuExpr<LF::Item>,
        LG::Expr: DeviceGpuExpr<LG::Item>,
        RA::Expr: DeviceGpuExpr<RA::Item>,
        RB::Expr: DeviceGpuExpr<RB::Item>,
        RC::Expr: DeviceGpuExpr<RC::Item>,
        RD::Expr: DeviceGpuExpr<RD::Item>,
        RE::Expr: DeviceGpuExpr<RE::Item>,
        RF::Expr: DeviceGpuExpr<RF::Item>,
        RG::Expr: DeviceGpuExpr<RG::Item>,
    {
        let (a, b, c, d, e, f) = self.apply_expr6(
            policy,
            (left.0, left.1, left.2, left.3, left.4, left.5),
            (right.0, right.1, right.2, right.3, right.4, right.5),
        )?;
        Ok((a, b, c, d, e, f, self.apply_expr(policy, left.6, right.6)?))
    }
}

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

pub(in crate::detail) struct IndexedWriteApply<'a, R: Runtime> {
    control: &'a crate::detail::control::PermutationControl<R>,
}

impl<'a, R: Runtime> IndexedWriteApply<'a, R> {
    pub(in crate::detail) fn new(
        control: &'a crate::detail::control::PermutationControl<R>,
    ) -> Self {
        Self { control }
    }

    pub(in crate::detail) fn scatter_expr_into<ValueSource>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        values: &ValueSource,
        output: &DeviceColumnMutView<R, ValueSource::Item>,
    ) -> Result<(), Error>
    where
        ValueSource: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueSource::Item: CubePrimitive + CubeElement,
        ValueSource::Expr: GpuExpr<ValueSource::Item>,
    {
        ensure_same_len(values.len(), self.control.len)?;
        let indices = self.control.indices(policy);
        expr::device_expr_scatter_into_with_policy(policy, values, &indices, output)
    }

    pub(in crate::detail) fn scatter_expr_where_into<ValueSource>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        values: &ValueSource,
        mask: &select::MaskControl,
        output: &DeviceColumnMutView<R, ValueSource::Item>,
    ) -> Result<(), Error>
    where
        ValueSource: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueSource::Item: CubePrimitive + CubeElement,
        ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    {
        ensure_same_len(values.len(), self.control.len)?;
        ensure_same_len(mask.len, self.control.len)?;
        let indices = self.control.indices(policy);
        expr::device_expr_scatter_where_into_with_control(policy, values, &indices, mask, output)
    }
}

pub(in crate::detail) struct MaterializePayloadApply;

impl MaterializePayloadApply {
    pub(in crate::detail) fn collect_expr<ExprSource>(
        policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
        expr: &ExprSource,
    ) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
    where
        ExprSource: KernelColumn + KernelColumnAt<S0>,
        ExprSource::Runtime: Runtime,
        ExprSource::Item: CubePrimitive + CubeElement,
        ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    {
        expr::device_expr_collect_with_policy(policy, expr)
    }
}

pub(in crate::detail) struct MaterializeWriteApply<'a, R: Runtime, T: CubePrimitive + CubeElement> {
    output: &'a DeviceColumnMutView<R, T>,
}

impl<'a, R, T> MaterializeWriteApply<'a, R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    pub(in crate::detail) fn new(output: &'a DeviceColumnMutView<R, T>) -> Self {
        Self { output }
    }

    pub(in crate::detail) fn collect_expr<ExprSource>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        expr: &ExprSource,
    ) -> Result<(), Error>
    where
        ExprSource: KernelColumn<Runtime = R, Item = T> + KernelColumnAt<S0>,
        ExprSource::Expr: DeviceGpuExpr<T>,
    {
        expr::device_expr_collect_into_with_policy(policy, expr, self.output)
    }

    pub(in crate::detail) fn copy_where_expr<ExprSource, Stencil, Pred>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        expr: &ExprSource,
        stencil: &Stencil,
        pred: Pred,
    ) -> Result<(), Error>
    where
        ExprSource: KernelColumn<Runtime = R, Item = T> + KernelColumnAt<S0>,
        ExprSource::Expr: DeviceGpuExpr<T>,
        Stencil: SelectionStencil<Pred, Runtime = R>,
    {
        expr::device_expr_copy_where_into_with_policy(policy, expr, stencil, self.output, pred)
    }
}

pub(in crate::detail) struct MaskWriteApply<'a, R: Runtime, T: CubePrimitive + CubeElement> {
    mask: &'a select::MaskControl,
    output: &'a DeviceColumnMutView<R, T>,
}

impl<'a, R, T> MaskWriteApply<'a, R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    pub(in crate::detail) fn new(
        mask: &'a select::MaskControl,
        output: &'a DeviceColumnMutView<R, T>,
    ) -> Self {
        Self { mask, output }
    }

    pub(in crate::detail) fn replace_value(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        replacement: T,
    ) -> Result<(), Error> {
        expr::replace_where_into_with_control(policy, replacement, self.mask, self.output)
    }
}

pub(in crate::detail) struct FillWriteApply<'a, R: Runtime, T: CubePrimitive + CubeElement> {
    output: &'a DeviceColumnMutView<R, T>,
}

impl<'a, R, T> FillWriteApply<'a, R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    pub(in crate::detail) fn new(output: &'a DeviceColumnMutView<R, T>) -> Self {
        Self { output }
    }

    pub(in crate::detail) fn fill_value(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        value: T,
    ) -> Result<(), Error> {
        crate::detail::primitives::fill_slice_with_policy(policy, value, self.output)
    }
}

pub(in crate::detail) struct MaskedIndexedExprApply;

impl MaskedIndexedExprApply {
    pub(in crate::detail) fn gather_where_expr_into<InputSource, IndexSource>(
        policy: &crate::policy::CubePolicy<InputSource::Runtime>,
        input: &InputSource,
        indices: &IndexSource,
        mask: &select::MaskControl,
        output: &DeviceColumnMutView<InputSource::Runtime, InputSource::Item>,
    ) -> Result<(), Error>
    where
        InputSource: KernelColumn + KernelColumnAt<S0>,
        InputSource::Runtime: Runtime,
        IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
        InputSource::Item: CubePrimitive + CubeElement,
        InputSource::Expr: GpuExpr<InputSource::Item>,
        IndexSource::Expr: DeviceGpuExpr<u32>,
    {
        expr::device_expr_gather_where_into_with_control(policy, input, indices, mask, output)
    }

    pub(in crate::detail) fn scatter_where_expr_into<ValueSource, IndexSource>(
        policy: &crate::policy::CubePolicy<ValueSource::Runtime>,
        values: &ValueSource,
        indices: &IndexSource,
        mask: &select::MaskControl,
        output: &DeviceColumnMutView<ValueSource::Runtime, ValueSource::Item>,
    ) -> Result<(), Error>
    where
        ValueSource: KernelColumn + KernelColumnAt<S0>,
        ValueSource::Runtime: Runtime,
        IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
        ValueSource::Item: CubePrimitive + CubeElement,
        ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
        IndexSource::Expr: DeviceGpuExpr<u32>,
    {
        expr::device_expr_scatter_where_into_with_control(policy, values, indices, mask, output)
    }
}

pub(in crate::detail) struct ConcatPayloadApply;

impl ConcatPayloadApply {
    pub(in crate::detail) fn apply_values<R, T>(
        policy: &crate::policy::CubePolicy<R>,
        left: &DeviceVec<R, T>,
        right: &DeviceVec<R, T>,
    ) -> Result<DeviceVec<R, T>, Error>
    where
        R: Runtime,
        T: CubePrimitive + CubeElement,
    {
        crate::detail::primitives::range::concat_device_with_policy(policy, left, right)
    }
}

pub(in crate::detail) struct IndexedExprApply;

impl IndexedExprApply {
    pub(in crate::detail) fn gather_expr<InputSource, IndexSource>(
        policy: &crate::policy::CubePolicy<InputSource::Runtime>,
        input: &InputSource,
        indices: &IndexSource,
    ) -> Result<DeviceVec<InputSource::Runtime, InputSource::Item>, Error>
    where
        InputSource: KernelColumn + KernelColumnAt<S0>,
        InputSource::Runtime: Runtime,
        IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
        InputSource::Item: CubePrimitive + CubeElement,
        InputSource::Expr: GpuExpr<InputSource::Item>,
        IndexSource::Expr: GpuExpr<u32>,
    {
        expr::device_expr_gather_with_policy(policy, input, indices)
    }

    pub(in crate::detail) fn gather_expr_into<InputSource, IndexSource>(
        policy: &crate::policy::CubePolicy<InputSource::Runtime>,
        input: &InputSource,
        indices: &IndexSource,
        output: &DeviceColumnMutView<InputSource::Runtime, InputSource::Item>,
    ) -> Result<(), Error>
    where
        InputSource: KernelColumn + KernelColumnAt<S0>,
        InputSource::Runtime: Runtime,
        IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
        InputSource::Item: CubePrimitive + CubeElement,
        InputSource::Expr: GpuExpr<InputSource::Item>,
        IndexSource::Expr: GpuExpr<u32>,
    {
        expr::device_expr_gather_into_with_policy(policy, input, indices, output)
    }

    pub(in crate::detail) fn scatter_expr_into<ValueSource, IndexSource>(
        policy: &crate::policy::CubePolicy<ValueSource::Runtime>,
        values: &ValueSource,
        indices: &IndexSource,
        output: &DeviceColumnMutView<ValueSource::Runtime, ValueSource::Item>,
    ) -> Result<(), Error>
    where
        ValueSource: KernelColumn + KernelColumnAt<S0>,
        ValueSource::Runtime: Runtime,
        IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
        ValueSource::Item: CubePrimitive + CubeElement,
        ValueSource::Expr: GpuExpr<ValueSource::Item>,
        IndexSource::Expr: GpuExpr<u32>,
    {
        expr::device_expr_scatter_into_with_policy(policy, values, indices, output)
    }
}

pub(in crate::detail) struct RangePayloadApply<'a> {
    control: &'a crate::detail::control::RangeControl,
}

impl<'a> RangePayloadApply<'a> {
    pub(in crate::detail) fn new(control: &'a crate::detail::control::RangeControl) -> Self {
        Self { control }
    }

    pub(in crate::detail) fn apply_expr<ExprSource>(
        &self,
        policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
        expr: &ExprSource,
    ) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
    where
        ExprSource: KernelColumn + KernelColumnAt<S0>,
        ExprSource::Runtime: Runtime,
        ExprSource::Item: CubePrimitive + CubeElement,
        ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    {
        ensure_same_len(expr.len(), self.control.len)?;
        match self.control.mapping {
            crate::detail::control::RangeMapping::Reverse => {
                expr::device_expr_reverse_collect(policy, expr)
            }
        }
    }

    pub(in crate::detail) fn apply_expr2<Left, Right>(
        &self,
        policy: &crate::policy::CubePolicy<Left::Runtime>,
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
        Left::Runtime: Runtime,
        Left::Item: CubePrimitive + CubeElement,
        Right::Item: CubePrimitive + CubeElement,
        Left::Expr: DeviceGpuExpr<Left::Item>,
        Right::Expr: DeviceGpuExpr<Right::Item>,
    {
        Ok((
            self.apply_expr(policy, left)?,
            self.apply_expr(policy, right)?,
        ))
    }

    pub(in crate::detail) fn apply_expr3<First, Second, Third>(
        &self,
        policy: &crate::policy::CubePolicy<First::Runtime>,
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
        First::Runtime: Runtime,
        First::Item: CubePrimitive + CubeElement,
        Second::Item: CubePrimitive + CubeElement,
        Third::Item: CubePrimitive + CubeElement,
        First::Expr: DeviceGpuExpr<First::Item>,
        Second::Expr: DeviceGpuExpr<Second::Item>,
        Third::Expr: DeviceGpuExpr<Third::Item>,
    {
        Ok((
            self.apply_expr(policy, first)?,
            self.apply_expr(policy, second)?,
            self.apply_expr(policy, third)?,
        ))
    }
}
