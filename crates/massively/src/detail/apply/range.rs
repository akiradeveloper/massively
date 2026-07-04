use super::*;

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
