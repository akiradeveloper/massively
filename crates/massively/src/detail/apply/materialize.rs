use super::*;

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
