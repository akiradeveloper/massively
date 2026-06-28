use std::any::Any;

use cubecl::prelude::Runtime;

use crate::Error;
use crate::detail::dispatch::{self as sealed, array_from_inner};
use crate::detail::op_adapter::{
    KernelOp, KernelTuple1InnerProductOp, KernelTuple1Op, StencilFlag,
};
use crate::error::ensure_same_len;
use crate::expr::DeviceGpuExpr;
use crate::iter::{MIter, MIterMut, SoA1, SoA2, SoA3};
use crate::op;
use crate::runtime::{DeviceSliceMut, DeviceVec, Executor, Scalar};
use crate::slice::MSlice;
use crate::value::{MItem, MVec};

mod item;
mod iter;

trait IntoMaterializedColumn<R, T>: Sized
where
    R: Runtime,
    T: Scalar + 'static,
{
    fn into_materialized_column(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<crate::detail::device::DeviceColumnView<R, T>, Error>;
}

impl<R, T> IntoMaterializedColumn<R, T> for crate::detail::device::DeviceColumnView<R, T>
where
    R: Runtime,
    T: Scalar + 'static,
{
    fn into_materialized_column(
        self,
        _policy: &crate::detail::CubePolicy<R>,
    ) -> Result<crate::detail::device::DeviceColumnView<R, T>, Error> {
        Ok(self)
    }
}

impl<R, T> IntoMaterializedColumn<R, T> for crate::slice::ConstantRead<R, T>
where
    R: Runtime,
    T: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
{
    fn into_materialized_column(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<crate::detail::device::DeviceColumnView<R, T>, Error> {
        let column = crate::detail::api::device_expr_collect_with_policy(policy, &self)?;
        Ok(crate::detail::device::DeviceColumnView::from_column(
            &column,
        ))
    }
}

impl<R> IntoMaterializedColumn<R, u32> for crate::slice::TabulateRead<R>
where
    R: Runtime,
{
    fn into_materialized_column(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<crate::detail::device::DeviceColumnView<R, u32>, Error> {
        let column = crate::detail::primitives::range::indices_u32(
            policy,
            crate::detail::device::KernelColumn::len(&self),
        )?;
        Ok(crate::detail::device::DeviceColumnView::from_column(
            &column,
        ))
    }
}

impl<R, Source, Op, T> IntoMaterializedColumn<R, T>
    for crate::slice::TransformRead<R, Source, Op, T>
where
    R: Runtime,
    T: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    crate::slice::TransformRead<R, Source, Op, T>: crate::detail::device::KernelColumn<Runtime = R, Item = T>
        + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
    <crate::slice::TransformRead<R, Source, Op, T> as crate::detail::device::KernelColumn>::Expr:
        DeviceGpuExpr<T>,
{
    fn into_materialized_column(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<crate::detail::device::DeviceColumnView<R, T>, Error> {
        let column = crate::detail::api::device_expr_collect_with_policy(policy, &self)?;
        Ok(crate::detail::device::DeviceColumnView::from_column(
            &column,
        ))
    }
}

fn lower_mslice_column<R, S, T>(
    slice: S,
    policy: &crate::detail::CubePolicy<R>,
) -> Result<crate::detail::device::DeviceColumnView<R, T>, Error>
where
    R: Runtime,
    S: MSlice<R, Item = T>,
    T: Scalar + 'static,
    S::Read: IntoMaterializedColumn<R, T>,
{
    slice.into_read(policy)?.into_materialized_column(policy)
}
