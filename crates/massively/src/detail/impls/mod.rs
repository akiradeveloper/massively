use std::any::Any;

use cubecl::prelude::*;

use crate::Error;
use crate::detail::dispatch::{self as sealed, array_from_inner};
use crate::detail::op_adapter::{
    KernelOp, KernelTuple1InnerProductOp, KernelTuple1Op, StencilFlag,
};
use crate::error::ensure_same_len;
use crate::expr::DeviceGpuExpr;
use crate::iter::{MIter, MIterMut, SoA1, SoA2, SoA3, SoA4, SoA5, SoA6, SoA7};
use crate::op;
use crate::runtime::{DeviceSliceMut, DeviceVec, Executor, Scalar};
use crate::slice::MSlice;
use crate::value::{MItem, MVec};

mod item;
mod iter;

pub(crate) trait IntoMaterializedColumn<R, T>: Sized
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
    for crate::slice::TransformRead<R, (Source,), Op, T>
where
    R: Runtime,
    T: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    crate::slice::TransformRead<R, (Source,), Op, T>: crate::detail::device::KernelColumn<Runtime = R, Item = T>
        + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
    <crate::slice::TransformRead<R, (Source,), Op, T> as crate::detail::device::KernelColumn>::Expr:
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

impl<R, Left, Right, Op, T> IntoMaterializedColumn<R, T>
    for crate::slice::TransformRead<
        R,
        (
            crate::detail::device::DeviceColumnView<R, Left>,
            crate::detail::device::DeviceColumnView<R, Right>,
        ),
        Op,
        T,
    >
where
    R: Runtime,
    Left: Scalar + cubecl::prelude::CubeElement + 'static,
    Right: Scalar + cubecl::prelude::CubeElement + 'static,
    T: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    Op: op::UnaryOp<R, (Left, Right), Output = (T,)>,
    (T,): crate::detail::TransformSoA2Output<R, Left, Right, KernelOp<R, Op>>,
    <(T,) as crate::detail::MItemStorage<R>>::Storage:
        crate::detail::MaterializeOutput<Runtime = R, Output = (crate::detail::DeviceVec<R, T>,)>,
{
    fn into_materialized_column(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<crate::detail::device::DeviceColumnView<R, T>, Error> {
        ensure_same_len(self.source.0.len, self.source.1.len)?;
        let storage = <(T,) as crate::detail::TransformSoA2Output<
            R,
            Left,
            Right,
            KernelOp<R, Op>,
        >>::run(policy, self.source.0, self.source.1)?;
        let (column,) = crate::detail::MaterializeOutput::materialize_output(storage, policy)?;
        Ok(crate::detail::device::DeviceColumnView::from_column(
            &column,
        ))
    }
}

impl<R, First, Second, Third, Op, T> IntoMaterializedColumn<R, T>
    for crate::slice::TransformRead<
        R,
        (
            crate::detail::device::DeviceColumnView<R, First>,
            crate::detail::device::DeviceColumnView<R, Second>,
            crate::detail::device::DeviceColumnView<R, Third>,
        ),
        Op,
        T,
    >
where
    R: Runtime,
    First: Scalar + cubecl::prelude::CubeElement + 'static,
    Second: Scalar + cubecl::prelude::CubeElement + 'static,
    Third: Scalar + cubecl::prelude::CubeElement + 'static,
    T: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    Op: op::UnaryOp<R, (First, Second, Third), Output = (T,)>,
    (T,): crate::detail::TransformSoA3Output<R, First, Second, Third, KernelOp<R, Op>>,
    <(T,) as crate::detail::MItemStorage<R>>::Storage:
        crate::detail::MaterializeOutput<Runtime = R, Output = (crate::detail::DeviceVec<R, T>,)>,
{
    fn into_materialized_column(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<crate::detail::device::DeviceColumnView<R, T>, Error> {
        ensure_same_len(self.source.0.len, self.source.1.len)?;
        ensure_same_len(self.source.0.len, self.source.2.len)?;
        let storage = <(T,) as crate::detail::TransformSoA3Output<
            R,
            First,
            Second,
            Third,
            KernelOp<R, Op>,
        >>::run(policy, self.source.0, self.source.1, self.source.2)?;
        let (column,) = crate::detail::MaterializeOutput::materialize_output(storage, policy)?;
        Ok(crate::detail::device::DeviceColumnView::from_column(
            &column,
        ))
    }
}

macro_rules! impl_transform_read_materialized_tuple1 {
    (
        $trait_name:ident,
        ($($in_ty:ident : $idx:tt),+)
    ) => {
        impl<R, $($in_ty,)+ Op, T> IntoMaterializedColumn<R, T>
            for crate::slice::TransformRead<
                R,
                ($(crate::detail::device::DeviceColumnView<R, $in_ty>,)+),
                Op,
                T,
            >
        where
            R: Runtime,
            $(
                $in_ty: Scalar
                    + cubecl::prelude::CubePrimitive
                    + cubecl::prelude::CubeElement
                    + 'static,
            )+
            T: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
            Op: op::UnaryOp<R, ($($in_ty,)+), Output = (T,)>,
            (T,): crate::detail::$trait_name<R, $($in_ty,)+ KernelOp<R, Op>>,
            <(T,) as crate::detail::MItemStorage<R>>::Storage:
                crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = (crate::detail::DeviceVec<R, T>,),
                >,
        {
            fn into_materialized_column(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<crate::detail::device::DeviceColumnView<R, T>, Error> {
                $(
                    ensure_same_len(self.source.0.len, self.source.$idx.len)?;
                )+
                let storage = <(T,) as crate::detail::$trait_name<
                    R,
                    $($in_ty,)+
                    KernelOp<R, Op>,
                >>::run(
                    policy,
                    $(self.source.$idx,)+
                )?;
                let (column,) =
                    crate::detail::MaterializeOutput::materialize_output(storage, policy)?;
                Ok(crate::detail::device::DeviceColumnView::from_column(&column))
            }
        }
    };
}

impl_transform_read_materialized_tuple1!(
    TransformSoA4Output,
    (A: 0, B: 1, C: 2, D: 3)
);
impl_transform_read_materialized_tuple1!(
    TransformSoA5Output,
    (A: 0, B: 1, C: 2, D: 3, E: 4)
);
impl_transform_read_materialized_tuple1!(
    TransformSoA6Output,
    (A: 0, B: 1, C: 2, D: 3, E: 4, F: 5)
);

impl<R, A, B, C, D, E, F, G, Op, T> IntoMaterializedColumn<R, T>
    for crate::slice::TransformRead<
        R,
        (
            crate::detail::device::DeviceColumnView<R, A>,
            crate::detail::device::DeviceColumnView<R, B>,
            crate::detail::device::DeviceColumnView<R, C>,
            crate::detail::device::DeviceColumnView<R, D>,
            crate::detail::device::DeviceColumnView<R, E>,
            crate::detail::device::DeviceColumnView<R, F>,
            crate::detail::device::DeviceColumnView<R, G>,
        ),
        Op,
        T,
    >
where
    R: Runtime,
    A: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    B: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    C: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    D: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    E: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    F: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    G: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    T: Scalar + cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    Op: op::UnaryOp<R, (A, B, C, D, E, F, G), Output = (T,)>,
    (T,): crate::detail::TransformSoA7Output<R, A, B, C, D, E, F, G, KernelOp<R, Op>>,
    <(T,) as crate::detail::MItemStorage<R>>::Storage:
        crate::detail::MaterializeOutput<Runtime = R, Output = (crate::detail::DeviceVec<R, T>,)>,
{
    fn into_materialized_column(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<crate::detail::device::DeviceColumnView<R, T>, Error> {
        ensure_same_len(self.source.0.len, self.source.1.len)?;
        ensure_same_len(self.source.0.len, self.source.2.len)?;
        ensure_same_len(self.source.0.len, self.source.3.len)?;
        ensure_same_len(self.source.0.len, self.source.4.len)?;
        ensure_same_len(self.source.0.len, self.source.5.len)?;
        ensure_same_len(self.source.0.len, self.source.6.len)?;
        let storage = <(T,) as crate::detail::TransformSoA7Output<
            R,
            A,
            B,
            C,
            D,
            E,
            F,
            G,
            KernelOp<R, Op>,
        >>::run(
            policy,
            self.source.0,
            self.source.1,
            self.source.2,
            self.source.3,
            self.source.4,
            self.source.5,
            self.source.6,
        )?;
        let (column,) = crate::detail::MaterializeOutput::materialize_output(storage, policy)?;
        Ok(crate::detail::device::DeviceColumnView::from_column(
            &column,
        ))
    }
}

pub(crate) fn lower_mslice_column<R, S, T>(
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

pub(crate) fn end_flags_from_head_flags<R>(
    policy: &crate::detail::CubePolicy<R>,
    head_flags: cubecl::server::Handle,
    len: usize,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
{
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let client = policy.client();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let end_flags = client.empty(len * std::mem::size_of::<u32>());
    let num_blocks = len.div_ceil(crate::detail::primitives::scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        crate::kernels::head_flags_to_end_flags_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(crate::detail::primitives::scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(head_flags.clone(), len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(end_flags.clone(), len),
        );
    }

    Ok(end_flags)
}
