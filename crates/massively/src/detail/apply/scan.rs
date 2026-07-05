use crate::{
    MItem,
    detail::{
        CubePolicy,
        api::{Tuple4AsTuple7BinaryOp, Tuple5AsTuple7BinaryOp, Tuple6AsTuple7BinaryOp},
        control::ScanByKeyControl,
        device::{
            DeviceColumnMutView, DeviceColumnView, DeviceVec, KernelColumn, KernelColumnAt,
            KernelColumnBindings, S0, SoA1 as DeviceSoA1, SoA2 as DeviceSoA2, SoA3 as DeviceSoA3,
        },
        op::kernel::BinaryOp,
        primitives::{range as primitive_range, scan as primitive_scan},
        read::by_key::scan::{
            exclusive_scan_by_flags_one, exclusive_scan_by_flags_one_into,
            exclusive_scan_by_flags_seven_views, exclusive_scan_by_flags_seven_views_into,
            exclusive_scan_by_flags_three, exclusive_scan_by_flags_three_into,
            exclusive_scan_by_flags_two, exclusive_scan_by_flags_two_into,
            inclusive_scan_by_flags_one, inclusive_scan_by_flags_one_into,
            inclusive_scan_by_flags_seven_views, inclusive_scan_by_flags_seven_views_into,
            inclusive_scan_by_flags_three, inclusive_scan_by_flags_three_into,
            inclusive_scan_by_flags_two, inclusive_scan_by_flags_two_into,
        },
    },
    error::Error,
    expr::DeviceGpuExpr,
    value::MStorageElement,
};
use cubecl::prelude::*;

pub(in crate::detail) struct LinearScanApply;

impl LinearScanApply {
    pub(in crate::detail) fn inclusive_expr1<R, T, Expr, Op>(
        policy: &CubePolicy<R>,
        input: &KernelColumnBindings,
        len: usize,
    ) -> Result<DeviceSoA1<DeviceVec<R, T>>, Error>
    where
        R: Runtime,
        T: MStorageElement + 'static,
        Expr: DeviceGpuExpr<T>,
        (T,): MItem<R>,
        Op: BinaryOp<(T,)>,
    {
        primitive_scan::inclusive_scan_tuple1_device_expr::<R, T, Expr, Op>(policy, input, len)
    }

    pub(in crate::detail) fn exclusive_expr1<R, T, Expr, Op>(
        policy: &CubePolicy<R>,
        input: &KernelColumnBindings,
        len: usize,
        init: (T,),
    ) -> Result<DeviceSoA1<DeviceVec<R, T>>, Error>
    where
        R: Runtime,
        T: MStorageElement + 'static,
        Expr: DeviceGpuExpr<T>,
        (T,): MItem<R>,
        Op: BinaryOp<(T,)>,
    {
        primitive_scan::exclusive_scan_tuple1_device_expr::<R, T, Expr, Op>(
            policy, input, len, init,
        )
    }

    pub(in crate::detail) fn adjacent_expr1<Source, Op>(
        policy: &CubePolicy<Source::Runtime>,
        source: &Source,
    ) -> Result<DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>, Error>
    where
        Source: KernelColumn + KernelColumnAt<S0>,
        Source::Item: MStorageElement + 'static,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Op: BinaryOp<Source::Item>,
    {
        let source = crate::detail::api::device_expr_adjacent_difference_with_policy::<Source, Op>(
            policy, source,
        )?;
        Ok(DeviceSoA1 { source })
    }

    pub(in crate::detail) fn inclusive_expr2<R, A, C, AExpr, CExpr, Op>(
        policy: &CubePolicy<R>,
        left: &KernelColumnBindings,
        right: &KernelColumnBindings,
        len: usize,
    ) -> Result<DeviceSoA2<DeviceVec<R, A>, DeviceVec<R, C>>, Error>
    where
        R: Runtime,
        A: MStorageElement + 'static,
        C: MStorageElement + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        (A, C): MItem<R>,
        Op: BinaryOp<(A, C)>,
    {
        primitive_scan::inclusive_scan_tuple2_device_expr::<R, A, C, AExpr, CExpr, Op>(
            policy, left, right, len,
        )
    }

    pub(in crate::detail) fn exclusive_expr2<R, A, C, AExpr, CExpr, Op>(
        policy: &CubePolicy<R>,
        left: &KernelColumnBindings,
        right: &KernelColumnBindings,
        len: usize,
        init: (A, C),
    ) -> Result<DeviceSoA2<DeviceVec<R, A>, DeviceVec<R, C>>, Error>
    where
        R: Runtime,
        A: MStorageElement + 'static,
        C: MStorageElement + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        (A, C): MItem<R>,
        Op: BinaryOp<(A, C)>,
    {
        primitive_scan::exclusive_scan_tuple2_device_expr::<R, A, C, AExpr, CExpr, Op>(
            policy, left, right, len, init,
        )
    }

    pub(in crate::detail) fn adjacent_expr2<R, A, C, AExpr, CExpr, Op>(
        policy: &CubePolicy<R>,
        left: &KernelColumnBindings,
        right: &KernelColumnBindings,
        len: usize,
    ) -> Result<DeviceSoA2<DeviceVec<R, A>, DeviceVec<R, C>>, Error>
    where
        R: Runtime,
        A: MStorageElement + 'static,
        C: MStorageElement + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        (A, C): MItem<R>,
        Op: BinaryOp<(A, C)>,
    {
        primitive_scan::adjacent_difference_tuple2_device_expr::<R, A, C, AExpr, CExpr, Op>(
            policy, left, right, len,
        )
    }

    pub(in crate::detail) fn inclusive_expr3<R, A, C, D, AExpr, CExpr, DExpr, Op>(
        policy: &CubePolicy<R>,
        first: &KernelColumnBindings,
        second: &KernelColumnBindings,
        third: &KernelColumnBindings,
        len: usize,
    ) -> Result<DeviceSoA3<DeviceVec<R, A>, DeviceVec<R, C>, DeviceVec<R, D>>, Error>
    where
        R: Runtime,
        A: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        DExpr: DeviceGpuExpr<D>,
        (A, C, D): MItem<R>,
        Op: BinaryOp<(A, C, D)>,
    {
        primitive_scan::inclusive_scan_tuple3_device_expr::<R, A, C, D, AExpr, CExpr, DExpr, Op>(
            policy, first, second, third, len,
        )
    }

    pub(in crate::detail) fn exclusive_expr3<R, A, C, D, AExpr, CExpr, DExpr, Op>(
        policy: &CubePolicy<R>,
        first: &KernelColumnBindings,
        second: &KernelColumnBindings,
        third: &KernelColumnBindings,
        len: usize,
        init: (A, C, D),
    ) -> Result<DeviceSoA3<DeviceVec<R, A>, DeviceVec<R, C>, DeviceVec<R, D>>, Error>
    where
        R: Runtime,
        A: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        DExpr: DeviceGpuExpr<D>,
        (A, C, D): MItem<R>,
        Op: BinaryOp<(A, C, D)>,
    {
        primitive_scan::exclusive_scan_tuple3_device_expr::<R, A, C, D, AExpr, CExpr, DExpr, Op>(
            policy, first, second, third, len, init,
        )
    }

    pub(in crate::detail) fn adjacent_expr3<R, A, C, D, AExpr, CExpr, DExpr, Op>(
        policy: &CubePolicy<R>,
        first: &KernelColumnBindings,
        second: &KernelColumnBindings,
        third: &KernelColumnBindings,
        len: usize,
    ) -> Result<DeviceSoA3<DeviceVec<R, A>, DeviceVec<R, C>, DeviceVec<R, D>>, Error>
    where
        R: Runtime,
        A: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        DExpr: DeviceGpuExpr<D>,
        (A, C, D): MItem<R>,
        Op: BinaryOp<(A, C, D)>,
    {
        primitive_scan::adjacent_difference_tuple3_device_expr::<R, A, C, D, AExpr, CExpr, DExpr, Op>(
            policy, first, second, third, len,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn inclusive_views4<R, A, B, C, D, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
        ),
        Error,
    >
    where
        R: Runtime,
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D)>,
    {
        let dummy4 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy4 = DeviceColumnView::from_column(&dummy4);
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, _, _, _) =
            Self::inclusive_views7::<R, A, B, C, D, u32, u32, u32, Tuple4AsTuple7BinaryOp<Op>>(
                policy, a, b, c, d, &dummy4, &dummy5, &dummy6,
            )?;
        Ok((a, b, c, d))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn inclusive_views5<R, A, B, C, D, E, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
        ),
        Error,
    >
    where
        R: Runtime,
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E)>,
    {
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, _, _) =
            Self::inclusive_views7::<R, A, B, C, D, E, u32, u32, Tuple5AsTuple7BinaryOp<Op>>(
                policy, a, b, c, d, e, &dummy5, &dummy6,
            )?;
        Ok((a, b, c, d, e))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn inclusive_views6<R, A, B, C, D, E, F, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
            DeviceVec<R, F>,
        ),
        Error,
    >
    where
        R: Runtime,
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        F: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E, F)>,
    {
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, f, _) =
            Self::inclusive_views7::<R, A, B, C, D, E, F, u32, Tuple6AsTuple7BinaryOp<Op>>(
                policy, a, b, c, d, e, f, &dummy6,
            )?;
        Ok((a, b, c, d, e, f))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn inclusive_views7<R, A, B, C, D, E, F, G, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
        g: &DeviceColumnView<R, G>,
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
            DeviceVec<R, F>,
            DeviceVec<R, G>,
        ),
        Error,
    >
    where
        R: Runtime,
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        F: MStorageElement + 'static,
        G: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E, F, G)>,
    {
        primitive_scan::inclusive_scan_tuple7_device_views::<R, A, B, C, D, E, F, G, Op>(
            policy, a, b, c, d, e, f, g,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn exclusive_views4<R, A, B, C, D, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        init: (A, B, C, D),
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
        ),
        Error,
    >
    where
        R: Runtime,
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D)>,
    {
        let dummy4 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy4 = DeviceColumnView::from_column(&dummy4);
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, _, _, _) =
            Self::exclusive_views7::<R, A, B, C, D, u32, u32, u32, Tuple4AsTuple7BinaryOp<Op>>(
                policy,
                a,
                b,
                c,
                d,
                &dummy4,
                &dummy5,
                &dummy6,
                (init.0, init.1, init.2, init.3, 0, 0, 0),
            )?;
        Ok((a, b, c, d))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn exclusive_views5<R, A, B, C, D, E, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        init: (A, B, C, D, E),
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
        ),
        Error,
    >
    where
        R: Runtime,
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E)>,
    {
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, _, _) =
            Self::exclusive_views7::<R, A, B, C, D, E, u32, u32, Tuple5AsTuple7BinaryOp<Op>>(
                policy,
                a,
                b,
                c,
                d,
                e,
                &dummy5,
                &dummy6,
                (init.0, init.1, init.2, init.3, init.4, 0, 0),
            )?;
        Ok((a, b, c, d, e))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn exclusive_views6<R, A, B, C, D, E, F, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
        init: (A, B, C, D, E, F),
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
            DeviceVec<R, F>,
        ),
        Error,
    >
    where
        R: Runtime,
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        F: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E, F)>,
    {
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, f, _) =
            Self::exclusive_views7::<R, A, B, C, D, E, F, u32, Tuple6AsTuple7BinaryOp<Op>>(
                policy,
                a,
                b,
                c,
                d,
                e,
                f,
                &dummy6,
                (init.0, init.1, init.2, init.3, init.4, init.5, 0),
            )?;
        Ok((a, b, c, d, e, f))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn exclusive_views7<R, A, B, C, D, E, F, G, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
        g: &DeviceColumnView<R, G>,
        init: (A, B, C, D, E, F, G),
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
            DeviceVec<R, F>,
            DeviceVec<R, G>,
        ),
        Error,
    >
    where
        R: Runtime,
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        F: MStorageElement + 'static,
        G: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E, F, G)>,
    {
        primitive_scan::exclusive_scan_tuple7_device_views::<R, A, B, C, D, E, F, G, Op>(
            policy, a, b, c, d, e, f, g, init,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn adjacent_views4<R, A, B, C, D, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
        ),
        Error,
    >
    where
        R: Runtime,
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D)>,
    {
        let dummy4 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy4 = DeviceColumnView::from_column(&dummy4);
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, _, _, _) =
            Self::adjacent_views7::<R, A, B, C, D, u32, u32, u32, Tuple4AsTuple7BinaryOp<Op>>(
                policy, a, b, c, d, &dummy4, &dummy5, &dummy6,
            )?;
        Ok((a, b, c, d))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn adjacent_views5<R, A, B, C, D, E, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
        ),
        Error,
    >
    where
        R: Runtime,
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E)>,
    {
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, _, _) =
            Self::adjacent_views7::<R, A, B, C, D, E, u32, u32, Tuple5AsTuple7BinaryOp<Op>>(
                policy, a, b, c, d, e, &dummy5, &dummy6,
            )?;
        Ok((a, b, c, d, e))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn adjacent_views6<R, A, B, C, D, E, F, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
            DeviceVec<R, F>,
        ),
        Error,
    >
    where
        R: Runtime,
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        F: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E, F)>,
    {
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, f, _) =
            Self::adjacent_views7::<R, A, B, C, D, E, F, u32, Tuple6AsTuple7BinaryOp<Op>>(
                policy, a, b, c, d, e, f, &dummy6,
            )?;
        Ok((a, b, c, d, e, f))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn adjacent_views7<R, A, B, C, D, E, F, G, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
        g: &DeviceColumnView<R, G>,
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
            DeviceVec<R, F>,
            DeviceVec<R, G>,
        ),
        Error,
    >
    where
        R: Runtime,
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        F: MStorageElement + 'static,
        G: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E, F, G)>,
    {
        primitive_scan::adjacent_difference_tuple7_device_views::<R, A, B, C, D, E, F, G, Op>(
            policy, a, b, c, d, e, f, g,
        )
    }
}

pub(in crate::detail) struct SegmentedScanApply<'a, R: Runtime> {
    control: &'a ScanByKeyControl<R>,
}

impl<'a, R: Runtime> SegmentedScanApply<'a, R> {
    pub(in crate::detail) fn new(control: &'a ScanByKeyControl<R>) -> Self {
        Self { control }
    }

    pub(in crate::detail) fn inclusive_expr<Source, Op>(
        &self,
        policy: &CubePolicy<R>,
        source: &Source,
    ) -> Result<DeviceVec<R, Source::Item>, Error>
    where
        Source: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        Source::Item: MStorageElement + 'static,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Op: BinaryOp<(Source::Item,)>,
    {
        inclusive_scan_by_flags_one::<Source, Op>(policy, source, self.control)
    }

    pub(in crate::detail) fn inclusive_expr_into<Source, Op>(
        &self,
        policy: &CubePolicy<R>,
        source: &Source,
        output: &DeviceColumnMutView<R, Source::Item>,
    ) -> Result<(), Error>
    where
        Source: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        Source::Item: MStorageElement + 'static,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Op: BinaryOp<(Source::Item,)>,
    {
        inclusive_scan_by_flags_one_into::<Source, Op>(policy, source, self.control, output)
    }

    pub(in crate::detail) fn exclusive_expr<Source, Op>(
        &self,
        policy: &CubePolicy<R>,
        source: &Source,
        init: Source::Item,
    ) -> Result<DeviceVec<R, Source::Item>, Error>
    where
        Source: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        Source::Item: MStorageElement + 'static,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Op: BinaryOp<(Source::Item,)>,
    {
        exclusive_scan_by_flags_one::<Source, Op>(policy, source, self.control, init)
    }

    pub(in crate::detail) fn exclusive_expr_into<Source, Op>(
        &self,
        policy: &CubePolicy<R>,
        source: &Source,
        init: Source::Item,
        output: &DeviceColumnMutView<R, Source::Item>,
    ) -> Result<(), Error>
    where
        Source: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        Source::Item: MStorageElement + 'static,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Op: BinaryOp<(Source::Item,)>,
    {
        exclusive_scan_by_flags_one_into::<Source, Op>(policy, source, self.control, init, output)
    }

    pub(in crate::detail) fn inclusive_expr2<A, C, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &A,
        c: &C,
    ) -> Result<DeviceSoA2<DeviceVec<R, A::Item>, DeviceVec<R, C::Item>>, Error>
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: MStorageElement + 'static,
        C::Item: MStorageElement + 'static,
        A::Expr: DeviceGpuExpr<A::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        (A::Item, C::Item): MItem<R>,
        Op: BinaryOp<(A::Item, C::Item)>,
    {
        inclusive_scan_by_flags_two::<A, C, Op>(policy, a, c, self.control)
    }

    pub(in crate::detail) fn exclusive_expr2<A, C, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &A,
        c: &C,
        init: (A::Item, C::Item),
    ) -> Result<DeviceSoA2<DeviceVec<R, A::Item>, DeviceVec<R, C::Item>>, Error>
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: MStorageElement + 'static,
        C::Item: MStorageElement + 'static,
        A::Expr: DeviceGpuExpr<A::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        (A::Item, C::Item): MItem<R>,
        Op: BinaryOp<(A::Item, C::Item)>,
    {
        exclusive_scan_by_flags_two::<A, C, Op>(policy, a, c, self.control, init)
    }

    pub(in crate::detail) fn inclusive_expr2_into<A, C, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &A,
        c: &C,
        out_a: &DeviceColumnMutView<R, A::Item>,
        out_c: &DeviceColumnMutView<R, C::Item>,
    ) -> Result<(), Error>
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: MStorageElement + 'static,
        C::Item: MStorageElement + 'static,
        A::Expr: DeviceGpuExpr<A::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        (A::Item, C::Item): MItem<R>,
        Op: BinaryOp<(A::Item, C::Item)>,
    {
        inclusive_scan_by_flags_two_into::<A, C, Op>(policy, a, c, self.control, out_a, out_c)
    }

    pub(in crate::detail) fn exclusive_expr2_into<A, C, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &A,
        c: &C,
        init: (A::Item, C::Item),
        out_a: &DeviceColumnMutView<R, A::Item>,
        out_c: &DeviceColumnMutView<R, C::Item>,
    ) -> Result<(), Error>
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: MStorageElement + 'static,
        C::Item: MStorageElement + 'static,
        A::Expr: DeviceGpuExpr<A::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        (A::Item, C::Item): MItem<R>,
        Op: BinaryOp<(A::Item, C::Item)>,
    {
        exclusive_scan_by_flags_two_into::<A, C, Op>(policy, a, c, self.control, init, out_a, out_c)
    }

    pub(in crate::detail) fn inclusive_expr3<A, C, D, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &A,
        c: &C,
        d: &D,
    ) -> Result<
        DeviceSoA3<DeviceVec<R, A::Item>, DeviceVec<R, C::Item>, DeviceVec<R, D::Item>>,
        Error,
    >
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: MStorageElement + 'static,
        C::Item: MStorageElement + 'static,
        D::Item: MStorageElement + 'static,
        A::Expr: DeviceGpuExpr<A::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        (A::Item, C::Item, D::Item): MItem<R>,
        Op: BinaryOp<(A::Item, C::Item, D::Item)>,
    {
        inclusive_scan_by_flags_three::<A, C, D, Op>(policy, a, c, d, self.control)
    }

    pub(in crate::detail) fn exclusive_expr3<A, C, D, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &A,
        c: &C,
        d: &D,
        init: (A::Item, C::Item, D::Item),
    ) -> Result<
        DeviceSoA3<DeviceVec<R, A::Item>, DeviceVec<R, C::Item>, DeviceVec<R, D::Item>>,
        Error,
    >
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: MStorageElement + 'static,
        C::Item: MStorageElement + 'static,
        D::Item: MStorageElement + 'static,
        A::Expr: DeviceGpuExpr<A::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        (A::Item, C::Item, D::Item): MItem<R>,
        Op: BinaryOp<(A::Item, C::Item, D::Item)>,
    {
        exclusive_scan_by_flags_three::<A, C, D, Op>(policy, a, c, d, self.control, init)
    }

    pub(in crate::detail) fn inclusive_expr3_into<A, C, D, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &A,
        c: &C,
        d: &D,
        out_a: &DeviceColumnMutView<R, A::Item>,
        out_c: &DeviceColumnMutView<R, C::Item>,
        out_d: &DeviceColumnMutView<R, D::Item>,
    ) -> Result<(), Error>
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: MStorageElement + 'static,
        C::Item: MStorageElement + 'static,
        D::Item: MStorageElement + 'static,
        A::Expr: DeviceGpuExpr<A::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        (A::Item, C::Item, D::Item): MItem<R>,
        Op: BinaryOp<(A::Item, C::Item, D::Item)>,
    {
        inclusive_scan_by_flags_three_into::<A, C, D, Op>(
            policy,
            a,
            c,
            d,
            self.control,
            out_a,
            out_c,
            out_d,
        )
    }

    pub(in crate::detail) fn exclusive_expr3_into<A, C, D, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &A,
        c: &C,
        d: &D,
        init: (A::Item, C::Item, D::Item),
        out_a: &DeviceColumnMutView<R, A::Item>,
        out_c: &DeviceColumnMutView<R, C::Item>,
        out_d: &DeviceColumnMutView<R, D::Item>,
    ) -> Result<(), Error>
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: MStorageElement + 'static,
        C::Item: MStorageElement + 'static,
        D::Item: MStorageElement + 'static,
        A::Expr: DeviceGpuExpr<A::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        (A::Item, C::Item, D::Item): MItem<R>,
        Op: BinaryOp<(A::Item, C::Item, D::Item)>,
    {
        exclusive_scan_by_flags_three_into::<A, C, D, Op>(
            policy,
            a,
            c,
            d,
            self.control,
            init,
            out_a,
            out_c,
            out_d,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn inclusive_views4<A, B, C, D, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
        ),
        Error,
    >
    where
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D)>,
    {
        let dummy4 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy4 = DeviceColumnView::from_column(&dummy4);
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, _, _, _) = self
            .inclusive_views7::<A, B, C, D, u32, u32, u32, Tuple4AsTuple7BinaryOp<Op>>(
                policy, a, b, c, d, &dummy4, &dummy5, &dummy6,
            )?;
        Ok((a, b, c, d))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn inclusive_views5<A, B, C, D, E, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
        ),
        Error,
    >
    where
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E)>,
    {
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, _, _) = self
            .inclusive_views7::<A, B, C, D, E, u32, u32, Tuple5AsTuple7BinaryOp<Op>>(
                policy, a, b, c, d, e, &dummy5, &dummy6,
            )?;
        Ok((a, b, c, d, e))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn inclusive_views6<A, B, C, D, E, F, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
            DeviceVec<R, F>,
        ),
        Error,
    >
    where
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        F: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E, F)>,
    {
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, f, _) = self
            .inclusive_views7::<A, B, C, D, E, F, u32, Tuple6AsTuple7BinaryOp<Op>>(
                policy, a, b, c, d, e, f, &dummy6,
            )?;
        Ok((a, b, c, d, e, f))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn inclusive_views7<A, B, C, D, E, F, G, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
        g: &DeviceColumnView<R, G>,
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
            DeviceVec<R, F>,
            DeviceVec<R, G>,
        ),
        Error,
    >
    where
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        F: MStorageElement + 'static,
        G: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E, F, G)>,
    {
        inclusive_scan_by_flags_seven_views::<R, A, B, C, D, E, F, G, Op>(
            policy,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            self.control,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn inclusive_views7_into<A, B, C, D, E, F, G, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
        g: &DeviceColumnView<R, G>,
        out_a: &DeviceColumnMutView<R, A>,
        out_b: &DeviceColumnMutView<R, B>,
        out_c: &DeviceColumnMutView<R, C>,
        out_d: &DeviceColumnMutView<R, D>,
        out_e: &DeviceColumnMutView<R, E>,
        out_f: &DeviceColumnMutView<R, F>,
        out_g: &DeviceColumnMutView<R, G>,
    ) -> Result<(), Error>
    where
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        F: MStorageElement + 'static,
        G: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E, F, G)>,
    {
        inclusive_scan_by_flags_seven_views_into::<R, A, B, C, D, E, F, G, Op>(
            policy,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            self.control,
            out_a,
            out_b,
            out_c,
            out_d,
            out_e,
            out_f,
            out_g,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn exclusive_views4<A, B, C, D, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        init: (A, B, C, D),
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
        ),
        Error,
    >
    where
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D)>,
    {
        let dummy4 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy4 = DeviceColumnView::from_column(&dummy4);
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, _, _, _) = self
            .exclusive_views7::<A, B, C, D, u32, u32, u32, Tuple4AsTuple7BinaryOp<Op>>(
                policy,
                a,
                b,
                c,
                d,
                &dummy4,
                &dummy5,
                &dummy6,
                (init.0, init.1, init.2, init.3, 0, 0, 0),
            )?;
        Ok((a, b, c, d))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn exclusive_views5<A, B, C, D, E, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        init: (A, B, C, D, E),
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
        ),
        Error,
    >
    where
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E)>,
    {
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, _, _) = self
            .exclusive_views7::<A, B, C, D, E, u32, u32, Tuple5AsTuple7BinaryOp<Op>>(
                policy,
                a,
                b,
                c,
                d,
                e,
                &dummy5,
                &dummy6,
                (init.0, init.1, init.2, init.3, init.4, 0, 0),
            )?;
        Ok((a, b, c, d, e))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn exclusive_views6<A, B, C, D, E, F, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
        init: (A, B, C, D, E, F),
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
            DeviceVec<R, F>,
        ),
        Error,
    >
    where
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        F: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E, F)>,
    {
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, f, _) = self
            .exclusive_views7::<A, B, C, D, E, F, u32, Tuple6AsTuple7BinaryOp<Op>>(
                policy,
                a,
                b,
                c,
                d,
                e,
                f,
                &dummy6,
                (init.0, init.1, init.2, init.3, init.4, init.5, 0),
            )?;
        Ok((a, b, c, d, e, f))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn exclusive_views7<A, B, C, D, E, F, G, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
        g: &DeviceColumnView<R, G>,
        init: (A, B, C, D, E, F, G),
    ) -> Result<
        (
            DeviceVec<R, A>,
            DeviceVec<R, B>,
            DeviceVec<R, C>,
            DeviceVec<R, D>,
            DeviceVec<R, E>,
            DeviceVec<R, F>,
            DeviceVec<R, G>,
        ),
        Error,
    >
    where
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        F: MStorageElement + 'static,
        G: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E, F, G)>,
    {
        exclusive_scan_by_flags_seven_views::<R, A, B, C, D, E, F, G, Op>(
            policy,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            self.control,
            init,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn exclusive_views7_into<A, B, C, D, E, F, G, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
        g: &DeviceColumnView<R, G>,
        init: (A, B, C, D, E, F, G),
        out_a: &DeviceColumnMutView<R, A>,
        out_b: &DeviceColumnMutView<R, B>,
        out_c: &DeviceColumnMutView<R, C>,
        out_d: &DeviceColumnMutView<R, D>,
        out_e: &DeviceColumnMutView<R, E>,
        out_f: &DeviceColumnMutView<R, F>,
        out_g: &DeviceColumnMutView<R, G>,
    ) -> Result<(), Error>
    where
        A: MStorageElement + 'static,
        B: MStorageElement + 'static,
        C: MStorageElement + 'static,
        D: MStorageElement + 'static,
        E: MStorageElement + 'static,
        F: MStorageElement + 'static,
        G: MStorageElement + 'static,
        Op: BinaryOp<(A, B, C, D, E, F, G)>,
    {
        exclusive_scan_by_flags_seven_views_into::<R, A, B, C, D, E, F, G, Op>(
            policy,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            self.control,
            init,
            out_a,
            out_b,
            out_c,
            out_d,
            out_e,
            out_f,
            out_g,
        )
    }
}
