use crate::{
    MItem,
    detail::{
        CubePolicy,
        api::{Tuple4AsTuple7BinaryOp, Tuple5AsTuple7BinaryOp, Tuple6AsTuple7BinaryOp},
        control::{ReduceByKeyControl, ScanByKeyControl},
        device::{
            DeviceColumnView, DeviceVec, KernelColumn, KernelColumnAt, KernelColumnBindings, S0,
            SoA2 as DeviceSoA2, SoA3 as DeviceSoA3,
        },
        op::kernel::BinaryOp,
        primitives::{
            range as primitive_range, reduce as primitive_reduce, scan as primitive_scan,
        },
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::{
        reduce_by_key_apply_init_kernel, reduce_by_key_tuple2_apply_init_kernel,
        reduce_by_key_tuple3_apply_init_kernel, reduce_by_key_tuple7_apply_init_kernel,
    },
    runtime::Scalar,
};
use cubecl::prelude::*;

pub(in crate::detail) struct LinearReduceApply;

impl LinearReduceApply {
    pub(in crate::detail) fn apply_expr1<R, T, Expr, Op>(
        policy: &CubePolicy<R>,
        input: &KernelColumnBindings,
        len: usize,
        init: (T,),
    ) -> Result<(T,), Error>
    where
        R: Runtime,
        T: Scalar + 'static,
        Expr: DeviceGpuExpr<T>,
        (T,): MItem<R>,
        Op: BinaryOp<(T,)>,
    {
        primitive_reduce::reduce_tuple1_device_expr::<R, T, Expr, Op>(policy, input, len, init)
    }

    pub(in crate::detail) fn apply_expr2<R, A, C, AExpr, CExpr, Op>(
        policy: &CubePolicy<R>,
        left: &KernelColumnBindings,
        right: &KernelColumnBindings,
        len: usize,
        init: (A, C),
    ) -> Result<(A, C), Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        C: Scalar + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        (A, C): MItem<R>,
        Op: BinaryOp<(A, C)>,
    {
        primitive_reduce::reduce_tuple2_device_expr::<R, A, C, AExpr, CExpr, Op>(
            policy, left, right, len, init,
        )
    }

    pub(in crate::detail) fn apply_expr3<R, A, C, D, AExpr, CExpr, DExpr, Op>(
        policy: &CubePolicy<R>,
        first: &KernelColumnBindings,
        second: &KernelColumnBindings,
        third: &KernelColumnBindings,
        len: usize,
        init: (A, C, D),
    ) -> Result<(A, C, D), Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        DExpr: DeviceGpuExpr<D>,
        (A, C, D): MItem<R>,
        Op: BinaryOp<(A, C, D)>,
    {
        primitive_reduce::reduce_tuple3_device_expr::<R, A, C, D, AExpr, CExpr, DExpr, Op>(
            policy, first, second, third, len, init,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn apply_views4<R, A, B, C, D, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        init: (A, B, C, D),
    ) -> Result<(A, B, C, D), Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        B: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        Op: BinaryOp<(A, B, C, D)>,
    {
        let dummy4 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy4 = DeviceColumnView::from_column(&dummy4);
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, _, _, _) =
            Self::apply_views7::<R, A, B, C, D, u32, u32, u32, Tuple4AsTuple7BinaryOp<Op>>(
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
    pub(in crate::detail) fn apply_views5<R, A, B, C, D, E, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        init: (A, B, C, D, E),
    ) -> Result<(A, B, C, D, E), Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        B: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        E: Scalar + 'static,
        Op: BinaryOp<(A, B, C, D, E)>,
    {
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, _, _) =
            Self::apply_views7::<R, A, B, C, D, E, u32, u32, Tuple5AsTuple7BinaryOp<Op>>(
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
    pub(in crate::detail) fn apply_views6<R, A, B, C, D, E, F, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
        init: (A, B, C, D, E, F),
    ) -> Result<(A, B, C, D, E, F), Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        B: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        E: Scalar + 'static,
        F: Scalar + 'static,
        Op: BinaryOp<(A, B, C, D, E, F)>,
    {
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, f, _) =
            Self::apply_views7::<R, A, B, C, D, E, F, u32, Tuple6AsTuple7BinaryOp<Op>>(
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
    pub(in crate::detail) fn apply_views7<R, A, B, C, D, E, F, G, Op>(
        policy: &CubePolicy<R>,
        a: &DeviceColumnView<R, A>,
        b: &DeviceColumnView<R, B>,
        c: &DeviceColumnView<R, C>,
        d: &DeviceColumnView<R, D>,
        e: &DeviceColumnView<R, E>,
        f: &DeviceColumnView<R, F>,
        g: &DeviceColumnView<R, G>,
        init: (A, B, C, D, E, F, G),
    ) -> Result<(A, B, C, D, E, F, G), Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        B: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        E: Scalar + 'static,
        F: Scalar + 'static,
        G: Scalar + 'static,
        Op: BinaryOp<(A, B, C, D, E, F, G)>,
    {
        let a_stage = KernelColumn::stage(a, policy)?;
        let b_stage = KernelColumn::stage(b, policy)?;
        let c_stage = KernelColumn::stage(c, policy)?;
        let d_stage = KernelColumn::stage(d, policy)?;
        let e_stage = KernelColumn::stage(e, policy)?;
        let f_stage = KernelColumn::stage(f, policy)?;
        let g_stage = KernelColumn::stage(g, policy)?;
        primitive_reduce::reduce_tuple7_device_expr::<
            R,
            A,
            B,
            C,
            D,
            E,
            F,
            G,
            <DeviceColumnView<R, A> as KernelColumn>::Expr,
            <DeviceColumnView<R, B> as KernelColumn>::Expr,
            <DeviceColumnView<R, C> as KernelColumn>::Expr,
            <DeviceColumnView<R, D> as KernelColumn>::Expr,
            <DeviceColumnView<R, E> as KernelColumn>::Expr,
            <DeviceColumnView<R, F> as KernelColumn>::Expr,
            <DeviceColumnView<R, G> as KernelColumn>::Expr,
            Op,
        >(
            policy, &a_stage, &b_stage, &c_stage, &d_stage, &e_stage, &f_stage, &g_stage, a.len,
            init,
        )
    }
}

pub(in crate::detail) struct SegmentedReduceApply<'a, R: Runtime> {
    control: &'a ReduceByKeyControl<R>,
}

impl<'a, R: Runtime> SegmentedReduceApply<'a, R> {
    pub(in crate::detail) fn new(control: &'a ReduceByKeyControl<R>) -> Self {
        Self { control }
    }

    pub(in crate::detail) fn apply_expr<ValueSource, Op>(
        &self,
        policy: &CubePolicy<R>,
        source: &ValueSource,
        init: ValueSource::Item,
    ) -> Result<DeviceVec<R, ValueSource::Item>, Error>
    where
        ValueSource: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueSource::Item: Scalar + 'static,
        ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
        Op: BinaryOp<(ValueSource::Item,)>,
    {
        let client = policy.client();
        let scan_control: ScanByKeyControl<R> = self.control.into();
        let scan_apply = crate::detail::apply::SegmentedScanApply::new(&scan_control);
        let inclusive = scan_apply.inclusive_expr::<ValueSource, Op>(policy, source)?;

        let len_handle = client.create_from_slice(u32::as_bytes(&[self.control.len_u32]));
        let init_handle = client.create_from_slice(ValueSource::Item::as_bytes(&[init]));
        let reduced_value_handle =
            client.empty(self.control.len * std::mem::size_of::<ValueSource::Item>());
        let num_blocks = self
            .control
            .len
            .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_apply_init_kernel::launch_unchecked::<
                ValueSource::Item,
                Op,
                ValueSource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts(inclusive.handle.clone(), self.control.len),
                BufferArg::from_raw_parts(init_handle.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_value_handle.clone(), self.control.len),
            );
        }

        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(
            &self.control.output_selection,
            self.control.output_count,
        );
        payload_apply.apply_value::<R, ValueSource::Item>(policy, reduced_value_handle)
    }

    pub(in crate::detail) fn apply_expr2<ValueA, ValueB, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &ValueA,
        b: &ValueB,
        init: (ValueA::Item, ValueB::Item),
    ) -> Result<DeviceSoA2<DeviceVec<R, ValueA::Item>, DeviceVec<R, ValueB::Item>>, Error>
    where
        ValueA: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueB: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueA::Item: Scalar + 'static,
        ValueB::Item: Scalar + 'static,
        ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
        ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
        (ValueA::Item, ValueB::Item): MItem<R>,
        Op: BinaryOp<(ValueA::Item, ValueB::Item)>,
    {
        let client = policy.client();
        let scan_control: ScanByKeyControl<R> = self.control.into();
        let scan_apply = crate::detail::apply::SegmentedScanApply::new(&scan_control);
        let inclusive = scan_apply.inclusive_expr2::<ValueA, ValueB, Op>(policy, a, b)?;

        let len_handle = client.create_from_slice(u32::as_bytes(&[self.control.len_u32]));
        let init_a = client.create_from_slice(ValueA::Item::as_bytes(&[init.0]));
        let init_b = client.create_from_slice(ValueB::Item::as_bytes(&[init.1]));
        let reduced_a_handle = client.empty(self.control.len * std::mem::size_of::<ValueA::Item>());
        let reduced_b_handle = client.empty(self.control.len * std::mem::size_of::<ValueB::Item>());
        let num_blocks = self
            .control
            .len
            .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_tuple2_apply_init_kernel::launch_unchecked::<
                ValueA::Item,
                ValueB::Item,
                Op,
                ValueA::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts(inclusive.left.handle.clone(), self.control.len),
                BufferArg::from_raw_parts(inclusive.right.handle.clone(), self.control.len),
                BufferArg::from_raw_parts(init_a.clone(), 1),
                BufferArg::from_raw_parts(init_b.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_a_handle.clone(), self.control.len),
                BufferArg::from_raw_parts(reduced_b_handle.clone(), self.control.len),
            );
        }

        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(
            &self.control.output_selection,
            self.control.output_count,
        );
        Ok(DeviceSoA2 {
            left: payload_apply.apply_value::<R, ValueA::Item>(policy, reduced_a_handle)?,
            right: payload_apply.apply_value::<R, ValueB::Item>(policy, reduced_b_handle)?,
        })
    }

    pub(in crate::detail) fn apply_expr3<ValueA, ValueB, ValueC, Op>(
        &self,
        policy: &CubePolicy<R>,
        a: &ValueA,
        b: &ValueB,
        c: &ValueC,
        init: (ValueA::Item, ValueB::Item, ValueC::Item),
    ) -> Result<
        DeviceSoA3<
            DeviceVec<R, ValueA::Item>,
            DeviceVec<R, ValueB::Item>,
            DeviceVec<R, ValueC::Item>,
        >,
        Error,
    >
    where
        ValueA: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueB: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueC: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueA::Item: Scalar + 'static,
        ValueB::Item: Scalar + 'static,
        ValueC::Item: Scalar + 'static,
        ValueA::Expr: DeviceGpuExpr<ValueA::Item>,
        ValueB::Expr: DeviceGpuExpr<ValueB::Item>,
        ValueC::Expr: DeviceGpuExpr<ValueC::Item>,
        (ValueA::Item, ValueB::Item, ValueC::Item): MItem<R>,
        Op: BinaryOp<(ValueA::Item, ValueB::Item, ValueC::Item)>,
    {
        let client = policy.client();
        let scan_control: ScanByKeyControl<R> = self.control.into();
        let scan_apply = crate::detail::apply::SegmentedScanApply::new(&scan_control);
        let inclusive =
            scan_apply.inclusive_expr3::<ValueA, ValueB, ValueC, Op>(policy, a, b, c)?;

        let len_handle = client.create_from_slice(u32::as_bytes(&[self.control.len_u32]));
        let init_a = client.create_from_slice(ValueA::Item::as_bytes(&[init.0]));
        let init_b = client.create_from_slice(ValueB::Item::as_bytes(&[init.1]));
        let init_c = client.create_from_slice(ValueC::Item::as_bytes(&[init.2]));
        let reduced_a_handle = client.empty(self.control.len * std::mem::size_of::<ValueA::Item>());
        let reduced_b_handle = client.empty(self.control.len * std::mem::size_of::<ValueB::Item>());
        let reduced_c_handle = client.empty(self.control.len * std::mem::size_of::<ValueC::Item>());
        let num_blocks = self
            .control
            .len
            .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

        unsafe {
            reduce_by_key_tuple3_apply_init_kernel::launch_unchecked::<
                ValueA::Item,
                ValueB::Item,
                ValueC::Item,
                Op,
                ValueA::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts(inclusive.first.handle.clone(), self.control.len),
                BufferArg::from_raw_parts(inclusive.second.handle.clone(), self.control.len),
                BufferArg::from_raw_parts(inclusive.third.handle.clone(), self.control.len),
                BufferArg::from_raw_parts(init_a.clone(), 1),
                BufferArg::from_raw_parts(init_b.clone(), 1),
                BufferArg::from_raw_parts(init_c.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_a_handle.clone(), self.control.len),
                BufferArg::from_raw_parts(reduced_b_handle.clone(), self.control.len),
                BufferArg::from_raw_parts(reduced_c_handle.clone(), self.control.len),
            );
        }

        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(
            &self.control.output_selection,
            self.control.output_count,
        );
        Ok(DeviceSoA3 {
            first: payload_apply.apply_value::<R, ValueA::Item>(policy, reduced_a_handle)?,
            second: payload_apply.apply_value::<R, ValueB::Item>(policy, reduced_b_handle)?,
            third: payload_apply.apply_value::<R, ValueC::Item>(policy, reduced_c_handle)?,
        })
    }
}

macro_rules! reduce_by_key_tuple7_scanned_values {
    ($policy:ident, $control:ident, $inclusive:ident, $init:expr; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; $op:ty) => {{
        let client = $policy.client();
        let len_handle = client.create_from_slice(u32::as_bytes(&[$control.len_u32]));
        let init_a = client.create_from_slice($ty0::as_bytes(&[$init.0]));
        let init_b = client.create_from_slice($ty1::as_bytes(&[$init.1]));
        let init_c = client.create_from_slice($ty2::as_bytes(&[$init.2]));
        let init_d = client.create_from_slice($ty3::as_bytes(&[$init.3]));
        let init_e = client.create_from_slice($ty4::as_bytes(&[$init.4]));
        let init_f = client.create_from_slice($ty5::as_bytes(&[$init.5]));
        let init_g = client.create_from_slice($ty6::as_bytes(&[$init.6]));
        let reduced_a_handle = client.empty($control.len * std::mem::size_of::<$ty0>());
        let reduced_b_handle = client.empty($control.len * std::mem::size_of::<$ty1>());
        let reduced_c_handle = client.empty($control.len * std::mem::size_of::<$ty2>());
        let reduced_d_handle = client.empty($control.len * std::mem::size_of::<$ty3>());
        let reduced_e_handle = client.empty($control.len * std::mem::size_of::<$ty4>());
        let reduced_f_handle = client.empty($control.len * std::mem::size_of::<$ty5>());
        let reduced_g_handle = client.empty($control.len * std::mem::size_of::<$ty6>());
        let num_blocks = $control
            .len
            .div_ceil(primitive_scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
        unsafe {
            reduce_by_key_tuple7_apply_init_kernel::launch_unchecked::<
                $ty0,
                $ty1,
                $ty2,
                $ty3,
                $ty4,
                $ty5,
                $ty6,
                $op,
                R,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(primitive_scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts($inclusive.0.handle.clone(), $control.len),
                BufferArg::from_raw_parts($inclusive.1.handle.clone(), $control.len),
                BufferArg::from_raw_parts($inclusive.2.handle.clone(), $control.len),
                BufferArg::from_raw_parts($inclusive.3.handle.clone(), $control.len),
                BufferArg::from_raw_parts($inclusive.4.handle.clone(), $control.len),
                BufferArg::from_raw_parts($inclusive.5.handle.clone(), $control.len),
                BufferArg::from_raw_parts($inclusive.6.handle.clone(), $control.len),
                BufferArg::from_raw_parts(init_a.clone(), 1),
                BufferArg::from_raw_parts(init_b.clone(), 1),
                BufferArg::from_raw_parts(init_c.clone(), 1),
                BufferArg::from_raw_parts(init_d.clone(), 1),
                BufferArg::from_raw_parts(init_e.clone(), 1),
                BufferArg::from_raw_parts(init_f.clone(), 1),
                BufferArg::from_raw_parts(init_g.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_a_handle.clone(), $control.len),
                BufferArg::from_raw_parts(reduced_b_handle.clone(), $control.len),
                BufferArg::from_raw_parts(reduced_c_handle.clone(), $control.len),
                BufferArg::from_raw_parts(reduced_d_handle.clone(), $control.len),
                BufferArg::from_raw_parts(reduced_e_handle.clone(), $control.len),
                BufferArg::from_raw_parts(reduced_f_handle.clone(), $control.len),
                BufferArg::from_raw_parts(reduced_g_handle.clone(), $control.len),
            );
        }
        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(
            &$control.output_selection,
            $control.output_count,
        );
        Ok::<_, Error>((
            payload_apply.apply_value::<R, $ty0>($policy, reduced_a_handle)?,
            payload_apply.apply_value::<R, $ty1>($policy, reduced_b_handle)?,
            payload_apply.apply_value::<R, $ty2>($policy, reduced_c_handle)?,
            payload_apply.apply_value::<R, $ty3>($policy, reduced_d_handle)?,
            payload_apply.apply_value::<R, $ty4>($policy, reduced_e_handle)?,
            payload_apply.apply_value::<R, $ty5>($policy, reduced_f_handle)?,
            payload_apply.apply_value::<R, $ty6>($policy, reduced_g_handle)?,
        ))
    }};
}

impl<'a, R: Runtime> SegmentedReduceApply<'a, R> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn apply_views4<A, B, C, D, Op>(
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
        A: Scalar + 'static,
        B: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        Op: BinaryOp<(A, B, C, D)>,
    {
        if self.control.len == 0 {
            return Ok((
                policy.empty_device_vec(),
                policy.empty_device_vec(),
                policy.empty_device_vec(),
                policy.empty_device_vec(),
            ));
        }

        let dummy4 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy4 = DeviceColumnView::from_column(&dummy4);
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, _, _, _) = self
            .apply_views7::<A, B, C, D, u32, u32, u32, Tuple4AsTuple7BinaryOp<Op>>(
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
    pub(in crate::detail) fn apply_views5<A, B, C, D, E, Op>(
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
        A: Scalar + 'static,
        B: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        E: Scalar + 'static,
        Op: BinaryOp<(A, B, C, D, E)>,
    {
        if self.control.len == 0 {
            return Ok((
                policy.empty_device_vec(),
                policy.empty_device_vec(),
                policy.empty_device_vec(),
                policy.empty_device_vec(),
                policy.empty_device_vec(),
            ));
        }

        let dummy5 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy5 = DeviceColumnView::from_column(&dummy5);
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, _, _) = self
            .apply_views7::<A, B, C, D, E, u32, u32, Tuple5AsTuple7BinaryOp<Op>>(
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
    pub(in crate::detail) fn apply_views6<A, B, C, D, E, F, Op>(
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
        A: Scalar + 'static,
        B: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        E: Scalar + 'static,
        F: Scalar + 'static,
        Op: BinaryOp<(A, B, C, D, E, F)>,
    {
        if self.control.len == 0 {
            return Ok((
                policy.empty_device_vec(),
                policy.empty_device_vec(),
                policy.empty_device_vec(),
                policy.empty_device_vec(),
                policy.empty_device_vec(),
                policy.empty_device_vec(),
            ));
        }

        let dummy6 = primitive_range::indices_mindex(policy, a.len)?;
        let dummy6 = DeviceColumnView::from_column(&dummy6);
        let (a, b, c, d, e, f, _) = self
            .apply_views7::<A, B, C, D, E, F, u32, Tuple6AsTuple7BinaryOp<Op>>(
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
    pub(in crate::detail) fn apply_views7<A, B, C, D, E, F, G, Op>(
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
        A: Scalar + 'static,
        B: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        E: Scalar + 'static,
        F: Scalar + 'static,
        G: Scalar + 'static,
        Op: BinaryOp<(A, B, C, D, E, F, G)>,
    {
        let scan_control: ScanByKeyControl<R> = self.control.into();
        let scan_apply = crate::detail::apply::SegmentedScanApply::new(&scan_control);
        let inclusive =
            scan_apply.inclusive_views7::<A, B, C, D, E, F, G, Op>(policy, a, b, c, d, e, f, g)?;
        let control = self.control;
        reduce_by_key_tuple7_scanned_values!(
            policy, control, inclusive, init;
            A, B, C, D, E, F, G;
            Op
        )
    }
}
