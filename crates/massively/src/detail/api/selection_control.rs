use super::*;

#[doc(hidden)]
pub(crate) trait SelectionStencil<Pred> {
    type Runtime: Runtime;

    fn len(&self) -> usize;
    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error>;

    fn selection_flags_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        self.selection_handles_with_policy(policy, invert)
    }
}

#[doc(hidden)]
pub struct PrecomputedSelection<R: Runtime> {
    len: usize,
    handles: select::SelectionHandles,
    _runtime: std::marker::PhantomData<R>,
}

impl<R: Runtime> PrecomputedSelection<R> {
    pub(crate) fn from_stencil_with_policy<Stencil, Pred>(
        policy: &crate::policy::CubePolicy<R>,
        stencil: &Stencil,
        invert: bool,
    ) -> Result<Self, Error>
    where
        Stencil: SelectionStencil<Pred, Runtime = R>,
    {
        Ok(Self {
            len: stencil.len(),
            handles: stencil.selection_handles_with_policy(policy, invert)?,
            _runtime: std::marker::PhantomData,
        })
    }

    pub(crate) fn from_stencil_flags_with_policy<Stencil, Pred>(
        policy: &crate::policy::CubePolicy<R>,
        stencil: &Stencil,
        invert: bool,
    ) -> Result<Self, Error>
    where
        Stencil: SelectionStencil<Pred, Runtime = R>,
    {
        Ok(Self {
            len: stencil.len(),
            handles: stencil.selection_flags_with_policy(policy, invert)?,
            _runtime: std::marker::PhantomData,
        })
    }

    pub(crate) fn control(&self) -> &select::SelectionControl {
        &self.handles
    }
}

impl<R, Pred> SelectionStencil<Pred> for PrecomputedSelection<R>
where
    R: Runtime,
{
    type Runtime = R;

    fn len(&self) -> usize {
        self.len
    }

    fn selection_handles_with_policy(
        &self,
        _policy: &crate::policy::CubePolicy<Self::Runtime>,
        _invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        Ok(self.handles.clone())
    }

    fn selection_flags_with_policy(
        &self,
        _policy: &crate::policy::CubePolicy<Self::Runtime>,
        _invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        Ok(self.handles.clone())
    }
}

impl<Stencil, Pred> SelectionStencil<Pred> for Stencil
where
    Stencil: KernelColumn + KernelColumnAt<S0>,
    Stencil::Runtime: Runtime,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<Stencil::Item, Env = ()>,
{
    type Runtime = Stencil::Runtime;

    fn len(&self) -> usize {
        KernelColumn::len(self)
    }

    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        device_expr_selection_handles_with_policy::<Stencil, Pred>(policy, self, invert, ())
    }

    fn selection_flags_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        device_expr_selection_flags_with_policy::<Stencil, Pred>(policy, self, invert, ())
    }
}

impl<Stencil, Pred> SelectionStencil<Pred> for (Stencil,)
where
    Stencil: KernelColumn + KernelColumnAt<S0>,
    Stencil::Runtime: Runtime,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<(Stencil::Item,), Env = ()>,
{
    type Runtime = Stencil::Runtime;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        device_expr_selection_handles_with_policy::<Stencil, Tuple1PredicateOp<Pred>>(
            policy,
            &self.0,
            invert,
            (),
        )
    }

    fn selection_flags_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        device_expr_selection_flags_with_policy::<Stencil, Tuple1PredicateOp<Pred>>(
            policy,
            &self.0,
            invert,
            (),
        )
    }
}

impl<A, B, Pred> SelectionStencil<Pred> for SoAView2<A, B>
where
    A: KernelColumn + KernelColumnAt<S0>,
    B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Runtime: Runtime,
    A::Item: CubePrimitive + CubeElement,
    B::Item: CubePrimitive + CubeElement,
    A::Expr: DeviceGpuExpr<A::Item>,
    B::Expr: DeviceGpuExpr<B::Item>,
    Pred: PredicateOp<(A::Item, B::Item), Env = ()>,
{
    type Runtime = A::Runtime;

    fn len(&self) -> usize {
        self.left.len()
    }

    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        self.left.validate()?;
        self.right.validate()?;
        ensure_same_len(self.left.len(), self.right.len())?;
        let len = self.left.len();
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let client = policy.client();
        let flag = client.empty(len * std::mem::size_of::<u32>());
        if len != 0 {
            let block_count_u32 = api_expr_block_count(len)?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let invert_values = [if invert { 1_u32 } else { 0_u32 }];
            let invert_handle = client.create_from_slice(u32::as_bytes(&invert_values));
            let left_bindings = self.left.stage(policy)?;
            let right_bindings = self.right.stage(policy)?;
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
                tuple2_predicate_device_expr_flags_kernel::launch_unchecked::<
                    A::Item,
                    B::Item,
                    A::Expr,
                    B::Expr,
                    Pred,
                    A::Runtime,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
                    (),
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
                    unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(invert_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(flag.clone(), len) },
                );
            }
        }
        select::handles_from_flags(policy, len, len_u32, flag, policy.empty_handle())
    }
}

impl<A, B, C, Pred> SelectionStencil<Pred> for SoAView3<A, B, C>
where
    A: KernelColumn + KernelColumnAt<S0>,
    B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Runtime: Runtime,
    A::Item: CubePrimitive + CubeElement,
    B::Item: CubePrimitive + CubeElement,
    C::Item: CubePrimitive + CubeElement,
    A::Expr: DeviceGpuExpr<A::Item>,
    B::Expr: DeviceGpuExpr<B::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    Pred: PredicateOp<(A::Item, B::Item, C::Item), Env = ()>,
{
    type Runtime = A::Runtime;

    fn len(&self) -> usize {
        self.first.len()
    }

    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        self.first.validate()?;
        self.second.validate()?;
        self.third.validate()?;
        ensure_same_len(self.first.len(), self.second.len())?;
        ensure_same_len(self.first.len(), self.third.len())?;
        let len = self.first.len();
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let client = policy.client();
        let flag = client.empty(len * std::mem::size_of::<u32>());
        if len != 0 {
            let block_count_u32 = api_expr_block_count(len)?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let invert_values = [if invert { 1_u32 } else { 0_u32 }];
            let invert_handle = client.create_from_slice(u32::as_bytes(&invert_values));
            let first_bindings = self.first.stage(policy)?;
            let second_bindings = self.second.stage(policy)?;
            let third_bindings = self.third.stage(policy)?;
            let first_slot_offsets = first_bindings.slot_offsets_handle(client)?;
            let second_slot_offsets = second_bindings.slot_offsets_handle(client)?;
            let third_slot_offsets = third_bindings.slot_offsets_handle(client)?;
            let first_slot0 = first_bindings.slots.first().unwrap();
            let first_slot1 = first_bindings.slots.get(1).unwrap_or(first_slot0);
            let first_slot2 = first_bindings.slots.get(2).unwrap_or(first_slot0);
            let first_slot3 = first_bindings.slots.get(3).unwrap_or(first_slot0);
            let second_slot0 = second_bindings.slots.first().unwrap();
            let second_slot1 = second_bindings.slots.get(1).unwrap_or(second_slot0);
            let second_slot2 = second_bindings.slots.get(2).unwrap_or(second_slot0);
            let second_slot3 = second_bindings.slots.get(3).unwrap_or(second_slot0);
            let third_slot0 = third_bindings.slots.first().unwrap();
            let third_slot1 = third_bindings.slots.get(1).unwrap_or(third_slot0);
            let third_slot2 = third_bindings.slots.get(2).unwrap_or(third_slot0);
            let third_slot3 = third_bindings.slots.get(3).unwrap_or(third_slot0);

            unsafe {
                tuple3_predicate_device_expr_flags_kernel::launch_unchecked::<
                    A::Item,
                    B::Item,
                    C::Item,
                    A::Expr,
                    B::Expr,
                    C::Expr,
                    Pred,
                    A::Runtime,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
                    (),
                    unsafe { BufferArg::from_raw_parts(first_slot0.0.clone(), first_slot0.1) },
                    unsafe { BufferArg::from_raw_parts(first_slot1.0.clone(), first_slot1.1) },
                    unsafe { BufferArg::from_raw_parts(first_slot2.0.clone(), first_slot2.1) },
                    unsafe { BufferArg::from_raw_parts(first_slot3.0.clone(), first_slot3.1) },
                    unsafe { BufferArg::from_raw_parts(first_slot_offsets.clone(), 4) },
                    unsafe { BufferArg::from_raw_parts(second_slot0.0.clone(), second_slot0.1) },
                    unsafe { BufferArg::from_raw_parts(second_slot1.0.clone(), second_slot1.1) },
                    unsafe { BufferArg::from_raw_parts(second_slot2.0.clone(), second_slot2.1) },
                    unsafe { BufferArg::from_raw_parts(second_slot3.0.clone(), second_slot3.1) },
                    unsafe { BufferArg::from_raw_parts(second_slot_offsets.clone(), 4) },
                    unsafe { BufferArg::from_raw_parts(third_slot0.0.clone(), third_slot0.1) },
                    unsafe { BufferArg::from_raw_parts(third_slot1.0.clone(), third_slot1.1) },
                    unsafe { BufferArg::from_raw_parts(third_slot2.0.clone(), third_slot2.1) },
                    unsafe { BufferArg::from_raw_parts(third_slot3.0.clone(), third_slot3.1) },
                    unsafe { BufferArg::from_raw_parts(third_slot_offsets.clone(), 4) },
                    unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(invert_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(flag.clone(), len) },
                );
            }
        }
        select::handles_from_flags(policy, len, len_u32, flag, policy.empty_handle())
    }
}

impl<Left, Right, Pred> SelectionStencil<Pred> for (Left, Right)
where
    Left: Copy,
    Right: Copy,
    SoAView2<Left, Right>: SelectionStencil<Pred>,
{
    type Runtime = <SoAView2<Left, Right> as SelectionStencil<Pred>>::Runtime;

    fn len(&self) -> usize {
        <SoAView2<Left, Right> as SelectionStencil<Pred>>::len(&SoAView2 {
            left: self.0,
            right: self.1,
        })
    }

    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        SoAView2 {
            left: self.0,
            right: self.1,
        }
        .selection_handles_with_policy(policy, invert)
    }
}

impl<First, Second, Third, Pred> SelectionStencil<Pred> for (First, Second, Third)
where
    First: Copy,
    Second: Copy,
    Third: Copy,
    SoAView3<First, Second, Third>: SelectionStencil<Pred>,
{
    type Runtime = <SoAView3<First, Second, Third> as SelectionStencil<Pred>>::Runtime;

    fn len(&self) -> usize {
        <SoAView3<First, Second, Third> as SelectionStencil<Pred>>::len(&SoAView3 {
            first: self.0,
            second: self.1,
            third: self.2,
        })
    }

    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        SoAView3 {
            first: self.0,
            second: self.1,
            third: self.2,
        }
        .selection_handles_with_policy(policy, invert)
    }
}
