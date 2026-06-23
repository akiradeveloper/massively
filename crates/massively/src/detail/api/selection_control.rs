use super::*;

#[doc(hidden)]
pub trait SelectionStencil<Pred> {
    type Runtime: Runtime;

    fn len(&self) -> usize;
    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error>;
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
}

impl<Stencil, Pred> SelectionStencil<Pred> for Stencil
where
    Stencil: KernelColumn + KernelColumnAt<S0>,
    Stencil::Runtime: Runtime,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp1<Stencil::Item>,
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
        device_expr_selection_handles_with_policy::<Stencil, Pred>(policy, self, invert)
    }
}

impl<Stencil, Pred> SelectionStencil<Pred> for (Stencil,)
where
    Stencil: KernelColumn + KernelColumnAt<S0>,
    Stencil::Runtime: Runtime,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp1<(Stencil::Item,)>,
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
            policy, &self.0, invert,
        )
    }
}

macro_rules! impl_tuple_selection_stencil {
    (
        $name:ident < $first:ident, $( $rest:ident ),+ > {
            $first_field:ident, $( $field:ident ),+
        },
        $kernel_name:ident
    ) => {
        impl<$first, $( $rest ),+, Pred> SelectionStencil<Pred>
            for $name<$first, $( $rest ),+>
        where
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Runtime: Runtime,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Pred: PredicateOp1<(
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            fn len(&self) -> usize {
                self.$first_field.len()
            }

            fn selection_handles_with_policy(
                &self,
                policy: &crate::policy::CubePolicy<Self::Runtime>,
                invert: bool,
            ) -> Result<select::SelectionHandles, Error> {
                self.$first_field.validate()?;
                $(
                    self.$field.validate()?;
                    ensure_same_len(self.$field.len(), self.$first_field.len())?;
                )+
                let $first_field = device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let len = $first_field.len();
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let client = policy.client();
                let flag = client.empty(len * std::mem::size_of::<u32>());
                if len != 0 {
                    let block_count_u32 = api_expr_block_count(len)?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let invert_values = [if invert { 1_u32 } else { 0_u32 }];
                    let invert_handle = client.create_from_slice(u32::as_bytes(&invert_values));
                    unsafe {
                        $kernel_name::launch_unchecked::<
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item, )+
                            Pred,
                            <$first as KernelColumn>::Runtime,
                        >(
                            client,
                            CubeCount::Static(block_count_u32, 1, 1),
                            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
                            unsafe { BufferArg::from_raw_parts($first_field.handle.clone(), len) },
                            $(
                                unsafe { BufferArg::from_raw_parts($field.handle.clone(), len) },
                            )+
                            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                            unsafe { BufferArg::from_raw_parts(invert_handle.clone(), 1) },
                            unsafe { BufferArg::from_raw_parts(flag.clone(), len) },
                        );
                    }
                }
                select::handles_from_flags(
                    policy,
                    len,
                    len_u32,
                    flag,
                    $first_field.handle.clone(),
                )
            }
        }
    };
}

impl_tuple_selection_stencil!(
    SoAView2<A, B> { left, right },
    tuple2_predicate_flags_kernel
);
impl_tuple_selection_stencil!(
    SoAView3<A, B, C> { first, second, third },
    tuple3_predicate_flags_kernel
);

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
