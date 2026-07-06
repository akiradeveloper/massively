use super::*;

impl<'a, R, T> MIter<R> for crate::runtime::DeviceSlice<'a, R, T>
where
    R: Runtime,
    T: MStorageElement + 'static,
{
    type Item = T;
    type Slice<'b>
        = crate::runtime::DeviceSlice<'b, R, T>
    where
        Self: 'b;
    type Inner = crate::detail::device::DeviceColumnView<R, T>;
    type Read = crate::detail::read::ColumnRead<R, T>;

    fn len(&self) -> MIndex {
        self.len()
    }

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
    where
        Bounds: std::ops::RangeBounds<MIndex>,
    {
        crate::runtime::DeviceSlice::slice(self, range)
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("read-only MIter lowering requires a CubePolicy")
    }

    fn lower_read(self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        Ok(crate::detail::read::ColumnRead::new(
            self.into_inner_with_policy(policy)?,
        ))
    }

    fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
        exec.ensure_policy_id(self.source.inner.policy_id())
    }

    fn into_inner_with_policy(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<Self::Inner, Error> {
        let _ = policy;
        Ok(self.slice(..).column_view())
    }

    fn into_alloc_view_with_policy(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<<Self::Item as MAlloc<R>>::View, Error>
    where
        Self::Item: MAlloc<R>,
    {
        let _ = policy;
        let view = self.slice(..).column_view();
        if std::mem::size_of::<Self::Inner>()
            != std::mem::size_of::<<Self::Item as MAlloc<R>>::View>()
            || std::mem::align_of::<Self::Inner>()
                != std::mem::align_of::<<Self::Item as MAlloc<R>>::View>()
        {
            return Err(Error::Launch {
                message: "alloc view lowering is not supported for this iterator shape".to_string(),
            });
        }
        let alloc_view = unsafe { std::mem::transmute_copy(&view) };
        std::mem::forget(view);
        Ok(alloc_view)
    }

    fn stencil_selection_with_policy(
        self,
        policy: &crate::detail::CubePolicy<R>,
        invert: bool,
        flags_only: bool,
    ) -> Result<crate::detail::api::PrecomputedSelection<R>, Error>
    where
        Self: MIter<R, Item = u32>,
    {
        let stencil = self.aux_u32_column_view();
        if flags_only {
            crate::detail::api::PrecomputedSelection::from_stencil_flags_with_policy::<
                _,
                KernelOp<R, StencilFlag>,
            >(policy, &(stencil,), invert)
        } else {
            crate::detail::api::PrecomputedSelection::from_stencil_with_policy::<
                _,
                KernelOp<R, StencilFlag>,
            >(policy, &(stencil,), invert)
        }
    }

    fn index_column_with_policy(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<crate::detail::device::DeviceColumnView<R, MIndex>, Error>
    where
        Self: MIter<R, Item = MIndex>,
    {
        let _ = policy;
        Ok(self.aux_u32_column_view())
    }
}

macro_rules! impl_single_zip_miter {
    ($name:ident) => {
        impl<R, Source> MIter<R> for $name<Source>
        where
            R: Runtime,
            Source: MIter<R>,
            (Source::Item,): MItem<R>,
            crate::detail::read::ZipRead1<Source::Read>:
                crate::detail::read::KernelRead<R, Item = (Source::Item,)>
                    + crate::detail::read::KernelReadAt<
                        R,
                        crate::detail::device::S0,
                        LogicalItem = (Source::Item,),
                    >
                    + crate::detail::read::KernelReadBoundMany<R, Item = (Source::Item,)>,
        {
            type Item = (Source::Item,);
            type Slice<'a>
                = $name<Source::Slice<'a>>
            where
                Self: 'a;
            type Inner = (Source::Inner,);
            type Read = crate::detail::read::ZipRead1<Source::Read>;

    fn len(&self) -> MIndex {
        self.0.len()
    }

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
    where
        Bounds: std::ops::RangeBounds<MIndex>,
    {
        $name(self.0.slice(range))
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("read-only MIter lowering requires a CubePolicy")
    }

    fn lower_read(self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        Ok(crate::detail::read::ZipRead1::new(self.0.lower_read(policy)?))
    }

    fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
        self.0.validate_executor(exec)
    }

    fn into_inner_with_policy(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<Self::Inner, Error> {
        Ok((self.0.into_inner_with_policy(policy)?,))
    }

    fn into_alloc_view_with_policy(
        self,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<<Self::Item as MAlloc<R>>::View, Error>
    where
        Self::Item: MAlloc<R>,
    {
        let inner = self.into_inner_with_policy(policy)?;
        if std::mem::size_of::<Self::Inner>()
            != std::mem::size_of::<<Self::Item as MAlloc<R>>::View>()
            || std::mem::align_of::<Self::Inner>()
                != std::mem::align_of::<<Self::Item as MAlloc<R>>::View>()
        {
            return Err(Error::Launch {
                message: "alloc view lowering is not supported for this iterator shape".to_string(),
            });
        }
        let view = unsafe { std::mem::transmute_copy(&inner) };
        std::mem::forget(inner);
        Ok(view)
    }
}
    };
}

impl_single_zip_miter!(Zip1);

impl<'a, R, T> MIterMut<R> for Zip1<DeviceSliceMut<'a, R, T>>
where
    R: Runtime,
    T: MStorageElement + 'static,
    (T,): MAlloc<R, Inner = (crate::detail::DeviceVec<R, T>,)>,
{
    type Item = (T,);
    type Slice<'b>
        = Zip1<crate::runtime::DeviceSlice<'b, R, T>>
    where
        Self: 'b;
    type SliceMut<'b>
        = Zip1<DeviceSliceMut<'b, R, T>>
    where
        Self: 'b;
    type Inner = (crate::detail::device::DeviceColumnMutView<R, T>,);

    fn len(&self) -> MIndex {
        self.0.len()
    }

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
    where
        Bounds: std::ops::RangeBounds<MIndex>,
    {
        Zip1(self.0.slice(range))
    }

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
    where
        Bounds: std::ops::RangeBounds<MIndex>,
    {
        Zip1(self.0.slice_mut(range))
    }

    fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
        exec.ensure_policy_id(self.0.source.inner.policy_id())
    }

    fn column_mut_view_inner<U: 'static>(
        &self,
    ) -> Result<Option<crate::detail::device::DeviceColumnMutView<R, U>>, Error>
    where
        U: MStorageElement,
    {
        let source = &*self.0.source as &dyn Any;
        let source = match source.downcast_ref::<DeviceVec<R, U>>() {
            Some(source) => source,
            None => return Ok(None),
        };
        Ok(Some(
            crate::detail::device::DeviceColumnMutView::from_slice(
                &source.inner,
                usize_from_mindex(self.0.offset),
                usize_from_mindex(self.0.len),
            ),
        ))
    }

    fn into_inner(self) -> Self::Inner {
        (crate::detail::device::DeviceColumnMutView::from_slice(
            &self.0.source.inner,
            usize_from_mindex(self.0.offset),
            usize_from_mindex(self.0.len),
        ),)
    }

    fn write_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error> {
        let output = <Self as MIterMut<R>>::into_inner(self).0;
        let input = crate::detail::device::DeviceColumnView::from_column(&inner.0);
        crate::detail::apply::MaterializeWriteApply::new(&output).collect_expr(policy, &input)
    }

    fn write_prefix_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error> {
        let mut output = <Self as MIterMut<R>>::into_inner(self).0;
        let input = crate::detail::device::DeviceColumnView::from_column(&inner.0);
        if input.len > output.len {
            return Err(Error::LengthMismatch {
                input: input.len,
                output: output.len,
            });
        }
        output.len = input.len;
        crate::detail::apply::MaterializeWriteApply::new(&output).collect_expr(policy, &input)
    }

    fn write_split_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        selected: <Self::Item as MAlloc<R>>::Inner,
        rejected: <Self::Item as MAlloc<R>>::Inner,
    ) -> Result<(), Error> {
        let output = <Self as MIterMut<R>>::into_inner(self).0;
        let selected_input = crate::detail::device::DeviceColumnView::from_column(&selected.0);
        let rejected_input = crate::detail::device::DeviceColumnView::from_column(&rejected.0);
        let input_len = selected_input.len + rejected_input.len;
        if input_len > output.len {
            return Err(Error::LengthMismatch {
                input: input_len,
                output: output.len,
            });
        }
        let mut selected_output = output.clone();
        selected_output.len = selected_input.len;
        crate::detail::apply::MaterializeWriteApply::new(&selected_output)
            .collect_expr(policy, &selected_input)?;

        let mut rejected_output = output;
        rejected_output.offset += selected_input.len;
        rejected_output.len = rejected_input.len;
        crate::detail::apply::MaterializeWriteApply::new(&rejected_output)
            .collect_expr(policy, &rejected_input)
    }

    fn write_where_from_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        inner: <Self::Item as MAlloc<R>>::Inner,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error> {
        let output = <Self as MIterMut<R>>::into_inner(self).0;
        let input = crate::detail::device::DeviceColumnView::from_column(&inner.0);
        crate::detail::apply::MaterializeWriteApply::new(&output).copy_where_expr(
            policy,
            &input,
            &stencil,
            KernelOp::<R, StencilFlag>::new(),
        )
    }

    fn replace_where_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        replacement: Self::Item,
        stencil: crate::detail::api::PrecomputedSelection<R>,
    ) -> Result<(), Error> {
        let output = <Self as MIterMut<R>>::into_inner(self).0;
        let mask = stencil.mask();
        crate::detail::apply::MaskWriteApply::new(&mask, &output)
            .replace_value(policy, replacement.0)
    }

    fn fill_inner(
        self,
        policy: &crate::detail::CubePolicy<R>,
        value: Self::Item,
    ) -> Result<(), Error> {
        let output = <Self as MIterMut<R>>::into_inner(self).0;
        crate::detail::apply::FillWriteApply::new(&output).fill_value(policy, value.0)
    }
}
