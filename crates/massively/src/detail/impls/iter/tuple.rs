use super::*;

macro_rules! impl_miter_zip {
    ($name:ident => $read:ident; $( $ty:ident : $idx:tt : $tmp:ident ),+ => $transform:ident) => {
        impl<R, $( $ty ),+> MIter<R> for $name<$( $ty ),+>
        where
            R: Runtime,
            $( $ty: MIter<R>, )+
            ($( <$ty as MIter<R>>::Item, )+): MItem<R>,
            $( <$ty as MIter<R>>::Item: Send + Sync, )+
            crate::detail::read::$read<$( <$ty as MIter<R>>::Read ),+>:
                crate::detail::read::KernelRead<R, Item = ($( <$ty as MIter<R>>::Item, )+)>
                    + crate::detail::read::KernelReadAt<
                        R,
                        crate::detail::device::S0,
                        LogicalItem = ($( <$ty as MIter<R>>::Item, )+),
                    >
                    + crate::detail::read::KernelReadBoundMany<
                        R,
                        Item = ($( <$ty as MIter<R>>::Item, )+),
                    >,
            <crate::detail::read::$read<$( <$ty as MIter<R>>::Read ),+> as crate::detail::read::KernelReadAt<
                R,
                crate::detail::device::S0,
            >>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<($( <$ty as MIter<R>>::Item, )+)>
                + crate::expr::LogicalDeviceExpr3<
                    ($( <$ty as MIter<R>>::Item, )+),
                    <<crate::detail::read::$read<$( <$ty as MIter<R>>::Read ),+> as crate::detail::read::KernelReadAt<
                        R,
                        crate::detail::device::S0,
                    >>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<($( <$ty as MIter<R>>::Item, )+)>>::LeafA,
                    <<crate::detail::read::$read<$( <$ty as MIter<R>>::Read ),+> as crate::detail::read::KernelReadAt<
                        R,
                        crate::detail::device::S0,
                    >>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<($( <$ty as MIter<R>>::Item, )+)>>::LeafB,
                    <<crate::detail::read::$read<$( <$ty as MIter<R>>::Read ),+> as crate::detail::read::KernelReadAt<
                        R,
                        crate::detail::device::S0,
                    >>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<($( <$ty as MIter<R>>::Item, )+)>>::LeafC,
                >,
            <<crate::detail::read::$read<$( <$ty as MIter<R>>::Read ),+> as crate::detail::read::KernelReadAt<
                R,
                crate::detail::device::S0,
            >>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<($( <$ty as MIter<R>>::Item, )+)>>::LeafA:
                MStorageElement,
            <<crate::detail::read::$read<$( <$ty as MIter<R>>::Read ),+> as crate::detail::read::KernelReadAt<
                R,
                crate::detail::device::S0,
            >>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<($( <$ty as MIter<R>>::Item, )+)>>::LeafB:
                MStorageElement,
            <<crate::detail::read::$read<$( <$ty as MIter<R>>::Read ),+> as crate::detail::read::KernelReadAt<
                R,
                crate::detail::device::S0,
            >>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<($( <$ty as MIter<R>>::Item, )+)>>::LeafC:
                MStorageElement,
        {
            type Item = ($( <$ty as MIter<R>>::Item, )+);
            type Slice<'a>
                = $name<$( <$ty as MIter<R>>::Slice<'a> ),+>
            where
                Self: 'a;
            type Inner = ($( <$ty as MIter<R>>::Inner, )+);
            type Read = crate::detail::read::$read<$( <$ty as MIter<R>>::Read ),+>;

            fn len(&self) -> MIndex {
                self.0.len()
            }

            fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
            where
                Bounds: std::ops::RangeBounds<MIndex>,
            {
                let range = crate::iter::normalize_zip_range(self.len(), range);
                $name($( self.$idx.slice(range.clone()) ),+)
            }

            fn into_inner(self) -> Self::Inner {
                unreachable!("read-only MIter lowering requires a CubePolicy")
            }

            fn lower_read(self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
                Ok(crate::detail::read::$read::new($( self.$idx.lower_read(policy)? ),+))
            }

            fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
                $(
                    self.$idx.validate_executor(exec)?;
                    ensure_same_len(self.$idx.len(), self.0.len())?;
                )+
                Ok(())
            }

            fn into_inner_with_policy(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<Self::Inner, Error> {
                Ok(($( self.$idx.into_inner_with_policy(policy)?, )+))
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

            fn copy_selected_with_policy<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Self::Item: MAlloc<R>,
                Output: MIterMut<R, Item = Self::Item>,
            {
                let read = self.lower_read(policy)?;
                let inner = crate::detail::read::materialize_logical3_read(read, policy)?;
                let source = crate::iter::materialized_view_with_policy::<R, Self::Item>(
                    policy, inner,
                )?;
                <Self::Item as MAlloc<R>>::copy_selected_from_view(
                    policy, source, stencil, output,
                )
            }

            fn gather_with_policy<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: crate::detail::device::DeviceColumnView<R, MIndex>,
                output: Output,
            ) -> Result<(), Error>
            where
                Self::Item: MAlloc<R>,
                Output: MIterMut<R, Item = Self::Item>,
            {
                let read = self.lower_read(policy)?;
                let inner = crate::detail::read::materialize_logical3_read(read, policy)?;
                let source = crate::iter::materialized_view_with_policy::<R, Self::Item>(
                    policy, inner,
                )?;
                <Self::Item as MAlloc<R>>::gather_from_view(policy, source, indices, output)
            }

            fn gather_where_with_policy<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: crate::detail::device::DeviceColumnView<R, MIndex>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Self::Item: MAlloc<R>,
                Output: MIterMut<R, Item = Self::Item>,
            {
                let read = self.lower_read(policy)?;
                let inner = crate::detail::read::materialize_logical3_read(read, policy)?;
                let source = crate::iter::materialized_view_with_policy::<R, Self::Item>(
                    policy, inner,
                )?;
                <Self::Item as MAlloc<R>>::gather_where_from_view(
                    policy, source, indices, stencil, output,
                )
            }

            fn scatter_with_policy<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: crate::detail::device::DeviceColumnView<R, MIndex>,
                output: Output,
            ) -> Result<(), Error>
            where
                Self::Item: MAlloc<R>,
                Output: MIterMut<R, Item = Self::Item>,
            {
                let read = self.lower_read(policy)?;
                let inner = crate::detail::read::materialize_logical3_read(read, policy)?;
                let source = crate::iter::materialized_view_with_policy::<R, Self::Item>(
                    policy, inner,
                )?;
                <Self::Item as MAlloc<R>>::scatter_from_view(policy, source, indices, output)
            }

            fn scatter_where_with_policy<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: crate::detail::device::DeviceColumnView<R, MIndex>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Self::Item: MAlloc<R>,
                Output: MIterMut<R, Item = Self::Item>,
            {
                let read = self.lower_read(policy)?;
                let inner = crate::detail::read::materialize_logical3_read(read, policy)?;
                let source = crate::iter::materialized_view_with_policy::<R, Self::Item>(
                    policy, inner,
                )?;
                <Self::Item as MAlloc<R>>::scatter_where_from_view(
                    policy, source, indices, stencil, output,
                )
            }
        }
    };
}

macro_rules! impl_miter_mut_zip {
    ($name:ident; $( $ty:ident : $idx:tt ),+) => {
        impl<'a, R, $( $ty ),+> MIterMut<R> for $name<$( DeviceSliceMut<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            ($( $ty, )+): MAlloc<R, Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
                View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Slice<'b>
                = $name<$( crate::runtime::DeviceSlice<'b, R, $ty> ),+>
            where
                Self: 'b;
            type SliceMut<'b>
                = $name<$( DeviceSliceMut<'b, R, $ty> ),+>
            where
                Self: 'b;
            type Inner = ($( crate::detail::device::DeviceColumnMutView<R, $ty>, )+);

            fn len(&self) -> MIndex {
                self.0.len()
            }

            fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
            where
                Bounds: std::ops::RangeBounds<MIndex>,
            {
                let range = crate::iter::normalize_zip_range(self.len(), range);
                $name($( self.$idx.slice(range.clone()) ),+)
            }

            fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
            where
                Bounds: std::ops::RangeBounds<MIndex>,
            {
                let range = crate::iter::normalize_zip_range(self.len(), range);
                $name($( self.$idx.slice_mut(range.clone()) ),+)
            }

            fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
                $(
                    exec.ensure_policy_id(self.$idx.source.inner.policy_id())?;
                )+
                $(
                    ensure_same_len(self.$idx.len(), self.0.len())?;
                )+
                Ok(())
            }

            fn column_mut_view_by_index_inner<U: 'static>(
                &self,
                index: usize,
            ) -> Result<
                Option<crate::detail::device::DeviceColumnMutView<R, U>>,
                Error,
            >
            where
                U: MStorageElement,
            {
                $(
                    if index == $idx {
                        let source = &*self.$idx.source as &dyn Any;
                        let source = match source.downcast_ref::<DeviceVec<R, U>>() {
                            Some(source) => source,
                            None => return Ok(None),
                        };
                        return Ok(Some(crate::detail::device::DeviceColumnMutView::from_slice(
                            &source.inner,
                            usize_from_mindex(self.$idx.offset),
                            usize_from_mindex(self.$idx.len),
                        )));
                    }
                )+
                Ok(None)
            }

            fn into_inner(self) -> Self::Inner {
                ($(
                    crate::detail::device::DeviceColumnMutView::from_slice(
                        &self.$idx.source.inner,
                        usize_from_mindex(self.$idx.offset),
                        usize_from_mindex(self.$idx.len),
                    ),
                )+)
            }

            fn write_from_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                inner: <Self::Item as MAlloc<R>>::Inner,
            ) -> Result<(), Error> {
                let output = <Self as MIterMut<R>>::into_inner(self);
                $(
                    {
                        let input =
                            crate::detail::device::DeviceColumnView::from_column(&inner.$idx);
                        crate::detail::apply::MaterializeWriteApply::new(&output.$idx)
                            .collect_expr(policy, &input)?;
                    }
                )+
                Ok(())
            }

            fn write_prefix_from_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                inner: <Self::Item as MAlloc<R>>::Inner,
            ) -> Result<(), Error> {
                let mut output = <Self as MIterMut<R>>::into_inner(self);
                $(
                    {
                        let input =
                            crate::detail::device::DeviceColumnView::from_column(&inner.$idx);
                        if input.len > output.$idx.len {
                            return Err(Error::LengthMismatch {
                                input: input.len,
                                output: output.$idx.len,
                            });
                        }
                        output.$idx.len = input.len;
                        crate::detail::apply::MaterializeWriteApply::new(&output.$idx)
                            .collect_expr(policy, &input)?;
                    }
                )+
                Ok(())
            }

            fn write_split_from_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                selected: <Self::Item as MAlloc<R>>::Inner,
                rejected: <Self::Item as MAlloc<R>>::Inner,
            ) -> Result<(), Error> {
                let output = <Self as MIterMut<R>>::into_inner(self);
                $(
                    {
                        let selected_input =
                            crate::detail::device::DeviceColumnView::from_column(&selected.$idx);
                        let rejected_input =
                            crate::detail::device::DeviceColumnView::from_column(&rejected.$idx);
                        let input_len = selected_input.len + rejected_input.len;
                        if input_len > output.$idx.len {
                            return Err(Error::LengthMismatch {
                                input: input_len,
                                output: output.$idx.len,
                            });
                        }
                        let mut selected_output = output.$idx.clone();
                        selected_output.len = selected_input.len;
                        crate::detail::apply::MaterializeWriteApply::new(&selected_output)
                            .collect_expr(policy, &selected_input)?;

                        let mut rejected_output = output.$idx.clone();
                        rejected_output.offset += selected_input.len;
                        rejected_output.len = rejected_input.len;
                        crate::detail::apply::MaterializeWriteApply::new(&rejected_output)
                            .collect_expr(policy, &rejected_input)?;
                    }
                )+
                Ok(())
            }

            fn write_where_from_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                inner: <Self::Item as MAlloc<R>>::Inner,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<(), Error>
            {
                let output = <Self as MIterMut<R>>::into_inner(self);
                $(
                    {
                        let input =
                            crate::detail::device::DeviceColumnView::from_column(&inner.$idx);
                        crate::detail::apply::MaterializeWriteApply::new(&output.$idx)
                            .copy_where_expr(
                                policy,
                                &input,
                                &stencil,
                                KernelOp::<R, StencilFlag>::new(),
                            )?;
                    }
                )+
                Ok(())
            }

            fn replace_where_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                replacement: Self::Item,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<(), Error>
            {
                let output = <Self as MIterMut<R>>::into_inner(self);
                let mask = stencil.mask();
                $(
                    crate::detail::apply::MaskWriteApply::new(&mask, &output.$idx)
                        .replace_value(policy, replacement.$idx)?;
                )+
                Ok(())
            }

            fn fill_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                value: Self::Item,
            ) -> Result<(), Error>
            {
                let output = <Self as MIterMut<R>>::into_inner(self);
                $(
                    crate::detail::apply::FillWriteApply::new(&output.$idx)
                        .fill_value(policy, value.$idx)?;
                )+
                Ok(())
            }
        }
    };
}

macro_rules! impl_wide_miter_zip {
    ($name:ident => $read:ident; $selected_apply:ident; $( $ty:ident : $idx:tt : $tmp:ident ),+) => {
        impl<R, $( $ty ),+> MIter<R> for $name<$( $ty ),+>
        where
            R: Runtime,
            $( $ty: MIter<R>, )+
            ($( <$ty as MIter<R>>::Item, )+): MItem<R>,
            crate::detail::read::$read<$( <$ty as MIter<R>>::Read ),+>:
                crate::detail::read::KernelRead<R, Item = ($( <$ty as MIter<R>>::Item, )+)>
                    + crate::detail::read::KernelReadAt<
                        R,
                        crate::detail::device::S0,
                        LogicalItem = ($( <$ty as MIter<R>>::Item, )+),
                    >
                    + crate::detail::read::KernelReadBoundMany<
                        R,
                        Item = ($( <$ty as MIter<R>>::Item, )+),
                    >,
        {
            type Item = ($( <$ty as MIter<R>>::Item, )+);
            type Slice<'a>
                = $name<$( <$ty as MIter<R>>::Slice<'a> ),+>
            where
                Self: 'a;
            type Inner = ($( <$ty as MIter<R>>::Inner, )+);
            type Read = crate::detail::read::$read<$( <$ty as MIter<R>>::Read ),+>;

            fn len(&self) -> MIndex {
                self.0.len()
            }

            fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
            where
                Bounds: std::ops::RangeBounds<MIndex>,
            {
                let range = crate::iter::normalize_zip_range(self.len(), range);
                $name($( self.$idx.slice(range.clone()) ),+)
            }

            fn into_inner(self) -> Self::Inner {
                unreachable!("read-only MIter lowering requires a CubePolicy")
            }

            fn lower_read(self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
                Ok(crate::detail::read::$read::new($( self.$idx.lower_read(policy)? ),+))
            }

            fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
                $(
                    self.$idx.validate_executor(exec)?;
                    ensure_same_len(self.$idx.len(), self.0.len())?;
                )+
                Ok(())
            }

            fn into_inner_with_policy(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<Self::Inner, Error> {
                Ok(($( self.$idx.into_inner_with_policy(policy)?, )+))
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

impl_miter_zip!(Zip2 => ZipRead2; A: 0: a, C: 1: c => transform_binary);
impl_miter_zip!(Zip3 => ZipRead3; A: 0: a, C: 1: c, D: 2: d => transform_ternary);
impl_wide_miter_zip!(Zip4 => ZipRead4; apply_expr4; A: 0: a, C: 1: c, D: 2: d, E: 3: e);
impl_wide_miter_zip!(Zip5 => ZipRead5; apply_expr5; A: 0: a, C: 1: c, D: 2: d, E: 3: e, F: 4: f);
impl_wide_miter_zip!(Zip6 => ZipRead6; apply_expr6; A: 0: a, C: 1: c, D: 2: d, E: 3: e, F: 4: f, G: 5: g);
impl_wide_miter_zip!(Zip7 => ZipRead7; apply_expr7; A: 0: a, C: 1: c, D: 2: d, E: 3: e, F: 4: f, G: 5: g, H: 6: h);
impl_miter_mut_zip!(Zip2; A: 0, C: 1);
impl_miter_mut_zip!(Zip3; A: 0, C: 1, D: 2);
impl_miter_mut_zip!(Zip4; A: 0, C: 1, D: 2, E: 3);
impl_miter_mut_zip!(Zip5; A: 0, C: 1, D: 2, E: 3, F: 4);
impl_miter_mut_zip!(Zip6; A: 0, C: 1, D: 2, E: 3, F: 4, G: 5);
impl_miter_mut_zip!(Zip7; A: 0, C: 1, D: 2, E: 3, F: 4, G: 5, H: 6);
