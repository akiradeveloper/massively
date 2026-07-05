use super::DeviceVec;
use crate::{
    error::{Error, ensure_same_len},
    expr::{DeviceGpuExpr, GpuExpr, Slot0, Slot1, Slot2, Slot3},
    index::usize_from_mindex,
    policy::CubePolicy,
};
use cubecl::prelude::*;

/// Two-component flat device expression.
pub struct Zip2<Left, Right> {
    pub(crate) left: Left,
    pub(crate) right: Right,
}

/// One-component flat device expression.
pub struct Zip1<Source> {
    pub(crate) source: Source,
}

/// One-component read-only virtual device expression.
#[doc(hidden)]
pub struct ZipView1<Source> {
    pub(crate) source: Source,
}

#[doc(hidden)]
pub struct ZipView2<Left, Right> {
    pub(crate) left: Left,
    pub(crate) right: Right,
}

/// Three-component flat device expression.
pub struct Zip3<First, Second, Third> {
    pub(crate) first: First,
    pub(crate) second: Second,
    pub(crate) third: Third,
}

#[doc(hidden)]
pub struct ZipView3<First, Second, Third> {
    pub(crate) first: First,
    pub(crate) second: Second,
    pub(crate) third: Third,
}

#[doc(hidden)]
pub struct ZipView4<A, B, C, D> {
    pub(crate) a: A,
    pub(crate) b: B,
    pub(crate) c: C,
    pub(crate) d: D,
}

#[doc(hidden)]
pub struct ZipView5<A, B, C, D, E> {
    pub(crate) a: A,
    pub(crate) b: B,
    pub(crate) c: C,
    pub(crate) d: D,
    pub(crate) e: E,
}

#[doc(hidden)]
pub struct ZipView6<A, B, C, D, E, F> {
    pub(crate) a: A,
    pub(crate) b: B,
    pub(crate) c: C,
    pub(crate) d: D,
    pub(crate) e: E,
    pub(crate) f: F,
}

#[doc(hidden)]
pub struct ZipView7<A, B, C, D, E, F, G> {
    pub(crate) a: A,
    pub(crate) b: B,
    pub(crate) c: C,
    pub(crate) d: D,
    pub(crate) e: E,
    pub(crate) f: F,
    pub(crate) g: G,
}

#[doc(hidden)]
pub struct S0;
#[doc(hidden)]
pub struct S1;
#[doc(hidden)]
pub struct S2;
#[doc(hidden)]
pub struct S3;
#[doc(hidden)]
pub struct S4;

/// Internal scalar-column expression that can be lowered into GPU kernels.
///
/// This is not a public API concept. Public iterator inputs deal in
/// `DeviceSlice` / `DeviceSliceMut` wrapped by `Zip`; this trait is the private
/// staging layer used by algorithms to pass one or more columns to kernels.
#[doc(hidden)]
pub trait KernelColumn {
    type Runtime: Runtime;
    type Item: CubePrimitive;
    type Expr: GpuExpr<Self::Item> + DeviceGpuExpr<Self::Item>;

    fn len(&self) -> usize;
    fn validate(&self) -> Result<(), Error>;

    fn stage(&self, policy: &CubePolicy<Self::Runtime>) -> Result<KernelColumnBindings, Error>
    where
        Self: KernelColumnAt<S0>,
    {
        let mut bindings = KernelColumnBindings::empty(policy.client());
        <Self as KernelColumnAt<S0>>::stage_at(self, &mut bindings)?;
        bindings.finish();
        Ok(bindings)
    }
}

/// Internal shorthand for storage-backed columns that can be staged for kernels.
///
/// This includes both owned `DeviceVec` outputs being materialized internally
/// and borrowed `&DeviceVec` public inputs.
pub(crate) trait StorageKernelColumn: KernelColumn {}

/// Internal shorthand for public algorithm inputs that must be borrowed.
pub(crate) trait ReadOnlyKernelColumn: StorageKernelColumn {}

/// Read-only view over one storage-backed device column.
///
/// This is the internal counterpart of public `DeviceSlice`. It carries
/// storage-local lowering metadata; the common iterator abstraction remains
/// logical `i -> T`.
#[doc(hidden)]
#[derive(Clone)]
pub struct DeviceColumnView<R: Runtime, T> {
    pub(crate) source: DeviceVec<R, T>,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

impl<R, T> DeviceColumnView<R, T>
where
    R: Runtime,
{
    pub(crate) fn from_column(source: &DeviceVec<R, T>) -> Self {
        Self {
            source: DeviceVec::from_handle(source.policy_id(), source.handle.clone(), source.len()),
            offset: 0,
            len: source.len(),
        }
    }

    pub(crate) fn from_slice(source: &DeviceVec<R, T>, offset: usize, len: usize) -> Self {
        Self {
            source: DeviceVec::from_handle(source.policy_id(), source.handle.clone(), source.len()),
            offset,
            len,
        }
    }
}

/// Writable view over one storage-backed device column.
#[doc(hidden)]
#[derive(Clone)]
pub struct DeviceColumnMutView<R: Runtime, T> {
    pub(crate) source: DeviceVec<R, T>,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

impl<R, T> DeviceColumnMutView<R, T>
where
    R: Runtime,
{
    pub(crate) fn from_slice(source: &DeviceVec<R, T>, offset: usize, len: usize) -> Self {
        Self {
            source: DeviceVec::from_handle(source.policy_id(), source.handle.clone(), source.len()),
            offset,
            len,
        }
    }
}

impl<R, T> ReadOnlyZip for DeviceColumnView<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type Runtime = R;
    type Item = (T,);
    type Scalar = T;

    fn len(&self) -> usize {
        self.len
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }
}

/// Internal read-only Zip compatibility layer.
///
/// Public API terminology is `Zip`; this trait remains as an implementation
/// detail for virtual/read-only expression inputs.
pub(crate) trait ReadOnlyZip {
    type Runtime: Runtime;
    type Item;
    type Scalar;

    fn len(&self) -> usize;
    fn validate(&self) -> Result<(), Error>;
}

/// Internal storage-backed Zip value.
pub trait Zip {
    type Runtime: Runtime;
    type Item;
    type Scalar;

    fn len(&self) -> usize;
    fn validate(&self) -> Result<(), Error>;
}

impl<Source> ReadOnlyZip for Source
where
    Source: Zip,
{
    type Runtime = Source::Runtime;
    type Item = Source::Item;
    type Scalar = Source::Scalar;

    fn len(&self) -> usize {
        Zip::len(self)
    }

    fn validate(&self) -> Result<(), Error> {
        Zip::validate(self)
    }
}

impl<R, T> Zip for DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type Runtime = R;
    type Item = (T,);
    type Scalar = T;

    fn len(&self) -> usize {
        DeviceVec::len(self)
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl<R, T> ReadOnlyZip for &DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type Runtime = R;
    type Item = (T,);
    type Scalar = T;

    fn len(&self) -> usize {
        DeviceVec::len(self)
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct KernelColumnBindings {
    pub(crate) input: cubecl::server::Handle,
    pub(crate) input_len: usize,
    pub(crate) input_offset: usize,
    pub(crate) rhs: cubecl::server::Handle,
    pub(crate) rhs_len: usize,
    pub(crate) rhs_offset: usize,
    pub(crate) slots: Vec<(cubecl::server::Handle, usize)>,
    pub(crate) slot_offsets: Vec<usize>,
}

impl<Left, Right> ReadOnlyZip for ZipView2<Left, Right>
where
    Left: KernelColumn
        + KernelColumnAt<S0>
        + ReadOnlyZip<
            Runtime = <Left as KernelColumn>::Runtime,
            Item = (<Left as KernelColumn>::Item,),
        >,
    Right: KernelColumn<Runtime = <Left as KernelColumn>::Runtime>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>
        + ReadOnlyZip,
    <Left as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Right as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Runtime = <Left as KernelColumn>::Runtime;
    type Item = (<Left as KernelColumn>::Item, <Right as KernelColumn>::Item);
    type Scalar = <Left as KernelColumn>::Item;

    fn len(&self) -> usize {
        KernelColumn::len(&self.left)
    }

    fn validate(&self) -> Result<(), Error> {
        KernelColumn::validate(&self.left)?;
        KernelColumn::validate(&self.right)?;
        ensure_same_len(
            KernelColumn::len(&self.right),
            KernelColumn::len(&self.left),
        )?;
        Ok(())
    }
}

impl<Left, Right> Zip for Zip2<Left, Right>
where
    Left: Zip + KernelColumn,
    Right: Zip<Runtime = <Left as Zip>::Runtime>
        + KernelColumn<Runtime = <Left as KernelColumn>::Runtime>,
{
    type Runtime = <Left as Zip>::Runtime;
    type Item = (<Left as KernelColumn>::Item, <Right as KernelColumn>::Item);
    type Scalar = Left::Scalar;

    fn len(&self) -> usize {
        Zip::len(&self.left)
    }

    fn validate(&self) -> Result<(), Error> {
        Zip::validate(&self.left)?;
        Zip::validate(&self.right)?;
        ensure_same_len(Zip::len(&self.right), Zip::len(&self.left))?;
        Ok(())
    }
}

impl<Source> ReadOnlyZip for ZipView1<Source>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    <Source as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Runtime = <Source as KernelColumn>::Runtime;
    type Item = (<Source as KernelColumn>::Item,);
    type Scalar = <Source as KernelColumn>::Item;

    fn len(&self) -> usize {
        KernelColumn::len(&self.source)
    }

    fn validate(&self) -> Result<(), Error> {
        KernelColumn::validate(&self.source)
    }
}

impl<Source> Zip for Zip1<Source>
where
    Source: StorageKernelColumn + KernelColumnAt<S0>,
{
    type Runtime = Source::Runtime;
    type Item = (Source::Item,);
    type Scalar = Source::Item;

    fn len(&self) -> usize {
        KernelColumn::len(&self.source)
    }

    fn validate(&self) -> Result<(), Error> {
        KernelColumn::validate(&self.source)
    }
}

impl<Left, Right, Start> KernelColumnAt<Start> for Zip2<Left, Right>
where
    Left: KernelColumn + KernelColumnAt<S0> + KernelColumnAt<Start>,
    Right: KernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>
        + KernelColumnAt<<Left as KernelColumnAt<Start>>::Next>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
{
    type ExprAt = (
        <Left as KernelColumnAt<Start>>::ExprAt,
        <Right as KernelColumnAt<<Left as KernelColumnAt<Start>>::Next>>::ExprAt,
    );
    type Next = <Right as KernelColumnAt<<Left as KernelColumnAt<Start>>::Next>>::Next;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        <Left as KernelColumnAt<Start>>::stage_at(&self.left, bindings)?;
        <Right as KernelColumnAt<<Left as KernelColumnAt<Start>>::Next>>::stage_at(
            &self.right,
            bindings,
        )
    }
}

impl<First, Second, Third> ReadOnlyZip for ZipView3<First, Second, Third>
where
    First: KernelColumn
        + KernelColumnAt<S0>
        + ReadOnlyZip<
            Runtime = <First as KernelColumn>::Runtime,
            Item = (<First as KernelColumn>::Item,),
        >,
    Second: KernelColumn<Runtime = <First as KernelColumn>::Runtime>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>
        + ReadOnlyZip,
    Third: KernelColumn<Runtime = <First as KernelColumn>::Runtime>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>
        + ReadOnlyZip,
    <First as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Second as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Third as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Runtime = <First as KernelColumn>::Runtime;
    type Item = (
        <First as KernelColumn>::Item,
        <Second as KernelColumn>::Item,
        <Third as KernelColumn>::Item,
    );
    type Scalar = <First as KernelColumn>::Item;

    fn len(&self) -> usize {
        KernelColumn::len(&self.first)
    }

    fn validate(&self) -> Result<(), Error> {
        KernelColumn::validate(&self.first)?;
        KernelColumn::validate(&self.second)?;
        KernelColumn::validate(&self.third)?;
        ensure_same_len(
            KernelColumn::len(&self.second),
            KernelColumn::len(&self.first),
        )?;
        ensure_same_len(
            KernelColumn::len(&self.third),
            KernelColumn::len(&self.first),
        )?;
        Ok(())
    }
}

macro_rules! impl_read_only_wide_zip_view {
    ($name:ident < $first:ident : $first_field:ident, $( $ty:ident : $field:ident ),+ > => ($($item:ty),+)) => {
        impl<$first, $( $ty ),+> ReadOnlyZip for $name<$first, $( $ty ),+>
        where
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $ty: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$ty as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Item = ($($item,)+);
            type Scalar = <$first as KernelColumn>::Item;

            fn len(&self) -> usize {
                KernelColumn::len(&self.$first_field)
            }

            fn validate(&self) -> Result<(), Error> {
                KernelColumn::validate(&self.$first_field)?;
                $(
                    KernelColumn::validate(&self.$field)?;
                    ensure_same_len(
                        KernelColumn::len(&self.$field),
                        KernelColumn::len(&self.$first_field),
                    )?;
                )+
                Ok(())
            }
        }
    };
}

impl_read_only_wide_zip_view!(ZipView4<A: a, B: b, C: c, D: d> => (
    <A as KernelColumn>::Item,
    <B as KernelColumn>::Item,
    <C as KernelColumn>::Item,
    <D as KernelColumn>::Item
));
impl_read_only_wide_zip_view!(ZipView5<A: a, B: b, C: c, D: d, E: e> => (
    <A as KernelColumn>::Item,
    <B as KernelColumn>::Item,
    <C as KernelColumn>::Item,
    <D as KernelColumn>::Item,
    <E as KernelColumn>::Item
));
impl_read_only_wide_zip_view!(ZipView6<A: a, B: b, C: c, D: d, E: e, F: f> => (
    <A as KernelColumn>::Item,
    <B as KernelColumn>::Item,
    <C as KernelColumn>::Item,
    <D as KernelColumn>::Item,
    <E as KernelColumn>::Item,
    <F as KernelColumn>::Item
));
impl_read_only_wide_zip_view!(ZipView7<A: a, B: b, C: c, D: d, E: e, F: f, G: g> => (
    <A as KernelColumn>::Item,
    <B as KernelColumn>::Item,
    <C as KernelColumn>::Item,
    <D as KernelColumn>::Item,
    <E as KernelColumn>::Item,
    <F as KernelColumn>::Item,
    <G as KernelColumn>::Item
));

impl<First, Second, Third> Zip for Zip3<First, Second, Third>
where
    First: Zip + KernelColumn,
    Second: Zip<Runtime = <First as Zip>::Runtime>
        + KernelColumn<Runtime = <First as KernelColumn>::Runtime>,
    Third: Zip<Runtime = <First as Zip>::Runtime>
        + KernelColumn<Runtime = <First as KernelColumn>::Runtime>,
{
    type Runtime = <First as Zip>::Runtime;
    type Item = (
        <First as KernelColumn>::Item,
        <Second as KernelColumn>::Item,
        <Third as KernelColumn>::Item,
    );
    type Scalar = First::Scalar;

    fn len(&self) -> usize {
        Zip::len(&self.first)
    }

    fn validate(&self) -> Result<(), Error> {
        Zip::validate(&self.first)?;
        Zip::validate(&self.second)?;
        Zip::validate(&self.third)?;
        ensure_same_len(Zip::len(&self.second), Zip::len(&self.first))?;
        ensure_same_len(Zip::len(&self.third), Zip::len(&self.first))?;
        Ok(())
    }
}

impl<First, Second, Third, Start> KernelColumnAt<Start> for Zip3<First, Second, Third>
where
    First: KernelColumn + KernelColumnAt<S0> + KernelColumnAt<Start>,
    Second: KernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>
        + KernelColumnAt<<First as KernelColumnAt<Start>>::Next>,
    Third: KernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<Start>>::Next>>::Next>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
{
    type ExprAt = (
        <First as KernelColumnAt<Start>>::ExprAt,
        <Second as KernelColumnAt<<First as KernelColumnAt<Start>>::Next>>::ExprAt,
        <Third as KernelColumnAt<
            <Second as KernelColumnAt<<First as KernelColumnAt<Start>>::Next>>::Next,
        >>::ExprAt,
    );
    type Next = <Third as KernelColumnAt<
        <Second as KernelColumnAt<<First as KernelColumnAt<Start>>::Next>>::Next,
    >>::Next;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        <First as KernelColumnAt<Start>>::stage_at(&self.first, bindings)?;
        <Second as KernelColumnAt<<First as KernelColumnAt<Start>>::Next>>::stage_at(
            &self.second,
            bindings,
        )?;
        <Third as KernelColumnAt<
            <Second as KernelColumnAt<<First as KernelColumnAt<Start>>::Next>>::Next,
        >>::stage_at(&self.third, bindings)
    }
}

impl KernelColumnBindings {
    pub(crate) fn empty<R: Runtime>(client: &ComputeClient<R>) -> Self {
        Self {
            input: crate::policy::empty_handle(client),
            input_len: 0,
            input_offset: 0,
            rhs: crate::policy::empty_handle(client),
            rhs_len: 0,
            rhs_offset: 0,
            slots: Vec::new(),
            slot_offsets: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, handle: cubecl::server::Handle, len: usize) {
        self.push_with_offset(handle, len, 0);
    }

    pub(crate) fn push_with_offset(
        &mut self,
        handle: cubecl::server::Handle,
        len: usize,
        offset: usize,
    ) {
        self.slots.push((handle, len));
        self.slot_offsets.push(offset);
    }

    fn finish(&mut self) {
        if self.slots.is_empty() {
            self.slots.push((self.input.clone(), self.input_len));
            self.slot_offsets.push(self.input_offset);
        }
        if let Some((handle, len)) = self.slots.first() {
            self.input = handle.clone();
            self.input_len = *len;
            self.input_offset = self.slot_offsets[0];
        }
        if let Some((handle, len)) = self.slots.get(1) {
            self.rhs = handle.clone();
            self.rhs_len = *len;
            self.rhs_offset = self.slot_offsets[1];
        } else {
            self.rhs = self.input.clone();
            self.rhs_len = self.input_len;
            self.rhs_offset = self.input_offset;
        }
    }

    pub(crate) fn slot_offsets_handle<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
    ) -> Result<cubecl::server::Handle, Error> {
        let mut offsets = [0_u32; 4];
        for (index, offset) in self.slot_offsets.iter().take(4).enumerate() {
            offsets[index] = crate::index::mindex_from_usize(*offset)?;
        }
        Ok(client.create_from_slice(u32::as_bytes(&offsets)))
    }

    pub(crate) fn slot_or_first(&self, index: usize) -> &(cubecl::server::Handle, usize) {
        let first = self
            .slots
            .first()
            .expect("kernel column has at least one slot");
        self.slots.get(index).unwrap_or(first)
    }
}

#[doc(hidden)]
pub trait KernelColumnAt<Start> {
    type ExprAt;
    type Next;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error>;
}

impl<'a, R, T> KernelColumn for &'a DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive,
{
    type Runtime = R;
    type Item = T;
    type Expr = Slot0<T>;

    fn len(&self) -> usize {
        usize_from_mindex(self.len)
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl<'a, R, T> StorageKernelColumn for &'a DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive,
{
}
impl<'a, R, T> ReadOnlyKernelColumn for &'a DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive,
{
}

impl<'a, R, T> KernelColumnAt<S0> for &'a DeviceVec<R, T>
where
    R: Runtime,
{
    type ExprAt = Slot0<T>;
    type Next = S1;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push(self.handle.clone(), usize_from_mindex(self.len));
        Ok(())
    }
}

impl<'a, R, T> KernelColumnAt<S1> for &'a DeviceVec<R, T>
where
    R: Runtime,
{
    type ExprAt = Slot1<T>;
    type Next = S2;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push(self.handle.clone(), usize_from_mindex(self.len));
        Ok(())
    }
}

impl<'a, R, T> KernelColumnAt<S2> for &'a DeviceVec<R, T>
where
    R: Runtime,
{
    type ExprAt = Slot2<T>;
    type Next = S3;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push(self.handle.clone(), usize_from_mindex(self.len));
        Ok(())
    }
}

impl<'a, R, T> KernelColumnAt<S3> for &'a DeviceVec<R, T>
where
    R: Runtime,
{
    type ExprAt = Slot3<T>;
    type Next = S4;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push(self.handle.clone(), usize_from_mindex(self.len));
        Ok(())
    }
}

impl<R, T> KernelColumn for DeviceColumnView<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type Runtime = R;
    type Item = T;
    type Expr = Slot0<T>;

    fn len(&self) -> usize {
        self.len
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl<R, T> ReadOnlyKernelColumn for DeviceColumnView<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
}

impl<R, T> StorageKernelColumn for DeviceColumnView<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
}

impl<R, T> KernelColumnAt<S0> for DeviceColumnView<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type ExprAt = Slot0<T>;
    type Next = S1;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push_with_offset(self.source.handle.clone(), self.source.len(), self.offset);
        Ok(())
    }
}

impl<R, T> KernelColumnAt<S1> for DeviceColumnView<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type ExprAt = Slot1<T>;
    type Next = S2;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push_with_offset(self.source.handle.clone(), self.source.len(), self.offset);
        Ok(())
    }
}

impl<R, T> KernelColumnAt<S2> for DeviceColumnView<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type ExprAt = Slot2<T>;
    type Next = S3;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push_with_offset(self.source.handle.clone(), self.source.len(), self.offset);
        Ok(())
    }
}

impl<R, T> KernelColumnAt<S3> for DeviceColumnView<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type ExprAt = Slot3<T>;
    type Next = S4;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push_with_offset(self.source.handle.clone(), self.source.len(), self.offset);
        Ok(())
    }
}

impl<R, T> KernelColumn for DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive,
{
    type Runtime = R;
    type Item = T;
    type Expr = Slot0<T>;

    fn len(&self) -> usize {
        usize_from_mindex(self.len)
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl<R, T> StorageKernelColumn for DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive,
{
}

impl<R, T> KernelColumnAt<S0> for DeviceVec<R, T>
where
    R: Runtime,
{
    type ExprAt = Slot0<T>;
    type Next = S1;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push(self.handle.clone(), usize_from_mindex(self.len));
        Ok(())
    }
}

impl<R, T> KernelColumnAt<S1> for DeviceVec<R, T>
where
    R: Runtime,
{
    type ExprAt = Slot1<T>;
    type Next = S2;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push(self.handle.clone(), usize_from_mindex(self.len));
        Ok(())
    }
}

impl<R, T> KernelColumnAt<S2> for DeviceVec<R, T>
where
    R: Runtime,
{
    type ExprAt = Slot2<T>;
    type Next = S3;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push(self.handle.clone(), usize_from_mindex(self.len));
        Ok(())
    }
}

impl<R, T> KernelColumnAt<S3> for DeviceVec<R, T>
where
    R: Runtime,
{
    type ExprAt = Slot3<T>;
    type Next = S4;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push(self.handle.clone(), usize_from_mindex(self.len));
        Ok(())
    }
}
