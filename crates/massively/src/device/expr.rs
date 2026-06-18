use super::DeviceVec;
use crate::{
    error::{Error, ensure_same_len},
    expr::{BinaryMap, Slot0, Slot1, Slot2, Slot3},
    policy::CubePolicy,
};
use cubecl::prelude::*;
use std::marker::PhantomData;

/// Binary transform expression used by fused kernels.
pub struct DeviceBinaryMap<Left, Right, Op> {
    pub(crate) left: Left,
    pub(crate) right: Right,
    pub(crate) _op: PhantomData<fn() -> Op>,
}

/// Two-component flat device expression.
pub struct SoA2<Left, Right> {
    pub(crate) left: Left,
    pub(crate) right: Right,
}

/// One-component flat device expression.
pub struct SoA1<Source> {
    pub(crate) source: Source,
}

/// One-component read-only virtual device expression.
#[doc(hidden)]
pub struct SoAView1<Source> {
    pub(crate) source: Source,
}

#[doc(hidden)]
pub struct SoAView2<Left, Right> {
    pub(crate) left: Left,
    pub(crate) right: Right,
}

/// Three-component flat device expression.
pub struct SoA3<First, Second, Third> {
    pub(crate) first: First,
    pub(crate) second: Second,
    pub(crate) third: Third,
}

#[doc(hidden)]
pub struct SoAView3<First, Second, Third> {
    pub(crate) first: First,
    pub(crate) second: Second,
    pub(crate) third: Third,
}

macro_rules! define_device_soa {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        /// Flat tuple of device expressions.
        pub struct $name<$first, $( $rest ),+> {
            pub(crate) $first_field: $first,
            $( pub(crate) $field: $rest, )+
        }

        impl<$first, $( $rest ),+> SoA for $name<$first, $( $rest ),+>
        where
            $first: SoA,
            $(
                $rest: SoA<Runtime = <$first as SoA>::Runtime>,
            )+
        {
            type Runtime = <$first as SoA>::Runtime;
            type Item = (<$first as SoA>::Item, $( <$rest as SoA>::Item ),+);
            type Scalar = <$first as SoA>::Scalar;

            fn policy(&self) -> &CubePolicy<Self::Runtime> {
                self.$first_field.policy()
            }

            fn len(&self) -> usize {
                self.$first_field.len()
            }

            fn validate(&self) -> Result<(), Error> {
                self.$first_field.validate()?;
                $(
                    self.$field.validate()?;
                    ensure_same_len(self.$field.len(), self.$first_field.len())?;
                )+
                Ok(())
            }
        }
    };
}

define_device_soa!(SoA4<A, B, C, D> { a, b, c, d });
define_device_soa!(SoA5<A, B, C, D, E> { a, b, c, d, e });
define_device_soa!(SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
define_device_soa!(SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
define_device_soa!(SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
define_device_soa!(SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
define_device_soa!(SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
define_device_soa!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
define_device_soa!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_soa_as_readonly {
    ($name:ident < $( $ty:ident ),+ >) => {
        impl<$( $ty ),+> ReadOnlySoA for $name<$( $ty ),+>
        where
            Self: SoA,
        {
            type Runtime = <Self as SoA>::Runtime;
            type Item = <Self as SoA>::Item;
            type Scalar = <Self as SoA>::Scalar;

            fn policy(&self) -> &CubePolicy<Self::Runtime> {
                SoA::policy(self)
            }

            fn len(&self) -> usize {
                SoA::len(self)
            }

            fn validate(&self) -> Result<(), Error> {
                SoA::validate(self)
            }
        }
    };
}

impl_soa_as_readonly!(SoA4<A, B, C, D>);
impl_soa_as_readonly!(SoA5<A, B, C, D, E>);
impl_soa_as_readonly!(SoA6<A, B, C, D, E, F>);
impl_soa_as_readonly!(SoA7<A, B, C, D, E, F, G>);
impl_soa_as_readonly!(SoA8<A, B, C, D, E, F, G, H>);
impl_soa_as_readonly!(SoA9<A, B, C, D, E, F, G, H, I>);
impl_soa_as_readonly!(SoA10<A, B, C, D, E, F, G, H, I, J>);
impl_soa_as_readonly!(SoA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_soa_as_readonly!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L>);

macro_rules! define_device_soa_view {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        #[doc(hidden)]
        pub struct $name<$first, $( $rest ),+> {
            pub(crate) $first_field: $first,
            $( pub(crate) $field: $rest, )+
        }

        impl<$first, $( $rest ),+> ReadOnlySoA for $name<$first, $( $rest ),+>
        where
            $first: KernelColumn + ReadOnlySoA<Runtime = <$first as KernelColumn>::Runtime, Item = <$first as KernelColumn>::Item>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + ReadOnlySoA,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
        {
            type Runtime = <$first as KernelColumn>::Runtime;
            type Item = (<$first as KernelColumn>::Item, $( <$rest as KernelColumn>::Item ),+);
            type Scalar = <$first as KernelColumn>::Item;

            fn policy(&self) -> &CubePolicy<Self::Runtime> {
                KernelColumn::policy(&self.$first_field)
            }

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

define_device_soa_view!(SoAView4<A, B, C, D> { a, b, c, d });
define_device_soa_view!(SoAView5<A, B, C, D, E> { a, b, c, d, e });
define_device_soa_view!(SoAView6<A, B, C, D, E, F> { a, b, c, d, e, f });
define_device_soa_view!(SoAView7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
define_device_soa_view!(SoAView8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
define_device_soa_view!(SoAView9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
define_device_soa_view!(SoAView10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
define_device_soa_view!(SoAView11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
define_device_soa_view!(SoAView12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

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
/// This is not a public API concept. Public code deals in `DeviceVec`, `zip`,
/// and `SoA`; this trait is the private staging layer used by algorithms to pass
/// one or more columns to kernels.
#[doc(hidden)]
pub trait KernelColumn {
    type Runtime: Runtime;
    type Item;
    type Expr;

    fn policy(&self) -> &CubePolicy<Self::Runtime>;
    fn len(&self) -> usize;
    fn validate(&self) -> Result<(), Error>;

    fn staged_value_handle(
        &self,
        _bindings: &KernelColumnBindings,
    ) -> Option<cubecl::server::Handle> {
        None
    }

    fn stage(&self) -> Result<KernelColumnBindings, Error>
    where
        Self: KernelColumnAt<S0>,
    {
        let mut bindings = KernelColumnBindings::empty(KernelColumn::policy(self).client());
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

/// Internal read-only SoA compatibility layer.
///
/// Public API terminology is `SoA`; this trait remains as an implementation
/// detail for virtual/read-only expression inputs.
pub(crate) trait ReadOnlySoA {
    type Runtime: Runtime;
    type Item;
    type Scalar;

    fn policy(&self) -> &CubePolicy<Self::Runtime>;
    fn len(&self) -> usize;
    fn validate(&self) -> Result<(), Error>;
}

/// Storage-backed structure-of-arrays.
pub trait SoA {
    type Runtime: Runtime;
    type Item;
    type Scalar;

    fn policy(&self) -> &CubePolicy<Self::Runtime>;
    fn len(&self) -> usize;
    fn validate(&self) -> Result<(), Error>;
}

impl<R, T> SoA for DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type Runtime = R;
    type Item = T;
    type Scalar = T;

    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        DeviceVec::policy(self)
    }

    fn len(&self) -> usize {
        DeviceVec::len(self)
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl<R, T> ReadOnlySoA for &DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type Runtime = R;
    type Item = T;
    type Scalar = T;

    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        DeviceVec::policy(self)
    }

    fn len(&self) -> usize {
        DeviceVec::len(self)
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl<R, T> SoA for &DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type Runtime = R;
    type Item = T;
    type Scalar = T;

    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        DeviceVec::policy(self)
    }

    fn len(&self) -> usize {
        DeviceVec::len(self)
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }
}

#[doc(hidden)]
pub struct KernelColumnBindings {
    pub(crate) input: cubecl::server::Handle,
    pub(crate) input_len: usize,
    pub(crate) rhs: cubecl::server::Handle,
    pub(crate) rhs_len: usize,
    pub(crate) slots: Vec<(cubecl::server::Handle, usize)>,
}

impl<Left, Right> ReadOnlySoA for SoAView2<Left, Right>
where
    Left: KernelColumn
        + KernelColumnAt<S0>
        + ReadOnlySoA<Runtime = <Left as KernelColumn>::Runtime, Item = <Left as KernelColumn>::Item>,
    Right: KernelColumn<Runtime = <Left as KernelColumn>::Runtime>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>
        + ReadOnlySoA,
    <Left as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Right as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Runtime = <Left as KernelColumn>::Runtime;
    type Item = (<Left as KernelColumn>::Item, <Right as KernelColumn>::Item);
    type Scalar = <Left as KernelColumn>::Item;
    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        KernelColumn::policy(&self.left)
    }

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

impl<Left, Right> SoA for SoA2<Left, Right>
where
    Left: SoA,
    Right: SoA<Runtime = Left::Runtime>,
{
    type Runtime = Left::Runtime;
    type Item = (Left::Item, Right::Item);
    type Scalar = Left::Scalar;

    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        self.left.policy()
    }

    fn len(&self) -> usize {
        self.left.len()
    }

    fn validate(&self) -> Result<(), Error> {
        self.left.validate()?;
        self.right.validate()?;
        ensure_same_len(self.right.len(), self.left.len())?;
        Ok(())
    }
}

impl<Left, Right> ReadOnlySoA for SoA2<Left, Right>
where
    Self: SoA,
{
    type Runtime = <Self as SoA>::Runtime;
    type Item = <Self as SoA>::Item;
    type Scalar = <Self as SoA>::Scalar;

    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        SoA::policy(self)
    }

    fn len(&self) -> usize {
        SoA::len(self)
    }

    fn validate(&self) -> Result<(), Error> {
        SoA::validate(self)
    }
}

impl<Source> ReadOnlySoA for SoAView1<Source>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    <Source as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Runtime = <Source as KernelColumn>::Runtime;
    type Item = <Source as KernelColumn>::Item;
    type Scalar = <Source as KernelColumn>::Item;
    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        KernelColumn::policy(&self.source)
    }

    fn len(&self) -> usize {
        KernelColumn::len(&self.source)
    }

    fn validate(&self) -> Result<(), Error> {
        KernelColumn::validate(&self.source)
    }
}

impl<Source> SoA for SoA1<Source>
where
    Source: StorageKernelColumn + KernelColumnAt<S0>,
{
    type Runtime = Source::Runtime;
    type Item = Source::Item;
    type Scalar = Source::Item;

    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        KernelColumn::policy(&self.source)
    }

    fn len(&self) -> usize {
        KernelColumn::len(&self.source)
    }

    fn validate(&self) -> Result<(), Error> {
        KernelColumn::validate(&self.source)
    }
}

impl<Left, Right, Start> KernelColumnAt<Start> for SoA2<Left, Right>
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

impl<First, Second, Third> ReadOnlySoA for SoAView3<First, Second, Third>
where
    First: KernelColumn
        + KernelColumnAt<S0>
        + ReadOnlySoA<
            Runtime = <First as KernelColumn>::Runtime,
            Item = <First as KernelColumn>::Item,
        >,
    Second: KernelColumn<Runtime = <First as KernelColumn>::Runtime>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>
        + ReadOnlySoA,
    Third: KernelColumn<Runtime = <First as KernelColumn>::Runtime>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>
        + ReadOnlySoA,
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
    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        KernelColumn::policy(&self.first)
    }

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

impl<First, Second, Third> SoA for SoA3<First, Second, Third>
where
    First: SoA,
    Second: SoA<Runtime = First::Runtime>,
    Third: SoA<Runtime = First::Runtime>,
{
    type Runtime = First::Runtime;
    type Item = (First::Item, Second::Item, Third::Item);
    type Scalar = First::Scalar;

    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        self.first.policy()
    }

    fn len(&self) -> usize {
        self.first.len()
    }

    fn validate(&self) -> Result<(), Error> {
        self.first.validate()?;
        self.second.validate()?;
        self.third.validate()?;
        ensure_same_len(self.second.len(), self.first.len())?;
        ensure_same_len(self.third.len(), self.first.len())?;
        Ok(())
    }
}

impl<First, Second, Third> ReadOnlySoA for SoA3<First, Second, Third>
where
    Self: SoA,
{
    type Runtime = <Self as SoA>::Runtime;
    type Item = <Self as SoA>::Item;
    type Scalar = <Self as SoA>::Scalar;

    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        SoA::policy(self)
    }

    fn len(&self) -> usize {
        SoA::len(self)
    }

    fn validate(&self) -> Result<(), Error> {
        SoA::validate(self)
    }
}

impl<First, Second, Third, Start> KernelColumnAt<Start> for SoA3<First, Second, Third>
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
    fn empty<R: Runtime>(client: &ComputeClient<R>) -> Self {
        Self {
            input: crate::policy::empty_handle(client),
            input_len: 0,
            rhs: crate::policy::empty_handle(client),
            rhs_len: 0,
            slots: Vec::new(),
        }
    }

    fn push(&mut self, handle: cubecl::server::Handle, len: usize) {
        self.slots.push((handle, len));
    }

    fn finish(&mut self) {
        if let Some((handle, len)) = self.slots.first() {
            self.input = handle.clone();
            self.input_len = *len;
        }
        if let Some((handle, len)) = self.slots.get(1) {
            self.rhs = handle.clone();
            self.rhs_len = *len;
        } else {
            self.rhs = self.input.clone();
            self.rhs_len = self.input_len;
        }
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
{
    type Runtime = R;
    type Item = T;
    type Expr = Slot0<T>;

    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        &self.policy
    }

    fn len(&self) -> usize {
        self.len
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }

    fn staged_value_handle(
        &self,
        bindings: &KernelColumnBindings,
    ) -> Option<cubecl::server::Handle> {
        Some(bindings.input.clone())
    }
}

impl<'a, R, T> StorageKernelColumn for &'a DeviceVec<R, T> where R: Runtime {}
impl<'a, R, T> ReadOnlyKernelColumn for &'a DeviceVec<R, T> where R: Runtime {}

impl<'a, R, T> KernelColumnAt<S0> for &'a DeviceVec<R, T>
where
    R: Runtime,
{
    type ExprAt = Slot0<T>;
    type Next = S1;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push(self.handle.clone(), self.len);
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
        bindings.push(self.handle.clone(), self.len);
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
        bindings.push(self.handle.clone(), self.len);
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
        bindings.push(self.handle.clone(), self.len);
        Ok(())
    }
}

impl<R, T> KernelColumn for DeviceVec<R, T>
where
    R: Runtime,
{
    type Runtime = R;
    type Item = T;
    type Expr = Slot0<T>;

    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        &self.policy
    }

    fn len(&self) -> usize {
        self.len
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }

    fn staged_value_handle(
        &self,
        bindings: &KernelColumnBindings,
    ) -> Option<cubecl::server::Handle> {
        Some(bindings.input.clone())
    }
}

impl<R, T> StorageKernelColumn for DeviceVec<R, T> where R: Runtime {}

impl<R, T> KernelColumnAt<S0> for DeviceVec<R, T>
where
    R: Runtime,
{
    type ExprAt = Slot0<T>;
    type Next = S1;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        bindings.push(self.handle.clone(), self.len);
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
        bindings.push(self.handle.clone(), self.len);
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
        bindings.push(self.handle.clone(), self.len);
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
        bindings.push(self.handle.clone(), self.len);
        Ok(())
    }
}

impl<Left, Right, Op> KernelColumn for DeviceBinaryMap<Left, Right, Op>
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    Left::Item: CubePrimitive + CubeElement,
{
    type Runtime = Left::Runtime;
    type Item = Left::Item;
    type Expr = BinaryMap<
        <Left as KernelColumnAt<S0>>::ExprAt,
        <Right as KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>>::ExprAt,
        Op,
    >;

    fn policy(&self) -> &CubePolicy<Self::Runtime> {
        self.left.policy()
    }

    fn len(&self) -> usize {
        self.left.len()
    }

    fn validate(&self) -> Result<(), Error> {
        self.left.validate()?;
        self.right.validate()?;
        ensure_same_len(self.right.len(), self.left.len())?;
        Ok(())
    }

    fn stage(&self) -> Result<KernelColumnBindings, Error> {
        let mut bindings = KernelColumnBindings::empty(KernelColumn::policy(self).client());
        <Self as KernelColumnAt<S0>>::stage_at(self, &mut bindings)?;
        bindings.finish();
        Ok(bindings)
    }
}

impl<Left, Right, Op, Start> KernelColumnAt<Start> for DeviceBinaryMap<Left, Right, Op>
where
    Left: KernelColumn + KernelColumnAt<S0> + KernelColumnAt<Start>,
    Right: KernelColumn<Runtime = Left::Runtime, Item = Left::Item>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>
        + KernelColumnAt<<Left as KernelColumnAt<Start>>::Next>,
    Left::Item: CubePrimitive + CubeElement,
{
    type ExprAt = BinaryMap<
        <Left as KernelColumnAt<Start>>::ExprAt,
        <Right as KernelColumnAt<<Left as KernelColumnAt<Start>>::Next>>::ExprAt,
        Op,
    >;
    type Next = <Right as KernelColumnAt<<Left as KernelColumnAt<Start>>::Next>>::Next;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        <Left as KernelColumnAt<Start>>::stage_at(&self.left, bindings)?;
        <Right as KernelColumnAt<<Left as KernelColumnAt<Start>>::Next>>::stage_at(
            &self.right,
            bindings,
        )
    }
}
