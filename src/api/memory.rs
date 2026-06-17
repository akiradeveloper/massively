use crate::{
    device::{
        DeviceMap, DeviceVec, KernelColumn, KernelColumnAt, OwnedKernelColumn, S0, SoA, SoA1, SoA2,
        SoA3, SoA4, SoA5, SoA6, SoA7, SoA8, SoA9, SoA10, SoA11, SoA12, SoVA, SoVA1, SoVA2, SoVA3,
        SoVA4, SoVA5, SoVA6, SoVA7, SoVA8, SoVA9, SoVA10, SoVA11, SoVA12,
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::{GpuOp, UnaryOp},
};
use cubecl::prelude::*;
use std::marker::PhantomData;

/// Owned input accepted by [`zip`].
#[doc(hidden)]
pub trait ZipInput {
    /// SoA source returned for this tuple shape.
    type Output;

    /// Builds the SoA source.
    fn zip(self) -> Self::Output;
}

#[doc(hidden)]
pub trait ZipColumn {
    type Source;

    fn into_source(self) -> Self::Source;
}

impl<R, T> ZipColumn for DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type Source = DeviceVec<R, T>;

    fn into_source(self) -> Self::Source {
        self
    }
}

impl<Source> ZipColumn for SoA1<Source>
where
    Source: OwnedKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
{
    type Source = Source;

    fn into_source(self) -> Self::Source {
        self.source
    }
}

impl<Left, Right> ZipInput for (Left, Right)
where
    Left: ZipColumn,
    Right: ZipColumn,
    Left::Source: KernelColumn + KernelColumnAt<S0>,
    Right::Source: KernelColumn<Runtime = <Left::Source as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left::Source as KernelColumnAt<S0>>::Next>,
    <Left::Source as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Right::Source as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Output = SoA2<Left::Source, Right::Source>;

    fn zip(self) -> Self::Output {
        SoA2 {
            left: self.0.into_source(),
            right: self.1.into_source(),
        }
    }
}

macro_rules! impl_raw_zip_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+> ZipInput for ($first, $( $rest ),+)
        where
            $first: OwnedKernelColumn,
            $(
                $rest: OwnedKernelColumn<Runtime = <$first as KernelColumn>::Runtime>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
        {
            type Output = $name<$first, $( $rest ),+>;

            fn zip(self) -> Self::Output {
                let ($first_field, $( $field ),+) = self;
                $name { $first_field, $( $field ),+ }
            }
        }
    };
}

impl_raw_zip_input!(SoA3<A, B, C> { first, second, third });
impl_raw_zip_input!(SoA4<A, B, C, D> { a, b, c, d });
impl_raw_zip_input!(SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_raw_zip_input!(SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_raw_zip_input!(SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_raw_zip_input!(SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_raw_zip_input!(SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_raw_zip_input!(SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_raw_zip_input!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_raw_zip_input!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_zip_column_left {
    ($name:ident < $( $ty:ident ),+ > { $( $field:ident : $var:ident ),+ }) => {
        impl<Column, $( $ty ),+> ZipInput for (Column, $name<$( $ty ),+>)
        where
            Column: ZipColumn,
            (<Column as ZipColumn>::Source, $( $ty ),+): ZipInput,
        {
            type Output = <(<Column as ZipColumn>::Source, $( $ty ),+) as ZipInput>::Output;

            fn zip(self) -> Self::Output {
                let source = self.0.into_source();
                let $name { $( $field: $var ),+ } = self.1;
                (source, $( $var ),+).zip()
            }
        }
    };
}

macro_rules! impl_zip_column_right {
    ($name:ident < $( $ty:ident ),+ > { $( $field:ident : $var:ident ),+ }) => {
        impl<Column, $( $ty ),+> ZipInput for ($name<$( $ty ),+>, Column)
        where
            Column: ZipColumn,
            ($( $ty, )+ <Column as ZipColumn>::Source): ZipInput,
        {
            type Output = <($( $ty, )+ <Column as ZipColumn>::Source) as ZipInput>::Output;

            fn zip(self) -> Self::Output {
                let $name { $( $field: $var ),+ } = self.0;
                let source = self.1.into_source();
                ($( $var, )+ source).zip()
            }
        }
    };
}

macro_rules! impl_zip_concat {
    (
        $left_name:ident < $( $left_ty:ident ),+ > { $( $left_field:ident : $left_var:ident ),+ }
        +
        $right_name:ident < $( $right_ty:ident ),+ > { $( $right_field:ident : $right_var:ident ),+ }
    ) => {
        impl<$( $left_ty ),+, $( $right_ty ),+> ZipInput
            for ($left_name<$( $left_ty ),+>, $right_name<$( $right_ty ),+>)
        where
            ($( $left_ty, )+ $( $right_ty ),+): ZipInput,
        {
            type Output = <($( $left_ty, )+ $( $right_ty ),+) as ZipInput>::Output;

            fn zip(self) -> Self::Output {
                let $left_name { $( $left_field: $left_var ),+ } = self.0;
                let $right_name { $( $right_field: $right_var ),+ } = self.1;
                ($( $left_var, )+ $( $right_var ),+).zip()
            }
        }
    };
}

impl_zip_column_left!(SoA2<A, B> { left: a, right: b });
impl_zip_column_left!(SoA3<A, B, C> { first: a, second: b, third: c });
impl_zip_column_left!(SoA4<A, B, C, D> { a: a, b: b, c: c, d: d });
impl_zip_column_left!(SoA5<A, B, C, D, E> { a: a, b: b, c: c, d: d, e: e });
impl_zip_column_left!(SoA6<A, B, C, D, E, F> { a: a, b: b, c: c, d: d, e: e, f: f });
impl_zip_column_left!(SoA7<A, B, C, D, E, F, G> { a: a, b: b, c: c, d: d, e: e, f: f, g: g });
impl_zip_column_left!(SoA8<A, B, C, D, E, F, G, H> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h });
impl_zip_column_left!(SoA9<A, B, C, D, E, F, G, H, I> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h, i: i });
impl_zip_column_left!(SoA10<A, B, C, D, E, F, G, H, I, J> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h, i: i, j: j });
impl_zip_column_left!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h, i: i, j: j, k: k });

impl_zip_column_right!(SoA2<A, B> { left: a, right: b });
impl_zip_column_right!(SoA3<A, B, C> { first: a, second: b, third: c });
impl_zip_column_right!(SoA4<A, B, C, D> { a: a, b: b, c: c, d: d });
impl_zip_column_right!(SoA5<A, B, C, D, E> { a: a, b: b, c: c, d: d, e: e });
impl_zip_column_right!(SoA6<A, B, C, D, E, F> { a: a, b: b, c: c, d: d, e: e, f: f });
impl_zip_column_right!(SoA7<A, B, C, D, E, F, G> { a: a, b: b, c: c, d: d, e: e, f: f, g: g });
impl_zip_column_right!(SoA8<A, B, C, D, E, F, G, H> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h });
impl_zip_column_right!(SoA9<A, B, C, D, E, F, G, H, I> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h, i: i });
impl_zip_column_right!(SoA10<A, B, C, D, E, F, G, H, I, J> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h, i: i, j: j });
impl_zip_column_right!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h, i: i, j: j, k: k });

impl_zip_concat!(SoA2<A, B> { left: a, right: b } + SoA2<C, D> { left: c, right: d });
impl_zip_concat!(SoA2<A, B> { left: a, right: b } + SoA3<C, D, E> { first: c, second: d, third: e });
impl_zip_concat!(SoA2<A, B> { left: a, right: b } + SoA4<C, D, E, F> { a: c, b: d, c: e, d: f });
impl_zip_concat!(SoA2<A, B> { left: a, right: b } + SoA5<C, D, E, F, G> { a: c, b: d, c: e, d: f, e: g });
impl_zip_concat!(SoA2<A, B> { left: a, right: b } + SoA6<C, D, E, F, G, H> { a: c, b: d, c: e, d: f, e: g, f: h });
impl_zip_concat!(SoA2<A, B> { left: a, right: b } + SoA7<C, D, E, F, G, H, I> { a: c, b: d, c: e, d: f, e: g, f: h, g: i });
impl_zip_concat!(SoA2<A, B> { left: a, right: b } + SoA8<C, D, E, F, G, H, I, J> { a: c, b: d, c: e, d: f, e: g, f: h, g: i, h: j });
impl_zip_concat!(SoA2<A, B> { left: a, right: b } + SoA9<C, D, E, F, G, H, I, J, K> { a: c, b: d, c: e, d: f, e: g, f: h, g: i, h: j, i: k });
impl_zip_concat!(SoA2<A, B> { left: a, right: b } + SoA10<C, D, E, F, G, H, I, J, K, L> { a: c, b: d, c: e, d: f, e: g, f: h, g: i, h: j, i: k, j: l });

impl_zip_concat!(SoA3<A, B, C> { first: a, second: b, third: c } + SoA2<D, E> { left: d, right: e });
impl_zip_concat!(SoA3<A, B, C> { first: a, second: b, third: c } + SoA3<D, E, F> { first: d, second: e, third: f });
impl_zip_concat!(SoA3<A, B, C> { first: a, second: b, third: c } + SoA4<D, E, F, G> { a: d, b: e, c: f, d: g });
impl_zip_concat!(SoA3<A, B, C> { first: a, second: b, third: c } + SoA5<D, E, F, G, H> { a: d, b: e, c: f, d: g, e: h });
impl_zip_concat!(SoA3<A, B, C> { first: a, second: b, third: c } + SoA6<D, E, F, G, H, I> { a: d, b: e, c: f, d: g, e: h, f: i });
impl_zip_concat!(SoA3<A, B, C> { first: a, second: b, third: c } + SoA7<D, E, F, G, H, I, J> { a: d, b: e, c: f, d: g, e: h, f: i, g: j });
impl_zip_concat!(SoA3<A, B, C> { first: a, second: b, third: c } + SoA8<D, E, F, G, H, I, J, K> { a: d, b: e, c: f, d: g, e: h, f: i, g: j, h: k });
impl_zip_concat!(SoA3<A, B, C> { first: a, second: b, third: c } + SoA9<D, E, F, G, H, I, J, K, L> { a: d, b: e, c: f, d: g, e: h, f: i, g: j, h: k, i: l });

impl_zip_concat!(SoA4<A, B, C, D> { a: a, b: b, c: c, d: d } + SoA2<E, F> { left: e, right: f });
impl_zip_concat!(SoA4<A, B, C, D> { a: a, b: b, c: c, d: d } + SoA3<E, F, G> { first: e, second: f, third: g });
impl_zip_concat!(SoA4<A, B, C, D> { a: a, b: b, c: c, d: d } + SoA4<E, F, G, H> { a: e, b: f, c: g, d: h });
impl_zip_concat!(SoA4<A, B, C, D> { a: a, b: b, c: c, d: d } + SoA5<E, F, G, H, I> { a: e, b: f, c: g, d: h, e: i });
impl_zip_concat!(SoA4<A, B, C, D> { a: a, b: b, c: c, d: d } + SoA6<E, F, G, H, I, J> { a: e, b: f, c: g, d: h, e: i, f: j });
impl_zip_concat!(SoA4<A, B, C, D> { a: a, b: b, c: c, d: d } + SoA7<E, F, G, H, I, J, K> { a: e, b: f, c: g, d: h, e: i, f: j, g: k });
impl_zip_concat!(SoA4<A, B, C, D> { a: a, b: b, c: c, d: d } + SoA8<E, F, G, H, I, J, K, L> { a: e, b: f, c: g, d: h, e: i, f: j, g: k, h: l });

impl_zip_concat!(SoA5<A, B, C, D, E> { a: a, b: b, c: c, d: d, e: e } + SoA2<F, G> { left: f, right: g });
impl_zip_concat!(SoA5<A, B, C, D, E> { a: a, b: b, c: c, d: d, e: e } + SoA3<F, G, H> { first: f, second: g, third: h });
impl_zip_concat!(SoA5<A, B, C, D, E> { a: a, b: b, c: c, d: d, e: e } + SoA4<F, G, H, I> { a: f, b: g, c: h, d: i });
impl_zip_concat!(SoA5<A, B, C, D, E> { a: a, b: b, c: c, d: d, e: e } + SoA5<F, G, H, I, J> { a: f, b: g, c: h, d: i, e: j });
impl_zip_concat!(SoA5<A, B, C, D, E> { a: a, b: b, c: c, d: d, e: e } + SoA6<F, G, H, I, J, K> { a: f, b: g, c: h, d: i, e: j, f: k });
impl_zip_concat!(SoA5<A, B, C, D, E> { a: a, b: b, c: c, d: d, e: e } + SoA7<F, G, H, I, J, K, L> { a: f, b: g, c: h, d: i, e: j, f: k, g: l });

impl_zip_concat!(SoA6<A, B, C, D, E, F> { a: a, b: b, c: c, d: d, e: e, f: f } + SoA2<G, H> { left: g, right: h });
impl_zip_concat!(SoA6<A, B, C, D, E, F> { a: a, b: b, c: c, d: d, e: e, f: f } + SoA3<G, H, I> { first: g, second: h, third: i });
impl_zip_concat!(SoA6<A, B, C, D, E, F> { a: a, b: b, c: c, d: d, e: e, f: f } + SoA4<G, H, I, J> { a: g, b: h, c: i, d: j });
impl_zip_concat!(SoA6<A, B, C, D, E, F> { a: a, b: b, c: c, d: d, e: e, f: f } + SoA5<G, H, I, J, K> { a: g, b: h, c: i, d: j, e: k });
impl_zip_concat!(SoA6<A, B, C, D, E, F> { a: a, b: b, c: c, d: d, e: e, f: f } + SoA6<G, H, I, J, K, L> { a: g, b: h, c: i, d: j, e: k, f: l });

impl_zip_concat!(SoA7<A, B, C, D, E, F, G> { a: a, b: b, c: c, d: d, e: e, f: f, g: g } + SoA2<H, I> { left: h, right: i });
impl_zip_concat!(SoA7<A, B, C, D, E, F, G> { a: a, b: b, c: c, d: d, e: e, f: f, g: g } + SoA3<H, I, J> { first: h, second: i, third: j });
impl_zip_concat!(SoA7<A, B, C, D, E, F, G> { a: a, b: b, c: c, d: d, e: e, f: f, g: g } + SoA4<H, I, J, K> { a: h, b: i, c: j, d: k });
impl_zip_concat!(SoA7<A, B, C, D, E, F, G> { a: a, b: b, c: c, d: d, e: e, f: f, g: g } + SoA5<H, I, J, K, L> { a: h, b: i, c: j, d: k, e: l });

impl_zip_concat!(SoA8<A, B, C, D, E, F, G, H> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h } + SoA2<I, J> { left: i, right: j });
impl_zip_concat!(SoA8<A, B, C, D, E, F, G, H> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h } + SoA3<I, J, K> { first: i, second: j, third: k });
impl_zip_concat!(SoA8<A, B, C, D, E, F, G, H> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h } + SoA4<I, J, K, L> { a: i, b: j, c: k, d: l });

impl_zip_concat!(SoA9<A, B, C, D, E, F, G, H, I> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h, i: i } + SoA2<J, K> { left: j, right: k });
impl_zip_concat!(SoA9<A, B, C, D, E, F, G, H, I> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h, i: i } + SoA3<J, K, L> { first: j, second: k, third: l });

impl_zip_concat!(SoA10<A, B, C, D, E, F, G, H, I, J> { a: a, b: b, c: c, d: d, e: e, f: f, g: g, h: h, i: i, j: j } + SoA2<K, L> { left: k, right: l });

/// Combines two owned SoA inputs into a wider owned SoA.
///
/// `zip` is an ownership boundary. It is for columns that may be consumed by
/// algorithms such as [`sort`](crate::sort), [`reverse`](crate::reverse), or
/// [`remove_if`](crate::remove_if). It does not allocate new device storage by
/// itself; it groups existing owned columns.
///
/// Use [`vzip`] instead when an algorithm only needs to read borrowed columns.
///
/// ```no_run
/// use massively::{CubeWgpu, sort, unzip, zip};
///
/// # struct Less;
/// # #[cubecl::cube]
/// # impl massively::op::BinaryPredicateOp<(f32, u32)> for Less {
/// #     fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> bool { lhs.0 < rhs.0 }
/// # }
/// # fn main() -> Result<(), massively::Error> {
/// let policy = CubeWgpu::new();
/// let values = policy.to_device(&[3.0_f32, 1.0, 2.0])?;
/// let tags = policy.to_device(&[30_u32, 10, 20])?;
///
/// let sorted = sort(zip(values, tags), Less)?;
/// let (_values, _tags) = unzip(sorted)?;
/// # Ok(())
/// # }
/// ```
pub fn zip<Left, Right>(left: Left, right: Right) -> <(Left, Right) as ZipInput>::Output
where
    (Left, Right): ZipInput,
{
    (left, right).zip()
}

/// Convenience wrapper over binary [`zip`] for three owned SoA inputs.
pub fn zip3<A, B, C>(a: A, b: B, c: C) -> <(A, B, C) as ZipInput>::Output
where
    (A, B, C): ZipInput,
{
    (a, b, c).zip()
}

macro_rules! define_zip_n {
    ($func:ident < $( $ty:ident : $var:ident ),+ >) => {
        /// Convenience wrapper over binary [`zip`] for owned SoA inputs.
        pub fn $func<$( $ty ),+>($( $var: $ty ),+) -> <($( $ty ),+) as ZipInput>::Output
        where
            ($( $ty ),+): ZipInput,
        {
            ($( $var ),+).zip()
        }
    };
}

define_zip_n!(zip4<A: a, B: b, C: c, D: d>);
define_zip_n!(zip5<A: a, B: b, C: c, D: d, E: e>);
define_zip_n!(zip6<A: a, B: b, C: c, D: d, E: e, F: f>);
define_zip_n!(zip7<A: a, B: b, C: c, D: d, E: e, F: f, G: g>);
define_zip_n!(zip8<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h>);
define_zip_n!(zip9<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i>);
define_zip_n!(zip10<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j>);
define_zip_n!(zip11<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k>);
define_zip_n!(zip12<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k, L: l>);

/// Virtual-vector input accepted by [`vzip`].
#[doc(hidden)]
pub trait VzipInput {
    /// Read-only SoVA returned for this tuple shape.
    type Output;

    /// Builds the read-only SoVA.
    fn vzip(self) -> Self::Output;
}

pub(crate) trait VzipSource: KernelColumn {}

impl<R, T> VzipSource for &DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
}

impl<Left, Right> VzipInput for (Left, Right)
where
    Left: VzipSource + KernelColumnAt<S0>,
    Right: VzipSource
        + KernelColumn<Runtime = <Left as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    <Left as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Right as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Output = SoVA2<Left, Right>;

    fn vzip(self) -> Self::Output {
        SoVA2 {
            left: self.0,
            right: self.1,
        }
    }
}

impl<Left, Right> VzipInput for (SoVA1<Left>, SoVA1<Right>)
where
    Left: VzipSource + KernelColumnAt<S0>,
    Right: VzipSource
        + KernelColumn<Runtime = <Left as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    <Left as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Right as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Output = SoVA2<Left, Right>;

    fn vzip(self) -> Self::Output {
        SoVA2 {
            left: self.0.source,
            right: self.1.source,
        }
    }
}

impl<First, Second, Third> VzipInput for (First, Second, Third)
where
    First: VzipSource + KernelColumnAt<S0>,
    Second: VzipSource
        + KernelColumn<Runtime = <First as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: VzipSource
        + KernelColumn<Runtime = <First as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    <First as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Second as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Third as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Output = SoVA3<First, Second, Third>;

    fn vzip(self) -> Self::Output {
        SoVA3 {
            first: self.0,
            second: self.1,
            third: self.2,
        }
    }
}

impl<First, Second, Third> VzipInput for (SoVA1<First>, SoVA1<Second>, SoVA1<Third>)
where
    First: VzipSource + KernelColumnAt<S0>,
    Second: VzipSource
        + KernelColumn<Runtime = <First as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: VzipSource
        + KernelColumn<Runtime = <First as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    <First as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Second as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Third as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Output = SoVA3<First, Second, Third>;

    fn vzip(self) -> Self::Output {
        SoVA3 {
            first: self.0.source,
            second: self.1.source,
            third: self.2.source,
        }
    }
}

macro_rules! impl_vzip_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+> VzipInput for ($first, $( $rest ),+)
        where
            $first: VzipSource + KernelColumnAt<S0>,
            $(
                $rest: VzipSource
                    + KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
        {
            type Output = $name<$first, $( $rest ),+>;

            fn vzip(self) -> Self::Output {
                let ($first_field, $( $field ),+) = self;
                $name { $first_field, $( $field ),+ }
            }
        }
    };
}

impl_vzip_input!(SoVA4<A, B, C, D> { a, b, c, d });
impl_vzip_input!(SoVA5<A, B, C, D, E> { a, b, c, d, e });
impl_vzip_input!(SoVA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_vzip_input!(SoVA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_vzip_input!(SoVA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_vzip_input!(SoVA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_vzip_input!(SoVA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_vzip_input!(SoVA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_vzip_input!(SoVA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_vzip_soa1_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+> VzipInput for (SoVA1<$first>, $( SoVA1<$rest> ),+)
        where
            $first: VzipSource + KernelColumnAt<S0>,
            $(
                $rest: VzipSource
                    + KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
        {
            type Output = $name<$first, $( $rest ),+>;

            fn vzip(self) -> Self::Output {
                let ($first_field, $( $field ),+) = self;
                $name {
                    $first_field: $first_field.source,
                    $( $field: $field.source, )+
                }
            }
        }
    };
}

impl_vzip_soa1_input!(SoVA4<A, B, C, D> { a, b, c, d });
impl_vzip_soa1_input!(SoVA5<A, B, C, D, E> { a, b, c, d, e });
impl_vzip_soa1_input!(SoVA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_vzip_soa1_input!(SoVA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_vzip_soa1_input!(SoVA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_vzip_soa1_input!(SoVA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_vzip_soa1_input!(SoVA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_vzip_soa1_input!(SoVA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_vzip_soa1_input!(SoVA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Combines two read-only columns into a wider read-only SoVA input.
///
/// `vzip` is for borrowing algorithms such as [`transform`], [`reduce`](crate::reduce), and
/// [`gather`](crate::gather). It does not allocate device storage; it creates a
/// typed read-only view over existing device columns.
///
/// Use [`zip`] instead when the grouped columns are owned storage passed to a
/// consuming algorithm.
///
/// ```no_run
/// use massively::{CubeWgpu, transform, unzip, vzip};
///
/// # struct Add;
/// # #[cubecl::cube]
/// # impl massively::op::UnaryOp<(f32, f32)> for Add {
/// #     type Output = f32;
/// #     fn apply(input: (f32, f32)) -> f32 { input.0 + input.1 }
/// # }
/// # fn main() -> Result<(), massively::Error> {
/// let policy = CubeWgpu::new();
/// let x = policy.to_device(&[1.0_f32, 2.0])?;
/// let y = policy.to_device(&[10.0_f32, 20.0])?;
///
/// let output = unzip(transform(vzip(&x, &y), Add)?)?;
/// assert_eq!(output.to_vec()?, vec![11.0, 22.0]);
/// # Ok(())
/// # }
/// ```
pub fn vzip<Left, Right>(left: Left, right: Right) -> <(Left, Right) as VzipInput>::Output
where
    (Left, Right): VzipInput,
{
    (left, right).vzip()
}

/// Convenience wrapper over binary [`vzip`] for three read-only columns.
pub fn vzip3<A, B, C>(a: A, b: B, c: C) -> <(A, B, C) as VzipInput>::Output
where
    (A, B, C): VzipInput,
{
    (a, b, c).vzip()
}

macro_rules! define_vzip_n {
    ($func:ident < $( $ty:ident : $var:ident ),+ >) => {
        /// Convenience wrapper over binary [`vzip`] for read-only columns.
        pub fn $func<$( $ty ),+>($( $var: $ty ),+) -> <($( $ty ),+) as VzipInput>::Output
        where
            ($( $ty ),+): VzipInput,
        {
            ($( $var ),+).vzip()
        }
    };
}

define_vzip_n!(vzip4<A: a, B: b, C: c, D: d>);
define_vzip_n!(vzip5<A: a, B: b, C: c, D: d, E: e>);
define_vzip_n!(vzip6<A: a, B: b, C: c, D: d, E: e, F: f>);
define_vzip_n!(vzip7<A: a, B: b, C: c, D: d, E: e, F: f, G: g>);
define_vzip_n!(vzip8<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h>);
define_vzip_n!(vzip9<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i>);
define_vzip_n!(vzip10<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j>);
define_vzip_n!(vzip11<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k>);
define_vzip_n!(vzip12<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k, L: l>);

/// Input accepted by [`transform`].
#[doc(hidden)]
pub trait TransformWriteInput<Op, Output> {
    /// Applies the transform in the same abstraction level as the input.
    fn transform_write_input(self, op: GpuOp<Op>, output: &mut Output) -> Result<(), Error>;
}

impl<Source, R, T, Op> TransformWriteInput<Op, DeviceVec<R, T>> for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn<Runtime = R, Item = T> + KernelColumnAt<S0>,
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    <Source as KernelColumnAt<S0>>::ExprAt: DeviceGpuExpr<Source::Item>,
    Op: UnaryOp<Source::Item, Output = Source::Item>,
{
    fn transform_write_input(
        self,
        _op: GpuOp<Op>,
        output: &mut DeviceVec<R, T>,
    ) -> Result<(), Error> {
        SoVA::validate(&self)?;
        let mapped = DeviceMap {
            source: self.source,
            _op: PhantomData::<fn() -> Op>,
        };
        super::device_expr_collect_into(&mapped, output)
    }
}

impl<Source, R, T, Op> TransformWriteInput<Op, DeviceVec<R, T>> for Source
where
    Source: KernelColumn<Runtime = R, Item = T> + KernelColumnAt<S0>,
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    <Source as KernelColumnAt<S0>>::ExprAt: DeviceGpuExpr<Source::Item>,
    Op: UnaryOp<Source::Item, Output = Source::Item>,
{
    fn transform_write_input(
        self,
        op: GpuOp<Op>,
        output: &mut DeviceVec<R, T>,
    ) -> Result<(), Error> {
        <SoVA1<Source> as TransformWriteInput<Op, DeviceVec<R, T>>>::transform_write_input(
            SoVA1 { source: self },
            op,
            output,
        )
    }
}

impl<Source, R, T, Op> TransformWriteInput<Op, SoA1<DeviceVec<R, T>>> for Source
where
    Source: TransformWriteInput<Op, DeviceVec<R, T>>,
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    fn transform_write_input(
        self,
        op: GpuOp<Op>,
        output: &mut SoA1<DeviceVec<R, T>>,
    ) -> Result<(), Error> {
        <Self as TransformWriteInput<Op, DeviceVec<R, T>>>::transform_write_input(
            self,
            op,
            &mut output.source,
        )
    }
}

impl<Left, Right, R, Out, Op> TransformWriteInput<Op, DeviceVec<R, Out>> for SoVA2<Left, Right>
where
    Self: SoVA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    R: Runtime,
    Out: CubePrimitive + CubeElement,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: UnaryOp<<Self as SoVA>::Item, Output = Out>,
{
    fn transform_write_input(
        self,
        _op: GpuOp<Op>,
        output: &mut DeviceVec<R, Out>,
    ) -> Result<(), Error> {
        SoVA::validate(&self)?;
        let left = super::device_expr_collect(&self.left)?;
        let right = super::device_expr_collect(&self.right)?;
        let len = left.len();
        if len != output.len() {
            return Err(Error::LengthMismatch {
                input: len,
                output: output.len(),
            });
        }
        if len != 0 {
            let client = left.policy().client();
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_tuple2_kernel::launch_unchecked::<Left::Item, Right::Item, Out, Op, R>(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    ArrayArg::from_raw_parts::<Left::Item>(&left.handle, len, 1),
                    ArrayArg::from_raw_parts::<Right::Item>(&right.handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<Out>(&output.handle, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok(())
    }
}

impl<First, Second, Third, R, Out, Op> TransformWriteInput<Op, DeviceVec<R, Out>>
    for SoVA3<First, Second, Third>
where
    Self: SoVA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
    First: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: KernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    R: Runtime,
    Out: CubePrimitive + CubeElement,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    Op: UnaryOp<<Self as SoVA>::Item, Output = Out>,
{
    fn transform_write_input(
        self,
        _op: GpuOp<Op>,
        output: &mut DeviceVec<R, Out>,
    ) -> Result<(), Error> {
        SoVA::validate(&self)?;
        let first = super::device_expr_collect(&self.first)?;
        let second = super::device_expr_collect(&self.second)?;
        let third = super::device_expr_collect(&self.third)?;
        let len = first.len();
        if len != output.len() {
            return Err(Error::LengthMismatch {
                input: len,
                output: output.len(),
            });
        }
        if len != 0 {
            let client = first.policy().client();
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_tuple3_kernel::launch_unchecked::<
                    First::Item,
                    Second::Item,
                    Third::Item,
                    Out,
                    Op,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    ArrayArg::from_raw_parts::<First::Item>(&first.handle, len, 1),
                    ArrayArg::from_raw_parts::<Second::Item>(&second.handle, len, 1),
                    ArrayArg::from_raw_parts::<Third::Item>(&third.handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<Out>(&output.handle, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok(())
    }
}

macro_rules! impl_transform_write_input {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    (
        $name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ },
        $kernel_name:ident
    ) => {
        impl<$first, $( $rest ),+, R, Out, Op> TransformWriteInput<Op, DeviceVec<R, Out>> for $name<$first, $( $rest ),+>
        where
            Self: SoVA<Scalar = <$first as KernelColumn>::Item>,
            $first: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            R: Runtime,
            Out: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Op: UnaryOp<(
                impl_transform_write_input!(@item_ty $first),
                $( impl_transform_write_input!(@item_ty $rest) ),+
            ), Output = Out>,
        {
            fn transform_write_input(self, _op: GpuOp<Op>, output: &mut DeviceVec<R, Out>) -> Result<(), Error> {
                SoVA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+

                let len = $first_field.len();
                if len != output.len() {
                    return Err(Error::LengthMismatch {
                        input: len,
                        output: output.len(),
                    });
                }
                let client = $first_field.policy().client();
                if len != 0 {
                    let len_u32 =
                        u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let block_size = 256_u32;
                    let block_count = len.div_ceil(block_size as usize);
                    let block_count_u32 = u32::try_from(block_count)
                        .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                    unsafe {
                        $kernel_name::launch_unchecked::<
                            <$first as KernelColumn>::Item,
                            $(
                                <$rest as KernelColumn>::Item,
                            )+
                            Out,
                            Op,
                            <$first as KernelColumn>::Runtime,
                        >(
                            client,
                            CubeCount::Static(block_count_u32, 1, 1),
                            CubeDim::new_1d(block_size),
                            ArrayArg::from_raw_parts::<<$first as KernelColumn>::Item>(
                                &$first_field.handle,
                                len,
                                1,
                            ),
                            $(
                                ArrayArg::from_raw_parts::<<$rest as KernelColumn>::Item>(
                                    &$field.handle,
                                    len,
                                    1,
                                ),
                            )+
                            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                            ArrayArg::from_raw_parts::<Out>(
                                &output.handle,
                                len,
                                1,
                            ),
                        )
                        .map_err(|err| Error::Launch {
                            message: format!("{err:?}"),
                        })?;
                    }
                }

                Ok(())
            }
        }
    };
}

impl_transform_write_input!(SoVA4<A, B, C, D> { a, b, c, d }, transform_tuple4_kernel);
impl_transform_write_input!(SoVA5<A, B, C, D, E> { a, b, c, d, e }, transform_tuple5_kernel);
impl_transform_write_input!(SoVA6<A, B, C, D, E, F> { a, b, c, d, e, f }, transform_tuple6_kernel);
impl_transform_write_input!(SoVA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g }, transform_tuple7_kernel);
impl_transform_write_input!(SoVA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h }, transform_tuple8_kernel);
impl_transform_write_input!(SoVA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i }, transform_tuple9_kernel);
impl_transform_write_input!(SoVA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j }, transform_tuple10_kernel);
impl_transform_write_input!(SoVA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k }, transform_tuple11_kernel);
impl_transform_write_input!(SoVA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l }, transform_tuple12_kernel);

/// Storage shape used for a transformed device value.
#[doc(hidden)]
pub trait StorageOutput<R: Runtime>: CubeType {
    type Storage;
}

macro_rules! impl_scalar_storage_output {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl<R> StorageOutput<R> for $ty
            where
                R: Runtime,
            {
                type Storage = SoA1<DeviceVec<R, $ty>>;
            }
        )+
    };
}

impl_scalar_storage_output!(f32, f64, u8, u16, u32, u64, i8, i16, i32, i64, bool);

impl<R, A, B> StorageOutput<R> for (A, B)
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
{
    type Storage = SoA2<DeviceVec<R, A>, DeviceVec<R, B>>;
}

impl<R, A, B, C> StorageOutput<R> for (A, B, C)
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
{
    type Storage = SoA3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>;
}

trait TransformUnaryOutput<R, T, Op>: StorageOutput<R>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Op: UnaryOp<T, Output = Self>,
{
    fn run(input: &DeviceVec<R, T>) -> Result<Self::Storage, Error>;
}

macro_rules! impl_scalar_transform_unary_output {
    ($($out:ty),+ $(,)?) => {
        $(
            impl<R, T, Op> TransformUnaryOutput<R, T, Op> for $out
            where
                R: Runtime,
                T: CubePrimitive + CubeElement,
                Op: UnaryOp<T, Output = $out>,
            {
                fn run(input: &DeviceVec<R, T>) -> Result<Self::Storage, Error> {
                    let len = input.len();
                    let client = input.policy().client();
                    let output_handle = client.empty(len * std::mem::size_of::<$out>());
                    if len != 0 {
                        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                        let block_size = 256_u32;
                        let block_count = len.div_ceil(block_size as usize);
                        let block_count_u32 = u32::try_from(block_count)
                            .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                        unsafe {
                            transform_unary_kernel::launch_unchecked::<T, $out, Op, R>(
                                client,
                                CubeCount::Static(block_count_u32, 1, 1),
                                CubeDim::new_1d(block_size),
                                ArrayArg::from_raw_parts::<T>(&input.handle, len, 1),
                                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                                ArrayArg::from_raw_parts::<$out>(&output_handle, len, 1),
                            )
                            .map_err(|err| Error::Launch {
                                message: format!("{err:?}"),
                            })?;
                        }
                    }
                    Ok(SoA1 {
                        source: DeviceVec::from_handle(input.policy().clone(), output_handle, len),
                    })
                }
            }
        )+
    };
}

impl_scalar_transform_unary_output!(f32, f64, u8, u16, u32, u64, i8, i16, i32, i64, bool);

impl<R, T, A, B, Op> TransformUnaryOutput<R, T, Op> for (A, B)
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    Op: UnaryOp<T, Output = (A, B)>,
{
    fn run(input: &DeviceVec<R, T>) -> Result<Self::Storage, Error> {
        let len = input.len();
        let client = input.policy().client();
        let output_a = client.empty(len * std::mem::size_of::<A>());
        let output_b = client.empty(len * std::mem::size_of::<B>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_unary_tuple2_kernel::launch_unchecked::<T, A, B, Op, R>(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    ArrayArg::from_raw_parts::<T>(&input.handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<A>(&output_a, len, 1),
                    ArrayArg::from_raw_parts::<B>(&output_b, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok(SoA2 {
            left: DeviceVec::from_handle(input.policy().clone(), output_a, len),
            right: DeviceVec::from_handle(input.policy().clone(), output_b, len),
        })
    }
}

impl<R, T, A, B, C, Op> TransformUnaryOutput<R, T, Op> for (A, B, C)
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    Op: UnaryOp<T, Output = (A, B, C)>,
{
    fn run(input: &DeviceVec<R, T>) -> Result<Self::Storage, Error> {
        let len = input.len();
        let client = input.policy().client();
        let output_a = client.empty(len * std::mem::size_of::<A>());
        let output_b = client.empty(len * std::mem::size_of::<B>());
        let output_c = client.empty(len * std::mem::size_of::<C>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_unary_tuple3_kernel::launch_unchecked::<T, A, B, C, Op, R>(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    ArrayArg::from_raw_parts::<T>(&input.handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<A>(&output_a, len, 1),
                    ArrayArg::from_raw_parts::<B>(&output_b, len, 1),
                    ArrayArg::from_raw_parts::<C>(&output_c, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok(SoA3 {
            first: DeviceVec::from_handle(input.policy().clone(), output_a, len),
            second: DeviceVec::from_handle(input.policy().clone(), output_b, len),
            third: DeviceVec::from_handle(input.policy().clone(), output_c, len),
        })
    }
}

/// Input accepted by returning [`transform`].
#[doc(hidden)]
pub trait TransformInput<Op> {
    type Output;

    fn transform_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error>;
}

impl<Source, Op> TransformInput<Op> for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Runtime: Runtime,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: UnaryOp<Source::Item>,
    Op::Output: TransformUnaryOutput<Source::Runtime, Source::Item, Op>,
{
    type Output = <Op::Output as StorageOutput<Source::Runtime>>::Storage;

    fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let input = super::device_expr_collect(&self.source)?;
        <Op::Output as TransformUnaryOutput<Source::Runtime, Source::Item, Op>>::run(&input)
    }
}

impl<Source, Op> TransformInput<Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Runtime: Runtime,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: UnaryOp<Source::Item>,
    Op::Output: TransformUnaryOutput<Source::Runtime, Source::Item, Op>,
{
    type Output = <Op::Output as StorageOutput<Source::Runtime>>::Storage;

    fn transform_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoVA1<Source> as TransformInput<Op>>::transform_input(SoVA1 { source: self }, op)
    }
}

#[doc(hidden)]
pub trait TransformSoA2Output<R, InA, InB, Op>: CubeType
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB), Output = Self>,
{
    type Storage;

    fn run(
        policy: &crate::policy::CubePolicy<R>,
        left: &DeviceVec<R, InA>,
        right: &DeviceVec<R, InB>,
    ) -> Result<Self::Storage, Error>;
}

macro_rules! impl_scalar_transform_soa2_output {
    ($($out:ty),+ $(,)?) => {
        $(
            impl<R, InA, InB, Op> TransformSoA2Output<R, InA, InB, Op> for $out
            where
                R: Runtime,
                InA: CubePrimitive + CubeElement,
                InB: CubePrimitive + CubeElement,
                Op: UnaryOp<(InA, InB), Output = $out>,
            {
                type Storage = SoA1<DeviceVec<R, $out>>;

                fn run(
                    policy: &crate::policy::CubePolicy<R>,
                    left: &DeviceVec<R, InA>,
                    right: &DeviceVec<R, InB>,
                ) -> Result<Self::Storage, Error> {
                    let len = left.len();
                    let client = policy.client();
                    let output_handle = client.empty(len * std::mem::size_of::<$out>());
                    if len != 0 {
                        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                        let block_size = 256_u32;
                        let block_count = len.div_ceil(block_size as usize);
                        let block_count_u32 = u32::try_from(block_count)
                            .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                        unsafe {
                            transform_tuple2_kernel::launch_unchecked::<InA, InB, $out, Op, R>(
                                client,
                                CubeCount::Static(block_count_u32, 1, 1),
                                CubeDim::new_1d(block_size),
                                ArrayArg::from_raw_parts::<InA>(&left.handle, len, 1),
                                ArrayArg::from_raw_parts::<InB>(&right.handle, len, 1),
                                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                                ArrayArg::from_raw_parts::<$out>(&output_handle, len, 1),
                            )
                            .map_err(|err| Error::Launch {
                                message: format!("{err:?}"),
                            })?;
                        }
                    }
                    Ok(SoA1 {
                        source: DeviceVec::from_handle(policy.clone(), output_handle, len),
                    })
                }
            }
        )+
    };
}

impl_scalar_transform_soa2_output!(f32, f64, u8, u16, u32, u64, i8, i16, i32, i64, bool);

impl<R, InA, InB, OutA, OutB, Op> TransformSoA2Output<R, InA, InB, Op> for (OutA, OutB)
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    OutA: CubePrimitive + CubeElement,
    OutB: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB), Output = (OutA, OutB)>,
{
    type Storage = SoA2<DeviceVec<R, OutA>, DeviceVec<R, OutB>>;

    fn run(
        policy: &crate::policy::CubePolicy<R>,
        left: &DeviceVec<R, InA>,
        right: &DeviceVec<R, InB>,
    ) -> Result<Self::Storage, Error> {
        let len = left.len();
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<OutA>());
        let output_b = client.empty(len * std::mem::size_of::<OutB>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_tuple2_to_tuple2_kernel::launch_unchecked::<InA, InB, OutA, OutB, Op, R>(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    ArrayArg::from_raw_parts::<InA>(&left.handle, len, 1),
                    ArrayArg::from_raw_parts::<InB>(&right.handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<OutA>(&output_a, len, 1),
                    ArrayArg::from_raw_parts::<OutB>(&output_b, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok(SoA2 {
            left: DeviceVec::from_handle(policy.clone(), output_a, len),
            right: DeviceVec::from_handle(policy.clone(), output_b, len),
        })
    }
}

impl<Left, Right, Op> TransformInput<Op> for SoVA2<Left, Right>
where
    Self: SoVA<Runtime = Left::Runtime, Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
    Left::Runtime: Runtime,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: UnaryOp<(Left::Item, Right::Item)>,
    Op::Output: TransformSoA2Output<Left::Runtime, Left::Item, Right::Item, Op>,
{
    type Output =
        <Op::Output as TransformSoA2Output<Left::Runtime, Left::Item, Right::Item, Op>>::Storage;

    fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let policy = self.policy().clone();
        let left = super::device_expr_collect(&self.left)?;
        let right = super::device_expr_collect(&self.right)?;
        <Op::Output as TransformSoA2Output<Left::Runtime, Left::Item, Right::Item, Op>>::run(
            &policy, &left, &right,
        )
    }
}

#[doc(hidden)]
pub trait TransformSoA3Output<R, InA, InB, InC, Op>: CubeType
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    InC: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB, InC), Output = Self>,
{
    type Storage;

    fn run(
        policy: &crate::policy::CubePolicy<R>,
        first: &DeviceVec<R, InA>,
        second: &DeviceVec<R, InB>,
        third: &DeviceVec<R, InC>,
    ) -> Result<Self::Storage, Error>;
}

macro_rules! impl_scalar_transform_soa3_output {
    ($($out:ty),+ $(,)?) => {
        $(
            impl<R, InA, InB, InC, Op> TransformSoA3Output<R, InA, InB, InC, Op> for $out
            where
                R: Runtime,
                InA: CubePrimitive + CubeElement,
                InB: CubePrimitive + CubeElement,
                InC: CubePrimitive + CubeElement,
                Op: UnaryOp<(InA, InB, InC), Output = $out>,
            {
                type Storage = SoA1<DeviceVec<R, $out>>;

                fn run(
                    policy: &crate::policy::CubePolicy<R>,
                    first: &DeviceVec<R, InA>,
                    second: &DeviceVec<R, InB>,
                    third: &DeviceVec<R, InC>,
                ) -> Result<Self::Storage, Error> {
                    let len = first.len();
                    let client = policy.client();
                    let output_handle = client.empty(len * std::mem::size_of::<$out>());
                    if len != 0 {
                        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                        let block_size = 256_u32;
                        let block_count = len.div_ceil(block_size as usize);
                        let block_count_u32 = u32::try_from(block_count)
                            .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                        unsafe {
                            transform_tuple3_kernel::launch_unchecked::<
                                InA,
                                InB,
                                InC,
                                $out,
                                Op,
                                R,
                            >(
                                client,
                                CubeCount::Static(block_count_u32, 1, 1),
                                CubeDim::new_1d(block_size),
                                ArrayArg::from_raw_parts::<InA>(&first.handle, len, 1),
                                ArrayArg::from_raw_parts::<InB>(&second.handle, len, 1),
                                ArrayArg::from_raw_parts::<InC>(&third.handle, len, 1),
                                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                                ArrayArg::from_raw_parts::<$out>(&output_handle, len, 1),
                            )
                            .map_err(|err| Error::Launch {
                                message: format!("{err:?}"),
                            })?;
                        }
                    }
                    Ok(SoA1 {
                        source: DeviceVec::from_handle(policy.clone(), output_handle, len),
                    })
                }
            }
        )+
    };
}

impl_scalar_transform_soa3_output!(f32, f64, u8, u16, u32, u64, i8, i16, i32, i64, bool);

impl<R, InA, InB, InC, OutA, OutB, OutC, Op> TransformSoA3Output<R, InA, InB, InC, Op>
    for (OutA, OutB, OutC)
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    InC: CubePrimitive + CubeElement,
    OutA: CubePrimitive + CubeElement,
    OutB: CubePrimitive + CubeElement,
    OutC: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB, InC), Output = (OutA, OutB, OutC)>,
{
    type Storage = SoA3<DeviceVec<R, OutA>, DeviceVec<R, OutB>, DeviceVec<R, OutC>>;

    fn run(
        policy: &crate::policy::CubePolicy<R>,
        first: &DeviceVec<R, InA>,
        second: &DeviceVec<R, InB>,
        third: &DeviceVec<R, InC>,
    ) -> Result<Self::Storage, Error> {
        let len = first.len();
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<OutA>());
        let output_b = client.empty(len * std::mem::size_of::<OutB>());
        let output_c = client.empty(len * std::mem::size_of::<OutC>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_tuple3_to_tuple3_kernel::launch_unchecked::<
                    InA,
                    InB,
                    InC,
                    OutA,
                    OutB,
                    OutC,
                    Op,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    ArrayArg::from_raw_parts::<InA>(&first.handle, len, 1),
                    ArrayArg::from_raw_parts::<InB>(&second.handle, len, 1),
                    ArrayArg::from_raw_parts::<InC>(&third.handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<OutA>(&output_a, len, 1),
                    ArrayArg::from_raw_parts::<OutB>(&output_b, len, 1),
                    ArrayArg::from_raw_parts::<OutC>(&output_c, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok(SoA3 {
            first: DeviceVec::from_handle(policy.clone(), output_a, len),
            second: DeviceVec::from_handle(policy.clone(), output_b, len),
            third: DeviceVec::from_handle(policy.clone(), output_c, len),
        })
    }
}

impl<First, Second, Third, Op> TransformInput<Op> for SoVA3<First, Second, Third>
where
    Self: SoVA<
            Runtime = First::Runtime,
            Item = (First::Item, Second::Item, Third::Item),
            Scalar = First::Item,
        >,
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Runtime: Runtime,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    Op: UnaryOp<(First::Item, Second::Item, Third::Item)>,
    Op::Output: TransformSoA3Output<First::Runtime, First::Item, Second::Item, Third::Item, Op>,
{
    type Output = <Op::Output as TransformSoA3Output<
        First::Runtime,
        First::Item,
        Second::Item,
        Third::Item,
        Op,
    >>::Storage;

    fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let policy = self.policy().clone();
        let first = super::device_expr_collect(&self.first)?;
        let second = super::device_expr_collect(&self.second)?;
        let third = super::device_expr_collect(&self.third)?;
        <Op::Output as TransformSoA3Output<
            First::Runtime,
            First::Item,
            Second::Item,
            Third::Item,
            Op,
        >>::run(&policy, &first, &second, &third)
    }
}

macro_rules! impl_transform_input {
    (@item_ty $field:ident) => {
        <$field as KernelColumn>::Item
    };

    ($name:ident < $first:ident, $( $rest:ident ),+ >) => {
        impl<$first, $( $rest ),+, Op> TransformInput<Op> for $name<$first, $( $rest ),+>
        where
            Self: SoVA<Runtime = <$first as KernelColumn>::Runtime>
                + TransformWriteInput<Op, DeviceVec<<$first as KernelColumn>::Runtime, Op::Output>>,
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime>,
            )+
            <$first as KernelColumn>::Runtime: Runtime,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            Op: UnaryOp<(
                impl_transform_input!(@item_ty $first),
                $( impl_transform_input!(@item_ty $rest) ),+
            )>,
            Op::Output: CubePrimitive + CubeElement,
        {
            type Output = SoA1<DeviceVec<<$first as KernelColumn>::Runtime, Op::Output>>;

            fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                let len = SoVA::len(&self);
                let output_handle = self
                    .policy()
                    .client()
                    .empty(len * std::mem::size_of::<Op::Output>());
                let mut output = DeviceVec::from_handle(self.policy().clone(), output_handle, len);
                <Self as TransformWriteInput<
                    Op,
                    DeviceVec<<$first as KernelColumn>::Runtime, Op::Output>,
                >>::transform_write_input(self, GpuOp::<Op>::new(), &mut output)?;
                Ok(SoA1 { source: output })
            }
        }
    };
}

impl_transform_input!(SoVA4<A, B, C, D>);
impl_transform_input!(SoVA5<A, B, C, D, E>);
impl_transform_input!(SoVA6<A, B, C, D, E, F>);
impl_transform_input!(SoVA7<A, B, C, D, E, F, G>);
impl_transform_input!(SoVA8<A, B, C, D, E, F, G, H>);
impl_transform_input!(SoVA9<A, B, C, D, E, F, G, H, I>);
impl_transform_input!(SoVA10<A, B, C, D, E, F, G, H, I, J>);
impl_transform_input!(SoVA11<A, B, C, D, E, F, G, H, I, J, K>);
impl_transform_input!(SoVA12<A, B, C, D, E, F, G, H, I, J, K, L>);

/// Input accepted by [`unzip`].
#[doc(hidden)]
pub trait UnzipInput {
    /// Materialized output produced by unzipping this input.
    type Output;

    /// Materializes this input.
    fn unzip_input(self) -> Result<Self::Output, Error>;
}

impl<Left, Right> UnzipInput for SoA2<Left, Right>
where
    Self: SoA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: OwnedKernelColumn + KernelColumnAt<S0>,
    Right: OwnedKernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
{
    type Output = (
        DeviceVec<Left::Runtime, Left::Item>,
        DeviceVec<Left::Runtime, Right::Item>,
    );

    fn unzip_input(self) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let left = super::device_expr_collect(&self.left)?;
        let right = super::device_expr_collect(&self.right)?;
        Ok((left, right))
    }
}

impl<Source> UnzipInput for SoA1<Source>
where
    Self: SoA<Item = Source::Item, Scalar = Source::Item>,
    Source: OwnedKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Output = DeviceVec<Source::Runtime, Source::Item>;

    fn unzip_input(self) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let source = super::device_expr_collect(&self.source)?;
        Ok(source)
    }
}

impl<R, T> UnzipInput for DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type Output = Self;

    fn unzip_input(self) -> Result<Self::Output, Error> {
        Ok(self)
    }
}

impl<First, Second, Third> UnzipInput for SoA3<First, Second, Third>
where
    Self: SoA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
    First: OwnedKernelColumn + KernelColumnAt<S0>,
    Second: OwnedKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: OwnedKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
{
    type Output = (
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    );

    fn unzip_input(self) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let first = super::device_expr_collect(&self.first)?;
        let second = super::device_expr_collect(&self.second)?;
        let third = super::device_expr_collect(&self.third)?;
        Ok((first, second, third))
    }
}

macro_rules! impl_zip_unzip_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+> UnzipInput for $name<$first, $( $rest ),+>
        where
            Self: SoA,
            $first: OwnedKernelColumn + KernelColumnAt<S0>,
            $(
                $rest: OwnedKernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
        {
            type Output = (
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            );

            fn unzip_input(self) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                Ok(($first_field, $( $field ),+))
            }
        }
    };
}

impl_zip_unzip_input!(SoA4<A, B, C, D> { a, b, c, d });
impl_zip_unzip_input!(SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_zip_unzip_input!(SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_zip_unzip_input!(SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_zip_unzip_input!(SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_zip_unzip_input!(SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_zip_unzip_input!(SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_zip_unzip_input!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_zip_unzip_input!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Recovers owned device columns from an owned SoA.
///
/// `unzip` consumes owned SoA storage created by [`zip`], returned by
/// algorithms, or represented by a one-column [`DeviceVec`](crate::DeviceVec).
/// It does not accept read-only [`vzip`] outputs, because that would hide a
/// materializing copy behind an ownership-recovery name.
pub fn unzip<Source>(source: Source) -> Result<<Source as UnzipInput>::Output, Error>
where
    Source: UnzipInput,
{
    source.unzip_input()
}

/// Applies a read-only transform and returns owned device storage.
///
/// The input may be a borrowed [`DeviceVec`](crate::DeviceVec) or a read-only
/// SoVA built with [`vzip`]. The returned value is owned SoA storage, so call
/// [`unzip`] to recover the output [`DeviceVec`](crate::DeviceVec) column or
/// columns.
pub fn transform<Source, Op>(
    source: Source,
    _op: Op,
) -> Result<<Source as TransformInput<Op>>::Output, Error>
where
    Source: TransformInput<Op>,
{
    source.transform_input(GpuOp::<Op>::new())
}
