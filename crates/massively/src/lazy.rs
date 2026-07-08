//! Lazy read-only massively iterator constructors.

use cubecl::{
    frontend::PartialEqExpand,
    prelude::{CubeElement, Runtime},
};
use std::marker::PhantomData;

use crate::{Error, MIndex, iter::MIter, op, value::MStorageElement};

/// Lazy constant stream before a finite length is assigned.
#[derive(Debug)]
pub struct Constant<T> {
    value: T,
}

impl<T> Constant<T> {
    /// Turns this lazy stream into a finite read-only iterator.
    pub fn take(self, len: MIndex) -> Taken<Self> {
        Taken::new(self, len)
    }
}

/// Lazy counting stream before a finite length is assigned.
#[derive(Debug)]
pub struct Counting {
    start: MIndex,
}

impl Counting {
    /// Turns this lazy stream into a finite read-only iterator.
    pub fn take(self, len: MIndex) -> Taken<Self> {
        Taken::new(self, len)
    }
}

/// Finite lazy read-only iterator.
#[derive(Debug)]
pub struct Taken<Expr> {
    pub(crate) expr: Expr,
    pub(crate) len: MIndex,
}

impl<Expr> Taken<Expr> {
    pub(crate) fn new(expr: Expr, len: MIndex) -> Self {
        Self { expr, len }
    }
}

/// Lazy permuted read-only iterator.
#[derive(Debug)]
pub struct Permute<Values, Indices> {
    values: Values,
    indices: Indices,
}

/// Lazy unary transform read-only iterator.
#[derive(Debug)]
pub struct Transform<Input, Op> {
    input: Input,
    _op: PhantomData<fn() -> Op>,
}

/// Lazy identity read-only iterator.
#[derive(Debug)]
pub struct Identity<Input> {
    input: Input,
}

/// Creates a lazy constant stream.
pub fn constant<T>(value: T) -> Constant<T> {
    Constant { value }
}

/// Creates a lazy counting stream whose first value is `start`.
pub fn counting(start: MIndex) -> Counting {
    Counting { start }
}

/// Creates a lazy permutation expression that reads `values[indices[i]]`.
pub fn permute<Values, Indices>(values: Values, indices: Indices) -> Permute<Values, Indices> {
    Permute { values, indices }
}

/// Creates a lazy transform expression that reads `op(input[i])`.
pub fn transform<Input, Op>(input: Input, _op: Op) -> Transform<Input, Op> {
    Transform {
        input,
        _op: PhantomData,
    }
}

/// Wraps an iterator as a lazy read-only identity expression.
pub fn identity<Input>(input: Input) -> Identity<Input> {
    Identity { input }
}

#[doc(hidden)]
pub trait ConstantItem<R: Runtime>: crate::MItem<R> {
    type Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>;

    fn lower_constant_read(
        value: Self,
        len: MIndex,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<Self::Read, Error>;
}

fn constant_leaf_read<R, T>(
    value: T,
    len: MIndex,
    policy: &crate::detail::CubePolicy<R>,
) -> crate::detail::read::ConstantRead<T>
where
    R: Runtime,
    T: MStorageElement + 'static,
{
    let handle = policy.client().create_from_slice(T::as_bytes(&[value]));
    crate::detail::read::ConstantRead::new(handle, len as usize)
}

macro_rules! impl_constant_item_scalar {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl<R> ConstantItem<R> for $ty
            where
                R: Runtime,
            {
                type Read = crate::detail::read::ConstantRead<$ty>;

                fn lower_constant_read(
                    value: Self,
                    len: MIndex,
                    policy: &crate::detail::CubePolicy<R>,
                ) -> Result<Self::Read, Error> {
                    Ok(constant_leaf_read::<R, $ty>(value, len, policy))
                }
            }
        )+
    };
}

impl_constant_item_scalar!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

#[doc(hidden)]
pub struct ConstantBool;

#[cubecl::cube]
impl<R> op::UnaryOp<R, u32> for ConstantBool
where
    R: Runtime,
{
    type Output = bool;

    fn apply(input: u32) -> bool {
        input != 0
    }
}

impl<R> ConstantItem<R> for bool
where
    R: Runtime,
{
    type Read =
        crate::detail::read::TransformRead<crate::detail::read::ConstantRead<u32>, ConstantBool>;

    fn lower_constant_read(
        value: Self,
        len: MIndex,
        policy: &crate::detail::CubePolicy<R>,
    ) -> Result<Self::Read, Error> {
        let value = if value { 1_u32 } else { 0_u32 };
        Ok(crate::detail::read::TransformRead::new(
            constant_leaf_read::<R, u32>(value, len, policy),
        ))
    }
}

macro_rules! impl_constant_item_tuple {
    ($zip:ident; $( $ty:ident : $var:ident ),+) => {
        impl<R, $( $ty ),+> ConstantItem<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: ConstantItem<R>, )+
            crate::detail::read::$zip<$( <$ty as ConstantItem<R>>::Read ),+>:
                crate::detail::read::KernelReadBoundMany<R, Item = ($( $ty, )+)>,
        {
            type Read = crate::detail::read::$zip<$( <$ty as ConstantItem<R>>::Read ),+>;

            fn lower_constant_read(
                value: Self,
                len: MIndex,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<Self::Read, Error> {
                let ($( $var, )+) = value;
                Ok(crate::detail::read::$zip::new($(
                    <$ty as ConstantItem<R>>::lower_constant_read($var, len, policy)?,
                )+))
            }
        }
    };
}

impl_constant_item_tuple!(ZipRead1; A: a);
impl_constant_item_tuple!(ZipRead2; A: a, B: b);
impl_constant_item_tuple!(ZipRead3; A: a, B: b, C: c);
impl_constant_item_tuple!(ZipRead4; A: a, B: b, C: c, D: d);
impl_constant_item_tuple!(ZipRead5; A: a, B: b, C: c, D: d, E: e);
impl_constant_item_tuple!(ZipRead6; A: a, B: b, C: c, D: d, E: e, F: f);
impl_constant_item_tuple!(ZipRead7; A: a, B: b, C: c, D: d, E: e, F: f, G: g);

impl<R, T> MIter<R> for Taken<Constant<T>>
where
    R: Runtime,
    T: ConstantItem<R>,
{
    type Item = T;
    type Inner = ();
    type Read = T::Read;

    fn len(&self) -> MIndex {
        self.len
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("lazy constant MIter has no storage inner")
    }

    fn lower_read_ref(&self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        T::lower_constant_read(self.expr.value, self.len, policy)
    }

    fn validate_executor(&self, _exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        Ok(())
    }
}

impl<R, Values, Indices> MIter<R> for Permute<Values, Indices>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: MIter<R, Item = MIndex>,
    crate::detail::read::GatherRead<Values::Read, Indices::Read>:
        crate::detail::read::KernelReadBoundMany<R, Item = Values::Item>,
{
    type Item = Values::Item;
    type Inner = ();
    type Read = crate::detail::read::GatherRead<Values::Read, Indices::Read>;

    fn len(&self) -> MIndex {
        self.indices.len()
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("lazy permute MIter has no storage inner")
    }

    fn lower_read_ref(&self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        let values = self.values.lower_read_ref(policy)?;
        let indices = self.indices.lower_read_ref(policy)?;
        Ok(crate::detail::read::GatherRead::new(values, indices))
    }

    fn validate_executor(&self, exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        self.values.validate_executor(exec)?;
        self.indices.validate_executor(exec)
    }
}

impl<R, Input, Op> MIter<R> for Transform<Input, Op>
where
    R: Runtime,
    Input: MIter<R>,
    Op: op::UnaryOp<R, Input::Item>,
    crate::detail::read::TransformRead<Input::Read, Op>:
        crate::detail::read::KernelReadBoundMany<R, Item = Op::Output>,
{
    type Item = Op::Output;
    type Inner = ();
    type Read = crate::detail::read::TransformRead<Input::Read, Op>;

    fn len(&self) -> MIndex {
        self.input.len()
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("lazy transform MIter has no storage inner")
    }

    fn lower_read_ref(&self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        let input = self.input.lower_read_ref(policy)?;
        Ok(crate::detail::read::TransformRead::new(input))
    }

    fn validate_executor(&self, exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        self.input.validate_executor(exec)
    }
}

impl<R> MIter<R> for Taken<Counting>
where
    R: Runtime,
{
    type Item = MIndex;
    type Inner = ();
    type Read = crate::detail::read::CountingRead;

    fn len(&self) -> MIndex {
        self.len
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("lazy counting MIter has no storage inner")
    }

    fn lower_read_ref(&self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        let handle = policy
            .client()
            .create_from_slice(MIndex::as_bytes(&[self.expr.start]));
        Ok(crate::detail::read::CountingRead::new(
            handle,
            self.len as usize,
        ))
    }

    fn validate_executor(&self, _exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        Ok(())
    }
}

impl<R, Input> MIter<R> for Identity<Input>
where
    R: Runtime,
    Input: MIter<R>,
{
    type Item = Input::Item;
    type Inner = ();
    type Read = Input::Read;

    fn len(&self) -> MIndex {
        self.input.len()
    }

    fn into_inner(self) -> Self::Inner {
        unreachable!("lazy identity MIter has no storage inner")
    }

    fn lower_read_ref(&self, policy: &crate::detail::CubePolicy<R>) -> Result<Self::Read, Error> {
        self.input.lower_read_ref(policy)
    }

    fn validate_executor(&self, exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
        self.input.validate_executor(exec)
    }
}
