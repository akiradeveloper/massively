//! Type-level read arity.

/// A supported number of physical leaves read by an expression.
pub trait ReadArity: private::Sealed + 'static {}

macro_rules! define_arities {
    ($($arity:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Read arity marker `", stringify!($arity), "`.")]
            #[derive(Clone, Copy, Debug, Eq, PartialEq)]
            pub struct $arity;

            impl ReadArity for $arity {}
            impl private::Sealed for $arity {}
        )+
    };
}

define_arities!(A1, A2, A3, A4, A5, A6, A7, A8);

/// Type-level addition for read arities whose sum is at most eight.
///
/// There is deliberately no implementation for sums greater than eight.
pub trait AddArity<Rhs: ReadArity>: ReadArity {
    type Output: ReadArity;
}

macro_rules! impl_add_arity {
    ($lhs:ty, $rhs:ty => $output:ty) => {
        impl AddArity<$rhs> for $lhs {
            type Output = $output;
        }
    };
}

impl_add_arity!(A1, A1 => A2);
impl_add_arity!(A1, A2 => A3);
impl_add_arity!(A1, A3 => A4);
impl_add_arity!(A1, A4 => A5);
impl_add_arity!(A1, A5 => A6);
impl_add_arity!(A1, A6 => A7);
impl_add_arity!(A1, A7 => A8);
impl_add_arity!(A2, A1 => A3);
impl_add_arity!(A2, A2 => A4);
impl_add_arity!(A2, A3 => A5);
impl_add_arity!(A2, A4 => A6);
impl_add_arity!(A2, A5 => A7);
impl_add_arity!(A2, A6 => A8);
impl_add_arity!(A3, A1 => A4);
impl_add_arity!(A3, A2 => A5);
impl_add_arity!(A3, A3 => A6);
impl_add_arity!(A3, A4 => A7);
impl_add_arity!(A3, A5 => A8);
impl_add_arity!(A4, A1 => A5);
impl_add_arity!(A4, A2 => A6);
impl_add_arity!(A4, A3 => A7);
impl_add_arity!(A4, A4 => A8);
impl_add_arity!(A5, A1 => A6);
impl_add_arity!(A5, A2 => A7);
impl_add_arity!(A5, A3 => A8);
impl_add_arity!(A6, A1 => A7);
impl_add_arity!(A6, A2 => A8);
impl_add_arity!(A7, A1 => A8);

mod private {
    pub trait Sealed {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    assert_impl_all!(A1: AddArity<A7, Output = A8>);
    assert_impl_all!(A4: AddArity<A4, Output = A8>);
    assert_not_impl_any!(A8: AddArity<A1>);
    assert_not_impl_any!(A7: AddArity<A2>);
}
