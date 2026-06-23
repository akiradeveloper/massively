//! Public Structure-of-Arrays wrappers.

/// Single-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA1<A>(pub A);

/// Two-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA2<A, B>(pub A, pub B);

/// Three-column structure-of-arrays container.
#[derive(Clone, Copy, Debug)]
pub struct SoA3<A, B, C>(pub A, pub B, pub C);

impl<A> From<(A,)> for SoA1<A> {
    fn from(value: (A,)) -> Self {
        Self(value.0)
    }
}

impl<A, B> From<(A, B)> for SoA2<A, B> {
    fn from(value: (A, B)) -> Self {
        Self(value.0, value.1)
    }
}

impl<A, B, C> From<(A, B, C)> for SoA3<A, B, C> {
    fn from(value: (A, B, C)) -> Self {
        Self(value.0, value.1, value.2)
    }
}
