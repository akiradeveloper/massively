//! Host-side operation traits mirroring `massively::op`.

pub trait UnaryOp<Input>: 'static + Send + Sync {
    type Env: Copy;
    type Output;

    fn apply(env: Self::Env, input: Input) -> Self::Output;
}

pub trait BinaryOp<X, Y>: 'static + Send + Sync {
    type Output;

    fn apply(lhs: X, rhs: Y) -> Self::Output;
}

pub trait ReductionOp<X>: 'static + Send + Sync {
    fn apply(lhs: X, rhs: X) -> X;
}

pub trait PredicateOp<T>: 'static + Send + Sync {
    type Env: Copy;

    fn apply(env: Self::Env, input: T) -> bool;
}

pub trait BinaryPredicateOp<T>: 'static + Send + Sync {
    fn apply(lhs: T, rhs: T) -> bool;
}
