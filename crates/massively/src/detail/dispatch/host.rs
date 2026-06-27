use super::*;

pub trait ToHostDispatch<B: Runtime> {
    type Output;

    fn to_host_with(&self, exec: &Executor<B>) -> Result<Self::Output, Error>;
}
