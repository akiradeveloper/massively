use super::*;

pub trait ToHostDispatch<R: Runtime> {
    type Output;

    fn to_host_with(&self, exec: &Executor<R>) -> Result<Self::Output, Error>;
}
