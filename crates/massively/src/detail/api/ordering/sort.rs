use super::*;

/// Sorts read-only Zip input and returns owned device storage.
pub fn sort<R, Input, Less>(
    policy: &CubePolicy<R>,
    input: Input,
    _less: Less,
) -> Result<
    <<Input as crate::detail::read::KernelSortInput<Less>>::Output as MaterializeOutput>::Output,
    Error,
>
where
    R: Runtime,
    Input: crate::detail::read::KernelSortInput<Less, Runtime = R>,
    <Input as crate::detail::read::KernelSortInput<Less>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(policy, input.sort_read(policy)?)
}
