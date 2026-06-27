use super::*;

/// Reverses read-only SoA input and returns new device storage.
pub fn reverse<Input>(
    policy: &CubePolicy<<Input as crate::detail::read::KernelReverseInput>::Runtime>,
    input: Input,
) -> Result<
    <<Input as crate::detail::read::KernelReverseInput>::Output as MaterializeOutput>::Output,
    Error,
>
where
    Input: crate::detail::read::KernelReverseInput,
    <Input as crate::detail::read::KernelReverseInput>::Output:
        MaterializeOutput<Runtime = <Input as crate::detail::read::KernelReverseInput>::Runtime>,
{
    materialize(policy, input.reverse_read(policy)?)
}
