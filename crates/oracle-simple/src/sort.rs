pub fn sort(input: &[u32]) -> Vec<u32> {
    let mut output = input.to_vec();
    output.sort_by(|lhs, rhs| {
        if crate::bucket_then_value_less(*lhs, *rhs) {
            std::cmp::Ordering::Less
        } else if crate::bucket_then_value_less(*rhs, *lhs) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    output
}
