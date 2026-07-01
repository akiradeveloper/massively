pub fn minmax_element(input: &[u32]) -> Option<(usize, usize)> {
    Some((crate::min_element(input)?, crate::max_element(input)?))
}
