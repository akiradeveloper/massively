pub fn max_element(input: &[u32]) -> Option<usize> {
    if input.is_empty() {
        return None;
    }
    let mut best = 0;
    for i in 1..input.len() {
        if crate::bucket_then_value_less(input[best], input[i]) {
            best = i;
        }
    }
    Some(best)
}
