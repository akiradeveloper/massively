pub fn sort_by_key(keys: &[u32], values: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut pairs = Vec::with_capacity(keys.len());
    for i in 0..keys.len() {
        pairs.push((keys[i], values[i]));
    }
    pairs.sort_by(|lhs, rhs| {
        if crate::bucket_then_value_less(lhs.0, rhs.0) {
            std::cmp::Ordering::Less
        } else if crate::bucket_then_value_less(rhs.0, lhs.0) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });

    let mut out_keys = vec![0; pairs.len()];
    let mut out_values = vec![0; pairs.len()];
    for i in 0..pairs.len() {
        out_keys[i] = pairs[i].0;
        out_values[i] = pairs[i].1;
    }
    (out_keys, out_values)
}
