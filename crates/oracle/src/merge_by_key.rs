pub fn merge_by_key(
    left_keys: &[u32],
    left_values: &[u32],
    right_keys: &[u32],
    right_values: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let mut out_keys = Vec::with_capacity(left_keys.len() + right_keys.len());
    let mut out_values = Vec::with_capacity(left_values.len() + right_values.len());
    let mut i = 0;
    let mut j = 0;
    while i < left_keys.len() && j < right_keys.len() {
        if crate::bucket_then_value_less(right_keys[j], left_keys[i]) {
            out_keys.push(right_keys[j]);
            out_values.push(right_values[j]);
            j += 1;
        } else {
            out_keys.push(left_keys[i]);
            out_values.push(left_values[i]);
            i += 1;
        }
    }
    while i < left_keys.len() {
        out_keys.push(left_keys[i]);
        out_values.push(left_values[i]);
        i += 1;
    }
    while j < right_keys.len() {
        out_keys.push(right_keys[j]);
        out_values.push(right_values[j]);
        j += 1;
    }
    (out_keys, out_values)
}
