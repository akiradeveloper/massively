pub fn unique_by_key(keys: &[u32], values: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut out_keys = Vec::new();
    let mut out_values = Vec::new();
    if values.is_empty() {
        return (out_keys, out_values);
    }

    out_keys.push(keys[0]);
    out_values.push(values[0]);
    for i in 1..values.len() {
        if !crate::same_low_nibble(keys[i - 1], keys[i]) {
            out_keys.push(keys[i]);
            out_values.push(values[i]);
        }
    }
    (out_keys, out_values)
}
