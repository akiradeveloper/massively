pub fn inclusive_scan_by_key(keys: &[u32], values: &[u32]) -> Vec<u32> {
    let mut output = vec![0; values.len()];
    if values.is_empty() {
        return output;
    }

    output[0] = values[0];
    for i in 1..values.len() {
        output[i] = if crate::same_low_nibble(keys[i - 1], keys[i]) {
            crate::max_op(output[i - 1], values[i])
        } else {
            values[i]
        };
    }
    output
}
