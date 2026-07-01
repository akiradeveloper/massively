pub fn exclusive_scan_by_key(keys: &[u32], values: &[u32], init: u32) -> Vec<u32> {
    let mut output = vec![0; values.len()];
    if values.is_empty() {
        return output;
    }

    let mut acc = init;
    for i in 0..values.len() {
        if i == 0 || !crate::same_low_nibble(keys[i - 1], keys[i]) {
            acc = init;
        }
        output[i] = acc;
        acc = crate::max_op(acc, values[i]);
    }
    output
}
