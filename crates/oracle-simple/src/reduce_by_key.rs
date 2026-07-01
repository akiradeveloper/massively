pub fn reduce_by_key(keys: &[u32], values: &[u32], init: u32) -> (Vec<u32>, Vec<u32>) {
    let mut out_keys = Vec::new();
    let mut out_values = Vec::new();
    if values.is_empty() {
        return (out_keys, out_values);
    }

    let mut current_key = keys[0];
    let mut acc = crate::max_op(init, values[0]);
    for i in 1..values.len() {
        if crate::same_low_nibble(keys[i - 1], keys[i]) {
            acc = crate::max_op(acc, values[i]);
        } else {
            out_keys.push(current_key);
            out_values.push(acc);
            current_key = keys[i];
            acc = crate::max_op(init, values[i]);
        }
    }
    out_keys.push(current_key);
    out_values.push(acc);
    (out_keys, out_values)
}
