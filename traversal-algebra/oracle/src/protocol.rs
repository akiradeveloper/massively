use std::sync::Arc;

use crate::graph::Terminal;

pub(crate) const VERSION: &str = "TAO1";

#[derive(Clone, Debug, Default)]
pub(crate) struct EncodedExpr {
    pub(crate) tokens: Vec<String>,
    pub(crate) vertex_columns: Vec<Arc<[u32]>>,
    pub(crate) edge_columns: Vec<Arc<[u32]>>,
}

impl EncodedExpr {
    pub(crate) fn expression(&self) -> String {
        self.tokens.join(":")
    }
}

pub(crate) fn terminal_name(terminal: Terminal) -> &'static str {
    match terminal {
        Terminal::Emit => "emit",
        Terminal::ReduceBySource => "source",
        Terminal::ReduceByDestination => "destination",
    }
}

pub(crate) fn csv(values: &[u32]) -> String {
    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn columns(columns: &[Arc<[u32]>]) -> String {
    columns
        .iter()
        .map(|column| {
            if column.is_empty() {
                "~".to_owned()
            } else {
                csv(column)
            }
        })
        .collect::<Vec<_>>()
        .join(";")
}

pub(crate) fn parse_csv(input: &str) -> std::result::Result<Vec<u32>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    input
        .split(',')
        .map(|component| {
            component
                .parse::<u32>()
                .map_err(|error| format!("invalid u32 {component:?}: {error}"))
        })
        .collect()
}

pub(crate) fn parse_u64_csv(input: &str) -> std::result::Result<Vec<u64>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    input
        .split(',')
        .map(|component| {
            component
                .parse::<u64>()
                .map_err(|error| format!("invalid u64 {component:?}: {error}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_nonempty_arrays_have_distinct_column_frames() {
        assert_eq!(columns(&[]), "");
        assert_eq!(columns(&[Arc::from([])]), "~");
        assert_eq!(columns(&[Arc::from([1, 2]), Arc::from([])]), "1,2;~");
    }
}
