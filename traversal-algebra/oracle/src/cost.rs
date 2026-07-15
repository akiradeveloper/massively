//! Proof-backed abstract CubeCL resource certificates.

use crate::{Error, Result};

/// Backend-neutral CubeCL hierarchy parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubeClMachine {
    pub workgroup_size: u32,
    pub subgroup_size: u32,
}

impl Default for CubeClMachine {
    fn default() -> Self {
        Self {
            workgroup_size: 256,
            subgroup_size: 32,
        }
    }
}

impl CubeClMachine {
    pub(crate) fn validate(self) -> Result<()> {
        if self.workgroup_size == 0 {
            return Err(Error::InvalidInput(
                "CubeCL workgroup size must be positive".into(),
            ));
        }
        if self.subgroup_size == 0 {
            return Err(Error::InvalidInput(
                "CubeCL subgroup size must be positive".into(),
            ));
        }
        if self.subgroup_size > self.workgroup_size {
            return Err(Error::InvalidInput(format!(
                "CubeCL subgroup size {} exceeds workgroup size {}",
                self.subgroup_size, self.workgroup_size
            )));
        }
        Ok(())
    }
}

/// Destination collision strategy represented by the certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DestinationStrategy {
    /// General path for every lawful commutative monoid.
    SortReduce,
    /// Single-pass abstract path requiring a carried atomic-action witness.
    Atomic,
}

impl DestinationStrategy {
    pub(crate) const fn protocol_name(self) -> &'static str {
        match self {
            Self::SortReduce => "sort",
            Self::Atomic => "atomic",
        }
    }

    const fn code(self) -> u64 {
        match self {
            Self::SortReduce => 0,
            Self::Atomic => 1,
        }
    }
}

/// Exact counters in the abstract CubeCL execution model.
///
/// These are symbolic resource counts, not elapsed-time predictions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CubeClCost {
    pub logical_threads: u64,
    pub scheduled_threads: u64,
    pub scheduled_subgroups: u64,
    pub scalar_work: u64,
    pub span: u64,
    pub global_loads: u64,
    pub global_stores: u64,
    pub host_read_words: u64,
    pub atomic_operations: u64,
    pub barriers: u64,
    pub launches: u64,
    pub allocated_words: u64,
    pub materializations: u64,
}

impl CubeClCost {
    const FIELD_COUNT: usize = 13;

    fn parse(fields: &[u64]) -> Result<Self> {
        let [
            logical_threads,
            scheduled_threads,
            scheduled_subgroups,
            scalar_work,
            span,
            global_loads,
            global_stores,
            host_read_words,
            atomic_operations,
            barriers,
            launches,
            allocated_words,
            materializations,
        ] = fields
        else {
            return Err(Error::Protocol(format!(
                "CubeCL cost expected {} fields, received {}",
                Self::FIELD_COUNT,
                fields.len()
            )));
        };
        Ok(Self {
            logical_threads: *logical_threads,
            scheduled_threads: *scheduled_threads,
            scheduled_subgroups: *scheduled_subgroups,
            scalar_work: *scalar_work,
            span: *span,
            global_loads: *global_loads,
            global_stores: *global_stores,
            host_read_words: *host_read_words,
            atomic_operations: *atomic_operations,
            barriers: *barriers,
            launches: *launches,
            allocated_words: *allocated_words,
            materializations: *materializations,
        })
    }
}

/// Shape, scalar-expression profile, fused terminal cost, the current
/// materialized CSR-control prefix, and their sequential composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubeClCertificate {
    pub vertices: u64,
    pub topology_edges: u64,
    pub frontier_occurrences: u64,
    pub active_edges: u64,
    pub expression_work: u64,
    pub expression_depth: u64,
    pub global_load_words_per_edge: u64,
    pub output_words: u64,
    pub strategy: DestinationStrategy,
    pub fused: CubeClCost,
    pub materialized_csr_control: CubeClCost,
    pub with_materialized_csr_control: CubeClCost,
}

impl CubeClCertificate {
    const HEADER_FIELDS: usize = 9;
    pub(crate) const FIELD_COUNT: usize = Self::HEADER_FIELDS + 3 * CubeClCost::FIELD_COUNT;

    pub(crate) fn parse(fields: &[u64], requested: DestinationStrategy) -> Result<Self> {
        if fields.len() != Self::FIELD_COUNT {
            return Err(Error::Protocol(format!(
                "CubeCL certificate expected {} fields, received {}",
                Self::FIELD_COUNT,
                fields.len()
            )));
        }
        let strategy = match fields[8] {
            0 => DestinationStrategy::SortReduce,
            1 => DestinationStrategy::Atomic,
            value => {
                return Err(Error::Protocol(format!(
                    "unknown CubeCL destination strategy code {value}"
                )));
            }
        };
        if strategy.code() != requested.code() {
            return Err(Error::Protocol(format!(
                "CubeCL certificate returned {strategy:?}, requested {requested:?}"
            )));
        }
        let fused_start = Self::HEADER_FIELDS;
        let materialized_csr_control_start = fused_start + CubeClCost::FIELD_COUNT;
        let with_materialized_csr_control_start =
            materialized_csr_control_start + CubeClCost::FIELD_COUNT;
        Ok(Self {
            vertices: fields[0],
            topology_edges: fields[1],
            frontier_occurrences: fields[2],
            active_edges: fields[3],
            expression_work: fields[4],
            expression_depth: fields[5],
            global_load_words_per_edge: fields[6],
            output_words: fields[7],
            strategy,
            fused: CubeClCost::parse(&fields[fused_start..materialized_csr_control_start])?,
            materialized_csr_control: CubeClCost::parse(
                &fields[materialized_csr_control_start..with_materialized_csr_control_start],
            )?,
            with_materialized_csr_control: CubeClCost::parse(
                &fields[with_materialized_csr_control_start..],
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_machine_geometry() {
        assert!(
            CubeClMachine {
                workgroup_size: 32,
                subgroup_size: 64,
            }
            .validate()
            .is_err()
        );
    }
}
