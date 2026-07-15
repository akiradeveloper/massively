use std::{
    ffi::OsStr,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

use crate::{
    CubeClCertificate, CubeClMachine, DestinationStrategy, Error, Result,
    graph::{Expression, ExpressionImpl, Observation, Query, ScalarExpr, Terminal},
    protocol::{self, EncodedExpr, VERSION},
};

/// Persistent client for the theorem-backed Lean oracle executable.
pub struct LeanOracle {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
    loaded_graph: Option<crate::graph::Csr>,
}

impl LeanOracle {
    /// Starts the default oracle built by `just ta::proof`.
    pub fn start() -> Result<Self> {
        Self::spawn(default_oracle_path())
    }

    /// Starts a specific Lean oracle executable in server mode.
    pub fn spawn(path: impl AsRef<OsStr>) -> Result<Self> {
        let mut child = Command::new(path)
            .arg("--server")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| Error::Protocol("failed to open Lean oracle stdin".into()))?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| Error::Protocol("failed to open Lean oracle stdout".into()))?;
        let mut oracle = Self {
            child,
            input: BufWriter::new(input),
            output: BufReader::new(output),
            loaded_graph: None,
        };
        oracle.send(&format!("{VERSION}|HELLO"))?;
        oracle.expect_ok("handshake")?;
        Ok(oracle)
    }

    /// Evaluates every scalar leaf in one semantic query and reconstructs its
    /// recursive product rows on the host.
    #[allow(private_bounds)]
    pub fn evaluate<Expr>(&mut self, query: Query<Expr>) -> Result<Vec<Expr::Item>>
    where
        Expr: Expression + ExpressionImpl,
    {
        Ok(self.observe(query)?.into_values())
    }

    /// Evaluates a query while preserving whether its result is emitted,
    /// source-indexed, or destination-indexed.
    #[allow(private_bounds)]
    pub fn observe<Expr>(&mut self, query: Query<Expr>) -> Result<Observation<Expr::Item>>
    where
        Expr: Expression + ExpressionImpl,
    {
        query.graph.validate()?;
        let mut leaves = Vec::with_capacity(query.expression.leaf_count());
        query.expression.encode_leaves(&mut leaves);
        for leaf in &leaves {
            validate_leaf(leaf, &query.graph)?;
        }
        self.load_graph(&query.graph)?;

        let mut columns = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            let command = format!(
                "{VERSION}|QUERY|{}|{}|{}|{}|{}",
                protocol::terminal_name(query.terminal),
                leaf.expression(),
                protocol::csv(&query.frontier),
                protocol::columns(&leaf.vertex_columns),
                protocol::columns(&leaf.edge_columns),
            );
            self.send(&command)?;
            let response = self.read_response()?;
            let fields = response.splitn(3, '|').collect::<Vec<_>>();
            match fields.as_slice() {
                [version, "RESULT", values] if *version == VERSION => {
                    columns.push(protocol::parse_csv(values).map_err(Error::Protocol)?);
                }
                [version, "ERROR", message] if *version == VERSION => {
                    return Err(Error::Protocol((*message).to_owned()));
                }
                _ => {
                    return Err(Error::Protocol(format!(
                        "invalid QUERY response {response:?}"
                    )));
                }
            }
        }
        let values = query.expression.assemble(&columns)?;
        Ok(match query.terminal {
            Terminal::Emit => Observation::Emitted(values),
            Terminal::ReduceBySource => Observation::SourceReduced(values),
            Terminal::ReduceByDestination => Observation::DestinationReduced(values),
        })
    }

    /// Returns the machine-checked abstract CubeCL resource certificate for a
    /// scalar query without executing its value observation.
    pub fn cubecl_certificate(
        &mut self,
        query: &Query<ScalarExpr>,
        machine: CubeClMachine,
        strategy: DestinationStrategy,
    ) -> Result<CubeClCertificate> {
        machine.validate()?;
        query.graph.validate()?;
        let mut leaves = Vec::with_capacity(1);
        query.expression.encode_leaves(&mut leaves);
        let [leaf] = leaves.as_slice() else {
            return Err(Error::Protocol(format!(
                "scalar CubeCL certificate expected one expression leaf, received {}",
                leaves.len()
            )));
        };
        self.load_graph(&query.graph)?;
        let command = format!(
            "{VERSION}|COST|{}|{}|{}|{}|{}|{}",
            strategy.protocol_name(),
            protocol::terminal_name(query.terminal),
            leaf.expression(),
            protocol::csv(&query.frontier),
            machine.workgroup_size,
            machine.subgroup_size,
        );
        self.send(&command)?;
        let response = self.read_response()?;
        let fields = response.splitn(3, '|').collect::<Vec<_>>();
        match fields.as_slice() {
            [version, "COST", values] if *version == VERSION => {
                let values = protocol::parse_u64_csv(values).map_err(Error::Protocol)?;
                CubeClCertificate::parse(&values, strategy)
            }
            [version, "ERROR", message] if *version == VERSION => {
                Err(Error::Protocol((*message).to_owned()))
            }
            _ => Err(Error::Protocol(format!(
                "invalid COST response {response:?}"
            ))),
        }
    }

    fn load_graph(&mut self, graph: &crate::graph::Csr) -> Result<()> {
        if self.loaded_graph.as_ref() == Some(graph) {
            return Ok(());
        }
        self.send(&format!(
            "{VERSION}|GRAPH|{}|{}",
            protocol::csv(graph.offsets()),
            protocol::csv(graph.destinations())
        ))?;
        self.expect_ok("GRAPH")?;
        self.loaded_graph = Some(graph.clone());
        Ok(())
    }

    fn send(&mut self, command: &str) -> Result<()> {
        self.input.write_all(command.as_bytes())?;
        self.input.write_all(b"\n")?;
        self.input.flush()?;
        Ok(())
    }

    fn read_response(&mut self) -> Result<String> {
        let mut response = String::new();
        let read = self.output.read_line(&mut response)?;
        if read == 0 {
            let status = self.child.try_wait()?;
            return Err(Error::Protocol(format!(
                "Lean oracle closed stdout unexpectedly (status {status:?})"
            )));
        }
        while matches!(response.as_bytes().last(), Some(b'\n' | b'\r')) {
            response.pop();
        }
        Ok(response)
    }

    fn expect_ok(&mut self, operation: &str) -> Result<()> {
        let response = self.read_response()?;
        if response == format!("{VERSION}|OK") {
            return Ok(());
        }
        if let Some(message) = response.strip_prefix(&format!("{VERSION}|ERROR|")) {
            return Err(Error::Protocol(message.to_owned()));
        }
        Err(Error::Protocol(format!(
            "invalid {operation} response {response:?}"
        )))
    }
}

impl Drop for LeanOracle {
    fn drop(&mut self) {
        let _ = self.send(&format!("{VERSION}|QUIT"));
        let _ = self.child.wait();
    }
}

fn validate_leaf(expression: &EncodedExpr, graph: &crate::graph::Csr) -> Result<()> {
    // Encoding a scalar leaf always originated from ScalarExpr, so validate
    // column lengths through an equivalent reconstructed traversal of its
    // registered columns here as a transport-level guard.
    let vertices = graph.offsets().len() - 1;
    let edges = graph.destinations().len();
    if expression
        .vertex_columns
        .iter()
        .any(|column| column.len() != vertices)
    {
        return Err(Error::InvalidInput(format!(
            "oracle vertex column length must equal {vertices}"
        )));
    }
    if expression
        .edge_columns
        .iter()
        .any(|column| column.len() != edges)
    {
        return Err(Error::InvalidInput(format!(
            "oracle edge column length must equal {edges}"
        )));
    }
    Ok(())
}

fn default_oracle_path() -> PathBuf {
    std::env::var_os("TRAVERSAL_ALGEBRA_LEAN_ORACLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../proof/.lake/build/bin/oracle")
        })
}
