import TraversalAlgebra.OracleCubeCL

open TraversalAlgebra

namespace TraversalAlgebra.OracleCli

def protocolVersion : String := "TAO1"

def join (separator : String) : List String → String
  | [] => ""
  | first :: rest => rest.foldl (fun output item => output ++ separator ++ item) first

def parseCsv (input : String) : Except String (List Nat) := do
  if input.isEmpty then
    return []
  input.splitOn "," |>.mapM fun component =>
    match component.toNat? with
    | some value =>
        if value ≤ 4294967295 then pure value
        else throw ("natural does not fit u32: " ++ component)
    | none => throw ("invalid natural number: " ++ component)

def parseColumns (input : String) : Except String (List (List Nat)) := do
  if input.isEmpty then
    return []
  input.splitOn ";" |>.mapM fun column =>
    if column = "~" then pure [] else parseCsv column

def parseIndexed (marker token : String) : Except String Nat := do
  if !token.startsWith marker then
    throw ("expected " ++ marker ++ " column, received " ++ token)
  let suffix := (token.drop marker.length).copy
  match suffix.toNat? with
  | some value => pure value
  | none => throw ("invalid column reference: " ++ token)

partial def parseExprTokens : List String → Except String (Oracle.Typed.Expr × List String)
  | [] => throw "unexpected end of expression"
  | token :: rest =>
      if token = "sid" then pure (.sourceId, rest)
      else if token = "did" then pure (.destinationId, rest)
      else if token = "eid" then pure (.edgeId, rest)
      else if token = "add" then do
        let (left, rest) ← parseExprTokens rest
        let (right, rest) ← parseExprTokens rest
        pure (.add left right, rest)
      else if token.startsWith "src" then do
        pure (.source (← parseIndexed "src" token), rest)
      else if token.startsWith "dst" then do
        pure (.destination (← parseIndexed "dst" token), rest)
      else if token.startsWith "edge" then do
        pure (.edge (← parseIndexed "edge" token), rest)
      else if token.startsWith "c" then do
        pure (.constant (← parseIndexed "c" token), rest)
      else
        throw ("unknown expression token: " ++ token)

def parseExpr (input : String) : Except String Oracle.Typed.Expr := do
  let (expression, rest) ← parseExprTokens (input.splitOn ":")
  if rest.isEmpty then pure expression
  else throw ("trailing expression tokens: " ++ join ":" rest)

def parseTerminal : String → Except String Oracle.Typed.Terminal
  | "emit" => pure .emit
  | "source" => pure .reduceBySource
  | "destination" => pure .reduceByDestination
  | terminal => throw ("unknown terminal: " ++ terminal)

def parseStrategy : String → Except String
    Verified.Typed.CubeCL.DestinationStrategy
  | "sort" => pure .sortReduce
  | "atomic" => pure .atomic
  | strategy => throw ("unknown destination strategy: " ++ strategy)

def graphFromCsr (offsets destinations : List Nat) : Except String Graph := do
  pure (← Oracle.Typed.checkCsr offsets destinations).graph

def renderNats (values : List Nat) : String :=
  join "," (values.map toString)

def renderEdges (edges : List EdgeContext) : String :=
  join ";" <| edges.map fun edge =>
    toString edge.source ++ "," ++ toString edge.destination ++ "," ++ toString edge.edge

/-- Backward-compatible one-shot evaluator retained for fixture generation and
manual artifact inspection. -/
def evaluate (offsetsText destinationsText frontierText : String) : Except String String := do
  let offsets ← parseCsv offsetsText
  let destinations ← parseCsv destinationsText
  let frontier ← parseCsv frontierText
  let graph ← graphFromCsr offsets destinations
  let case : Oracle.Case := { name := "property", graph, frontier }
  if !Oracle.isValid case then
    throw "frontier contains a vertex outside the graph"
  pure <|
    renderEdges (Oracle.expectedEdges case) ++ "|" ++
    renderNats (Oracle.expectedSourceCounts case) ++ "|" ++
    renderNats (Oracle.expectedDestinationCounts case)

def columnsHaveLength (columns : List (List Nat)) (length : Nat) : Bool :=
  columns.all fun column => column.length = length

def evaluateTyped
    (csr : Oracle.Typed.CheckedCsr)
    (terminalText expressionText frontierText vertexColumnsText edgeColumnsText : String) :
    Except String String := do
  let terminal ← parseTerminal terminalText
  let expression ← parseExpr expressionText
  let frontier ← parseCsv frontierText
  let vertexColumns ← parseColumns vertexColumnsText
  let edgeColumns ← parseColumns edgeColumnsText
  if !columnsHaveLength vertexColumns csr.graph.vertexCount then
    throw "vertex column length does not equal the graph vertex count"
  if !columnsHaveLength edgeColumns csr.graph.edgeCount then
    throw "edge column length does not equal the graph edge count"
  if !expression.referencesValid vertexColumns.length edgeColumns.length then
    throw "expression references a missing host column"
  if frontierEq : isValidFrontier csr.graph frontier = true then
    let values := Oracle.Typed.evaluateCsr csr frontier
      ((Oracle.Typed.frontier_isValid_iff csr.graph frontier).mp frontierEq)
      vertexColumns edgeColumns terminal expression
    if values.all fun value => value ≤ 4294967295 then
      pure (renderNats values)
    else
      throw "typed oracle result does not fit u32"
  else
    throw "frontier contains a vertex outside the graph"

def costNats (cost : Verified.Typed.CubeCL.Cost) : List Nat :=
  [cost.logicalThreads, cost.scheduledThreads, cost.scheduledSubgroups,
   cost.scalarWork, cost.span, cost.globalLoads, cost.globalStores, cost.hostReadWords,
   cost.atomicOperations, cost.barriers, cost.launches,
   cost.allocatedWords, cost.materializations]

def strategyCode : Verified.Typed.CubeCL.DestinationStrategy → Nat
  | .sortReduce => 0
  | .atomic => 1

def certificateNats
    (certificate : Oracle.Typed.CubeCL.Certificate) : List Nat :=
  [certificate.vertices, certificate.topologyEdges,
   certificate.frontierOccurrences, certificate.activeEdges,
   certificate.expressionWork, certificate.expressionDepth,
   certificate.globalLoadWordsPerEdge, certificate.outputWords,
   strategyCode certificate.strategy] ++
  costNats certificate.fused ++
  costNats certificate.materializedCsrControl ++
  costNats certificate.withMaterializedCsrControl

def evaluateCost
    (csr : Oracle.Typed.CheckedCsr)
    (strategyText terminalText expressionText frontierText
      workgroupText subgroupText : String) : Except String String := do
  let strategy ← parseStrategy strategyText
  let terminal ← parseTerminal terminalText
  let expression ← parseExpr expressionText
  let frontier ← parseCsv frontierText
  let some workgroupSize := workgroupText.toNat?
    | throw ("invalid workgroup size: " ++ workgroupText)
  let some subgroupSize := subgroupText.toNat?
    | throw ("invalid subgroup size: " ++ subgroupText)
  if workgroupSize = 0 then throw "workgroup size must be positive"
  if subgroupSize = 0 then throw "subgroup size must be positive"
  if subgroupSize > workgroupSize then
    throw "subgroup size must not exceed workgroup size"
  if frontierEq : isValidFrontier csr.graph frontier = true then
    let certificate := Oracle.Typed.CubeCL.certificate
      { workgroupSize, subgroupSize }
      csr frontier
      ((Oracle.Typed.frontier_isValid_iff csr.graph frontier).mp frontierEq)
      strategy terminal expression
    pure (renderNats (certificateNats certificate))
  else
    throw "frontier contains a vertex outside the graph"

structure ServerReply where
  keepRunning : Bool
  graph : Option Oracle.Typed.CheckedCsr
  response : String

def ok (graph : Option Oracle.Typed.CheckedCsr) (response := protocolVersion ++ "|OK") :
    ServerReply :=
  { keepRunning := true, graph, response }

def handleServerCommand
    (cached : Option Oracle.Typed.CheckedCsr) (line : String) : Except String ServerReply := do
  let fields := line.splitOn "|"
  let version :: command :: arguments := fields
    | throw "protocol frame must contain version and command"
  if version != protocolVersion then
    throw ("unsupported protocol version: " ++ version)
  match command, arguments with
  | "HELLO", [] => pure (ok cached)
  | "GRAPH", [offsetsText, destinationsText] =>
      let offsets ← parseCsv offsetsText
      let destinations ← parseCsv destinationsText
      let graph ← Oracle.Typed.checkCsr offsets destinations
      pure (ok (some graph))
  | "QUERY", [terminal, expression, frontier, vertexColumns, edgeColumns] =>
      let some graph := cached | throw "QUERY received before GRAPH"
      let result ← evaluateTyped graph terminal expression frontier vertexColumns edgeColumns
      pure (ok cached (protocolVersion ++ "|RESULT|" ++ result))
  | "COST", [strategy, terminal, expression, frontier, workgroup, subgroup] =>
      let some graph := cached | throw "COST received before GRAPH"
      let result ← evaluateCost graph strategy terminal expression frontier workgroup subgroup
      pure (ok cached (protocolVersion ++ "|COST|" ++ result))
  | "QUIT", [] =>
      pure { keepRunning := false, graph := cached, response := protocolVersion ++ "|OK" }
  | _, _ => throw ("invalid command frame for " ++ command)

def sanitizeError (message : String) : String :=
  (message.replace "|" "/").replace "\n" " "

partial def serverLoop
    (input output : IO.FS.Stream) (cached : Option Oracle.Typed.CheckedCsr) : IO UInt32 := do
  let raw ← input.getLine
  if raw.isEmpty then
    return 0
  let line := raw.trimAscii.copy
  match handleServerCommand cached line with
  | .ok reply =>
      output.putStr (reply.response ++ "\n")
      output.flush
      if reply.keepRunning then serverLoop input output reply.graph else return 0
  | .error message =>
      output.putStr (protocolVersion ++ "|ERROR|" ++ sanitizeError message ++ "\n")
      output.flush
      serverLoop input output cached

def server : IO UInt32 := do
  serverLoop (← IO.getStdin) (← IO.getStdout) none

end TraversalAlgebra.OracleCli

def main (arguments : List String) : IO UInt32 := do
  match arguments with
  | ["--server"] => TraversalAlgebra.OracleCli.server
  | [offsets, destinations, frontier] =>
      match TraversalAlgebra.OracleCli.evaluate offsets destinations frontier with
      | .ok output =>
          IO.println output
          return 0
      | .error message =>
          IO.eprintln message
          return 1
  | _ =>
      IO.eprintln "usage: oracle --server | oracle OFFSETS DESTINATIONS FRONTIER"
      return 2
