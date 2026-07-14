import TraversalAlgebra.Oracle

open TraversalAlgebra

namespace TraversalAlgebra.OracleCli

def join (separator : String) : List String → String
  | [] => ""
  | first :: rest => rest.foldl (fun output item => output ++ separator ++ item) first

def parseCsv (input : String) : Except String (List Nat) := do
  if input.isEmpty then
    return []
  input.splitOn "," |>.mapM fun component =>
    match component.toNat? with
    | some value => pure value
    | none => throw ("invalid natural number: " ++ component)

def graphFromCsr (offsets destinations : List Nat) : Except String Graph := do
  if offsets.isEmpty then
    throw "CSR offsets must contain the initial zero"
  if offsets.head? != some 0 then
    throw "CSR offsets must start at zero"
  let rows := (offsets.zip offsets.tail).map fun (start, stop) =>
    (destinations.drop start).take (stop - start)
  let graph : Graph := ⟨rows⟩
  if graph.csrOffsets != offsets || graph.csrDestinations != destinations then
    throw "CSR offsets are not canonical for the destination stream"
  if !graph.isValid then
    throw "CSR contains a destination outside the vertex set"
  pure graph

def renderNats (values : List Nat) : String :=
  join "," (values.map toString)

def renderEdges (edges : List EdgeContext) : String :=
  join ";" <| edges.map fun edge =>
    toString edge.source ++ "," ++ toString edge.destination ++ "," ++ toString edge.edge

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

end TraversalAlgebra.OracleCli

def main (arguments : List String) : IO UInt32 := do
  match arguments with
  | [offsets, destinations, frontier] =>
      match TraversalAlgebra.OracleCli.evaluate offsets destinations frontier with
      | .ok output =>
          IO.println output
          return 0
      | .error message =>
          IO.eprintln message
          return 1
  | _ =>
      IO.eprintln "usage: oracle OFFSETS DESTINATIONS FRONTIER"
      return 2
