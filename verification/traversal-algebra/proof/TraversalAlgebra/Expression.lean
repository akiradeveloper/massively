import TraversalAlgebra.Graph

namespace TraversalAlgebra

/-- The structural fragment of edge expressions. Attribute pullbacks are added
to the same evaluator interface in later slices. -/
inductive IdExpr where
  | source
  | destination
  | edge
deriving Repr, DecidableEq

namespace IdExpr

/-- Denotation of a structural expression on an edge context. -/
def eval : IdExpr → EdgeContext → Nat
  | .source, context => context.source
  | .destination, context => context.destination
  | .edge, context => context.edge

end IdExpr

end TraversalAlgebra
