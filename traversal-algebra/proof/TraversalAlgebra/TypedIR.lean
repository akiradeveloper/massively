import TraversalAlgebra.Algebra

namespace TraversalAlgebra.Verified.Typed

/-- Object-language value types. `product` is the structural basis for
multi-column values; `base` names scalar types supplied by a fixed signature. -/
inductive ValueType (Base : Type) where
  | boolean
  | index
  | base (name : Base)
  | product (left right : ValueType Base)

namespace ValueType

/-- Denotation of an object-language type for a graph with `n` vertices. -/
def denote (n : Nat) (denoteBase : Base → Type) : ValueType Base → Type
  | .boolean => Bool
  | .index => Fin n
  | .base name => denoteBase name
  | .product left right =>
      denote n denoteBase left × denote n denoteBase right

end ValueType

/-- A fixed, many-sorted scalar signature.

Term constructors contain symbols from this signature rather than their
denotation functions. `denote*` gives the mathematical interpretation.
`Signature.Lowering` separately gives optional proof-carrying translations
between symbol vocabularies; graph-language equivalence uses one fixed
signature on both sides and does not depend on such a translation. -/
structure Signature (n : Nat) (Base : Type) (denoteBase : Base → Type) where
  Literal : ValueType Base → Type
  denoteLiteral : {output : ValueType Base} →
    Literal output → output.denote n denoteBase
  Primitive : ValueType Base → ValueType Base → Type
  denotePrimitive : {input output : ValueType Base} →
    Primitive input output →
      input.denote n denoteBase → output.denote n denoteBase
  Monoid : ValueType Base → Type
  denoteMonoid : {value : ValueType Base} →
    Monoid value → Reduction (value.denote n denoteBase)
  lawfulMonoid : {value : ValueType Base} → (monoid : Monoid value) →
    LawfulCommutativeReduction (denoteMonoid monoid)

/-- A well-typed de Bruijn variable. -/
inductive Variable : List (ValueType Base) → ValueType Base → Type
  | here : Variable (output :: context) output
  | there : Variable context output → Variable (head :: context) output

/-- A heterogeneous environment matching a type context exactly. -/
inductive Environment (n : Nat) (denoteBase : Base → Type) :
    List (ValueType Base) → Type
  | nil : Environment n denoteBase []
  | cons : head.denote n denoteBase → Environment n denoteBase tail →
      Environment n denoteBase (head :: tail)

namespace Variable

/-- Total lookup; an ill-typed variable/environment pair is unrepresentable. -/
def lookup
    {n : Nat} {Base : Type} {denoteBase : Base → Type}
    {context : List (ValueType Base)} {output : ValueType Base}
    (reference : Variable context output)
    (environment : Environment n denoteBase context) :
    output.denote n denoteBase :=
  match reference, environment with
  | .here, .cons value _ => value
  | .there rest, .cons _ tail => rest.lookup tail

end Variable

/-- A closed, typed, first-order term relative to a fixed signature.

All multi-argument scalar operations consume a `product`; this mirrors
Massively's `zip` plus unary-map interface and avoids an arity-product in the
formal language. -/
inductive Term
    (signature : Signature n Base denoteBase) :
    List (ValueType Base) → ValueType Base → Type
  | read {context output} :
      Variable context output → Term signature context output
  | literal {context output} :
      signature.Literal output → Term signature context output
  | pair {context left right} :
      Term signature context left → Term signature context right →
      Term signature context (.product left right)
  | apply {context input output} : signature.Primitive input output →
      Term signature context input → Term signature context output
  | let1 {context input output} : Term signature context input →
      Term signature [input] output → Term signature context output

namespace Term

/-- Denotational evaluation of a well-typed first-order term. -/
def evaluate
    {n : Nat} {Base : Type} {denoteBase : Base → Type}
    {signature : Signature n Base denoteBase}
    {context : List (ValueType Base)}
    (environment : Environment n denoteBase context) :
    {output : ValueType Base} →
      Term signature context output → output.denote n denoteBase
  | _, .read reference => reference.lookup environment
  | _, .literal symbol => signature.denoteLiteral symbol
  | _, .pair left right =>
      (left.evaluate environment, right.evaluate environment)
  | _, .apply primitive argument =>
      signature.denotePrimitive primitive (argument.evaluate environment)
  | _, .let1 input body =>
      body.evaluate (.cons (input.evaluate environment) .nil)

end Term

end TraversalAlgebra.Verified.Typed
