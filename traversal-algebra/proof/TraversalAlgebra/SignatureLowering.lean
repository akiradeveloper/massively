import TraversalAlgebra.TypedIR

namespace TraversalAlgebra.Verified.Typed

variable {n : Nat} {Base : Type} {denoteBase : Base → Type}
variable {source middle target signature : Signature n Base denoteBase}
variable {context : List (ValueType Base)} {output : ValueType Base}

namespace Signature

/-- A semantics-preserving translation between two signatures over the same
object-language types.

This is the formal core of signature transport: every source symbol is mapped
to a target symbol, and the certificate proves equality of their mathematical
denotations. It is independent of graph-language expressiveness and of any
concrete implementation. -/
structure Lowering
    (source target : Signature n Base denoteBase) where
  literal : {output : ValueType Base} →
    source.Literal output → target.Literal output
  primitive : {input output : ValueType Base} →
    source.Primitive input output → target.Primitive input output
  monoid : {value : ValueType Base} →
    source.Monoid value → target.Monoid value
  literal_correct : {output : ValueType Base} →
    (symbol : source.Literal output) →
      target.denoteLiteral (literal symbol) = source.denoteLiteral symbol
  primitive_correct : {input output : ValueType Base} →
    (symbol : source.Primitive input output) →
    (value : input.denote n denoteBase) →
      target.denotePrimitive (primitive symbol) value =
        source.denotePrimitive symbol value
  monoid_correct : {value : ValueType Base} →
    (symbol : source.Monoid value) →
      target.denoteMonoid (monoid symbol) = source.denoteMonoid symbol

namespace Lowering

/-- Every signature lowers to itself. -/
def identity (signature : Signature n Base denoteBase) :
    Lowering signature signature where
  literal := fun symbol => symbol
  primitive := fun symbol => symbol
  monoid := fun symbol => symbol
  literal_correct := fun _ => rfl
  primitive_correct := fun _ _ => rfl
  monoid_correct := fun _ => rfl

/-- Semantics-preserving signature lowerings compose. -/
def trans
    (first : Lowering source middle)
    (second : Lowering middle target) : Lowering source target where
  literal := fun symbol => second.literal (first.literal symbol)
  primitive := fun symbol => second.primitive (first.primitive symbol)
  monoid := fun symbol => second.monoid (first.monoid symbol)
  literal_correct := fun symbol =>
    (second.literal_correct (first.literal symbol)).trans
      (first.literal_correct symbol)
  primitive_correct := fun symbol value =>
    (second.primitive_correct (first.primitive symbol) value).trans
      (first.primitive_correct symbol value)
  monoid_correct := fun symbol =>
    (second.monoid_correct (first.monoid symbol)).trans
      (first.monoid_correct symbol)

end Lowering

end Signature

namespace Term

/-- Translate every symbol in a typed term through a certified signature
lowering. Variables and product structure are preserved exactly. -/
def lower
    (lowering : Signature.Lowering source target) :
    {context : List (ValueType Base)} → {output : ValueType Base} →
      Term source context output → Term target context output
  | _, _, .read reference => .read reference
  | _, _, .literal symbol => .literal (lowering.literal symbol)
  | _, _, .pair left right =>
      .pair (left.lower lowering) (right.lower lowering)
  | _, _, .apply primitive argument =>
      .apply (lowering.primitive primitive) (argument.lower lowering)
  | _, _, .let1 input body =>
      .let1 (input.lower lowering) (body.lower lowering)

/-- Fundamental lowering theorem: translating a term cannot change its
value. -/
@[simp]
theorem evaluate_lower
    (lowering : Signature.Lowering source target)
    (term : Term source context output)
    (environment : Environment n denoteBase context) :
    Term.evaluate (signature := target) environment (term.lower lowering) =
      Term.evaluate (signature := source) environment term := by
  induction term with
  | read => rfl
  | literal symbol => exact lowering.literal_correct symbol
  | pair left right leftInduction rightInduction =>
      simp only [lower, evaluate]
      rw [leftInduction environment, rightInduction environment]
  | apply primitive argument induction =>
      simp only [lower, evaluate]
      calc
        target.denotePrimitive (lowering.primitive primitive)
            ((argument.lower lowering).evaluate environment) =
          source.denotePrimitive primitive
            ((argument.lower lowering).evaluate environment) :=
          lowering.primitive_correct primitive _
        _ = source.denotePrimitive primitive (argument.evaluate environment) :=
          congrArg (source.denotePrimitive primitive) (induction environment)
  | let1 input body inputInduction bodyInduction =>
      simp only [lower, evaluate]
      rw [inputInduction environment, bodyInduction]

/-- Term lowering respects identity. -/
@[simp]
theorem lower_identity
    (term : Term signature context output) :
    term.lower (Signature.Lowering.identity signature) = term := by
  induction term with
  | read => rfl
  | literal => rfl
  | pair left right leftInduction rightInduction =>
      simp only [lower, leftInduction, rightInduction]
  | apply primitive argument induction =>
      change Term.apply primitive
          (argument.lower (Signature.Lowering.identity signature)) =
        Term.apply primitive argument
      rw [induction]
  | let1 input body inputInduction bodyInduction =>
      simp only [lower, inputInduction, bodyInduction]

/-- Term lowering respects composition. -/
theorem lower_trans
    (first : Signature.Lowering source middle)
    (second : Signature.Lowering middle target)
    (term : Term source context output) :
    term.lower (first.trans second) =
      (term.lower first).lower second := by
  induction term with
  | read => rfl
  | literal => rfl
  | pair left right leftInduction rightInduction =>
      simp only [lower, leftInduction, rightInduction]
  | apply primitive argument induction =>
      change Term.apply (second.primitive (first.primitive primitive))
          (argument.lower (first.trans second)) =
        Term.apply (second.primitive (first.primitive primitive))
          ((argument.lower first).lower second)
      rw [induction]
  | let1 input body inputInduction bodyInduction =>
      simp only [lower, inputInduction, bodyInduction]

end Term

end TraversalAlgebra.Verified.Typed
