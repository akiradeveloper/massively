import TraversalAlgebra.TypedLowering

namespace TraversalAlgebra.Verified.Typed.Example

/-- A minimal concrete signature witnessing that the typed development is
inhabited. It models natural-number path-mass propagation. -/
inductive Base where
  | natural

def denoteBase : Base → Type
  | .natural => Nat

abbrev natural : ValueType Base := .base .natural

inductive Literal : ValueType Base → Type
  | boolean (value : Bool) : Literal .boolean
  | natural (value : Nat) : Literal natural

def denoteLiteral {n : Nat} : {output : ValueType Base} →
    Literal output → output.denote n denoteBase
  | _, .boolean value => value
  | _, .natural value => value

inductive Primitive : ValueType Base → ValueType Base → Type
  | add : Primitive (.product natural natural) natural
  | different : Primitive (.product natural natural) .boolean

def denotePrimitive {n : Nat} : {input output : ValueType Base} →
    Primitive input output →
      input.denote n denoteBase → output.denote n denoteBase
  | _, _, .add => fun values =>
      Nat.add (show Nat from values.1) (show Nat from values.2)
  | _, _, .different => fun values =>
      (show Nat from values.1) != (show Nat from values.2)

inductive MonoidSymbol : ValueType Base → Type
  | naturalAdd : MonoidSymbol natural

def denoteMonoid {n : Nat} : {value : ValueType Base} →
    MonoidSymbol value → Reduction (value.denote n denoteBase)
  | _, .naturalAdd => natAdd

theorem lawfulMonoid {n : Nat} : {value : ValueType Base} →
    (monoid : MonoidSymbol value) →
      LawfulCommutativeReduction (denoteMonoid (n := n) monoid)
  | _, .naturalAdd => natAdd_lawful

def signature (n : Nat) : Signature n Base denoteBase where
  Literal := Literal
  denoteLiteral := denoteLiteral
  Primitive := Primitive
  denotePrimitive := denotePrimitive
  Monoid := MonoidSymbol
  denoteMonoid := denoteMonoid
  lawfulMonoid := lawfulMonoid

def pathMessage (n : Nat) :
    Term (signature n) (MessageContext natural natural) natural :=
  .read MessageContext.sourceState

def pathUpdate (n : Nat) :
    Term (signature n) (UpdateContext natural natural) natural :=
  .apply .add
    (.pair (.read UpdateContext.oldState) (.read UpdateContext.inbox))

def stateChanged (n : Nat) :
    Term (signature n) (SelectionContext natural) .boolean :=
  .apply .different
    (.pair (.read SelectionContext.oldState)
      (.read SelectionContext.newState))

/-- Each active source contributes its current natural value to every outgoing
destination; inboxes are added into state and changed vertices form the next
canonical frontier. -/
def pathProgram (n : Nat) :
    MonoidalFrontierBSP.Program (signature n) natural natural natural where
  messageTerm := pathMessage n
  reduction := .naturalAdd
  updateTerm := pathUpdate n
  frontier := .dense (stateChanged n)

theorem pathUpdate_evaluates :
    Term.evaluate (signature := signature 1)
      (UpdateContext.environment
        (denoteBase := denoteBase) (vertex := natural) (message := natural)
        (0 : Fin 1) (2 : Nat) (3 : Nat))
      (pathUpdate 1) = (5 : Nat) := rfl

theorem stateChanged_evaluates :
    Term.evaluate (signature := signature 1)
      (SelectionContext.environment
        (denoteBase := denoteBase) (vertex := natural)
        (0 : Fin 1) (2 : Nat) (5 : Nat))
      (stateChanged 1) = true := rfl

namespace Alternate

/-- A second, deliberately distinct syntax for the same scalar operations.
It witnesses non-identity signature transport: correctness below cannot hold
merely by reusing the source symbols because the constructor types differ. -/
inductive Literal : ValueType Base → Type
  | booleanImmediate (value : Bool) : Literal .boolean
  | naturalImmediate (value : Nat) : Literal natural

def denoteLiteral {n : Nat} : {output : ValueType Base} →
    Literal output → output.denote n denoteBase
  | _, .booleanImmediate value => value
  | _, .naturalImmediate value => value

inductive Primitive : ValueType Base → ValueType Base → Type
  | naturalAdd : Primitive (.product natural natural) natural
  | naturalNotEqual : Primitive (.product natural natural) .boolean

def denotePrimitive {n : Nat} : {input output : ValueType Base} →
    Primitive input output →
      input.denote n denoteBase → output.denote n denoteBase
  | _, _, .naturalAdd => fun values =>
      Nat.add (show Nat from values.1) (show Nat from values.2)
  | _, _, .naturalNotEqual => fun values =>
      (show Nat from values.1) != (show Nat from values.2)

inductive MonoidSymbol : ValueType Base → Type
  | naturalSum : MonoidSymbol natural

def denoteMonoid {n : Nat} : {value : ValueType Base} →
    MonoidSymbol value → Reduction (value.denote n denoteBase)
  | _, .naturalSum => natAdd

theorem lawfulMonoid {n : Nat} : {value : ValueType Base} →
    (monoid : MonoidSymbol value) →
      LawfulCommutativeReduction (denoteMonoid (n := n) monoid)
  | _, .naturalSum => natAdd_lawful

def signature (n : Nat) : Signature n Base denoteBase where
  Literal := Literal
  denoteLiteral := denoteLiteral
  Primitive := Primitive
  denotePrimitive := denotePrimitive
  Monoid := MonoidSymbol
  denoteMonoid := denoteMonoid
  lawfulMonoid := lawfulMonoid

end Alternate

/-- A proof-carrying translation from the example source vocabulary to a
distinct alternate vocabulary. Each certificate field is an exact denotational
equation, not a sampled test. -/
def alternateLowering (n : Nat) :
    Signature.Lowering (signature n) (Alternate.signature n) where
  literal := fun symbol =>
    match symbol with
    | .boolean value => .booleanImmediate value
    | .natural value => .naturalImmediate value
  primitive := fun symbol =>
    match symbol with
    | .add => .naturalAdd
    | .different => .naturalNotEqual
  monoid := fun symbol =>
    match symbol with
    | .naturalAdd => .naturalSum
  literal_correct := fun symbol => by cases symbol <;> rfl
  primitive_correct := fun symbol _ => by cases symbol <;> rfl
  monoid_correct := fun symbol => by cases symbol <;> rfl

def translatedPathProgram (n : Nat) :
    MonoidalFrontierBSP.Program (Alternate.signature n)
      natural natural natural :=
  (pathProgram n).lower (alternateLowering n)

/-- The concrete signature translation preserves every finite source run. -/
theorem translatedPathProgram_run_correct
    (graph : OrderedGraph n Nat)
    (fuel : Nat)
    (configuration : Configuration n Nat) :
    MonoidalFrontierBSP.run graph (translatedPathProgram n)
        fuel configuration =
      MonoidalFrontierBSP.run graph (pathProgram n)
        fuel configuration :=
  MonoidalFrontierBSP.run_lower
    graph (pathProgram n) (alternateLowering n) fuel configuration

/-- Compilation and the concrete signature translation commute for every
finite execution of the example. -/
theorem translatedPathProgram_compile_run_correct
    (graph : OrderedGraph n Nat)
    (fuel : Nat)
    (configuration : Configuration n Nat) :
    MonoidalFrontierBSP.run graph (translatedPathProgram n)
        fuel configuration =
      PullMapPush.run graph
        ((compile (pathProgram n)).lower (alternateLowering n))
        fuel configuration :=
  lower_compile_run_correct
    graph (pathProgram n) (alternateLowering n) fuel configuration

/-- Directly, the original example source equals the compiled program in the
distinct target vocabulary for every finite execution. -/
theorem pathProgram_translated_compile_run_correct
    (graph : OrderedGraph n Nat)
    (fuel : Nat)
    (configuration : Configuration n Nat) :
    MonoidalFrontierBSP.run graph (pathProgram n) fuel configuration =
      PullMapPush.run graph
        ((compile (pathProgram n)).lower (alternateLowering n))
        fuel configuration :=
  compile_lowered_run_correct
    graph (pathProgram n) (alternateLowering n) fuel configuration

end TraversalAlgebra.Verified.Typed.Example
