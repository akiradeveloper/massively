namespace TraversalAlgebra

/-- Executable reduction data. Its laws are kept separately so the same object
can be evaluated after proof erasure. -/
structure Reduction (α : Type u) where
  identity : α
  combine : α → α → α

/-- Laws needed for reassociation and parallel tree reduction. -/
structure LawfulReduction {α : Type u} (reduction : Reduction α) : Prop where
  associative : ∀ a b c,
    reduction.combine (reduction.combine a b) c =
      reduction.combine a (reduction.combine b c)
  leftIdentity : ∀ a, reduction.combine reduction.identity a = a
  rightIdentity : ∀ a, reduction.combine a reduction.identity = a

/-- Additional law needed when a parallel implementation may reorder
destination collisions. -/
structure LawfulCommutativeReduction {α : Type u}
    (reduction : Reduction α) : Prop extends LawfulReduction reduction where
  commutative : ∀ a b, reduction.combine a b = reduction.combine b a

/-- Natural-number addition as an executable reduction. -/
def natAdd : Reduction Nat where
  identity := 0
  combine := Nat.add

theorem natAdd_lawful : LawfulCommutativeReduction natAdd where
  associative := Nat.add_assoc
  leftIdentity := Nat.zero_add
  rightIdentity := Nat.add_zero
  commutative := Nat.add_comm

end TraversalAlgebra
