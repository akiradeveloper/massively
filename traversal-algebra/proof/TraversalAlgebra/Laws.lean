import TraversalAlgebra.Semantics

namespace TraversalAlgebra

@[simp]
theorem traverse_nil (graph : Graph) : graph.traverse [] = [] := by
  simp [Graph.traverse]

@[simp]
theorem traverse_append (graph : Graph) (left right : Frontier) :
    graph.traverse (left ++ right) = graph.traverse left ++ graph.traverse right := by
  simp [Graph.traverse]

@[simp]
theorem expand_length (graph : Graph) (source : Nat) :
    (graph.expand source).length = (graph.row source).length := by
  have expandFromLength : ∀ (index : Nat) (destinations : List Nat),
      (Graph.expandFrom graph source index destinations).length = destinations.length := by
    intro index destinations
    induction destinations generalizing index with
    | nil => simp [Graph.expandFrom]
    | cons destination destinations induction =>
        simp [Graph.expandFrom, induction]
  simpa [Graph.expand] using expandFromLength 0 (graph.row source)

theorem source_of_mem_expand (graph : Graph) (source : Nat) (context : EdgeContext)
    (member : context ∈ graph.expand source) : context.source = source := by
  have sourceOfMem : ∀ (index : Nat) (destinations : List Nat),
      context ∈ Graph.expandFrom graph source index destinations → context.source = source := by
    intro index destinations
    induction destinations generalizing index with
    | nil => simp [Graph.expandFrom]
    | cons destination destinations induction =>
        simp only [Graph.expandFrom, List.mem_cons]
        intro membership
        cases membership with
        | inl head =>
            subst context
            rfl
        | inr tail => exact induction (index + 1) tail
  exact sourceOfMem 0 (graph.row source) member

@[simp]
theorem emit_append (graph : Graph) (left right : Frontier)
    (map : EdgeContext → α) :
    emit graph (left ++ right) map = emit graph left map ++ emit graph right map := by
  simp [emit]

/-- Mapping after emission fuses into the edge map. This is the elementary
map-fusion law used by a fused GPU lowering. -/
theorem emit_map_fusion (graph : Graph) (frontier : Frontier)
    (first : EdgeContext → α) (second : α → β) :
    (emit graph frontier first).map second =
      emit graph frontier (second ∘ first) := by
  simp [emit, Function.comp_def]

@[simp]
theorem reduceBySource_append (graph : Graph) (left right : Frontier)
    (reduction : Reduction α) (map : EdgeContext → α) :
    reduceBySource graph (left ++ right) reduction map =
      reduceBySource graph left reduction map ++
      reduceBySource graph right reduction map := by
  simp [reduceBySource]

@[simp]
theorem reduceByDestinationAt_nil (graph : Graph) (reduction : Reduction α)
    (map : EdgeContext → α) (destination : Nat) :
    reduceByDestinationAt graph [] reduction map destination = reduction.identity := by
  simp [reduceByDestinationAt]

end TraversalAlgebra
