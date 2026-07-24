# Proposal: Power Diagram の dense all-pairs を疎な階層探索へ置き換える

- 状態: Draft
- 基準: Massively v0.87
- 対象: Massively core と、その上に構築する spatial companion module
- 主な利用事例: Power/Voronoi diagram、kNN、半径近傍、衝突 broad phase、
  粒子法、ray/AABB query、階層的クラスタ探索

## 結論

現在の Power Diagram 実装のオーダーを下げるために最も必要なのは、
`uniform_grid` や `PowerDiagram` という固有 API ではない。

オーダーを下げる直接の主体は、Power 側の power-aware BVH algorithm である。
Massively API を増やすだけでは下がらない。その algorithm を少ない専用機構で書くため、
まず追加候補にする最小抽象は次である。

1. **lane context + 全 lane 共通の read-only indexed view を処理する `map_shared`**
   - fixed `N` dispatch のまま、全 query が同じ BVH を dynamic index できる。
   - allocation、device-produced length、汎用 worklist を導入しない。
   - BVH 以外にも lookup table、mesh、辞書、static graph へ再利用できる。

その上で、Power 専用 kernel と kNN/AABB の第二用途から同じ形が確認できた場合だけ、
次を public API へ抽出する。

2. **共有 read-only hierarchy に対する batched stateful query**
   - 一つの query を一つの GPU lane が担当する。
   - 各 lane は query-local state と bounded local queue/stack を持つ。
   - node predicate により部分木を保守的に prune する。
   - leaf の処理により query state が更新され、その後の prune 判定も変化できる。
3. **`Segmentation` を slot plan として使う bounded segmented output**
   - query ごとの物理出力上限を表す。
   - required count、written count、overflow status は device 上に保持する。
   - 動的長を通常の host-exact `MIter` として偽装しない。
   - exact terminal で一度だけ長さと overflow を観測する。

Power Diagram では、最初は `map_shared` または現 API の仮想 shared segment を使う
専用 kernel として次を実装する。

```text
sites
  -> Morton sort
  -> LBVH(AABB + subtree max weight)
  -> one query / power cell
  -> query-local best-first traversal
  -> conservative directional culling
  -> leaf site による逐次 clipping
  -> polygon / moments / adjacency
```

この形なら、全 site pair を事前生成しない。メモリは dense な `O(N²)` から、
BVH の `O(N)` と query ごとの bounded scratch へ変わる。仕事量は
「全 pair 数」ではなく、訪問 node、queue 操作、実際の clipping cost に比例する。

一方、以下は最初の Massively core API には入れない。

- Power Diagram 固有の culling 式
- `UniformGrid2d` と `Lbvh` を無理に統一する巨大な `SpatialIndex` trait
- 任意 algorithm を回す public `Worklist` / `Iteration`
- dynamic length を通常の `MIter` として公開する API
- silent truncation を許す atomic append

## この proposal が扱う問題

以前の Power Diagram proposal は、可変長 clipping の readback と dispatch 数を
主な問題としていた。v0.87 では、その議論を経て次の方針が採用されている。

- 通常 API は host-exact に保つ。
- `MVal` と device logical extent は crate-private に置く。
- 可変長 result は public return boundary で正確な長さへ解決する。
- CSR/ragged data の共通表現には `seg::Segmentation` を使う。
- 汎用 public `Worklist` / `Iteration` は追加しない。

この proposal の問題はそれとは異なる。

現在の性能限界は、同じ dense relation に対する dispatch を減らすことではなく、
最初から `N × N` の relation を作っていることである。kernel fusion、workspace
再利用、高性能 dGPU は定数倍を改善するが、この cardinality は変えない。

したがって必要なのは「dense relation を高速に処理する API」ではなく、
**不要な relation entry を生成せずに済む query abstraction** である。

## 現在の計算量

domain の辺数を `d`、site 数を `N` とする。現在は各 cell に対し、他の全 site と
domain edge を制約にする。

```text
m_i = N - 1 + d
```

全制約数は次になる。

```text
C = Σ_i m_i = N(N - 1 + d) = Θ(N²)
```

さらに、各 candidate constraint は同じ cell の constraint を最大 `m_i - 1` 個
調べる。

```text
T_upper = Σ_i m_i² = Θ(N³)
```

これは上限であり、通常入力では interval fold の早期終了により大部分の candidate が
全 constraint を走査する前に脱落する。そのため実測 GPU wall time の傾きは 3 より
かなり小さくなり得るが、それを algorithmic exponent と解釈してはいけない。再測定時は
各 `N` の raw time、入力 seed/distribution、adapter、path、repeat 数、log-slope の
算出法を一緒に保存する。

全 path で必ず `Θ(N²)` になるのは次である。

- constraint topology
- constraint columns
- 最低 memory traffic

polygon materialization と large-cell moments path では、さらに candidate output slots、
sort/filter 対象も `Θ(N²)` になる。small-cell direct moments path はそれらを確保しないが、
全 constraint の resident topology/columns は残る。

`N=10,000` では約 1 億 constraint、resident memory の下限だけで約 1.86 GiB
だった。一方、最終 polygon の頂点数は 59,633、すなわち約 `5.96N` だった。
最終的に必要な境界が線形規模なのに、その約 1,677 倍の constraint を生成している。

## 目標とする sparse formulation

cell `i` について、実際に調べる competitor site の集合を `R_i`、その件数を
`k_i` とする。

現在の interval 法をそのまま使う場合でも、計算量は次になる。

```text
storage = O(Σ_i (d + k_i))
work    = O(Σ_i (d + k_i)²)
```

`Σ k_i = O(N)` かつ `k_i` の二次モーメントも bounded なら、両方とも実質線形になる。
平均だけが定数でも、一部の `k_i` が極端に大きければ `Σ k_i²` は線形にならない。
退化配置では二乗へ戻る可能性があるが、dense all-pairs より大幅に小さい。

さらに、hierarchy traversal と clipping を一つの query に融合すれば、
candidate relation 自体を materialize する必要もない。

```text
work = O(visited BVH nodes
       + priority queue operations
       + Σ accepted leaf の clip cost)
```

2D convex polygon の一回の clip cost は、その時点の polygon 頂点数に比例する。
この work は分布、weight、BVH quality、culling bound に依存する。一般入力に対し
常に `O(N log N)` を保証するという proposal ではない。general-position などの前提を
置いた理論保証を優先するなら、regular triangulation を構築する別実装も検討対象に
なる。

本 proposal の目標は次である。

- dense `O(N²)` allocation を除去する。
- arbitrary weight でも missed cutter がない保守的 pruning を可能にする。
- visited node 数と accepted leaf 数を計測可能にする。
- Power Diagram 以外でも同じ traversal/output engine を再利用する。

## 固定近傍 grid だけでは不十分

unweighted Voronoi や fixed-radius neighbor search では、uniform grid や spatial hash
が有効である。しかし arbitrary weight の Power Diagram では、固定 `3×3` stencil、
固定半径、固定 `k`-nearest は exact ではない。

非常に大きい weight を持つ遠方 site の bisector が、現在の cell を切る場合がある。
したがって次のどちらかが必要になる。

1. 未訪問領域に cell を切れる site が存在しないことを証明する。
2. regular triangulation を構築し、真の隣接 site だけを使う。

grid は warm start や fixed-radius 用途には残す価値があるが、Power Diagram の
correctness primitive にはしない。

## Exact な階層 pruning

Power distance を次とする。

```text
π_i(x) = ||x - p_i||² - w_i
```

site `j` との bisector までの `i` からの符号付き距離は、次で表せる。

```text
d_ij = (||p_i - p_j||² + w_i - w_j) / (2 ||p_i - p_j||)
```

進行中 cell の、candidate 方向への conservative radius を `r_i` とすると、
`d_ij > r_i` である site `j` は cell を切れない。

BVH node が次を保持すると、leaf を一つずつ調べずに部分木全体を prune できる。

- subtree の site を含む AABB
- subtree 内の `max(weight)`
- child topology

node AABB と `max(weight)` から、その node 内にある全 site の
`d_ij` の下限を計算する。query site から node AABB までの Euclidean distance を `d`、
`δ = w_i - max_weight(node)` とすると、論文の保守的 lower bound は次である。

```text
δ <= 0 and d > 0: lower = d/2 + δ/(2d)
δ > 0:            lower = d/2
d == 0:           prune しない
```

2D node AABB が query site に対して複数 quadrant をまたぐ場合は、占有する全 quadrant の
directional radius を計算し、その最大値を `r_i(node)` に使う。

```text
lower > r_i(node) + safety_margin
```

の時だけ node 全体を捨てる。lower は小さい側、radius は大きい側へ丸める。等号、
非有限値、overflow、丸め誤差の符号を証明できない場合は prune せず descend する。
従って「exact」は実数上の culling criterion を指し、`f32` 実装では有限な入力範囲、
外向き tolerance、status/fallback を含めて false negative を防ぐ。

実装では各 diagram の domain AABB を基準に position を bounded range へ正規化し、
weight を length scale の二乗で同時に変換する。正規化後の position²、weight difference、
bound の全中間値が文書化した `f32` safety range に入ることを launch 前に検証する。
安全に正規化できない diagram は高精度 CPU path へ送る。

### coincident site

`p_i == p_j` では `d_ij` の分母が 0 になるため、通常の bisector 式へ入れない。
同一位置の site は build 前に deterministic に処理する。

- 最大 weight の site だけが残り、それより小さい weight の site は empty。
- 最大 weight が同値なら最小 global site ID を owner とし、他を empty。
- owner query では同位置の dominated site を無視する。

near-coincident で `f32` の安全な範囲を外れる場合は uncertain status を立て、
高精度 CPU oracle または保守的 dense path へ fallback する。

重要なのは cell が query 中に変化することである。

```text
近い leaf を clip
  -> cell が縮む
  -> cell AABB / directional radius が縮む
  -> 以前 queue に入れた node も prune 可能になる
```

従って、単なる stateless range query では足りない。leaf 処理によって query state が
更新され、その state を後続 node predicate が参照できる必要がある。

2026 年の GPU Power Diagram 研究でも、進行中 cell の directional bounds と
`max(weight)` を持つ BVH の best-first traversal を組み合わせ、任意の空間分布と
weight への対応を狙っている。ただし同研究は 3D/CUDA 実装であり、本 proposal の
2D/WGPU 性能を直接保証するものではない。ここでは pruning semantics とデータ構造の
実証例として参照する。

## Massively v0.87 に既にある部品

次は既存 primitive で表現できる。

| 処理 | v0.87 の部品 |
|---|---|
| Morton/cell key の生成 | `lazy::map` / `vector::map` |
| key sort | `radix_sort_by_key` |
| run boundary | `adjacent_difference` / `unique_by_key` |
| group offsets | `Segmentation::from_lengths` / `from_segment_ids` |
| prefix allocation | scan |
| node aggregate | segmented/reduce/scatter-reduce |
| fixed-shape result | `vector::map` / caller-provided output |
| CSR traversal | `graph::traverse` |
| SoA context | `zipN`、`lazy::permute` |

uniform grid の次の compound recipe も既存 API だけで記述できる。

```text
map point -> cell key
radix sort (cell key, original id)
Segmentation from sorted cell ids
map query -> neighbor cell keys
lower_bound / upper_bound
flat_map matched ranges
permute payload
filter exact predicate
```

従って `group_by_key` や equi-join を直ちに新 primitive とする必要はない。
まず標準 recipe と benchmark を用意し、複数用途で同じ余分な materialization が
確認された場合に compound algorithm へ昇格すればよい。

なお v0.87 の public dynamic operation は host-exact result を返すため、この recipe の
`flat_map` などは return boundary で長さを解決する。既存 API で記述可能であることと、
複数段を同期なしに fusion できることは同義ではない。同期/temporary が支配的だと
測定された場合に限り、意味を保った compound implementation で内部 fusion する。

## v0.87 に不足している意味

### 1. 全 query から共有される random-access table の一級表現

通常の `UnaryOp` は一つの logical row を受け取る。`Segment` 自体は `FlatRow` では
ないため、通常 iterator の row として単純に `zip` することはできない。

ただし、現行公開 API だけでも試作は可能である。`N × node_count` の仮想 stream を
`counting -> div/mod -> permute(nodes/queries) -> zip` で作り、
`SegmentIterator::new(offsets = i * node_count)` へ渡せば、物理複製なしで各 query に
同じ BVH を見せられる。各 `Segment<(Node, Query)>` は dynamic `at()` できる。

従って shared table は新 public primitive の絶対条件ではない。一方、この lowering は
`N * node_count` の `MIndex` overflow、余分な div/mod、read binding 上限、意図の
分かりにくさを伴う。専用 kernel と比較して有意な差が出るなら、read-only table view
を性能と使い勝手のための一級表現として追加する価値がある。

### 2. leaf 処理で更新される query-local state

通常の map、filter、graph edge traversal は、現在の query state に応じて次に読む
node の順序と pruning predicate が変わる探索を表さない。

Power cell polygon、AABB、directional radii、priority queue は query 中だけ存在する
local state である。

### 3. query ごとの bounded variable output

moments は一つの fixed row として返せるが、polygon、adjacency、candidate IDs は
query ごとに長さが異なる。

この動的結果を capacity 全体の通常 `MIter` として公開すると、invalid slot や前回値を
後続 algorithm が読む。v0.87 の host-exact length 契約とも矛盾する。

### 4. `graph::Traversal` は近いが十分ではない

`graph::Traversal` は private device length と host-known capacity を持ち、CSR row を
frontier から展開できる。しかし hierarchy query とは次が異なる。

- graph vertex frontier とは異なり、query domain と node/leaf domain が明確に別である。
- 同じ node を複数の独立 query が訪問し、query owner を保持する必要がある。
- leaf 処理が query state を更新する。
- enqueue priority は push 時の state に対する snapshot である。
- pop 後、更新済み state に対して node viability を必ず再評価する。
- one-thread-per-query の local queue lowering が有力である。

global frontier の反復だけへ落とすと、各 tree level の dynamic sequence と host
boundary、または public Worklist が再び必要になる。Power cell のように query-local
state が小さい場合は、local traversal の方が自然である。

queued node 全体を cell 更新のたびに re-key する必要はない。priority/order は性能だけに
影響し、正確性は pop-time viability check が担う。初版の best-first は min-priority とし、
finite priority の比較、同値時の node ID tie-break を定義する。NaN priority は status
を立てて prune しない。DFS と best-first は runtime branch ではなく別 kernel
specialization にする。

## 提案 1: `map_shared`

最小の Massively 追加候補として、lane ごとの context と、全 lane に共有される
read-only random-access view を一つの fixed-length map へ渡せるようにする。
名前と正確な trait 境界は議論用である。

```rust,ignore
pub struct SharedView<Item> {
    // launch 中だけ有効な semantic kernel argument。storage を所有しない。
}

impl<Item: CubeType> SharedView<Item> {
    pub fn len(&self) -> MIndex;
    pub fn get(&self, index: MIndex) -> Option<Item>;

    // caller が index < len() を証明済みの内部高速 path。
    pub(crate) unsafe fn at_unchecked(&self, index: MIndex) -> Item;
}

pub trait SharedUnaryOp<Context, Item>: CubeType {
    type Output: CubeType;

    fn apply(context: Context, shared: &SharedView<Item>) -> Self::Output;
}

pub fn map_shared<R, Contexts, Values, Op>(
    exec: &Executor<R>,
    contexts: Contexts,
    values: Values,
    op: Op,
) -> Result<MVec<R, Op::Output>, Error>
where
    R: Runtime,
    Contexts: MIter<R>,
    Values: MIter<R>,
    Op: SharedUnaryOp<Contexts::Item, Values::Item>,
    Op::Output: MAlloc<R>;
```

`Values::Item` は `zipN` により SoA node row にできる。`SharedView::get` は GPU 内で
dynamic index し、範囲外なら `None` を返す。実装上は guarded load と status tuple に
lower してよい。validated `BinaryForest` を読む
crate-private lowering だけが `at_unchecked` を使う。view は launch に borrow される
immutable snapshot で、元 storage への mutable alias を保持しない。

この API は allocation、readback、可変 extent を増やさない。`Contexts` に
`SegmentIterator` を使える lowering なら、query-local initial guess segment も
context にできる。固定 row の moments/centroid/status は通常の host-exact `MVec` として
返る。

ただし、現行 API の仮想 segmented view でも同じ kernel semantics を試作できる。
まず両者を実装し、`N * node_count` overflow 回避、div/mod 削減、read binding 削減が
実測できた時点で `map_shared` を採用する。hierarchy、queue、overflow policy はこの
低レベル API 自体には埋め込まない。

## 提案 2: batched stateful hierarchy query

専用 Power kernel と第二用途で semantics が一致した場合の generic extraction candidate
を `hierarchy::query` とする。

```rust,ignore
pub struct NodeRef {
    // internal node / leaf の tagged validated index
}

pub struct BinaryForest<Nodes, Leaves, Roots> {
    nodes: Nodes,   // child refs と query-specific aggregate を含む
    leaves: Leaves,
    roots: Roots,   // independent tree ごとの NodeRef
}

pub struct QueryConfig<const QUEUE_CAPACITY: usize, Policy> {
    policy: PhantomData<Policy>,
}

pub struct DepthFirst;
pub struct BestFirstMin;

pub fn query_bounded<
    'w, R, Queries, Nodes, Leaves, Roots, Op, Policy, const Q: usize
>(
    exec: &Executor<R>,
    forest: BinaryForest<Nodes, Leaves, Roots>,
    queries: Queries, // item = (tree_id, Query)
    op: Op,
    config: QueryConfig<Q, Policy>,
    workspace: &'w mut QueryWorkspace<R, Op::Hit>,
) -> Result<QueryRun<'w, R, Op::Summary, Op::Hit>, Error>
where
    R: Runtime,
    Op::Summary: MAlloc<R>,
    Op::Hit: MAlloc<R>;
```

初版では binary hierarchy だけでよい。arbitrary arity は CSR children へ一般化できるが、
API と lowering を同時に広げない。`QUEUE_CAPACITY` は host runtime の単なる値ではなく、
CubeCL の local `Array` 長を決める compile-time specialization key とする。Power
polygon の最大頂点数も operation 側の同様な compile-time parameter にする。`Policy`
も type-level specialization とし、初版は deterministic DFS と finite priority の
`BestFirstMin` に限定する。

可変出力が不要な場合は、workspace を要求しない `query_reduce` overload を用意する。
これが Treemap iteration の moments-only path になる。

`BinaryForest` は sealed/validated snapshot とする。constructor は root/tag/index range、
tree 境界、child topology、cycle、leaf range を検証し、元 buffer への mutable alias を
保持しない。query engine は `(tree_id, query)` から root を選ぶため、独立 diagram 間を
誤って traversal しない。内部 builder が生成した topology には crate-private な
validated constructor を使えるが、public unchecked constructor は設けない。

### Query operation

```rust,ignore
pub trait HierarchyQueryOp<Query, Node, Leaf>: CubeType {
    type State: CubeType;
    type Priority: CubePrimitive;
    type Summary: CubeType + Send + Sync + 'static;
    type Hit: CubeType + Send + Sync + 'static;

    fn init(query: Query) -> Self::State;

    // push 時と pop 後に呼ぶ。false なら subtree を prune する。
    fn should_visit(
        query: Query,
        state: &Self::State,
        node: Node,
    ) -> bool;

    // viable child を enqueue する時の snapshot score。
    fn enqueue_priority(
        query: Query,
        state: &Self::State,
        node: Node,
    ) -> Self::Priority;

    // state を更新する。必要なら query 専用 sink へ出力できる。
    fn leaf(
        query: Query,
        state: &mut Self::State,
        leaf: Leaf,
        output: &mut BoundedSink<Self::Hit>,
    );

    // Power polygon や top-k のように、最終 state から初めて決まる item も
    // この hook で query 専用 sink へ列挙できる。
    fn finish(
        query: Query,
        state: Self::State,
        output: &mut BoundedSink<Self::Hit>,
    ) -> Self::Summary;
}
```

これは概念 API であり、`BoundedSink` を Rust/CubeCL のこの形で公開するという決定では
ない。実装では `NoOutput` と per-segment bounded sink を別 kernel specialization にし、
tuple/status 形式へ lower してよい。重要な意味は次である。

- hierarchy storage は全 query で共有される。
- state は query-local である。
- pop 後の `should_visit` は更新済み state を読む。
- queue policy は Massively 側が所有する。
- leaf processing と state update は同じ query kernel 内に留められる。
- leaf 時の streaming hit と、finish 時の final-state item の両方を表現できる。
- summary は一 query 一 row の fixed-shape result である。

### Query-local storage

Power Diagram の `State` は概ね次を持つ。

```text
bounded convex polygon
polygon length
cell AABB
directional radii
status
```

queue と polygon は compile-time capacity の CubeCL local `Array` で実装できる可能性が
ある。ただし、capacity が大きいと register pressure と private-memory spill が増える。
参照研究は per-thread state をすべて register/local array に置かず、lane 間で
coalesced access になる transposed global SoA scratch も使っている。従って public
semantics は storage placement を固定せず、次を比較する。

- queue capacity
- polygon vertex capacity
- query chunk size
- local `Array` と slot-major/transposed global SoA scratch
- one lane/query と one subgroup/query
- DFS と best-first

global SoA が複数用途で勝つ場合だけ、次の storage abstraction を別 candidate とする。

```rust,ignore
pub struct LaneScratchWorkspace<R, T, const CAP: usize> {
    // private slot-major storage
}

pub fn map_shared_with_scratch<..., const CAP: usize>(
    ...,
    scratch: &mut LaneScratchWorkspace<R, StateItem, CAP>,
    ...
) -> Result<CheckedRun<...>, Error>;
```

意味は「lane `i` だけが logical row `i` の `CAP` slots を一 launch 中だけ変更できる」
であり、物理 layout は coalescing のため slot-major にしてよい。scratch は result
`MIter` ではなく、raw handle を外へ出さない。priority queue、top-k、小 hash table、
bounded dynamic programming state に再利用できる。local `Array` で十分なら追加しない。

初期実装では one lane/query を優先する。Power/Voronoi cell、kNN、small BVH query は
query 間の並列性が十分に大きく、query 内 cooperative scheduling より単純である。
dispatch extent は常に host-known な query count である。tree level ごとの
device-active frontier、indirect dispatch、各 round の length readback を必要とせず、
traversal と moments calculation を一つの fixed-`N` kernel に閉じ込められる。

## 提案 3: bounded segmented output

P5 で polygon/adjacency が必要になり、別用途でも同じ契約が確認できた場合、
query ごとの hit、polygon vertex、adjacency を安全に出力する共通構造を追加する。
物理 slot plan には既存 `Segmentation` を使う。

hierarchy に結合しない最小 surface は `bounded_map` とする。

```rust,ignore
pub trait BoundedEmitOp<Input>: CubeType {
    type Item: CubeType;
    type Summary: CubeType;

    fn apply(
        input: Input,
        output: &mut BoundedSink<Self::Item>,
    ) -> Self::Summary;
}

pub fn bounded_map<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    slots: Segmentation<R>,
    op: Op,
) -> Result<BoundedRun<R, Op::Summary, Op::Item>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Op: BoundedEmitOp<Input::Item>,
    Op::Item: MAlloc<R>,
    Op::Summary: MAlloc<R>;
```

一様な `CAP` には `bounded_map_uniform<const CAP: usize>` convenience overload を用意できる。
物理表現は host-known な fixed slots + per-row count/status であり、device length を持つ
public `MVec` ではない。CSR が必要な terminal だけ exact compaction する。

allocation を反復再利用する必要がある場合の lowering が `QueryWorkspace` である。

```rust,ignore
pub struct QueryWorkspace<R: Runtime, T: MAlloc<R>> {
    // private:
    // slots: Segmentation<R>
    // payload storage
    // required/written counts
    // overflow status
    // compaction scratch
}

impl<R: Runtime, T: MAlloc<R>> QueryWorkspace<R, T> {
    pub fn new(
        exec: &Executor<R>,
        slots: Segmentation<R>,
    ) -> Result<Self, Error>;
}
```

`slots.segment_count() == query_count` とする。
workspace は reusable だが `Clone` にはせず、一回の `QueryRun` が `&mut` borrow する。
run が生きている間の再利用と、別 query から同じ slots への alias write を型で防ぐ。

```text
required[i] = query i が必要とした出力数
capacity[i] = slots.lengths[i]
written[i]  = min(required[i], capacity[i])
```

global capacity だけを共有しない。query ごとの segment を使うことで、先頭 query が
容量を使い切って後半 query を飢餓させることを防ぐ。
one-lane/query lowering では、各 lane は自分の segment だけへ順に書くため global
atomic append は不要である。sink は全 item に対して `required[i]` を増やし、
`local_index < capacity[i]` の時だけ payload と `written[i]` を更新する。

`QueryRun` は通常の `MIter` にはせず、workspace-backed storage の raw view も返さない。
v0.87 の device handles/views は clone 可能なため、借用した `&DeviceVec` を公開すると
workspace 再利用中の alias write を防げないからである。

```rust,ignore
pub struct QueryOutput<R, Summary, Hit> {
    pub summaries: MVec<R, Summary>,     // owned snapshot
    pub hits: MVec<R, Hit>,              // host-exact
    pub segmentation: Segmentation<R>,
}

pub struct QueryFailure {
    pub per_query: Vec<QueryFailureRecord>,
}

impl<'w, R, Summary, Hit> QueryRun<'w, R, Summary, Hit> {
    // 全 status と長さを観測後にだけ owned result を返す。
    pub fn finish_exact(
        self,
        exec: &Executor<R>,
    ) -> Result<QueryOutput<R, Summary, Hit>, QueryFailure>;

    // HIT_SLOT_OVERFLOW だけを明示的に許す terminal。
    pub fn finish_prefix(
        self,
        exec: &Executor<R>,
    ) -> Result<QueryOutput<R, Summary, Hit>, QueryFailure>;
}
```

per-query status は少なくとも `HIT_SLOT_OVERFLOW`、`QUEUE_OVERFLOW`、
`STATE_OVERFLOW`、`COUNT_OVERFLOW`、`INVALID_INDEX`、`NON_FINITE` を区別する。
`finish_exact` は一つでも status があれば summary を含む部分結果を返さない。
`finish_prefix` が成功扱いできるのは `HIT_SLOT_OVERFLOW` だけであり、queue/state/index
などに失敗した traversal の prefix は返さない。`QueryFailure::per_query` により、
capacity specialization を変えた再実行または failed chunk の dense fallback ができる。

成功 path の host observation を一回にするには、query kernel の後へ count scan と
status reduction を enqueue し、packed status/total を一度 readback する。その後、
kernel 自身が生成した validated offsets から crate-private constructor で
`Segmentation` を組み立てる。public
`Segmentation::from_lengths/from_offsets` を再度呼ぶと validation readback が増えるため
使わない。public unchecked constructor は追加しない。

moments-only の `query_reduce` も同じ status gate を持つ
`QueryReduceRun::finish_exact` とし、成功確認前に summary handle を外へ出さない。
device 上の iteration を status check をまたいで連鎖させる必要があるなら、
workspace-scoped consumer または checked device value の別設計が必要である。この契約を
決める前に generic hierarchy API を公開しない。

workspace は作成元 `Executor` の owner/policy と query count を保持する。launch 前に
`queries.len() == slots.segment_count()`、root/tree ID、総 slot 数、総 byte 数を
checked arithmetic で検証する。`required` は capacity 超過後も saturating count し、
`MIndex` overflow 前に sticky status を立てる。

### empty と順序

初版の出力順は deterministic にする。

```text
query input order
  -> traversal policy が定める visit order
    -> leaf-local output order
```

- DFS の child push order、best-first の min comparator、`-0/+0` と同値 priority、
  node ID tie-break、leaf 内 order を固定する。NaN priority は error。
- query 0 件なら offsets は `[0]`。
- hit 0 件の query も repeated offset で保存する。
- trailing empty query を失わない。
- atomic append は既定にしない。
- unordered 高速版を追加する場合は `_unordered` を名前に含める。

default の保証は「同じ backend、同じ float mode、同じ入力で repeatable」とする。
backend をまたぐ bitwise-identical polygon は保証せず、正確性は missed cutter がないこと、
area/centroid/vertex が tolerance 内であること、adjacency の canonical ID set が一致する
ことで判定する。backend 横断で canonical vertex order が必要な terminal だけ、最終
angle/key sort を明示的に行う。

## Power Diagram での利用

### BVH build

1. diagram ごとの normalized site position から Morton code を作る。
2. `(diagram_id, morton, original_site_id)` を radix sort する。
3. diagram 境界ごとに binary radix tree/LBVH topology と root を作る。
4. leaf から AABB と `max(weight)` を bottom-up aggregate する。

LBVH の全 internal node を並列構築する方法は既知であり、Morton sort、prefix/radix
relation、node-local writes が中心になる。ただし、Karras 型 topology build は
sorted key table を data-dependent index するため、ここでも read-only indexed context
が有用である。同一 Morton code は `original_site_id` まで含む一意 key で tie-break
し、prefix 計算が duplicate spatial key を誤って同一 leaf と扱わないようにする。

参照した Power Diagram 研究は cuBQL の binary BVH builder を使っており、Morton LBVH
で同じ tree quality/性能が出ることを示してはいない。Karras LBVH は Massively v0.87
で構成しやすい初期選択であり、SAH/quality-aware builder と独立に benchmark する。

LBVH を最初から Massively core の汎用型にしない。まず `massively-spatial` 相当の
companion module か、この project 内の実験実装で build/query contract を固める。

Voronoi Treemap のように独立 diagram が複数ある場合は、group ごとに root を持つ
forest として node arrays を連結し、query は自分の root だけを読む。小 group は
後述の dense path に残す。read-only hierarchy は一回の query batch に対する snapshot
であり、site position/weight が変われば rebuild または保守的な refit が必要になる。
topology reuse/refit/rebuild の選択は spatial layer の policy とし、core query API には
埋め込まない。Treemap 反復では毎回 AABB と `max(weight)` を更新し、refit tree の
overlap/visited-node 数が悪化したら Morton topology を rebuild する。古い aggregate を
一回でも query に使うことは禁止する。

### Cell query

query `i` は次を行う。

```text
state = domain polygon
queue = [BVH root]

while queue is not empty:
    node = pop best node

    if conservative_bound(node, state) says impossible:
        continue

    if node is internal:
        score children from current state
        push viable children
    else:
        for site j in leaf:
            if j != i:
                clip state polygon by bisector(i, j)
                update AABB/directional bounds

return polygon, moments, adjacency, status
```

Power operation は leaf visit 時に polygon/adjacency item を emit しない。その plane は
後続 clip で最終境界から消える可能性があるためである。boundary label を polygon state
と一緒に更新し、`finish` で final state だけを列挙する。collision pair のような
leaf-streaming output は別 terminal mode を使う。

moments-only Treemap iterationでは、`Summary` を
`(area, centroid_x, centroid_y, valid)` とし、variable hit を出さなくてよい。
これにより constraint relation と polygon materialization の両方を省ける。

最終描画時だけ、polygon vertex または adjacency を `QueryWorkspace` の per-query slots
へ出す。

### Hybrid dispatch

小さい diagram に BVH build/query の固定費を払わない。

```text
small group:
    現在の fused all-pairs path

large group:
    BVH hierarchy query path
```

Voronoi Treemap の一つの parent が持つ child 数は小さい場合が多い。workspace 全体の
`N` ではなく、各独立 diagram の group size で path を選ぶ。

## Grouping をどう扱うか

grid、LBVH leaf、edge-list-to-CSR に共通する compound pattern は次である。

```rust,ignore
pub struct Grouping<R: Runtime> {
    order: DeviceVec<R, MIndex>,
    segmentation: Segmentation<R>,
}

pub fn group_indices_by_id<R, Ids>(
    exec: &Executor<R>,
    ids: Ids,
    group_count: MIndex,
) -> Result<Grouping<R>, Error>;
```

契約は次とする。

- `ids[order[k]]` は非減少。
- 同一 ID 内は original index 順で deterministic。
- `segmentation.segment_count() == group_count`。
- empty/trailing group を保持する。
- payload を所有せず、`lazy::permute(payload, order)` で再利用する。
- ID range を検証する。
- fixed-width integer ID は radix sort を使う。

ただし、これは既に次の合成で表現できる。

```text
order = values from radix_sort_by_key((ids, original_index), original_index)
sorted_ids = permute(ids, order)
Segmentation::from_segment_ids(sorted_ids, group_count)
```

radix sort 自体の stability に依存する場合は、その契約を明記する。依存しない実装では
`(id, original_index)` を複合 key にして同値 ID の順序を一意にする。

従って実装順は次とする。

1. documented compound recipe と benchmark を追加する。
2. grid、BVH leaf grouping、edge-list-to-CSR の三用途で再利用する。
3. 同じ intermediate materialization/validation が繰り返され、内部 fusion に価値がある
   と測定できた時点で `Grouping` を public compound algorithm へ昇格する。

`join` はさらに後でよい。inner/left/semi/anti、duplicate、stable order、巨大 output の
契約を一度に追加せず、まず `lower_bound`、range expansion、`Segmentation` の合成を
使う。

## Generic use cases

### kNN / radius query

- query state: current top-k distance、search radius
- node predicate: AABB lower-bound distance
- leaf: candidate distanceを評価し top-k を更新
- summary: kNN distance/statistics
- hits: neighbor IDs

### Collision broad phase

- query state: query AABB / swept volume
- node predicate: overlap test
- leaf: exact shape test
- hits: candidate pair

### Particle method / molecular dynamics

- query state: interaction radius
- node predicate: AABB distance
- leaf: force contribution
- summary: accumulated force
- hits: optional neighbor list

### Ray / geometric query

- query state: closest hit distance
- node predicate: ray-AABB distance bound
- leaf: primitive intersection
- summary: nearest hit

### Inverted index / hierarchical clustering

- query state: score threshold / current top-k
- node predicate: subtree score upper bound
- leaf: exact score
- summary/hits: selected records

### Mesh and graph algorithms

- hierarchy: cluster tree、separator tree、mesh patch tree
- query state: local error bound
- node predicate: affected-region test
- hits: affected vertices/edges

これらはいずれも「shared read-only hierarchy + mutable query-local bound + bounded output」
という同じ意味を持つ。ただし Power polygon/top-k のような final-state output と、
collision pair のような leaf-streaming output は terminal policy を分ける。両者を
一つの曖昧な atomic append API に押し込まない。

## regular triangulation との関係

2D Power Diagram の dual である regular triangulation を先に構築すれば、各 cell は
regular neighbor だけで構築できる。非退化な planar embedding では active site の
総 adjacency は `O(N)` である。

これは理論的に強いが、GPU 実装には次が必要になる。

- robust orientation / in-circle または lifted predicates
- dynamic triangle/half-edge topology
- cavity/edge-flip frontier
- conflict-free claim
- stale work item detection
- bounded mutable slot storage

これらの多くは `del2d` と共有できる。しかし現時点で Massively core へ
`MutableMesh`、`ClaimAll`、汎用 topology allocator を追加する根拠はまだ弱い。

まず hierarchy query path を実装する。regular triangulation は次の条件を満たした時に
別 proposal とする。

- Power Diagram と `del2d` の二実装で同じ slot/generation/claim 不変条件が現れた。
- domain 固有処理を除いた共通部分が API として説明できた。
- hierarchy clipping path より優れる入力領域が benchmark で確認できた。

## Memory と capacity contract

### Host-known allocation

WebGPU kernel は device から新しい buffer を確保できない。従って物理 capacity は
host-known でなければならない。

初版では次を明示的に指定する。

- compile-time specialization される local queue capacity
- compile-time specialization される local state capacity
- global SoA lowering を選ぶ場合の host-known per-query scratch capacity
- query output slot lengths
- optional chunk size

### Overflow

次はすべて sticky status として記録する。

- local queue overflow
- local polygon/state overflow
- per-query hit slot overflow
- `MIndex` count overflow
- invalid node/leaf index
- non-finite query or hierarchy data

exact terminal は status を一度観測し、いずれかが立っていれば成功 result を返さない。

Power Diagram 側の recovery は次から選べる。

1. capacity を増やして chunk を再実行する。
2. overflow query だけ dense path へ fallback する。
3. CPU reference path へ fallback する。

candidate や polygon vertex を黙って捨てることは禁止する。Power Diagram では一件の
miss でも topology が変わるためである。

### Workspace lifetime

`QueryRun<'w>` は `&'w mut QueryWorkspace` を保持し、`Clone` できない。

- active run 中の workspace 再利用を型で防ぐ。
- status/count buffer の古い handle を外へ出さない。
- exact result は別 allocation へ compact/snapshot する。
- run drop 後の同一 queue 上の再利用には追加 `sync` を要求しない。
- `reserve` は active run 中に呼べない。

## CubeCL / WGPU feasibility

CubeCL には compile-time length の function-local `Array<T>` があるため、
small queue と small polygon state の実験は可能である。

ただし以下は feasibility spike で確認する。

- custom query `State` に local arrays を持たせられるか。
- trait method 間で state を不要に copy せず更新できるか。
- WGPU/WGSL、Vulkan、CPU backend で同じ制御フローが通るか。
- queue/polygon capacity ごとの register pressure と spill。
- dynamic index による bounds contract。
- input/context/output を pack し、現行 Massively の 13 read / 12 write slot 上限を
  超えないか。
- `f32` NaN/Inf を node priority と sort key へ入れない validation。

もし generic `State` が CubeCL frontend 上で不安定なら、初版は次の二段階に分ける。

1. Power Diagram 内部の専用 kernel で semantics と性能を実証する。
2. kNN または AABB query の二つ目の実装と共通形を比較してから
   `HierarchyQueryOp` を Massively へ抽出する。

専用 kernel による実証は、Massively API の失敗ではない。未検証の巨大 abstraction を
先に固定しないための設計手順である。

## 実装段階

### P0: 計測と oracle

- dense path の constraint 数、intersection step 数、temporary bytes を計測する。
- CPU all-pairs を exact oracle とする。
- uniform、clustered、density gradient、extreme weight、empty cell を用意する。
- group size ごとの all-pairs/BVH crossover を測る。

### P1: Grouping recipe

- Morton/cell ID grouping を既存 v0.87 primitiveだけで作る。
- deterministic order、empty group、invalid ID、`MIndex` overflow をテストする。
- まだ新 public API は追加しない。

### P2: 2D LBVH prototype

- Morton sort。
- binary radix tree topology。
- bottom-up AABB / `max(weight)` aggregate。
- build と query を別々に benchmark する。
- hierarchy storage は read-only snapshot とし、元 site buffer の mutable alias を
  保持しない。

### P3: Power-specific query kernel

- one lane/cell。
- bounded best-first queue。local `Array` と transposed global SoA を比較する。
- bounded convex polygon。
- directional node culling。
- moments-only fixed output。
- overflow chunk の dense fallback。

BVH 成功 path から dense `N²` allocation を除去し、正確性と scaling を確認する。
fallback が発生した run の peak memory/time は成功 path と分けて計測する。

### P4: Massively generic extraction

まず現行 API の仮想 shared segment と専用 lowering を比較し、効果が確認できた場合だけ
`map_shared` を追加する。その後、Power query と kNN または AABB query を比較し、
共通部分を extraction candidate とする。

- `map_shared` / read-only indexed context
- `HierarchyQueryOp`
- query-local traversal policy
- local `Array` spill が支配的なら `LaneScratchWorkspace`

第二用途と status-gated output contract が揃う前に `HierarchyQueryOp` を public にしない。

### P5: polygon/adjacency output

- per-query slot plan。
- `QueryWorkspace` または独立した `bounded_map` の比較。
- per-query status と exact/prefix terminal。
- exact compaction。
- `Segmentation` result。
- final polygon/adjacency graph を CPU oracle と比較。

### P6: optional spatial companion module

- `Lbvh2d` または dimension-generic LBVH。
- `UniformGrid2d` は fixed-radius 用途として別実装。
- 実装が二つ揃うまで共通 `SpatialIndex` trait は作らない。

## Acceptance criteria

### Correctness

1. CPU all-pairs と area、centroid、polygon geometry が tolerance 内で一致し、
   adjacency の canonical site-ID set が一致する。
2. 文書化した有限 coordinate/weight range 内で positive/negative/extreme weight の
   missed cutter が 0。範囲外または不確かな判定は fallback する。
3. empty cell、coincident/near-coincident site、domain boundary を扱える。
4. queue/state/output overflow は error または明示 fallback になり、silent miss がない。
5. empty/trailing segment が保持される。
6. default output order が同一 backend/float mode で deterministic。

### Complexity

1. large path に長さ `N(N - 1 + d)` の allocation が存在しない。
2. BVH storage は `O(N)`。
3. query scratch は `O(N × configured local/slot capacity)` または chunked bound。
4. benchmark は時間だけでなく次を出力する。
   - visited nodes/query
   - accepted leaves/query
   - actual clips/query
   - max/median queue depth
   - false-positive leaf 数
   - overflow/fallback count
   - peak temporary bytes

### Performance

1. `N=1,000 / 2,000 / 5,000 / 10,000` を同じ入力系列で測る。
2. build、moments query、polygon output を分離する。
3. small-group dense path の crossover を測る。
4. repeated Treemap iteration では BVH rebuild/update を含む end-to-end を測る。
5. Radeon 680M と、利用可能なら大容量 dGPU の両方で測る。
6. wall time、GPU timestamp、host observation 数を分ける。

### Genericity

Massively へ API を昇格する前に、同じ `HierarchyQueryOp` / bounded output contract で
少なくとも次の二つを実装する。

1. Power Diagram
2. kNN、radius query、AABB collision のいずれか

Power 固有の名称や数式が core API に漏れないことを確認する。

## Rejected alternatives

### 1. 高性能 dGPU だけで解決する

memory bandwidth と capacity は改善するが、`N²` storage は残る。順序を下げない。

### 2. 固定 k-nearest

任意 weight と非一様分布で exact guarantee がない。

### 3. 固定 3×3 grid

fixed-radius 用途には良いが、Power Diagram の遠方 heavy-weight cutter を落とす。

### 4. 全 pair を作ってから filter

filter 後が sparse でも、生成・storage・sort は既に `O(N²)` である。

### 5. public dynamic `MIter`

v0.87 の host-exact contract を壊し、全 vector/seg/graph algorithm の extent semantics、
allocation、dispatch を再設計する必要がある。本 proposal の最小解ではない。

### 6. global atomic append

順序が不定で、per-query fairness がなく、overflow semantics が曖昧になる。

### 7. 汎用 global Worklist を最初に追加する

Power cell は query-local state が小さく、one-lane local traversal が有力である。
まずその lowering を実証する。global frontier が必要な algorithm は別 proposal で
扱う。

### 8. Massively core に Power-aware BVH を入れる

node の `max(weight)` と directional culling は domain policy であり、
generic hierarchy engine の上に置く。

## 推奨判断

Massively の次の開発項目として、直ちに巨大な spatial framework を追加するのではなく、
次の順序を推奨する。

1. 現 API の仮想 shared segment で Power-aware 2D BVH moments kernel を実証する。
2. 仮想 extent/div-mod/read-slot が実測 bottleneck なら、最小 API `map_shared` を入れる。
3. `Grouping` は既存 primitive の標準 compound recipe として整理する。
4. polygon/adjacency が必要になった P5 で bounded output/status contract を検証する。
5. kNN/AABB query と共通化できた時点で `hierarchy::query` を Massively へ入れる。
6. grid/LBVH は core abstraction ではなく、まず companion module とする。

この順序なら、Massively v0.87 の「少数 primitive と semantic compound operation」
という方針を保ちながら、Power Diagram の dense cardinality を実際に除去できる。

## References

- Bernardo Taveira et al.,
  [Scalable GPU Construction of 3D Voronoi and Power Diagrams](https://arxiv.org/abs/2605.06408),
  SIGGRAPH 2026.
- Zenseact,
  [Paragram reference implementation](https://github.com/zenseact/paragram).
- Tero Karras,
  [Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees),
  HPG 2012.
- Michael P. Howard et al.,
  [Quantized bounding volume hierarchies for neighbor search in molecular simulations on graphics processing units](https://arxiv.org/abs/1901.08088),
  2019.
- Franz Aurenhammer,
  [Power Diagrams: Properties, Algorithms and Applications](https://epubs.siam.org/doi/10.1137/0216006),
  SIAM Journal on Computing, 1987.
