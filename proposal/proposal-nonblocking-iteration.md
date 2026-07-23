# proposal-nonblocking-iteration: 公開 device value と bounded iteration

- 状態: 不採用（実験記録）
- 対象: Massively v0.87（breaking release）
- 主な利用事例: `del2d`、Power Diagram、Traversal Algebra、反復型数値計算
- 置き換える案: `proposal-worklist.md` の同期 API 併存案
- 現在の正本: [`proposal-compound-operations.md`](proposal-compound-operations.md)

> 2026-07-23再評価: Phase 1はcorrectnessを検証したが、del2d／Power Diagramによる
> application performance validationより先に公開API全体へ変更を波及させていた。
> fixed iterationはhostのRust loopと等価であり、conditional iterationもcommand構築、
> kernel launch、一時allocationを削減しない。この案は公開APIとして採用せず、
> `MVal`とdevice logical extentはcrate内部に限定する。反復は汎用`Iteration`ではなく、
> 通常のRust loopと少数primitiveの合成で表す。workspaceはアプリケーションが保持し、
> 最適化する場合も既存合成をexecutor／compiler内部で扱う。
> 本文は判断過程と実験内容を保存するため、そのまま残す。

## 当時の結論（不採用）

Massively の公開 algorithm を、host から見た完了を待たない nonblocking API に
統一する。

```text
algorithm call = GPU work を enqueue し、device result handle を返す
read           = device result を host へ観測する
```

可変長列、scalar result、反復終了条件を同じ規則で扱う。

- `MVal<R, T>` を immutable な device-resident scalar として公開する。
- `MExtent` を immutable な logical length として公開する。
- `MVec<R, T>` は host-known `capacity` と `MExtent` を持つ。
- `MVec::len()` は `MIndex` ではなく `MExtent` を返す。
- sequence shape は runtime 非依存の `MSequence` trait で統一する。
- scalar を返していた algorithm は `MVal<R, T>` を返す。
- 公開 algorithm は内部で `MVal::read`、`MExtent::read`、`to_host`、
  `Executor::sync` を呼ばない。
- host 観測は `.read(&exec)`、`Executor::to_host`、`Executor::sync`、
  `IterationRun::finish` に限定する。
- 可変長 algorithm は host-known capacity を使って allocation し、実際の長さを
  device 上で生成して後続 algorithm へ渡す。
- `Iteration<State>` は任意の積状態を bounded bulk-synchronous に反復する。
- Worklist は新しい storage 型ではなく、device extent を持つ `MVec` の利用形態とする。
- Traversal Algebra は graph 固有の semantic layer として `Iteration` の上に残す。

この決定により、同期版と非同期版、通常 vector と Worklist vector、
`Executor` と iteration 専用 algorithm context という二重 API を作らない。

### 実装境界

v0.87 の Phase 1 は次を実装する。

- public `MVal` / `MExtent` / `MSequence`
- scalar query と可変長 result を含む公開 algorithm の nonblocking 化
- fallible な structural `zip2..zip12`
- `map_with_value` と device-side boolean algebra
- 固定回数 iteration
- finite epoch、packed control observation、device `alive` gate を持つ条件付き iteration
- body 内 observation の拒否
- 通常の `vector` / `seg` / `graph` algorithm に対する共通 logical-extent gate

次は公開意味論を変えない Phase 2 以降の項目である。

- backend indirect dispatch
- device error の sticky status carrier
- `zip_min` / `zip_equal`
- prepared iteration plan と workspace reuse
- `seg::adjacent_expand_bounded`（現在は不採用。既存primitiveで合成する）

Phase 1 の fallback は capacity dispatch と device logical-length guard である。したがって
host readback の除去と iteration の意味論は完成しているが、sparse extent の無効 lane、
command construction、temporary allocation の最適化は別途 benchmark して進める。

ここで nonblocking は Rust の `async fn` や `Future` を意味しない。公開 algorithm は
通常の Rust 関数のまま、host-side validation、allocation、GPU work の enqueue が
終わった時点で返る。結果の完成を待たないことだけが契約である。

```rust
// Rust API は async ではない。
let sum: MVal<R, f32> = vector::reduce(&exec, input, 0.0, Add)?;

// 必要な場合だけ、この行が host observation になる。
let sum: f32 = sum.read(&exec)?;
```

したがって `_async` suffix、runtime、reactor は追加しない。

## 背景

### 現在の公開境界にある同期

v0.86 の内部実装は既に device-resident scalar と logical extent を持つが、
公開 API の返却契約に合わせて最後に host read を行う。

selection の概略は次である。

```rust
let len = copy_where_into(exec, input, stencil, output.slice_mut(..))?;
output.set_fixed_len(len.read(exec)?);
Ok(output)
```

Traversal の `emit` と `relax_min_by_destination` も同様に、内部で生成済みの
`MVal<R, MIndex>` を返却前に読む。

この同期は、利用者が結果件数を host で必要としていない場合にも起こる。
後続処理が `map`、`sort`、`reduce_by_key`、次 frontier の生成であるなら、
producer と consumer は同じ device scalar を queue 上で受け渡せる。

### `del2d`

`del2d` では少なくとも次の二つの host-driven loop がある。

```text
未挿入 site がなくなるまで:
    winner を生成
    compact
    winner 数を host が取得
    split と再配置

illegal edge がなくなるまで:
    illegal edge を検出
    winner を生成
    compact
    winner 数を host が取得
    flip
```

各 round の計算量とは独立に、件数の readback、CPU wakeup、後続 command の
構築と submit が発生する。また、edge legalization は affected edge だけでなく
全 edge を再検査している。

device extent と iteration は前者を解決する基盤である。affected-edge frontier は
後者を解決する algorithmic improvement であり、別々に測定する。

### Power Diagram

Power Diagram の半平面 clipping では、各 cell を表す segment の長さが round ごとに
変化する。既存の実験では、256 sites に対して次が報告されている。

```text
dynamic compaction:
    wall-clock     約 708 ms
    GPU timestamp   約 23.9 ms
    kernel launch   11,008
    round ごとに少なくとも 2 回の length readback

fixed-capacity:
    wall-clock     約 1.09 s
    GPU timestamp  約 133 ms
    kernel launch  8,708
```

dynamic compaction は host roundtrip を生み、fixed-capacity は padding lane を
増やす。必要なのは `capacity` と `logical length` を分離し、logical length から
device-side dispatch size を生成することである。

ただし、この測定だけで wall-clock 差の全てを readback に帰属させてはならない。
kernel launch、command construction、submission、temporary allocation、
algorithmic work を個別に計測する。

## 設計原則

### 1. algorithm と observation を分離する

GPU 上で値を計算することと、CPU がその値を知ることは別の操作である。

```rust
let sum = vector::reduce(&exec, input, 0.0, Add)?;
let normalized = vector::map_with_value(&exec, input, &sum, Divide)?;

// host が必要とした時だけ観測する。
let sum = sum.read(&exec)?;
```

`reduce` は scalar を生成する algorithm であり、host query ではない。
`.read()` が host query である。

### 2. 同期可能な箇所を名前と型で限定する

次の操作だけが host を待ってよい。

- `MVal::read`
- `MExtent::read`
- `Executor::to_host`
- `Executor::sync`
- `IterationRun::finish`

`read` は「必ず GPU wait を発生させる」という意味ではない。host-known な値なら
直ちに返してよい。契約は「この操作だけが GPU wait を発生させてよい」である。

### 3. public device value は immutable にする

公開 `MVal` と `MExtent` は、作成後に意味上の値が変化してはならない。

Phase 1 の iteration runner は step ごとに新しい immutable `alive` value を生成する。
将来 mutable count、status、ping-pong control buffer を再利用する場合は、公開 `MVal`
とは別の crate-private control cell とし、公開 handle として escape させない。

runner の結果を外へ返す時は、次のいずれかを行う。

- immutable device scalar へ snapshot する。
- host へ resolve する。
- ownership により control buffer の再利用を禁止する。

### 4. 物理 shape は host-known、論理 shape は device-known

反復中に GPU から新しい連続 allocation を作ることは前提にしない。

```text
capacity: host-known allocation bound
extent:   fixed または device-produced logical length
```

常に次を満たす。

```text
0 <= extent <= capacity <= MIndex::MAX
```

bounded iteration では物理 capacity と column schema を固定し、extent と値だけを
変化させる。

### 5. 非同期版を別名で作らない

次の二重 API は導入しない。

```text
copy_where / copy_where_async
reduce / reduce_deferred
MVec / WorklistVec
Executor algorithm / IterationContext algorithm
```

通常の algorithm 名を nonblocking の正本とする。

## 公開型

### `MVal<R, T>`

`MVal` は一つの immutable な device-resident logical value を表す。

```rust
#[derive(Clone)]
pub struct MVal<R: Runtime, T> {
    // type-erased private storage と reader
}
```

論理型と物理 storage の対応は公開 trait にせず、`MVal` 内部で type erase する。
少なくとも次を扱う。

- 数値 scalar
- `MIndex`
- `bool`
- Massively が既に扱う flat tuple
- `Option<MIndex>` など既存 query result に必要な有限 tagged value

CubeCL の `bool` が storage element でないことは公開 API に漏らさない。内部では
`u32` flag などへ lower してよい。

通常の `MAlloc<R>` item には既存の SoA storage mapping を用いる。query 用の
`bool`、`Option<MIndex>`、`Option<(MIndex, MIndex)>` には semantic mapping を
Massively 側で実装する。column arity ごとの `MVal` implementation を増やさず、
既存の flat storage lowering を再利用する。

基本 API は次である。

```rust
impl<R, T> MVal<R, T>
where
    R: Runtime,
{
    /// Host observation. Producer の完了を必要に応じて待つ。
    pub fn read(&self, exec: &Executor<R>) -> Result<T, Error>;
}

impl<R: Runtime> Executor<R> {
    /// 通常の allocatable row value を device value として利用可能にする。
    pub fn value<T: MAlloc<R>>(&self, value: T) -> Result<MVal<R, T>, Error>;
}
```

公開 `MVal` から mutable storage は取得できない。
`bool` と tagged query value は algorithm および boolean algebra が生成し、その物理
flag representation は公開しない。host 由来の boolean constructor が必要になった時は、
storage ABI を公開せず semantic constructor を追加する。

scalar result を GPU 上で再利用するため、最小限の scalar algebra を提供する。

```rust
pub mod value {
    pub fn map<R, T, Op>(
        exec: &Executor<R>,
        input: &MVal<R, T>,
        op: Op,
    ) -> Result<MVal<R, Op::Output>, Error>;

    pub fn zip_map<R, A, B, Op>(
        exec: &Executor<R>,
        left: &MVal<R, A>,
        right: &MVal<R, B>,
        op: Op,
    ) -> Result<MVal<R, Op::Output>, Error>;
}
```

vector へ broadcast する場合は scalar を host へ戻さない。

```rust
let sum = vector::reduce(&exec, input, 0.0, Add)?;
let normalized = vector::map_with_value(&exec, input, &sum, Divide)?;
```

### `MExtent`

`MExtent` は logical sequence length を表す immutable value である。

```rust
pub struct MExtent {
    // Fixed(MIndex) または immutable device source と host-known upper bound
}
```

公開 API は次とする。

```rust
impl MExtent {
    /// Host が既に知る場合だけ値を返す。同期しない。
    pub fn known(&self) -> Option<MIndex>;

    /// 同期せずに得られる安全な物理上限。
    pub fn upper_bound(&self) -> MIndex;

    /// Host observation.
    pub fn read<R: Runtime>(&self, exec: &Executor<R>) -> Result<MIndex, Error>;

    /// Device 上で zero test を作る。同期しない。
    pub fn is_zero<R: Runtime>(
        &self,
        exec: &Executor<R>,
    ) -> Result<MVal<R, bool>, Error>;

    /// Device 上で non-zero test を作る。同期しない。
    pub fn is_nonzero<R: Runtime>(
        &self,
        exec: &Executor<R>,
    ) -> Result<MVal<R, bool>, Error>;
}
```

`MExtent` は単なる `MVal<R, MIndex>` ではない。次の structural metadata も持つ。

- host-known upper bound
- executor ownership
- host-known slice start と limit
- extent identity

`MExtent` 自体に runtime 型引数を付けない。現行の `DeviceSlice<T>` は runtime-erased
view であり、extent も同じ性質を持つためである。device source には executor owner
identity が保存され、`read` や device 演算時に渡した `Executor<R>` と照合する。
これにより `DeviceSlice<R, T>` への全面的な型変更を避けつつ、foreign executor は
`Error::ForeignExecutor` にできる。

`known()` は長さが structural に host-known かだけを答える。producer の完了や
deferred status を観測したことにはならない。

内部の `LogicalExtent` 演算は再利用する。add、min、ceil-div、equality などのうち、
利用者が必要とするものだけを公開し、raw buffer handle は公開しない。

### `MVec<R, T>`

`MVec` は physical storage と logical extent を持つ。

```rust
impl<R, T> MVec<R, T> {
    /// Logical length handle。同期しない。
    pub fn len(&self) -> MExtent;

    /// Physical allocation bound。同期しない。
    pub fn capacity(&self) -> MIndex;

    /// Device-side empty test。同期しない。
    pub fn is_empty(&self, exec: &Executor<R>) -> Result<MVal<R, bool>, Error>;
}
```

`len()` は cheap な handle clone であり、host integer を返さない。

```rust
let selected = vector::copy_where(&exec, input, stencil)?;

// 同期なし
let extent = selected.len();
let capacity = selected.capacity();

// host 観測
let count = extent.read(&exec)?;
```

fixed extent でも同じ API を使う。

```rust
let count = fixed_vector.len().read(&exec)?;
```

この `read` は fixed fast path では GPU を待たない。

`MVec` は alias なので、実装上は同じ契約を全ての public sequence surface に適用する。
長さとcapacityはruntimeに依存しない `MSequence` へ分離する。

```rust
pub trait MSequence {
    fn len(&self) -> MExtent;
    fn capacity(&self) -> MIndex;
}

pub trait MStorage<R: Runtime>: MSequence { /* storage API */ }
pub trait MIter<R: Runtime>: MSequence { /* read lowering API */ }
pub trait MIterMut<R: Runtime>: MSequence { /* write lowering API */ }
```

`DeviceVec`、read-only `DeviceSlice`、`BoolVec`、segmented values、および lazy iterator
もこの規則に揃える。mutable view の `capacity()` は書き込み可能な物理範囲、
`len()` は同じ view を読んだ時の論理範囲を表す。

`MSequence` に runtime parameter を持たせないことで、runtime-erased な
`Zipped<DeviceSlice<_>, ...>` に対しても `zipped.len()` の `R` 推論を要求しない。
algorithm loweringだけが従来どおり `MIter<R>` / `MIterMut<R>` を使う。

iterator construction 時に解決できる size mismatch はそこで `Err` にする。
特に `zip2` から `zip12` は fallible constructor に変更し、構築済み `MIter` が
常に一つの有効な extent を持つ invariant を作る。

```rust
let zipped = zip2(left, right)?;
let extent = zipped.len();
```

独立した device extent は値が偶然等しいだけでは通常の `zip` にできない。後述の
identity 規則を満たさない場合は、明示的な `zip_min` または device equality assertion
を使う。

`read_len` は追加しない。長さを取得する操作は常に `len()`、host observation は
常に `.read(&exec)` とし、`values.len().read(&exec)?` の一つの綴りに統一する。

### multi-column

multi-column `MVec` は従来どおり SoA storage を使い、全 column が同一の
`MExtent` identity を共有する。

別々に生成された device extent を暗黙に zip しない。

- 同じ extent identity なら通常の `zip` を許す。
- fixed operand は device extent の upper bound を覆う場合に限り device extent へ
  narrow できる。
- 独立 extent は `zip_min` または明示的 `zip_equal` を要求する。

Phase 1 は、独立 extent を通常の `zip` が拒否するところまでを実装する。
`zip_min` と `zip_equal` は sticky status と併せて Phase 2 で追加する。

```rust
// Structural identity を検査するだけなので enqueue も同期も不要。
let rows = zip2(keys, values)?;

// 意味として短い側まで処理する。
let prefix = zip_min(left, right)?;

// Device 上で equality を検査し、違えば sticky status を立てる。
let asserted = zip_equal(&exec, left, right)?;
```

`zip_equal` の論理 extent は left を使い、right の capacity が left の upper bound を
覆うことを host で検証する。実際の長さが違う場合は out-of-bounds access をせず、
後続 algorithm を no-op にして observation 時に `Error::LengthMismatch` を返す。

この規則は column 数に依存せず、単列専用 path を作らない。

## Algorithm の返却契約

`Result` は「結果が完成した」ことを表さない。call 時点で host が判定できる validation、
allocation、enqueue の成否だけを表す。

| 現在の意味 | 新しい返却値 |
| --- | --- |
| `T`、`bool`、`MIndex` | `MVal<R, T>`、`MVal<R, bool>`、`MVal<R, MIndex>` |
| `Option<T>`、小さな tuple query | semantic type を保った `MVal<R, ...>` |
| length-preserving sequence | input extent を共有する `MVec` |
| data-dependent sequence | 新しい device extent を持つ `MVec` |
| partition boundary | 二つの logical sequence |
| preallocated output への書き込み | 従来どおり `Result<(), Error>`。完了待ちはしない |

### scalar result

scalar を返していた algorithm は `MVal` を返す。

```rust
let sum: MVal<R, T> =
    vector::reduce(&exec, input, init, Add)?;

let count: MVal<R, MIndex> =
    vector::count_if(&exec, input, pred)?;

let any: MVal<R, bool> =
    vector::any_of(&exec, input, pred)?;

let found: MVal<R, Option<MIndex>> =
    vector::find_if(&exec, input, pred)?;
```

host が必要なら同じ形で読む。

```rust
let count = count.read(&exec)?;
```

### length-preserving result

`map`、`copy`、permutation などは input capacity から output を確保し、input extent を
そのまま伝播する。

```rust
let output = vector::map(&exec, input, op)?;
assert_eq!(output.capacity(), input.capacity());
```

公開 `map` は現在の internal `map_preserving_extent` 相当を正本とする。

### selection / compaction

`copy_where`、`remove_where`、`unique` などは output capacity を host-known bound から
確保し、selected count を output extent に接続する。

```rust
let selected = vector::copy_where(&exec, input, stencil)?;
let mapped = vector::map(&exec, selected.slice(..), op)?;
```

この二つの call の間に readback はない。

### partition

host integer の partition boundary を返す API は廃止する。二つの logical sequence を
返す。

```rust
let (selected, rejected) =
    vector::partition(&exec, input, pred)?;
```

両者は必要に応じて storage や control を共有できるが、それぞれ独立した immutable
extent を持つ。

### reduction / predicate / search

reduction、predicate query、index query は device value を返す。後続 operation が
device value を直接消費できる path を用意する。

host branch を行いたい利用者は明示的に `.read()` する。

```rust
if vector::any_of(&exec, input, pred)?.read(&exec)? {
    // ここは明示的な同期を選んだ host branch
}
```

### sort / scan / by-key

sort、scan、by-key algorithm は input extent を kernel control へ渡す。

- allocation は capacity を使う。
- 実際の処理範囲は extent を使う。
- stage 数を capacity から安全に上限計画してよい。
- 各 stage の実 dispatch size は extent から生成する。
- output extent は algorithm の意味に従い preserve または device 上で生成する。

stable / deterministic order は operation ごとに定義する。atomic lowering と
sort-reduce lowering を交換する場合も公開順序契約を維持する。

`reduce_by_key` は少なくとも次を仕様化する。

- key equality / ordering
- init の適用位置
- reduction の associativity 要件
- commutativity 要件
- equal-key 内の順序
- deterministic variant と unordered variant の区別

### Traversal Algebra

Traversal terminal は exact-length 化の read を行わず、device extent 付き `MVec` を
返す。

```rust
let next = graph::traverse(...)
    .map(...)
    .relax_min_by_destination(...)?;
```

`next` はそのまま別 traversal または `Iteration` へ渡せる。

Traversal Algebra は次を保持する。

- CSR
- source / destination / edge semantics
- proposal collision semantics
- terminal semantics

Worklist や Iteration はこれらを置き換えない。

### segmented algorithm（不採用の記録）

以下の`adjacent_expand_bounded`案は採用しない。cyclic predecessorとsegment contextは
`counting`、offsets、`segment_ids`、`permute`、`zip`でentryごとに構成し、通常の
`flat_map`へ渡せる。本文は当時の設計案として残す。

segmented output は、fixed-length offsets と device extent 付き values を組み合わせる。
各 segment length と total values length を host へ戻さず再構築できるようにする。

Power Diagram に必要な次の operation を通常の `seg` algorithm として追加する。

```rust
seg::adjacent_expand_bounded
```

これは同一 segment の `(previous, current)` と segment context を読み、一入力から
`0..=K` 件を生成する。Power Diagram 固有ではなく、polyline clipping、mesh rewrite、
疎行の書き換えにも利用できる。

### bounded expansion

上限のない exact-size `flat_map` は、連続 WGPU buffer の byte size を host が決める
必要があるため、nonblocking contract と両立しない。

`flat_map` は host-known bound を要求する設計へ変更する。

```rust
let output = vector::flat_map_bounded(
    &exec,
    input,
    output_capacity,
    op,
)?;
```

または operation が安全な一入力当たり上限を与える。

```rust
pub trait BoundedExpandOp<Input> {
    type Output;

    fn max_outputs(&self) -> MIndex;
    fn apply(input: Input, output: &mut BoundedEmitter<Self::Output>);
}
```

trait の宣言だけを信用して unchecked write を行わない。実装は `required_len` と
`materialized_len` を分離し、capacity を越える write を禁止する。

```text
materialized_len = min(required_len, capacity)
overflow        = required_len > capacity
```

## Deferred error

nonblocking algorithm は、GPU 上で初めて判明する error を返却時の `Err` にできない。

以下は Phase 2 の契約である。Phase 1 は host-known validation を `Result` で返し、
各 kernel の既存 bounds guard を維持するが、汎用 sticky status carrier はまだ持たない。

`Result<Handle, Error>` は次だけを表す。

- foreign executor
- host-known capacity overflow
- invalid host range
- allocation / enqueue failure
- unsupported backend capability

GPU 上で判明する error は handle に紐づく sticky device status として伝播する。

- capacity overflow
- invalid device index
- stale topology generation
- algorithm-specific invariant violation

後続 operation は error status が立っていれば no-op となり、status を伝播する。
`.read()`、`to_host`、`IterationRun::finish` は producer completion 後に status を確認し、
対応する `Error` を返す。

公開 `MVal` / `MExtent` は immutable だが、completion status は future の完了状態として
変化してよい。利用者が device status buffer を直接変更することはできない。

`MVec`、`MVal`、`MExtent` は内部で completion/status carrier を共有できる。複数 input
を取る algorithm は carrier を merge し、`Executor::sync` は executor-global な
未観測 status も確認する。status carrier の物理共有方法は公開 ABI にしない。

## 同期契約

| 操作 | host synchronization |
| --- | --- |
| `capacity` | なし |
| `MVec::len` | なし |
| `MExtent::known` / `upper_bound` | なし |
| `map` / selection / sort / scan | なし |
| `reduce` / predicate / search | なし |
| Traversal terminal | なし |
| `Iteration::enqueue` | なし |
| `MVal::read` | 必要な場合あり |
| `MExtent::read` | 必要な場合あり |
| `Executor::to_host` | あり |
| `IterationRun::finish` | あり |
| `Executor::sync` | あり |

公開 algorithm が暗黙に observation API を呼んでいないことを trace test で検証する。

device extent 付き `MVec` に対する `to_host` は、extent を resolve してから logical
prefix だけを転送してよい。この二段階は `to_host` 自体が明示的 observation boundary
なので契約違反ではない。

## Indirect dispatch

device extent を保持しても、常に capacity 分を dispatch すると sparse input で
無効 lane が支配する。

```text
groups = ceil(extent / items_per_group)
dispatch = [groups, 1, 1]
```

Phase 2 では、対応 backend でこの三要素を device buffer に書き、indirect dispatch
を行う。

Phase 1、および backend が indirect dispatch を提供しない場合は次へ lower する。

```text
capacity dispatch + index < extent guard
```

意味論は同一であり、性能特性だけが異なる。

### zero dispatch

extent が zero の時に zero-group dispatch が有効か、backend ごとに conformance test
する。必要なら one-group dispatch と device guard を使う。

output extent と status は、input dispatch が zero でも必ず初期化されなければならない。
zero-group kernel は古い ping-pong scalar を clear できないため、各 step の前に
常時実行される control kernel または明示的 clear を置く。

空 sequence は次を満たす absorbing state とする。

```text
map(empty)              = empty
filter(empty)           = empty
expand_bounded(empty)   = empty
```

### indirect dispatch が解決しないもの

indirect dispatch は無効 lane と host readback を減らすが、kernel launch 数そのものは
減らさない。

Power Diagram の多数の map、gather、scan、scatter は、次も必要とする。

- safe kernel fusion
- segmented adjacent expansion
- temporary reuse
- pipeline / binding cache
- submission batching

これらを device extent と混同せず個別に測定する。

## Iteration

### 位置

`Iteration` は graph 固有でも vector 固有でもないため、公開 top-level module に置く。

```rust
pub mod iteration;
```

利用側は次となる。

```rust
use massively::iteration::{Iteration, StopReason, Transition};
```

### 意味論

Iteration は任意の積状態を進める bounded bulk-synchronous state machine である。

```text
(S[i+1], continue[i]) = body(S[i], i)
// Phase 2 では status[i] を同じ transition に合成する。
```

`State` は次を任意に含められる。

- 一つ以上の異なる item 型の `MVec`
- fixed dense state
- `MVal`
- topology storage
- segmented storage
- algorithm-specific metadata

一種類の `Item` や一つの Worklist に固定しない。

各 step は snapshot semantics を持つ。

- step `i` は `S[i]` を読む。
- step `i` が生成した work item は `S[i+1]` から見える。
- 同一 step 内の FIFO enqueue / dequeue semantics は提供しない。

ここで step は kernel launch 一回でも work item 一個でもない。body に書かれた
`vector`、`seg`、`graph` algorithm 群を一度実行し、その全 producer-consumer
依存が完了してから次の body へ進む、一つの BSP transition である。

### 基本 API

以下をpublic contractとする。Rust上のlifetimeやprivate workspace parameterは、
この返却値と同期意味論を変えない範囲で実装時に調整してよい。

```rust
let run = Iteration::new(initial_state)
    .max_steps(max_steps)
    .epoch_steps(8)
    .enqueue(&exec, |step, state| {
        let illegal = vector::copy_where(
            &exec,
            state.edges.slice(..),
            IsIllegal,
        )?;

        let winners = vector::reduce_by_key(
            &exec,
            illegal.slice(..),
            ClaimKey,
            PickWinner,
        )?;

        let next_edges = graph::apply_and_emit(
            &exec,
            &state.topology,
            winners,
            Flip,
        )?;

        let continue_if = next_edges.len().is_nonzero(&exec)?;

        Ok(Transition {
            state: State {
                topology: state.topology,
                edges: next_edges,
            },
            continue_if,
        })
    })?;

let outcome = run.finish(&exec)?;
assert!(matches!(
    outcome.reason,
    StopReason::Converged | StopReason::StepLimit,
));
```

body 内で呼ぶのは通常の `vector`、`seg`、`graph` algorithm である。これらが既に
nonblocking なので、iteration 専用 algorithm facade は不要である。

条件付き iteration の `State` は `Clone` を要求する。clone は epoch 内の候補 state
handle を保持するために使う。Massively storage の clone は device buffer の深い copy
ではなく immutable handle clone である。利用者定義 state も同じ cheap snapshot
semantics を持つべきである。固定回数 variant は候補選択が不要なので `Clone` を
要求しない。

`enqueue` は最初の epoch だけを enqueue して直ちに `IterationRun` を返す。
`finish` は明示的な iteration observation boundary であり、必要なら後続 epoch の
enqueue と epoch-end observation を繰り返して最終 state を返す。background thread
や async runtime は要求しない。

body は host observation を行ってはならない。

- `.read`
- `to_host`
- `sync`
- device result による host branch

debug / plan-build mode ではこれらを `Error::ObservationInsideIteration` として検出する。
runner は body の encode 中だけ private な iteration scope と device `alive` gate を
設定する。`LogicalExtent` はその gate を `alive ? extent : 0` として捕捉する。
通常 algorithm は同じ公開 signature のまま、既存の logical-length guard を通して
dispatch と write を無効化する。Phase 2 の status も同じ gate に合成する。
raw backend launch は gate を迂回するため body 内では許可しない。

body の host-side call count は API 契約にしない。実装は body を epoch ごとに
enqueue しても、一度 record して replay してもよい。したがって body 内の host side
effect に依存してはならない。

### 固定回数

Power Diagram、Jacobi iteration、固定round algorithmは device終了条件を必要としない。

```rust
let state = Iteration::new(state)
    .steps(num_sites - 1)
    .enqueue(&exec, |step, state| {
        let rank: MVal<R, MIndex> = step.index();
        let next = clip_rank(&exec, state, rank)?;
        Ok(Transition::next(next))
    })?;
```

iteration index は compile-time const generic にせず、runtime scalar または
parameter binding とする。

固定回数 variant は最終 state の handle が host encoding 時に確定するため、全 step
を enqueue して `State` を nonblocking に返せる。Power Diagram の後続 algorithm は
その state を `.read()` なしで消費できる。条件付き variant だけが
`IterationRun::finish` を必要とする。

この返却型の違いはbuilderの`Fixed` / `Conditional`という二つのtypestateだけで表す。
step数やstateのcolumn arityを型へ埋め込まず、iteration回数に比例する
monomorphizationを作らない。

### device 条件

Worklist 型 algorithm は `MVal<R, bool>` を継続条件に使う。

```rust
let continue_if = next_frontier.len().is_nonzero(&exec)?;
```

複数 frontier は scalar algebra で合成する。

```rust
let sites_active = next_sites.len().is_nonzero(&exec)?;
let edges_active = next_edges.len().is_nonzero(&exec)?;
let continue_if = value::zip_map(
    &exec,
    &sites_active,
    &edges_active,
    BoolOr,
)?;
```

residual convergence も同じ形で表す。

```rust
let residual = vector::reduce(&exec, errors, 0.0, Max)?;
let continue_if = value::map(&exec, &residual, GreaterThan(tolerance))?;
```

### mandatory fuel

device条件だけに依存する無制限 loop は提供しない。

```rust
Iteration::new(state).max_steps(max_steps)
```

`max_steps` は必須であり、完了理由を区別する。

```rust
pub enum StopReason {
    Converged,
    StepLimit,
}

pub struct IterationOutcome<State> {
    pub state: State,
    pub steps: MIndex,
    pub reason: StopReason,
}
```

device error は stop reason ではなく `IterationRun::finish` の `Err` として返す。
これにより partial state を成功値として外へ出さない。

### alive と status

epoch内で収束した後の事前encode済みstepを安全なno-opにするため、runnerは
step ごとの immutable device-resident `alive` を持つ。

```text
alive[0] = true

if alive[i]:
    (S[i+1], continue[i]) = body(S[i])
else:
    S[i+1] = S[i]

alive[i+1] =
    alive[i]
    && continue[i]
    // Phase 2: && status[i].is_ok
```

すべての input/output logical extent は `alive` を考慮する。収束した step より後に
encode 済みの通常 algorithm は GPU 上で data write を行わない。Phase 2 では status
と indirect dispatch も同じ条件を使う。

### epoch

WebGPU portable lowering は有限個のstepを一つのepochとしてenqueueする。

```text
CPU: step 0..K-1 を host observation なしで enqueue
GPU: step 0 -> step 1 -> ... -> step K-1
CPU: finish が epoch末尾で packed alive flags を一度観測
```

`epoch_steps` はruntime tuning parameterであり、型や公開意味論に含めない。

```rust
.epoch_steps(8)
```

runner は epoch 内で生成した候補 `State` handle と、各 step 後の alive flag を一つの
packed device buffer に保持する。`finish` はこの buffer を一度読み、実際に収束した
step の immutable state を選ぶ。これにより、任意の積状態を device-side conditional
copy する必要がなく、memory は epoch size で bound される。継続時は最後の候補 state
を次 epoch の入力にする。

`epoch_steps = 1` は一 step 一 observation の基準実装である。値を大きくすると
observation 回数は減るが、収束後に encode 済みの control command は残る。indirect
dispatch 対応 backend では data work は zero にできる。固定回数が安全に分かる場合は
全 step を enqueue し、中間 observation を行わない。

### workspace と escape

Iteration の物理 workspace は開始前に計画する。

- current / next の少なくとも二組
- 初期 lowering では epoch 内の候補 state handle
- algorithm scratch
- extent/control scalar
- Phase 2 の indirect dispatch args
- Phase 2 の sticky status

初期実装は有限epochをhostでencodeし、候補 handle を保持してよい。この時の peak
memory は `epoch_steps` に依存するので、tuning では observation 回数だけでなく memory
も測る。最適化実装はbodyをrecordし、buffer slotをtype-erased planへ割り当てる。

Phase 1 は buffer slot を再利用しないため、body 内の中間 `MVec` / `MVal` が clone
されても immutable handle として有効である。prepared plan が workspace reuse を
導入する場合は、次のいずれかで安全性を保証する。

- body build scopeから中間handleをescapeさせない。
- cloneが残るbufferを再利用しない。
- final stateだけをimmutable storage/extentへsnapshotする。

公開 API は「escapeしたhandleの内容が後から変わる」意味論を採用しない。

### 初期 lowering と prepared plan

Iteration の最初の実装に完全なprogram IRを要求しない。

Phase 1:

- hostのRust loopで有限epochをencode
- bodyは毎step呼ばれてよい
- device valueをreadしない
- `enqueue`は最初のepoch後に返り、`finish`だけがepoch controlを観測
- epoch内の候補stateからstop offsetに対応するstateを選択
- correctnessとreadback削減を検証

Phase 2:

- bodyをtype-erased `IterationPlan`へrecord
- scratchとbuffer slotを再利用
- command constructionを削減
- 同じplanを異なる入力に再利用

record/replayが必要かは、readback除去後のprofileで判断する。

### backend specialization

公開意味論はbackendに依存させない。

- WGPU Phase 1: finite epoch + capacity dispatch + device guard
- WGPU Phase 2: finite epoch + indirect dispatch
- indirect非対応backend: capacity dispatch + device guard
- CUDA: conditional CUDA Graphへの特殊化を許す
- persistent task queue: 別のexecution policyとして明示する

persistent asynchronous queueはBSP barrierを緩和し得る。処理順や追加workが変化する
ため、単なるhidden loweringとして使わない。algorithmがconfluenceまたは同値性を
保証する場合だけ選択する。Atosの研究も、task-parallel schedulingはBSPより高い
並列性を得られる一方、relaxed dependencyを前提とする別modelであることを示している。

## Worklist と Traversal Algebra

Worklist は storage 型ではない。

```text
Worklist = device extent を持つ MVec を
           bulk-synchronous frontier として使う規約
```

必要なら convenience module を追加できる。

```rust
iteration::until_empty(...)
graph::frontier_iteration(...)
```

しかし `Worklist<R, T>` という別 storage、別 `map`、別 `filter` は作らない。

Traversal Algebra は次の位置にある。

```text
public MVal / MExtent / MVec
        ↓
vector / seg nonblocking algorithms
        ↓
Iteration
        ↓
Traversal Algebra / graph rewrite / del2d / Power Diagram
```

Gunrockの研究結果は、frontierを中心としたbulk-synchronous abstractionがGPU graph
analyticsで有効であることを示す。一方でPower Diagramやdense convergenceは
vertex / edge frontierではない。したがってfrontierを捨てるのではなく、
`MVec + Iteration`の上のgraph semantic layerとして位置付ける。

## del2d

del2d は一つの `WorklistLoop<State, Item>` ではなく、複数の異種列を持つstateとして
表す。

```rust
struct Del2dState<R> {
    topology: Topology<R>,
    sites: MVec<R, SiteTask>,
    edges: MVec<R, EdgeTask>,
}
```

初期実装ではsite insertionとedge legalizationを別phaseにしてよい。

### site insertion

```text
remaining sites
    -> locate
    -> priority winner
    -> claim
    -> split/update
    -> next remaining sites
```

### edge legalization

```text
affected edges
    -> legality filter
    -> conflict winner
    -> flip
    -> newly affected edges
```

全edge再検査とaffected-edge frontierの差を、readback削減とは別に測定する。

### claim

`claim_all` はIteration coreではなくgraph rewrite layerに置く。

正しさには次が必要である。

- candidate-globalな全順序
- 同一resourceを要求する全candidateで一貫したwinner
- updateが書く全resourceをclaim対象に含める
- topology generationによるstale task検出
- claim失敗時にpartial updateを行わない

この契約を形式化してから汎用operatorとして追加する。

## Power Diagram（当時案・不採用）

当時はPower Diagramがfixed-step Iterationを使う案だった。現在は通常のRust loopと
既存primitiveの合成を使う。

```text
state:
    segmented polygons
    clipping context

for rank in 0..num_sites-1:
    adjacent_expand_bounded
    segment offsets rebuild
    current/next swap
```

当時案ではdevice extentが各cell polygonの変化する長さを保持し、host readbackなしで
次rankへ渡す想定だった。

現在はIterationも`seg::adjacent_expand_bounded`も追加しない。同期、padding、launch数
は既存合成のまま別々に測る。

## API migration

これはbreaking changeである。同期 compatibility variant は残さない。

### length

旧:

```rust
let len: MIndex = values.len();
```

新:

```rust
let extent: MExtent = values.len();
let len: MIndex = extent.read(&exec)?;
```

host-known値だけを同期なしで使う場合:

```rust
if let Some(len) = values.len().known() {
    // no synchronization
}
```

host-known allocation boundが必要な場合はlogical lengthを読まない。

```rust
let bound: MIndex = values.capacity();
```

### empty

旧:

```rust
if values.is_empty() {
    ...
}
```

新しいhost branch:

```rust
if values.is_empty(&exec)?.read(&exec)? {
    ...
}
```

device-side branchやiteration conditionでは返された`MVal<R, bool>`をそのまま使う。

### scalar

旧:

```rust
let sum: T = vector::reduce(&exec, input, init, Add)?;
```

新:

```rust
let sum: MVal<R, T> = vector::reduce(&exec, input, init, Add)?;
let sum: T = sum.read(&exec)?;
```

### predicates

旧:

```rust
if vector::any_of(&exec, input, pred)? {
    ...
}
```

新:

```rust
if vector::any_of(&exec, input, pred)?.read(&exec)? {
    ...
}
```

### data-dependent sequence

旧:

```rust
let selected = vector::copy_where(&exec, input, stencil)?;
let count = selected.len();
```

新:

```rust
let selected = vector::copy_where(&exec, input, stencil)?;
let count = selected.len().read(&exec)?;
```

algorithm chainだけなら追加操作はない。

```rust
let selected = vector::copy_where(&exec, input, stencil)?;
let mapped = vector::map(&exec, selected.slice(..), op)?;
let sorted = vector::sort(&exec, mapped.slice(..), less)?;
```

このchainはhost observationを行わない。

### zip

旧:

```rust
let rows = zip2(keys, values);
```

新:

```rust
let rows = zip2(keys, values)?;
```

constructorをfallibleにすることで、その後の`rows.len()`はfallibleでなくなる。

## 互換性を保たない理由

同期版を残すと次の問題が生じる。

- `_async` / `_deferred` variantでAPI数が倍になる。
- iteration bodyで誤って同期版を呼べる。
- 通常APIとiterationAPIで同じalgorithmを二重に公開する必要がある。
- どのcallがreadbackするかを名前ではなくvariant一覧で覚える必要がある。
- 新algorithmごとに二種類の公開契約を保守する必要がある。

次期breaking releaseで一つの規則へ移行する方が単純である。

## 実装状況と次段階

### v0.87 Phase 1: 完了

1. immutable public `MVal` と semantic bool / option query representation
2. public `MExtent` と internal `LogicalExtent`
3. `MSequence`、public `capacity()`、全 sequence の `len() -> MExtent`
4. fallible `zip2..zip12`
5. vector / segmented / Traversal terminal の nonblocking 化
6. bounded `flat_map`
7. arbitrary product state の固定回数 iteration
8. mandatory fuel と runtime epoch size を持つ条件付き iteration
9. immutable alive chain と logical-extent gate
10. packed epoch control、candidate retention、stop-offset state selection
11. body 内 observation の拒否

この実装は body を host の Rust loop で有限回 encode する。record/replay は要求しない。

検証結果（2026-07-23）:

- workspace test 823 件成功、2 件は設定上の skip
- Massively doctest 81 件成功
- public API leakage / placement check 成功
- Massively Core Lean proof 14 jobs 成功
- Traversal Algebra Lean proof 26 jobs 成功
- Massively / graph-algorithms benchmark target の compile check 成功

### 次: profile と application validation

次を個別に計測する。

- host observation 回数
- queue submission 数
- kernel launch 数
- command construction 時間
- GPU timestamp
- temporary allocation 数
- active work item 数
- epoch candidate による peak memory

適用順は、既存 oracle を持つ BFS / SSSP / connected components、Power Diagram の
fixed-step clipping、del2d の affected-edge legalization、site insertion とする。
Power Diagram の adjacent expansion、del2d の affected-edge 化の効果は control
abstraction の効果から分離する。

### Phase 2: status と dispatch

1. device error status carrier
2. `zip_min` / `zip_equal`
3. extent から indirect dispatch args を作る共通 lowering
4. control buffer pool
5. zero-group conformance
6. capacity-guard fallback との trace 比較
7. empty input の output clear

### Phase 3: prepared plan（profile が要求した場合だけ）

- type-erased plan nodes
- parameter binding
- buffer slot planning
- scratch reuse
- compiled pipeline reuse
- backend command graph

iteration 数や body graph を Rust のネスト型、tuple arity、const generic へ埋め込まない。

## Test

### value / extent

- fixed `MExtent::read`
- device `MExtent::read`
- `MVal` scalar / bool / tuple / option
- foreign executor
- immutable clone
- sliceによるclamp
- independent extent zip rejection
- same-identity multi-column zip
- zero extent
- `MIndex::MAX`境界

### algorithm

- algorithm chainにhost observationがない
- selection -> map -> sort -> reduce
- empty input
- all-selected / none-selected
- multi-column 1、3、7、12列
- segmented empty segment
- bounded expansion overflow
- `to_host`がlogical prefixだけを返す

### Iteration

- zero step
- fixed step
- initial empty
- one step convergence
- epoch途中の収束
- stop offsetに対応するcandidate stateの選択
- step limit
- multiple worklists
- dense state + sparse frontier
- output extent snapshot
- body内のobservation拒否
- body内の通常algorithmへのalive gate適用
- no stale ping-pong length
- runtime epochを変えても同じ結果

### Phase 2 status / backend

- deferred error propagation
- iteration device error
- zero indirect dispatch
- indirect dispatch args
- capacity fallback
- queue ordering
- status clear

### compile-time

- iteration countでmonomorphization数が増えない
- body typeがstep数に応じてネストしない
- multi-column input/outputのcross-product specializationを増やさない

## Performance acceptance

### 共通

- algorithm chain中のhost readbackが0
- explicit `read`ごとに必要最小限の同期
- fixed入力で不要なdevice scalar materializationを避ける
- output capacityに比例しないcontrol allocation

### Power Diagram

- clipping loop内のhost readbackが0
- dynamic extent版がfixed-padding版よりGPU workを減らす
- adjacent expansion後にkernel launch数が明確に減る
- wall-clockとGPU timestampの差をsubmission/launch別に説明できる

### del2d

- roundごとのwinner count readbackが0
- conditional control observationが高々`ceil(steps / epoch_steps)`
- edge legalizationがaffected-edge frontierを使う
- step数、active edge数、readback、launch数を記録する
- CPU比較とのcorrectnessを維持する

## 代替案

### 同期API + iteration scope

通常 `vector::*` は同期し、iteration内だけ別contextでnonblockingにする案。

不採用理由:

- `AlgorithmContext`によるcontext-dependent return typeが必要。
- iteration専用 `IterVec` / `IterScalar` が必要。
- 通常とiterationでalgorithm facadeが分裂する。
- helper functionがcontext genericになる。
- 全algorithmについて二つの返却契約を維持する必要がある。

### 同期版と非同期版の併存

不採用理由:

- APIが倍増する。
- iteration内で同期版を誤用できる。
- 新algorithm追加ごとにvariant選択が必要になる。

### `MVal` / `MExtent`をprivateのままにする

不採用理由:

- 通常algorithm間でdevice resultを表すpublic typeがなくなる。
- scalar broadcast、device条件、可変長chainをiteration専用facadeへ押し込める。
- 同じ計算をiterationの内外で別signatureにする必要がある。

公開するのは値とextentのsafe handleであり、raw storageやmutable controlではない。

### 全fuelの事前encode

条件付きiterationでも`max_steps`まで全てencodeし、最後だけ観測する案。

一般の積状態では、収束したstepのstateを選ぶために全candidateを保持するか、
device-side conditional copyが必要になる。前者はmemoryが`max_steps`に比例し、後者は
dense stateに大きなcopy costを加える。WebGPUには汎用のconditional command nodeも
ない。

したがってportableな条件付きiterationのdefaultには有限epochを使う。fixed-step、
自然にabsorbingなworklist、conditional graphを持つbackendでは中間観測を省く
specializationを許す。

### 全algorithmをRustの`async fn`にする

不採用理由:

- enqueue自体はCPU上ですぐ完了し、await対象がない。
- device handleが既にfuture valueの役割を果たす。
- executor選択、lifetime、runtime依存を全callへ伝播させる。
- `.read()`という明示的なobservation境界より同期箇所が見えにくい。

### Worklist専用storage

不採用理由:

- `MVec`とstorage、SoA、algorithmを二重化する。
- dense iterationやscalar convergenceを表現しない。
- del2dのSite/Edgeという複数異種列を一つのItem型へ押し込める。

### 一般再帰

不採用理由:

- termination bound、capacity、barrier、errorを明示できない。
- WebGPU portable loweringが定義できない。
- 型レベルunrollによるcompile-time explosionを招きやすい。

### persistent task queue

初期の汎用loweringとしては不採用。

- BSPと異なる実行順序を持ち得る。
- barrier緩和により追加workや結果順序が変わり得る。
- backend portabilityが低い。

algorithmが意味論上許す場合の明示的execution policyとして将来検討する。

## 参考設計

- CUDA はkernel launchとstream operationをhostに対して非同期とし、結果が必要な時に
  明示的同期を行う。
  <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html>
- CUB device-wide algorithmは結果をdevice outputへ書き、hostと同期しない。さらに
  `DeviceReduce` はdevice memory上のdeferred problem sizeをstream orderで読む。
  これは`MVal`と`MExtent`を次algorithmへ渡す設計に直接対応する。
  <https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html>
- rocPRIM selectionはselected countをdevice output iteratorへ書く。
  <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/device_ops/select.html>
- GraphBLASはnonblocking modeで未完成のoutput objectを次operationへ渡し、
  `GrB_wait`をmaterialization境界とする。execution errorもwaitまでdeferできる。
  <https://graphblas.org/docs/GraphBLAS_API_C_v2.1.0.pdf>
- JAX arrayはfutureとして後続計算へ渡せ、shape/typeは待たずに取得できる一方、
  host value inspectionで待機する。
  <https://docs.jax.dev/en/latest/async_dispatch.html>
- Futharkは明示loopを持ち、反復状態をcompiler管理下に置く。
  <https://futhark.readthedocs.io/en/stable/language-reference.html>
- Accelerateはembedded array computationとscalar computationを区別し、`awhile`で
  array stateを反復する。
  <https://hackage.haskell.org/package/accelerate/docs/Data-Array-Accelerate.html>
- Gunrockはvertex / edge frontierを中心としたbulk-synchronous GPU graph abstraction
  として、advance、filter、computeを合成する。
  <https://arxiv.org/abs/1701.01170>
- Atosはrelaxed dependencyを持つtask-parallel GPU schedulingがBSPを上回り得ることを
  示す。persistent queueを単なるBSP loweringではなく別policyにする根拠である。
  <https://arxiv.org/abs/2112.00132>

## 当時の最終推奨（不採用）

Massively を次の一つのperformance modelへ統一する。

```text
公開algorithmはdevice resultを生成する。
公開algorithmはhostを待たない。
device resultは通常のalgorithmへそのまま渡せる。
hostが値を必要とした時だけreadする。
Iterationは同じalgorithmをboundedに反復する。
```

公開するのは `MVal`、`MExtent`、`MSequence`、device extent対応 `MVec`、
`Iteration` である。
公開しないのはraw buffer handle、mutable control cell、indirect dispatch args、
workspace plan、backend command graphである。

この境界により、通常のalgorithm composition、del2d、Power Diagram、Traversal
Algebra、dense convergenceを同じAPIと同じstorage modelで表現する。
