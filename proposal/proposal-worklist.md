# proposal-worklist: device-resident Worklist と遅延長観測

- 状態: 公開API案は不採用（測定・内部設計の記録）
- 対象: Massively v0.86 以降
- 主な利用事例: `del2d`、Traversal Algebra、frontier 型グラフアルゴリズム
- 現在の正本: [`proposal-compound-operations.md`](proposal-compound-operations.md)

> 2026-07-23再評価: 公開遅延長、公開Worklist、epoch runnerは導入しない。
> 通常APIはhost-exactのまま保ち、device-resident count、caller-provided output、
> scratch再利用は既存primitiveまたはアプリケーション内部で使う。性能差だけを
> 理由に新しい公開runnerを追加しない。以下の性能測定と
> affected-frontier分析は引き続き設計根拠として有効である。

## 要約

Massively の可変長アルゴリズムは、GPU 上で計算した要素数を公開 API の
境界で host の整数へ変換することがある。`copy_where`、`unique_by_key`、
`flat_map`、Traversal の `emit` や次 frontier の生成がその例である。

scan や reduction 自体は問題ではない。問題は、その結果に対して
`read` を行うと、producer kernel の完了、staging copy、map、CPU の起床、
後続 command の構築と submit という GPU -> CPU -> GPU の制御往復が入る
ことである。反復アルゴリズムが毎ラウンド長さや空判定を読むと、GPU の
仕事よりこの固定費が支配的になる。

本提案では、可変長の処理対象を Worklist として扱う。ただし、Worklist のために
`MVec` と競合する新しい storage/extent 型を作ることは提案しない。v0.86 の `MVec` が
既に保持できる `LogicalExtent::Device` をデータ表現として使い、Worklist はその上の
演算規約と反復実行 abstraction とする。

- 物理 `capacity` は host が知る。
- 論理 `length` は `MVec` 内部の device-resident extent として保持する。
- Worklist 演算は論理長を GPU 上で生成、合成、伝播する。
- 可変長 algorithm は結果の `MVec` handle を直ちに返し、長さを eager read しない。
- host が `read_len`、`read_is_empty`、`resolve_len`、`to_host` を明示的に
  呼ぶまで長さの同期を行わない。
- 後続 kernel は同じ device 論理長から indirect dispatch する。
- buffer は利用者に要求せず、host-known capacity に基づいて Massively が
  確保し、反復中は ping-pong で再利用する。
- 終了判定は毎ラウンドではなく、複数 step をまとめた epoch 境界で行う。

Worklist は Traversal Algebra を置き換えるものではない。Worklist を共通の
実行基盤とし、Traversal Algebra は CSR、source、destination、edge、terminal
というグラフ固有の意味を保持する上位層として残す。

## 背景

### 現在の `del2d` の性能

Radeon 680M、Criterion sample size 10 の再計測では、入力 upload を計測外、
三角形分割と最後の GPU sync を計測内として次の結果になった。

| 点数 | Massively / GPU | CPU | GPU / CPU |
| ---: | ---: | ---: | ---: |
| 256 | 95.3 ms | 30.8 us | 3,090x |
| 1,024 | 123.4 ms | 159 us | 775x |
| 4,096 | 163.4 ms | 822 us | 199x |
| 16,384 | 193.8 ms | 3.94 ms | 49x |

入力を 64 倍にしても GPU 時間は約 2 倍にしか増えていない。この形は、
幾何計算の throughput より、ラウンド数、dispatch 数、readback、command
submission などの固定費が支配的であることを示している。

現在の実装には概ね次の反復がある。

```text
未挿入 site がなくなるまで:
    location で radix sort
    競合しない winner を選ぶ
    copy_where で winner を materialize
    host が winner 数を取得
    split と再配置

illegal edge がなくなるまで:
    全 edge の legality を検査
    競合しない winner を選ぶ
    copy_where で winner を materialize
    host が winner 数を取得
    flip
```

各ラウンドは GPU 内で並列でも、ラウンド境界の長さ取得によって command
stream が host に戻っている。また edge flip は、直前の flip によって影響を
受けた edge だけでなく、毎ラウンド全 edge を再検査している。

### v0.86 に既に存在する device-resident 部品

v0.86 は必要な機構の一部を既に持つ。

- `MVal<R, T>` は 1 行の device-resident storage を表す。
- `LogicalExtent::Device` は device-produced length を保持する。
- `MStorage::len` と `MIter::len` は device extent を暗黙に read せず、現在は
  `Error::UnresolvedLength` を返す。
- `Executor::to_host` は device extent を明示的に read して logical prefix を返す。
- scan、radix、selection、graph traversal の内部 lowerings は device extent を
  kernel へ渡せる。
- `TraversalControl` は `capacity` に加え、`output_len` と `required_len` を
  `MVal<R, MIndex>` として保持する。
- CubeCL の現在の依存 revision は `CubeCount::Dynamic` を持ち、WGPU backend
  は `dispatch_workgroups_indirect` を実装している。

一方、公開 API の同期的な返却契約に合わせるため、次のような eager read が
存在する。

```rust
let len = copy_where_into(exec, input, stencil, output.slice_mut(..))?;
output.set_fixed_len(len.read(exec)?);
```

Traversal でも、内部では device length を使っているが、`emit` と
`relax_min_by_destination` の返却前に同じ fixed-length 化を行う。

v0.86 の再利用箇所と不足箇所をまとめると次になる。

| 領域 | v0.86 で既にあるもの | Worklist path に不足するもの |
| --- | --- | --- |
| storage | `MVec` の `LogicalExtent::Device` | 可変長出力で eager read せず extent を返す public path |
| length scalar | crate-private `MVal<R, MIndex>` | public に出さず、結果 `MVec` へ一貫して接続する規約 |
| transform | internal `map_preserving_extent` | capacity allocation を使う public/deferred entry point |
| selection | count、scan、device extent の内部部品 | `copy_where` 返却前の `len.read` の除去 |
| flat-map | count、scan、owner mapping | exact allocation 以外の bounded-capacity path |
| traversal | `TraversalControl::{output_len, required_len}` | terminal の device-extent 返却 |
| dispatch | CubeCL `CubeCount::Dynamic`、WGPU indirect dispatch | logical extent から dispatch arguments を作る共通 lowering |
| iteration | queue 上の通常の launch ordering | ping-pong、epoch、終了観測をまとめる runner |

また、公開 algorithm の一部は allocation のために `input.len()?` を要求する。
device extent を受け取ると `UnresolvedLength` になるため、Worklist chain では
logical length ではなく host-known physical capacity から allocation する必要がある。

したがって必要なのは device scalar や第二の可変長 vector を新しく発明することでは
ない。v0.86 の既存 device extent を公開アルゴリズム間で失わずに伝播し、allocation は
capacity から行い、host 観測だけを明示的な同期境界にすることである。

## 問題設定

### 1. device 計算と host 観測が結合している

GPU 上で長さを計算することと、CPU がその長さを知ることは別の操作である。
現在の同期的な可変長 API はこの二つを一つの関数内で行うため、利用者がその
長さを CPU で使わなくても readback が起こる。

### 2. 次の GPU 処理へ渡すだけでも fixed length 化される

可変長出力を直後の `map`、`scan`、`sort`、`gather`、`reduce_by_key`、graph
traversal へ渡す場合、CPU は論理長を知る必要がない。同じ queue 上で producer
が length buffer へ書き、consumer がそれを読めばよい。

### 3. host-driven な `while` が毎ラウンド queue を分断する

`while !frontier.is_empty()` を host で評価すると、各 frontier の生成後に同期が
必要になる。既に submit 済みの GPU work は進められるが、次のラウンドは CPU
が結果を受け取るまで encode も submit もできない。

### 4. full-capacity 実行だけでは遅い

最大 capacity 分を毎回 dispatch し、kernel 内で `index < device_len` を判定すれば
readback は除去できる。しかし疎な frontier に対して無効 lane が増える。`del2d`
で試した完全固定長版は正しく動作したが、特に大きい入力で現在の compact 版より
遅かった。

device length の保持だけでなく、device length に応じた dispatch と疎な仕事の
再投入が必要である。

### 5. 一般の exact-size `flat_map` は host allocation に依存する

各入力の出力数を `count` とすると、総出力数は scan または reduction で安価に
計算できる。

```text
M = sum(count(input[i]))
```

しかし WGPU の連続 buffer 作成には host の byte size が必要である。GPU 上の
`M` から正確なサイズの buffer を作る場合、`M` の host read が必要になる。

本提案はこの制約を隠さない。Worklist の expansion は host-known な安全な上限を
使う。完全に一般的で上限のない exact-size `flat_map` は、同期版として残すか、
別の chunked storage を必要とする。

## 目標

- 可変長の中間結果を CPU に戻さず、複数の Massively operation へ渡せる。
- v0.86 の `MVec` / `LogicalExtent` を carrier として使い、同じ目的の storage 型を
  二重に導入しない。
- `filter`、bounded expansion、key reduction、state update、次 frontier 生成を
  `Worklist -> Worklist` として合成できる。
- 長さ取得による同期は、利用者が明示的に host 観測した時だけ起こる。
- device length から indirect dispatch 数を生成できる。
- Worklist の storage を反復ごとに再確保せず再利用できる。
- Traversal Algebra の frontier と edge stream が同じ Worklist 実装を使う。
- fixed-capacity な mutable topology 上の graph rewrite を表現できる。
- 既存の同期版 API を互換性のため維持できる。
- multi-column SoA storage が一つの extent を共有する。
- overflow、foreign executor、stale work item を安全に検出できる。

## 非目標

- GPU kernel から WGPU buffer を新規作成すること。
- 上限のない任意の flat-map を同期なしで exact-size allocation すること。
- WebGPU 上で無制限の device-side `while` や global barrier を提供すること。
- Worklist を追加するだけで小規模入力の GPU が CPU より速くなると保証すること。
- Traversal Algebra の source、destination、edge という意味論を削除すること。
- 最初の変更ですべての vector / segment / graph algorithm を Worklist 対応すること。
- `MVal` や `LogicalExtent` をそのまま public API に露出すること。

## 基本モデル

### Physical capacity と logical length

Worklist の carrier となる `MVec` は、物理容量と論理長を分離する。

```text
capacity: host-known、allocation と安全性の上限
length:   device-known、現在有効な work item 数
```

常に次の不変条件を満たす。

```text
0 <= length <= capacity <= MIndex::MAX
```

長さを保存する device scalar は crate-private のままでよい。利用者が GPU scalar
を直接操作する必要はない。

v0.86 の `MStorage` は既に物理 `capacity` と `LogicalExtent` を持つ。ここへ別の
`DeviceExtent` を重ねず、可変長 algorithm の出力 storage に
`LogicalExtent::Device` を設定する。tuple/SoA の全 column は同じ extent identity を
共有する。

### Worklist は storage 型ではなく演算規約

本書の `Worklist<R, T>` は、device extent を持つ `MVec<R, T>` を Worklist として
扱っていることを示す仕様上の表記である。

```rust
// 仕様上の表記。新しい公開 storage 型を要求するものではない。
type Worklist<R, T> = MVec<R, T>;
```

実 API は既存 vector algorithm の deferred variant、`worklist` module、または
`MVec` の extension method として提供できる。反復用の ping-pong buffer と scratch の
寿命を管理する `WorklistLoop` は別途有用だが、item storage と logical extent は
あくまで既存 `MVec` のものを使う。

Worklist は逐次 queue の FIFO pop/push を意味しない。各 step で有限の work item 列を
一括処理する bulk-synchronous frontier である。すべての Worklist operator は次を
守る。

1. allocation size は host-known な input capacity と operator の構造上限から求める。
2. 実際に処理する範囲は input の `LogicalExtent` から求める。
3. 出力件数は device scalar として出力 `MVec` の extent へ接続する。
4. 返却前にその scalar を read しない。

演算ごとの順序契約は個別に定義する。

- `map` は入力順を保持する。
- `filter` は stable compaction を行う。
- `expand_bounded` は入力順、次に local output index 順を保持する。
- `reduce_by_key` は key と明示された tie-break 規則で決定的にする。
- 順序を保証しない高速 variant を追加する場合は別名にする。

### 明示的な host 観測

`MStorage::len()` と `MIter::len()` は現在と同じく暗黙に同期しない。fixed extent なら
host 値を返し、device extent なら `Error::UnresolvedLength` を返す。同期して長さを
知りたい場合だけ、名前から同期が分かる API を明示的に呼ぶ。

API 名と trait 境界は議論用である。

```rust
impl<R: Runtime> Executor<R> {
    /// extent scalar だけを読む、明示的な同期境界。
    pub fn read_len<S: MStorage<R>>(&self, input: &S) -> Result<MIndex, Error>;

    /// 明示的な同期境界。
    pub fn read_is_empty<S: MStorage<R>>(&self, input: &S) -> Result<bool, Error>;

    /// device length を読み、同じ storage の extent を Fixed に cache する。
    /// item storage は copy しない。
    pub fn resolve_len<S: MStorage<R>>(&self, input: &mut S) -> Result<MIndex, Error>;
}
```

`read_len` は値を観測するだけで extent を変更しない。`resolve_len` は同期後に
`set_fixed_len` 相当の処理を行うため、その後の `.len()` は同期なしで返せる。
device length buffer が将来再利用される反復 runner では、escape した `MVec` が mutable
control buffer を参照し続けないよう、返却時に extent を snapshot するか所有権で禁止
する。

`MVec` の handle を返すこと自体は同期ではない。producer kernel と length kernel を
queue へ enqueue した後、storage と device extent の handle を CPU 上で組み立てて返せる。
同期が生じるのは、その extent の数値を host が要求した時である。

既存 `Executor::to_host` は既に明示的な観測境界であり、device extent を読んで logical
prefix を返す。この path はそのまま利用できる。将来は length と物理 storage を一つの
read submission へまとめる実装も検討できる。capacity 全体の転送が高価な場合は、length
read と正確な範囲の read を分ける同期版との比較が必要である。

### 同期契約

| 操作 | host synchronization |
| --- | --- |
| `MStorage::len` / `MIter::len` | なし。device extent なら `UnresolvedLength` |
| `capacity` | なし |
| `map` / `filter` / `expand_bounded` | なし |
| `reduce_by_key` / `unique` / `claim_all` | なし |
| `graph::traverse` / Worklist terminal | なし |
| `read_len` / `read_is_empty` | あり |
| `resolve_len` / `to_host` | あり |
| `Executor::sync` | あり |

同期版の既存 API は、この明示的な観測を内部で呼ぶ compatibility wrapper として残せる。
重要なのは「API 関数が返ること」と「GPU work が完了すること」を同一視しないことで
ある。deferred API の返却は enqueue 完了を意味し、計算完了は後続 queue dependency
または明示的な観測で保証する。

### 可変長出力の lifecycle

selection を例にすると、内部実装は eager read を次の接続へ置き換える。

```rust
let count = copy_where_into(exec, input, stencil, output.slice_mut(..))?;
output.set_logical_extent(LogicalExtent::from_device(
    count.storage(),
    output_capacity,
));
Ok(output)
```

ここで `count.storage()` は説明用の表記であり、`MVal` を公開する意図はない。producer
kernel、count kernel、後続 consumer は同じ queue の buffer dependency で順序付けられる。

利用側は次のようになる。

```rust
let selected = worklist::filter(&exec, input, pred)?; // readback なし
let next = worklist::map(&exec, selected, op)?; // extent を伝播
let reduced = worklist::reduce_by_key(&exec, next, reduce)?; // readback なし

// CPU が本当に必要としたここが、初めての同期点。
let count = exec.read_len(&reduced)?;
```

`read_len` を呼ばず、そのまま別の GPU algorithm や `to_host` へ渡してもよい。後者では
`to_host` 自体が明示的な最終同期になる。

## Worklist operation

### 固定長からの生成

```rust
pub fn from_iter<R, Input>(
    exec: &Executor<R>,
    input: Input,
) -> Result<Worklist<R, Input::Item>, Error>
where
    R: Runtime,
    Input: MIter<R>;
```

host-known length の入力では、device length を新しく計算する必要はない。fixed extent を
device scalar として materialize するのは、最初に consumer が必要とした時でよい。

### Length-preserving transform

```rust
pub fn map<R, Input, Op>(
    exec: &Executor<R>,
    input: Worklist<R, Input>,
    op: Op,
) -> Result<Worklist<R, Op::Output>, Error>;
```

出力 capacity と device length は入力と同じである。kernel は device length から
indirect dispatch する。

### Filter / compact

```rust
pub fn filter<R, T, Pred>(
    exec: &Executor<R>,
    input: Worklist<R, T>,
    pred: Pred,
) -> Result<Worklist<R, T>, Error>;
```

実装は predicate flag、scan、scatter を使える。scan 末尾の selected count は
`MVal` として出力 extent に格納し、read しない。出力 capacity は入力 capacity 以下
なので host で安全に確保できる。

`copy_where` と `remove_where` はこの operation の fixed-input convenience wrapper に
できる。

### Partition

現在の `partition` は output の総要素数自体は入力と同じだが、selected/rejected の境界を
host の `MIndex` で返すため、その境界の read が同期になる。後続 GPU 処理が二つの集合を
必要とするだけなら、host boundary は不要である。

Worklist 向けには、一つの contiguous output と device boundary を公開するより、次の
semantic API を優先する。

```rust
pub fn partition_worklists<R, T, Pred>(
    exec: &Executor<R>,
    input: Worklist<R, T>,
    pred: Pred,
) -> Result<(Worklist<R, T>, Worklist<R, T>), Error>;
```

selected count と rejected count はそれぞれの出力 extent に接続され、read しない。
storage を一つに保ちたい低レベル実装では device boundary を内部 control として共有して
よいが、v0.86 の `LogicalExtent::Device` は host-known `start` の slice を表す設計なので、
device start を public range API へ無理に持ち込まない。

既存 `partition -> (MVec, MIndex)` は host boundary が必要な同期版として維持できる。
つまり partition 全体の API を一律に変更するのではなく、device-resident な意味を持つ
二分 Worklist terminal を追加する。

### Bounded expansion

```rust
pub trait BoundedExpandOp<Input>: ExpandOp<Input> {
    /// 1 input item が生成する最大 item 数。host で取得できる安全な上限。
    fn max_outputs(&self) -> MIndex;
}

pub fn expand_bounded<R, Input, Op>(
    exec: &Executor<R>,
    input: Worklist<R, Input>,
    op: Op,
) -> Result<Worklist<R, Op::Output>, Error>
where
    Op: BoundedExpandOp<Input>;
```

出力 capacity は次で求める。

```text
output_capacity = input.capacity * op.max_outputs()
```

各 input の count と scan offset は GPU で計算する。総出力数は device extent へ保存
する。生成 kernel は次のいずれかを選べる。

1. input item ごとに dispatch し、各 lane が自分の `count` 件を offset から書く。
2. output item ごとに indirect dispatch し、owner mapping から input を求める。

小さく均一な上限では 1、出力数の偏りが大きい場合は 2 が有利になり得る。これは
Worklist program の意味を変えずに lowering が選択すべき事項である。

利用者へ出力 buffer を渡させない。Massively が capacity を確保し、反復中の
workspace は pool または ping-pong storage で再利用する。

上限のない既存 `flat_map` は exact-size 同期版として残せる。別案として chunked
Worklist を将来追加できるが、本提案の初期範囲には含めない。

### Key reduction と重複除去

```rust
pub fn reduce_by_key<R, K, V, Reduce>(
    exec: &Executor<R>,
    input: Worklist<R, (K, V)>,
    reduce: Reduce,
) -> Result<Worklist<R, (K, V)>, Error>;
```

結果数は入力数以下なので、入力 capacity を上限にできる。sort-reduce、atomic、
hierarchical reduce の選択は lowering に隠す。

`unique` は key ごとの代表を返す specialization とする。

### Multi-resource claim

Delaunay の insertion や flip は、一つの candidate が複数の triangle / edge resource
を同時に占有する。単一 destination の reduction だけでは commit 条件を表せない。

一般化した意味は次である。

```text
candidate
  -> bounded_expand claims(candidate)
  -> resource ごとに priority reduce
  -> candidate が全 resource を所有するか検査
  -> winner だけ commit
```

議論用 API は次のように表せる。

```rust
pub fn claim_all<R, Candidate, Resource, Priority, Claims>(
    exec: &Executor<R>,
    candidates: Worklist<R, Candidate>,
    claims: Claims,
) -> Result<Worklist<R, Candidate>, Error>;
```

`Claims` は candidate ごとの最大 claim 数、resource ID、priority を定義する。priority
と candidate ID に total order を要求すれば、実行順に依存しない決定的な winner を
選べる。

この operation は Delaunay 専用ではない。matching、mesh refinement、parallel graph
rewrite、複数 resource を占有する scheduling に利用できる。

### Stateful update と次 Worklist

```rust
pub fn apply_and_emit<R, Item, State, Apply>(
    exec: &Executor<R>,
    input: Worklist<R, Item>,
    state: State,
    apply: Apply,
) -> Result<Worklist<R, Apply::Next>, Error>;
```

`apply` は winner を fixed-capacity state table へ書き、各 winner から bounded な次
work item を生成する。実装は更新と emit を一つの kernel に fuse しても、別々の
kernel へ lower してもよい。

## Device extent の演算

後続 operation が host length を必要としないためには、単一 scalar を保存するだけで
なく、派生長を GPU 上で合成できる必要がある。

必要な基本演算は次である。

- preserve: `map` などの同一長
- clamp / min: capacity への制限
- scale: fixed expansion の `length * K`
- add: concat の `left + right`
- subtract: filter 後の complement など
- ceil-div: block count と indirect dispatch 引数
- equality / zero test: device flag と終了状態
- slice: host-known offset と limit による clamp

v0.86 の `LogicalExtent` には既に slice/clamp、add、min、equality、less、ceil-div などの
device 演算がある。まずこれらを再利用し、scale や subtract など不足分だけを追加する。
各演算は host 側で capacity overflow を検査し、必要な device scalar 演算だけを enqueue
する。

別々に生成された extent を暗黙に zip してはならない。同じ extent identity を共有する
場合のみ通常の zip を許可し、それ以外は `zip_min` や明示的な length check を要求する。

multi-column Worklist の全 column は同じ extent identity を共有する。

## Indirect dispatch

device-resident length を保持しても、毎回 capacity 分を dispatch すると疎な Worklist
で無効 lane が支配する。Worklist operation は device length から dispatch arguments
を生成する。

```text
groups = ceil(length / items_per_group)
dispatch = [groups, 1, 1]
```

この 3 要素の buffer を `CubeCount::Dynamic` へ渡す。length が 0 の場合は 0 group
となり、host の空判定なしで no-op にできる。実装時には各 backend の 0-group indirect
dispatch の扱いを conformance test し、必要なら 1 group + device guard へ lower する。

出力 length buffer は、入力 dispatch が 0 でも必ず 0 になる必要がある。前 epoch の値が
残った ping-pong control buffer を、0-group kernel だけで上書きすることはできない。
各 step は dispatch 前に次 extent/status を clear するか、常に実行される control kernel
で初期化する。これは空 Worklist を absorbing state にするための correctness 条件である。

backend が indirect dispatch を提供しない場合は、capacity dispatch と device guard を
fallback とする。その場合も意味は同じだが、性能特性が異なることを明示する。

indirect dispatch buffer は extent ごとに毎回新規確保せず、extent から lazily
materialize して共有するか、Executor の小さな control-buffer pool を使う。

indirect dispatch は host readback と無効 lane を減らすが、kernel launch 数そのものは
減らさない。scan、radix、sort のような multi-stage algorithm は capacity から最大 stage
DAG を host で構築し、各 stage の実 group 数だけを device extent から与える。空 stage
でも後続 control 値を正しく初期化する必要がある。

したがって性能実装では、extent 伝播と同時に次も行う。

- predicate + count、scatter + next-task emit など安全な kernel fusion
- 同じ extent から作る indirect arguments の共有
- scratch と pipeline/cache の再利用
- epoch 内の command submission batching

## Host との反復を減らす Epoch 実行

Worklist が次 Worklist を同期なしで返しても、host が毎回 `read_is_empty` を呼べば
同期は残る。反復 abstraction は複数 step を一つの epoch として enqueue する。

議論用 API:

```rust
pub struct WorklistLoop<R: Runtime, State, Item>
where
    Item: MAlloc<R>,
{
    state: State,
    current: MVec<R, Item>,
    next: MVec<R, Item>,
}

impl<R, State, Item> WorklistLoop<R, State, Item>
where
    R: Runtime,
    Item: MAlloc<R>,
{
    /// `steps` 回を結果の host 観測なしで enqueue する。
    pub fn enqueue_epoch<Step>(
        &mut self,
        exec: &Executor<R>,
        steps: u32,
        step: Step,
    ) -> Result<(), Error>;

    /// epoch 終了後の明示的な同期境界。
    pub fn read_done(
        &self,
        exec: &Executor<R>,
    ) -> Result<bool, Error>;
}
```

Rust の `for` は各 step の command を構築するが、device length の値を読まないため
GPU 完了を待たない。各 step は前 step が書いた length と state buffer を同じ queue
上で読む。

```text
CPU: step 0, 1, ... 7 の command を enqueue
GPU: step 0 -> step 1 -> ... -> step 7
CPU: epoch 末尾で done を 1 回だけ観測
```

空になった後の step は indirect dispatch が 0 になり no-op となる。

epoch 化できる step には次の契約を要求する。

- current/next と scratch の capacity が host で事前に決まる。
- 一 step の expansion が bounded である。
- empty input は empty output を生成する。つまり空状態が absorbing である。
- 次 step の構築に host scalar や algorithm-specific な分岐を必要としない。
- ping-pong する length/status buffer の clear と queue dependency を runner が管理する。

この契約を満たさない step は、無理に epoch へ入れず明示的な観測境界で分ける。

WebGPU は device が新しい command を自己生成する無制限 loop を提供しない。そのため
完全な `run_until_empty` は次のいずれかになる。

- 安全な最大 step 数を host が事前 enqueue し、最後だけ観測する。
- 8、16、32 などの固定 epoch ごとに一度だけ観測する。
- 特定 algorithm を persistent kernel として実装する。ただし汎用 Worklist の初期
  lowering には含めない。

epoch size は固定 API 契約ではなく、algorithm または autotuning が選べるようにする。

## Traversal Algebra との関係

Worklist は計算表現力として Traversal Algebra の frontier execution を包括できる。

```text
Traversal Algebra
= Worklist
+ CSR による bounded expansion
+ source / destination / edge context
+ graph 固有 terminal
```

しかし Traversal Algebra は削除しない。TA は次の意味を保持する。

- source、destination、edge の型付き context
- CSR row と edge position
- reduce-by-source / destination の terminal 契約
- push / pull、segmented reduction、atomic、sort-reduce などの graph 専用 lowering
- 既存の意味論、検証 artifact、graph algorithm API

推奨する層構造は次である。

```text
Worklist Core
|- device extent / indirect dispatch
|- filter / bounded expansion / reduce-by-key
|- buffer reuse / epoch execution
|
+- Traversal Algebra
|  |- CSR traversal semantics
|  |- edge expressions
|  `- graph terminals
|
`- Dynamic Graph Rewrite
   |- mutable stable-slot topology
   |- multi-resource claim
   `- state update + affected-item emission
```

実装上、`TraversalControl` が独自に持つ `output_len` と capacity-backed edge storage は、
`MVec` の device extent と共通の indirect-dispatch helper へ接続する。Worklist 専用の
第二の extent 実装へ移すのではない。

```rust
pub struct Traversal<R: Runtime> {
    edges: Worklist<R, EdgeContext>,
    graph_context: TraversalContext<R>,
}
```

新しい terminal は device frontier を返す。

```rust
pub fn emit_worklist(...) -> Result<Worklist<R, Output>, Error>;

pub fn relax_min_by_destination_worklist(...)
    -> Result<Worklist<R, MIndex>, Error>;
```

既存 `emit` と `relax_min_by_destination` は、互換性のため `resolve_len` を呼んで
host-visible fixed-length `MVec` を返す wrapper にできる。

将来の major version では Worklist 返却を既定にし、同期版を `emit_resolved` などの
明示名へ変更する案を検討できる。

## `del2d` への適用

### State

現在の stable triangle slot と stable edge slot を fixed-capacity な mutable graph state
として継続利用する。

```text
TriangleTable
|- vertices: a, b, c
`- incidences: e0, e1, e2

EdgeTable
|- endpoints: a, b
|- forward incidence
`- reverse incidence
```

slot は再配置せず、generation / epoch tag で stale work item を無効化する。

### Site insertion Worklist

```rust
struct SiteTask {
    point: u32,
    triangle: u32,
    triangle_generation: u32,
}
```

一つの step は次の意味を持つ。

```text
SiteTask
  -> stale task を filter
  -> triangle / adjacent triangle claims を bounded expand
  -> resource ごとに winner を決定
  -> 全 claim を所有する site を commit
  -> triangle split
  -> loser と未挿入 site の location を更新
  -> 次 SiteWorklist を emit
  -> 新しく影響を受けた edge を EdgeWorklist へ emit
```

この形では winner 数を host へ返す必要がない。site を triangle ごとに管理できれば、
毎ラウンド全 remaining site を location で radix sortする処理も削減できる。

### Edge flip Worklist

```rust
struct EdgeTask {
    edge: u32,
    edge_generation: u32,
}
```

初期 frontier には active interior edge を一度だけ入れる。その後は flip によって影響を
受けた周辺 edge だけを再投入する。

```text
EdgeTask
  -> stale / boundary edge を filter
  -> incircle predicate
  -> adjacent triangle claims
  -> winner flip を commit
  -> 最大 4 本程度の周辺 edge を emit
  -> unique / generation tag で重複を抑制
```

これにより、現在の「各 flip round で全 edge を検査」を event-driven な処理へ変更
できる。

### Phase scheduling

最初の実装は insertion と flip を別 phase に保つ。

1. SiteWorklist を epoch 単位で空にする。
2. EdgeWorklist を epoch 単位で空にする。
3. super triangle を除去し、最終出力だけ `resolve_len` または `to_host` する。

将来は split 直後に EdgeWorklist を処理する interleaved schedule を検討できるが、
正当性と決定性の検証範囲が増えるため初期目標には含めない。

## Allocation と buffer reuse

利用者に output buffer を渡させることを基本 API にしない。Worklist operator は
host-known capacity から必要な storage を内部確保する。

反復中は次を再利用する。

- current / next の ping-pong item storage
- predicate flags
- scan positions
- key sort / reduction scratch
- claim owner table
- indirect dispatch arguments
- overflow / error flags

同じ shape と item layout の scratch は Executor pool から取得する。`WorklistLoop` の
workspace lease または一時 `MVec` の drop 時に物理 allocation 全体を pool へ戻し、
logical length に合わせた再 allocation は行わない。

capacity の導出は operation の意味に基づく。

| Operation | Output capacity upper bound |
| --- | ---: |
| `map` | input capacity |
| `filter` / `unique` | input capacity |
| `expand_bounded(K)` | input capacity * K |
| `concat` | left capacity + right capacity |
| `reduce_by_key` | input capacity |
| `claim_all(K)` claim storage | candidate capacity * K |

capacity overflow は host で事前検査できる。

安全な構造上限がない operation は、同期 exact-size path か、利用者が明示する semantic
bound を要求する。単に大きそうな buffer を確保して silent truncation してはならない。

## Overflow と error の扱い

反復を管理する `WorklistLoop` または algorithm-specific control は device-resident status
を持てる。これは `MVec` の length extent とは別の control である。

```text
OK
capacity overflow
invalid index
algorithm-specific error
```

stale generation は通常のエラーではなく、古い work item を捨てる filter 条件として扱う。

保証された bounded operator の通常経路では overflow は起こらない。実行時 bound を
受け取る operator は、required length と materialized length を区別し、overflow flag
を伝播する。

state を部分更新した後で overflow が分かる設計は再試行不能になるため避ける。
stateful terminal は次のどちらかを要求する。

- capacity が構造的に十分であることを host で証明できる。
- state 更新前に `read_fits` する明示的同期 path を選ぶ。

`resolve_len` は length だけを読む。device status を持つ runner は、最終 `finish` または
明示的な `read_status` で status も観測して `Error` として返す。通常の bounded operator
は構造上 overflow しないため、length 観測へ不要な status read を追加しない。

## 既存 API との互換性

初期実装は additive にする。

- `MVec` は v0.86 と同じく fixed extent と device extent の両方を保持できる carrier と
  する。
- 新しい Worklist storage 型は追加しない。
- `map_preserving_extent` など既存の内部 lowering は再利用する。公開 API が
  `input.len()?` を要求する箇所には、capacity から allocation する deferred variant を
  用意する。
- `worklist::filter`、`partition_worklists`、`expand_bounded` など、readback しない
  operation を additive に追加する。
- 既存 `copy_where`、`remove_where`、`partition`、`unique_by_key`、`flat_map` は当面、現在の
  同期的な契約を維持し、deferred operation + `resolve_len` の compatibility wrapper に
  する。
- graph terminal も `emit_worklist` などの deferred variant を追加し、既存 terminal は
  `resolve_len` wrapper とする。

この rollout なら既存の `.len()` が即座に host 値を返す call site を壊さない。一方、
Worklist path の `.len()` は device extent に対して現在どおり `UnresolvedLength` を返し、
高価な同期を見えない場所へ移動しない。

関数 signature だけを見れば、既存可変長 algorithm が device extent 付き `MVec` を返す
ように変更することも可能である。しかし、返却直後の `.len()` 成功を前提にした既存 code
には observable な変更になる。標準 API を deferred に切り替えるなら major version で
行い、同期版を `*_resolved` という明示名へ移す。

## 実装計画

### Phase 1: Worklist Core

- 既存 `MVec`、`LogicalExtent::Device`、`MVal<MIndex>` をそのまま carrier として使う。
- `read_len`、`read_is_empty`、`resolve_len` という明示的な観測 API を追加する。
- algorithm の allocation を logical `.len()` ではなく physical capacity / upper bound から
  行える内部 helper を整える。
- multi-column extent identity と owner check を実装する。
- device length から indirect dispatch arguments を生成する。
- 既存 `map_preserving_extent` を Worklist path から使い、deferred `map` と `filter` を
  実装する。

### Phase 2: Selection と expansion

- `copy_where_into` の count を read せず出力 `MVec` extent に接続する。
- `partition_worklists` を追加し、host boundary を不要にする。
- `unique` と `reduce_by_key` を Worklist 対応する。
- bounded expansion と fixed expansion を実装する。
- scratch / ping-pong buffer reuse を導入する。

### Phase 3: Traversal Algebra

- `TraversalControl` の edge stream を共通の extent / indirect-dispatch helper へ接続する。
- `emit_worklist` と device-resident next frontier を追加する。
- `reduce_by_source` / destination の fixed-state terminal は既存 API を維持する。
- graph algorithm の host frontier loop を epoch 実行へ移せるようにする。

### Phase 4: Epoch と resource claim

- indirect dispatch を使う multi-step epoch を追加する。
- deterministic `claim_all` を実装する。
- device status と delayed error observation を追加する。
- readback / submission 数を計測できる tracing を追加する。

### Phase 5: `del2d`

- まず edge flip を affected-edge Worklist へ移す。
- correctness と性能を確認後、site insertion を Worklist 化する。
- 毎ラウンドの radix sort を triangle-local claim / bucket へ置換する。
- 最終 output 以外の host length observation を除去する。

## 検証

### Correctness

- empty、capacity ぴったり、単一 item、複数 column
- filter / expand / reduce の連鎖
- fixed extent と device extent の zip
- foreign Executor の拒否
- capacity arithmetic overflow
- deterministic tie-break
- stale generation の除外
- Worklist graph traversal と既存 TA oracle の一致
- `del2d` と CPU Delaunay の canonical triangle 一致
- cocircular、collinear、edge insertion、duplicate point

### Synchronization contract

test backend または tracing counter で read operation を数える。

```text
Worklist chain の途中: read count = 0
read_len 直後:         read count = 1
epoch K step:          epoch 境界まで read count = 0
resolve_len / to_host: 明示した回数だけ read
```

公開 documentation には、同期する operation を明示する。

### Performance

次を別々に計測する。

- kernel / dispatch 数
- queue submission 数
- scalar readback 数
- epoch size 別の時間
- static capacity dispatch と indirect dispatch
- buffer allocation / pool hit 数
- Worklist item 数と重複率
- `del2d` insertion round と flip event 数

現在の `del2d` benchmark を baseline とし、特に次を確認する。

- 256 -> 16,384 点で固定費が支配する形が改善するか。
- affected-edge Worklist により全 edge legality 検査数が減るか。
- readback 削減だけでなく dispatch と処理 item 数も減っているか。
- 小規模入力で過剰な Worklist setup cost を増やしていないか。

Worklist 導入だけで CPU 超えを成功条件にはしない。最初の成功条件は、host-driven な
ラウンド同期を測定可能な形で除去し、GPU 時間が入力仕事量に応じてスケールすること
である。

## 代替案

### Worklist 専用の dynamic storage 型を追加する

`Worklist { storage, extent }` という新型を作る案。型だけで device-resident な可変長を
区別できる利点はある。一方、v0.86 の `MVec` / `MStorage` は既に `LogicalExtent::Device`
を保持でき、map、scan、radix などの内部 lowering もその extent を伝播する。別型にすると既存 algorithm
との adapter、SoA storage 実装、zip/slice 規則を二重化する。

初期案では採用しない。buffer reuse と epoch の所有権を表す `WorklistLoop` は追加しても、
item carrier は `MVec` のままにする。

### `.len()` が自動で遅延 read する

可変長 algorithm の signature を変えず、結果 `MVec` の `.len()` が初回だけ同期して値を
cache する案。長さを求めた瞬間に同期するという利用感は自然だが、現在 O(1) か
`UnresolvedLength` と分かる `.len()` が queue wait になり得る。さらに extent 内へ
Executor client と同期 cache を保持する必要がある。

本提案では `.len()` を非同期のまま保ち、`read_len` / `resolve_len` という名前で同期を
明示する。将来 `.len()` の自動 resolve を選ぶ場合でも、Worklist の内部伝播設計自体は
変わらない。

### `MVal` を公開する

表現力は得られるが、利用者が scalar storage、owner、extent identity、capacity 証明を
扱う必要が生じる。Worklist operation が内部利用すれば十分なので、初期案では公開
しない。

### 利用者が output buffer を渡す

既に再利用可能な buffer を持つ場合は有効だが、毎回 reduce -> read -> allocate ->
`*_into` とすると同期を外へ移しただけになる。低レベル optimization hook としては
残せるが、基本 API にはしない。

### 毎回 capacity 全体を処理する

readback は消えるが、疎な frontier で無効 work が増える。indirect dispatch または
active block compaction の fallback が必要である。

### `read_async` にする

caller thread を解放し、独立した仕事と overlap できる。しかし後続 allocation や
dispatch が length に依存する場合、critical path の GPU -> host -> GPU 往復は残る。

### Device-side allocator / chunked storage

大きな arena や page pool から GPU が chunk を予約すれば、上限のない expansion を
表現できる可能性がある。一方、標準 `MVec` の連続 SoA layout、sort、scan、gather を
すべて chunk 対応する必要がある。将来案とし、最初の Worklist には含めない。

## 未決事項

- 公開名を `Worklist`、`Frontier`、`DeviceSequence` のどれにするか。
- stable order を既定にするか、unordered variant を既定にするか。
- bounded expansion の上限を associated const、method、constructor argument のどこに
  持たせるか。
- Worklist という名前を module、extension trait、runner のどこへ公開するか。
- indirect dispatch 非対応 backend の性能保証をどう記述するか。
- epoch size の既定値と autotuning。
- 複数 Worklist を同時に生成する step の API。
- device status をいつ強制観測するか。
- graph verification artifact に Worklist lowering の refinement proof を追加する範囲。

## 当時の推奨結論（不採用）

Massively の可変長反復を、host-visible vector length の連鎖として扱うのをやめ、
既存 `MVec` の device extent を carrier とする Worklist algebra として扱う。

1. `MVec` が `capacity + device logical length` を保持し、Worklist 専用 storage は増やさない。
2. Worklist operation は length を read せず、次の `MVec` へ伝播する。
3. host synchronization は `read_*`、`resolve_len`、`to_host` に限定する。
4. indirect dispatch と buffer reuse を Worklist の実装契約に含める。
5. 複数 step を epoch として enqueue し、host 終了判定を間引く。
6. Traversal Algebra は Worklist Core 上の graph-specific semantic layer とする。
7. `del2d` は SiteWorklist と EdgeWorklist により event-driven な動的 graph rewrite として
   実装する。

この設計は `del2d` 固有の最適化ではない。frontier graph algorithm、mesh processing、
matching、relaxation、parallel rewrite など、device 上で「次に処理すべき疎な集合」を
反復生成するアルゴリズムに共通する基盤となる。

本提案書は設計検討のみであり、Massively 本体のソース変更は行わない。
