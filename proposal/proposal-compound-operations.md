# proposal-compound-operations: Segmentationと少数primitiveによる合成

- 状態: 採用
- 対象: Massively v0.87
- 主な利用事例: Power Diagram、del2d、CSR／ragged data
- 不採用案: 公開`MVal`／`MExtent`、全algorithmのnonblocking化、汎用`Iteration`、
  context専用segmented adapter

## 最終判断

Massivelyの通常APIはhost-exactな一種類に保つ。

- `MVec::len()`は`MIndex`を返す。
- reduction、predicate、searchは通常のhost valueを返す。
- 可変長algorithmは公開return boundaryで長さを観測し、正確な`MVec`を返す。
- 固定長処理とcaller-provided outputは、長さのためのreadbackを行わない。
- `MVal`とdevice logical extentはcrate内部に限定する。
- 利用者へ容量上限を要求する`flat_map_bounded`を主要APIにしない。
- 任意の処理を反復する汎用`Iteration`を公開しない。

今回追加する基礎抽象は、反復runnerではなく`seg::Segmentation`である。

## 設計原則

Massivelyの目的は、少数の直交するGPU primitiveを合成して複雑なalgorithmを完全に
並列実行できることにある。便利関数や一つの利用形に対応するadapterを増やすことでは
ない。

- 一様な値は`constant`、indexは`counting`で表す。
- 関係は`permute`、行の結合は`zip`、変換は`map`で表す。
- partitionは`Segmentation`、segment単位のviewは既存`SegmentIterator`で表す。
- empty segmentのようにentryが存在しない場合は、lengthsとcontextに対する別の
  segment-level passを並列実行し、結果を`zip`／`map`で合成する。

一つのkernelで最速にならなくても、既存primitiveの有限個のGPU passで表現できるなら
それを優先する。性能上の理由だけで新しい意味論的抽象を公開せず、必要なら既存の
合成式を保ったまま内部fusionする。新しい公開primitiveが必要なのは、既存primitiveで
意味を表現できない場合だけである。

## 問題の捉え直し

次の三つは別のデータではない。

```text
segment lengths  [1, 2, 3]
segment IDs      [0, 1, 1, 2, 2, 2]
offsets          [0, 1, 3, 6]
```

いずれも「長さ6のflat rangeを三つの連続segmentへ分割する」という同じ値を表す。
これまでのAPIはoffsetsを各algorithmへ個別に渡していたため、次の一般的な操作が
明示されていなかった。

- segmentごとのcontextをflat entryへbroadcastする。
- lengthsからoffsetsを作る。
- offsetsからowner IDを作る。
- 同じ分割を別のsingle-column／multi-column valuesへ再利用する。
- empty segmentを失わずに表現間を変換する。

Power Diagramとdel2dに必要だったものの一部は「iteration」ではなく、この分割の
所有と変換である。

## 公開API

```rust,ignore
pub struct Segmentation<R: Runtime> {
    // canonical private offsets
    // host-exact value_count
}

impl<R: Runtime> Segmentation<R> {
    pub fn from_offsets<Offsets>(
        exec: &Executor<R>,
        offsets: Offsets,
    ) -> Result<Self, Error>
    where
        Offsets: MIter<R, Item = MIndex>;

    pub fn from_lengths<Lengths>(
        exec: &Executor<R>,
        lengths: Lengths,
    ) -> Result<Self, Error>
    where
        Lengths: MIter<R, Item = MIndex>;

    pub fn from_segment_ids<Ids>(
        exec: &Executor<R>,
        ids: Ids,
        segment_count: MIndex,
    ) -> Result<Self, Error>
    where
        Ids: MIter<R, Item = MIndex>;

    pub fn offsets(&self) -> DeviceSlice<MIndex>;
    pub fn lengths(&self, exec: &Executor<R>)
        -> Result<MVec<R, MIndex>, Error>;
    pub fn segment_ids(&self, exec: &Executor<R>)
        -> Result<MVec<R, MIndex>, Error>;

    pub fn segment_count(&self) -> MIndex;
    pub fn value_count(&self) -> MIndex;

    pub fn segments<Values>(
        &self,
        values: Values,
    ) -> Result<SegmentIterator<Values, DeviceSlice<MIndex>>, Error>
    where
        Values: MIter<R>;
}
```

`seg::expand`という名前は追加しない。`ExpandOp`や`FlatMap`の可変個出力と意味が
衝突し、生成される値がsegment IDであることも表さないため、
`Segmentation::segment_ids()`を使う。

## canonical representation

正本はoffsetsとする。

- 長さは`segment_count + 1`であり、通常はflat IDsより小さい。
- `offsets[i]..offsets[i + 1]`からsegment境界を直接読める。
- repeated offsetによりempty segmentを保存できる。
- CSR、ragged array、adjacency listで既に一般的な表現である。

構築時にoffsetsをprivate storageへmaterializeする。元の`DeviceVec`はclone後に
`slice_mut`できるため、そのhandleをそのまま保持すると検証済み不変条件が外部から
破壊される。`offsets()`が返すのはread-onlyな`DeviceSlice`だけとする。

初期版ではderived representationを自動cacheしない。IDsはO(N) storageを必要とし、
一度しか使わない処理も多い。必要なら利用者が`segment_ids()`の結果を保持する。
実アプリのprofileなしに、hidden cacheと失効規則を追加しない。

## 不変条件

offsets:

- 1要素以上。
- `offsets[0] == 0`。
- 非減少。
- 最後の値が`value_count`。

segment IDs:

- 0-basedであり、すべて`0 <= id < segment_count`。
- 非減少。leading empty segmentがあれば先頭IDは0とは限らない。
- 欠番はempty segmentを表す。

IDsだけではtrailing empty segmentもall-empty segmentationも復元できない。
そのため`from_segment_ids`は`segment_count`を必須とする。

```text
lengths       [1, 0, 2, 0]
IDs           [0, 2, 2] + segment_count 4
offsets       [0, 1, 1, 3, 3]
```

`segment_count == MIndex::MAX`はoffset数が`MIndex::MAX + 1`となるため拒否する。
lengthsのprefix sumが`MIndex`を超える場合も、O(segment_count)のscan／offset
metadataを作った時点で検出し、value countに依存するallocationを行う前に
`Error::LengthTooLarge`を返す。

## 変換

### lengthsからoffsets

1. inclusive scanでcumulative endsを作る。
2. u32 prefixが前要素より小さくなる箇所をoverflowとして検出する。
3. `[0, cumulative ends...]`をprivate offsetsへ書く。
4. 検証statusと最終value countを一度のhost observationで取得する。

任意長のexact allocationを返す通常APIであり、この境界の一度の観測は許容する。

### IDsからoffsets

1. IDsを正確な論理長だけprivate storageへmaterializeする。
2. 非減少と範囲をGPUで検証し、一度観測する。
3. sorted IDsに対するbatched lower boundで`0..=segment_count`の開始位置を求める。

各entryが同じcounterを更新するatomic histogramは、巨大な一segmentで競合するため
初期実装に採用しない。

### offsetsからIDs

1. 長さ`value_count`のzero-filled headsを作る。
2. 各nonempty segmentの先頭へ0-based segment IDを書く。
3. inclusive max scanで全entryへIDを伝播する。

これはO(segment_count + value_count)であり、長い一segmentを一threadで埋めない。

## context broadcast

segmentごとのcontextを各flat entryへ渡す処理は既存primitiveで表せる。

```rust,ignore
let ids = segmentation.segment_ids(&exec)?;
let entry_contexts = lazy::permute(contexts, ids.slice(..));
let rows = zip2(values, entry_contexts);
```

`contexts`はsegmentごとに1 row、すなわち`segment_count` rowを持たなければならない。
`permute`はunchecked indexed viewなので、この長さの検証は利用者側の前提とする。

`contexts`はsingle-columnでもmulti-columnでもよい。IDsは必要な間だけ保持し、
複数stageで再利用できる。

一様なcontextならIDsも不要である。

```rust,ignore
let entry_contexts =
    lazy::constant(context).take(segmentation.value_count());
let rows = zip2(values, entry_contexts);
```

segment全体とsegmentごとのcontextを同時に読む場合も専用adapterは要らない。
contextをentryへbroadcastしてからpartitionを適用する。

```rust,ignore
let ids = segmentation.segment_ids(&exec)?;
let decorated = zip2(
    values,
    lazy::permute(contexts, ids.slice(..)),
);
let segments = segmentation.segments(decorated)?;
// item type: Segment<(Value, Context)>
```

empty segmentにはbroadcast先のentryがない。emptyからcontext依存の値を生成する場合は、
`zip2(segmentation.lengths(&exec)?, contexts)`に対するsegment-level `map`を別に実行し、
nonempty側の結果と`zip2`／`map`で合成する。これはhost loopへ退避せず、すべてのpassが
GPU並列である。

empty segmentから可変個の値を生成する場合も一般の`FlatMap`で表せる。lengthsと
contextsをzipしたS行に、`counting(0).take(S + 1)`をoffsetsとして与えれば各行が
singleton segmentになる。これへ`ForEachSegment(FlatMap(op))`を適用すると、元の
segment境界を保った可変長結果になる。

cyclic predecessorやlocal indexが必要なら、`Segment`、offsets、`counting`、
`segment_ids`、`permute`を組み合わせる。

## Power Diagramへの適用

Power cellのvertices／edgesとcell contextは同じSegmentationで関連付けられる。

- offsetsを正本としてpolygon segmentを既存`ForEachSegment`へ渡す。
- per-edge clippingを試す場合はIDsと`permute`でcell contextをbroadcastする。
- cyclic predecessorはentryの`counting`、IDsから`permute`したsegment start／endで
  predecessor indexを`map`し、そのindexでverticesを`permute`する。これにより
  previous、current、cell contextをzipした全edge並列処理になり、adjacent専用
  flat-mapは不要である。
- clipping後のexact offsetsから次roundのSegmentationを構築する。公開primitiveを
  合成する場合は`from_offsets`で検証する。

ただし、Segmentation自体はkernel launch数やround readbackを自動的に減らさない。
whole-segment方式とper-entry方式のどちらが速いかはsegment長分布に依存する。
どちらも同じ少数primitiveで記述し、Power Diagram sourceを使った同一入力の
benchmarkでアプリ側の構成を選ぶ。差が大きくてもcontext専用APIは増やさない。

## del2dへの適用

del2dでは、candidate、winner、affected edgeを「どのsite／triangle／frontier itemに
由来するか」というowner relationへ変換する用途に使える。

一方、legalization loop、resource claim、topology update、stale task rejectionは
domain固有である。Segmentationはこれらを反復するrunnerではなく、flat work itemsと
owner groupの対応を表す部品に留める。

del2d側ではまず次を行う。

- 全edge再検査をaffected-edge frontierへ変更する。
- deterministic claimとstable IDを導入する。
- frontierとscratchをround間で再利用する。
- 一roundの終了時にpacked statusを一度だけ観測する。

これらはdel2d側で既存primitiveを合成して実装する。測定結果だけを理由に
Massivelyへdomain固有の公開algorithmを追加しない。既存primitiveでは意味を
表現できない一般的な不足が見つかった場合だけ、新しいprimitiveを検討する。

## context専用adapterを追加しない理由

per-entry context、一様なcontext、whole-segment context、empty segmentの
context依存出力は、上記の既存primitiveだけで表現できる。専用adapterを追加すると、
`zip`／`permute`／`SegmentIterator`と重複する意味、固有の戻り型、専用compiler
loweringが増える。

従って、実アプリで専用kernelの方が速い場合でも新しい公開抽象は追加しない。
最適化が必要なら、既存の合成を内部fusionするかアプリケーション内部の実装に置く。
利用者が理解すべきMassivelyの意味論は少数のprimitiveのまま保つ。

## `Iteration`を公開しない理由

固定回数のiterationは通常のRust `for`と同じcommand列を作るだけである。
条件付きiterationも、runnerがbuffer alias、容量上限、workspace lifetime、
kernel fusionを知らなければ、launch、allocation、command constructionを減らさない。

反復制御は通常のRust `for`／`while`または利用アプリケーションに置き、各roundを
Massively primitiveで並列実行する。workspace再利用はアプリケーションが所有できる。
backend command graphやfusionを導入する場合もexecutor内部のexecution policyとし、
新しい公開iteration意味論にはしない。

## 同期方針

最適化目標を「readback数を常に0にする」とはしない。

```text
通常の可変長algorithm:
    GPU stages -> exact shapeに必要な値を一度観測 -> result

primitive composition:
    algorithm -> 必要ならhost-exact境界 -> 次のalgorithm
```

API境界でhostがshapeやscalarを必要とするならblockしてよい。合成により
GPU -> CPU -> GPU往復が増える場合も、それだけを理由に公開device scalarやrunnerを
追加しない。まず少数primitiveで意味を完全に表し、最適化する場合は公開合成を変えずに
primitive実装、executor、またはcompiler loweringの内部で行う。

## 性能検証

repository内の`segmentation` benchmarkは三つのgroupを持つ。

- `segmentation_conversions`: 三表現からの構築、`lengths()`、`segment_ids()`、
  `from_lengths`からIDsまでの合成を、uniform／empty-heavy／skewedな分布で測る。
- `segmentation_round_transition`: segmented `FlatMap`単体に対し、
  `Segmentation`再構築、IDs生成、context consumerを順に加えた増分を測る。
- `segmentation_repeated_rounds`: 256 segment、0／1K／2K／4K entriesで2回または8回の
  `FlatMap`を連鎖し、raw offsets、最終roundだけの`Segmentation`構築、全roundでの
  構築を比較する。入力は0-output／2-outputを交互に生成し、各roundのentry数を一定に
  保つため、work量の増減と同期境界の費用を混同しない。

単一roundではIDs生成の増分が再materialize／再検証より大きい。しかしこれは
`from_offsets`のhost observationがbenchmark末尾の同期を置き換えるため、round間の
追加同期を評価できない。連続roundではこの費用が累積する。

repositoryのWGPU計測では、8 roundで最終roundだけ構築する経路と全roundで構築する
経路に次の差が出た。

```text
256 segments, all empty:  約3.00 ms -> 約4.35 ms
256 segments, length 4:   約7.83 ms -> 約9.82 ms
256 segments, length 16:  約7.74 ms -> 約9.27 ms
```

この差は再構築が無料ではないことを示すが、新しい公開経路を追加する根拠にはしない。
caller-provided expected lengthや`SegmentIterator::new`で作った任意offsetsを
trusted扱いすると、Segmentationの不変条件を利用者へ移してしまう。

従ってv0.87では公開unchecked constructor、trusted segmented-result、
domain compound algorithmのいずれも追加しない。将来、外部aliasを作らず既知の
metadataを引き継げる内部最適化を既存APIのまま実装できるなら採用し、できなければ
再検証費用を受け入れる。

外部アプリごとに次を記録する。

1. round数とactive item数。
2. kernel launch数。
3. temporary allocation数とbytes。
4. queue submission数。
5. CPU encode／submit時間。
6. GPU timestamp。
7. host observation回数と待ち時間。
8. wall-clock時間。

比較順は、host-exact baseline、Segmentationによるowner/context合成、
アプリケーションでのworkspace reuse、既存合成を保つ内部fusion、
backend specializationとする。前段で残った支配要因に対応する場合だけ後段へ進む。

repository内benchmarkはprimitiveの性能回帰と合成の増分を分離するためのものであり、
Power Diagram／del2dのapplication benchmarkを代替しない。

## 完了条件

- 公開docsに`MVal`、`MExtent`、`MSequence`、`Iteration`、
  context専用segmented adapterが存在しない。
- `Segmentation`がlengths／0-based IDs／offsetsを相互変換できる。
- empty、trailing empty、all-empty、zero-segmentを保存する。
- 不正offsets、IDs、sum overflow、offset-count overflowを拒否する。
- 同じ分割をsingle-column／multi-column valuesへ適用できる。
- IDsと`lazy::permute`でmulti-column contextをbroadcastできる。
- foreign executorを派生変換時に拒否する。
- Power Diagramとdel2dの外部benchmarkを変更前後で同一入力・環境により計測する。

最後の項目は外部application sourceが利用可能になった時に行う。Massively repository
内のsynthetic testだけでアプリケーション性能の完了を主張しない。
