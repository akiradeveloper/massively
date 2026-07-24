# proposal-del2d: Massively v0.87によるDelaunay実装から得た提案

- 状態: 実装・実測に基づく提案
- 対象: Massively v0.87以降
- 対象revision: `a960fd5410d39fedf4603020437f66e618236f3c`
- 実装: `del2d`
- 記録日: 2026-07-24

## 位置付け

本書は、Massively v0.87の公開primitiveだけで二次元Delaunay三角形分割を実装し、
CPU実装との比較とGPU profileを行った後のapplication evidenceである。

v0.87の正本である
[`proposal-compound-operations.md`](https://github.com/akiradeveloper/massively/blob/v0.87/proposal/proposal-compound-operations.md)
は、通常APIをhost-exactに保ち、公開`MVal`／`MExtent`／汎用`Iteration`を追加せず、
`Segmentation`と少数の直交primitiveを合成する方針を採用した。

本書は、過去の[`proposal-worklist.md`](proposal-worklist.md)をそのまま復活させる
提案ではない。今回の結果から、次の二点を分けて評価する。

1. 少数primitiveで複雑なalgorithmの意味を表現できるか。
2. その合成を現在の実行境界のまま十分高速に実行できるか。

前者には肯定的な証拠が得られた。後者は未達であり、その間を埋める限定された
execution/lowering abstractionが次の検討対象である。

## 最終判断

今回の経験から得た最も重要な判断は次である。

> Massivelyに直ちに必要なのは、DelaunayやWorklistという新しい高水準algorithm
> primitiveではない。有限個の既存primitiveを、device-residentな選択数、
> segmentation metadata、scratch、submissionとともに一つの有限DAGとしてlowering
> できる実行境界である。

評価を表にすると次になる。

| 仮説 | 判断 | 根拠 |
|---|---|---|
| primitive合成でDelaunayを表現できる | 支持 | Massively本体を変更せず、挿入、競合解決、CSR再配置、edge flipまで実装できた |
| segmentation algebra / CSRが複雑なowner relationに有効 | 支持 | raw offsetsと`SegmentIterator`によりroundごとのglobal sortと`lower_bound`を除去できた |
| 各round内を並列化できる | 支持 | geometry、claim、split、scan、scatter、flipはすべてGPU並列passになった |
| 有限passへ分解すれば現在のAPIのまま速い | 不支持 | 16K点でGPU kernel約15.4 msに対しwall約97 ms、1,219 dispatch、48回の長さ観測 |
| 現実装がGPUへ到達していない | 否定 | WGPU adapterがRadeon 680Mを選択し、amdgpu busy counterも最大98%を記録 |
| 現時点でCPUより速い | 否定 | 524K点でもsingle-thread CPUより3.67倍遅い |

したがって、v0.87の「表現の代数」は維持する。一方、primitive間の制御値と
materializationをすべて公開host-exact境界へ戻す実行方式は、有限scopeの内部に限って
再検討する価値がある。

## 実験条件

### Hardware

- CPU: AMD Ryzen 7 7735U、8 core / 16 thread
- GPU: WGPU adapterがRadeon 680Mを選択し、amdgpuを使用
- amdgpu busy counter: 計測中最大98%

### Benchmark契約

- 入力点生成は計測外。
- GPU uploadは計測外。
- shader pipelineは事前にwarm upする。
- GPU結果はdevice-residentのままにする。
- 最後のGPU synchronizationは計測内。
- CPUはsingle-threaded Delaunatorを使用する。
- CPU側は`f32 -> f64`入力変換とhost出力構築を計測内に含む。
- Criterion sample sizeは10、sampling modeはFlat、warm-upは3秒、measurementは5秒。
- 乱数seedは`0x6d65_7368_2d32_6401 ^ N`。
- `.cargo/config.toml`の`CUBECL_WGPU_MAX_TASKS=128`を使用する。
- CubeCL revisionは`0a62060a2c7a66c94f717d7cac0be0dc259bb607`。
- `cargo bench`のrelease相当bench profileを使用する。

再現コマンドの例は次である。

```console
DEL2D_BENCH_SIZES=256,1024,4096,16384,65536,131072,262144,524288 \
  cargo bench -p del2d --bench random_points
```

これはGPUに有利な条件である。したがって現在の差を、入力転送だけの問題として
説明することはできない。

### CPU/GPU比較

| sites | GPU | CPU | GPU / CPU |
|---:|---:|---:|---:|
| 256 | 48.805 ms | 0.031572 ms | 1,546x |
| 1,024 | 59.770 ms | 0.16181 ms | 369x |
| 4,096 | 80.936 ms | 0.82921 ms | 97.6x |
| 16,384 | 96.963 ms | 4.0241 ms | 24.1x |
| 65,536 | 157.40 ms | 22.455 ms | 7.01x |
| 131,072 | 235.55 ms | 50.447 ms | 4.67x |
| 262,144 | 490.79 ms | 119.13 ms | 4.12x |
| 524,288 | 1.0505 s | 286.44 ms | 3.67x |

入力増加に対するGPUのscaleはCPUとの差を大きく縮めている。しかし、測定した範囲に
crossoverはない。CPUとGPUではalgorithmと仕事量が異なるため、これは純粋なhardware
throughput比較ではない。それでも実用上の比較として、GPU版が未達であることは明確で
ある。

歴史的なv0.86時点の同hardware計測は
[`proposal-worklist.md`](proposal-worklist.md)に16K点約193.8 msと記録されている。
現在の96.963 msはそこから約2倍の改善である。ただしMassively revisionとdel2d実装の
両方が変わっており、元のraw traceもrepositoryにないため、この差を一つの最適化だけの
効果や厳密な回帰benchmarkとはみなさない。

### 16K profile

16K点の代表的なprofileは次の通りだった。

このprofileと上表のCriterion medianは別runである。profile counterとCriterion wall
timeを同一sampleの厳密な時間内訳として加算してはならない。

| 指標 | 値 |
|---|---:|
| wall time | 約97 ms |
| GPU kernel aggregate | 約15.4 ms |
| dispatch | 1,219 |
| insertion round | 約19 |
| legalization pass | 約27 |
| exact-length observation | 48 |
| `MaterializeA13` dispatch count | 564 |
| `IndexedCopyA13` dispatch count | 298 |
| `PaddedScanA13` dispatch count | 121 |
| `SegmentedScanA13` dispatch count | 35 |

GPU kernel aggregateとCriterion wallの大きな差は、robust geometry以外の
launch、materialization、allocation/binding、encode/submit、host observationを合わせた
費用が大きいことを強く示唆する。各要因の個別内訳はまだ分離できておらず、後述の
標準traceで測る必要がある。

48回という値は`CopyLastKernel`を伴うselection由来のexact-length観測数であり、
全host observation数ではない。前処理のbounding-box/finite reductionとcollinearity
predicateにも少なくとも2回のscalar観測がある。総数は現profileでは未集計であり、
標準traceで確定する。

GPU選択はWGPU adapter identityで確認し、amdgpu busy counterで実行中のactivityを
補強した。busy counter単独はadapter identityの証明ではなく、98%が有効な幾何計算
だったことも意味しない。

## v0.87のprimitiveだけで実装できたもの

次の構成はMassively本体を変更せず実装できた。

1. `map`、`zip`、`permute`、`reduce`による入力正規化と幾何metadata生成。
2. lexicographic sortと`unique_by_key`による重複点除去。
3. bit-reversal順序の直接生成。
4. triangle-major CSRによる未挿入点の保持。
5. `SegmentIterator`によるsegment先頭winner候補の読み出し。
6. 固定次数3の`map + permute`によるdeterministic resource claim。
7. one-hot bucket分類、tuple segmented scan、segment tail、prefix scan、scatterによる
   CSR再構築。
8. stable triangle/edge IDと既知destination scatterによるin-place topology update。
9. epoch-tagged tableによる全table clearの除去。
10. affected-edge frontierによる疎なlegalization。
11. dense、full-scan compact、frontierの三schedule。
12. robust orientation/incircle predicateと決定的なcocircular tie-break。

これは表現力について強い肯定的証拠である。del2dは多数のDelaunay固有CubeCL
`UnaryOp`を定義しており、それらはMassivelyの生成kernelへ融合される。一方、
Massively本体にDelaunay専用primitiveや手書きlaunchを追加せず、複雑なmutable
topology algorithmを有限primitiveへ分解できた。

## 効いた設計

### Stable IDと固定容量topology

triangleとedgeをroundごとに詰め直さず、stable slotとして保持した。入力数`N`から
安全な容量上限をhostで計算できるため、GPU allocatorは不要だった。

この設計により、incidenceのglobal sort/rebuildを既知destinationへのscatterへ変更
できた。これはDelaunayに限らず、bounded mutable graph rewriteに有効である。

### Domainの固定次数を使う

triangleが持つedgeは常に3本である。汎用sort/reduceによるclaimではなく、各triangleが
三候補から直接ownerを選ぶことでglobal claim sortを避けた。

一般的なprimitiveだけを使うことと、domain構造を捨てて常に最も汎用なalgorithmを
使うことは同義ではない。次数上限、fan-out上限、stable IDは積極的に利用すべきである。

### Epoch tag

winner、illegal edge、changed triangleのtableを毎round zero-fillせず、
`(epoch, value)`として再利用した。全体clearを局所scatterへ変える一般的な方法である。

epoch wraparound時だけ明示的clearまたはgeneration再初期化が必要になる。

### Triangle-major CSR

未挿入点をtriangle-major CSRでround間に保持したことで、以前のroundごとのcomparison
sortと`lower_bound`を削除できた。

profile上、`MergePermutation`は91回から6回へ減った。raw offsets、
`SegmentIterator`、CSRを使うsegmentation algebraが「flat itemとowner groupの関係」を
表すという判断は、この用途で正しかった。公開`Segmentation` concrete type自体は
hot loopで構築していない。

### Lazy expression内のfusion

keep flagと三bucket flag、offsetの先頭zero追加、winner由来の追加lengthなどをtuple
`map`へまとめた。初期CSR版は6〜32%回帰したが、このfusionにより16Kでは旧実装と同等
か僅かに改善するところまで戻せた。

### Density別schedule

- 小規模: dense処理を4 round enqueueしてから観測。
- 中規模: 全edgeを検査してwinnerだけcompact。
- 大規模: affected-edge frontier。

262K点ではfull-scan compactの約641.65 msから、正当性保証を含むfrontierの
約490.79 msへ約23.5%改善した。

一つのscheduleを全密度へ適用するより、dense/sparseの境界を明示する方が有効だった。

## 効かなかった設計と、その理由

### Sortをsegmented scanへ置き換えるだけでは速くならない

開発中に取得したCSR化前後の16K profileでは、dispatchは約1,214回から約1,219回へ
微増した。raw traceはrepositoryに保存されていないため、今後の標準traceで再現する。
global merge passを削除しても、segmented scan、segment tail materialization、
length scan、scatterが追加されたためである。

これはsegmentation algebraの失敗ではない。work complexityを改善しても、launch
complexityが改善されなければwall timeは下がらないという結果である。

### 全capacity dispatchは疎なfrontierに弱い

device lengthをhostへ戻さず、常にcapacity全体へdispatchする経路も検討した。
readbackは減るが、frontierが疎になるほどinactive laneが支配する。

必要なのはdevice countだけではなく、そのcountから実行範囲を作るindirect dispatch
または同等のsparse execution domainである。

### Hot loopで毎回`Segmentation`を再構築できない

安全な`Segmentation::from_offsets`はoffsetsを検証し、host-exactな`value_count`を
確定する。通常APIとしては正しいが、roundごとに構築すると同期が累積する。

現在のhot pathはraw offsetsと`SegmentIterator::new`を使う。offsetsが

- 先頭0、
- 非減少、
- 末尾がpending count、
- offset数がtriangle count + 1、

であることをdel2d側の生成規則で保証している。

これは公開unchecked constructorを追加すべき証拠ではない。Massively自身が生成した
metadataのprovenanceを内部で引き継ぐ余地があることを示す。

### Boundary-only frontierは正しくない

flip winnerが作る四境界辺だけを次frontierにすると、claimに負けた
「未変更だがillegalなedge」が消える場合がある。manifold検査だけではこの誤りを
検出できなかった。

現在はsparse waveが停止した時点でfull legality passを行い、

- winnerがなければ完了証明、
- winnerがあれば次のsparse waveをseed、

としている。

frontier最適化には、未処理itemを保存する規則か、quiescence時の完全なcertificateが
必要である。性能のためにこの検査を無条件に削除してはならない。

### 汎用`Iteration`は、それだけでは速くしない

Rustの`for`を別の型で包んでも、同じkernel列、allocation、materialization、
submissionを作るだけなら性能は変わらない。

必要なのは反復という名前ではなく、有限DAG全体を見てprivate extent、scratch、
fusion、observationをまとめるlowering能力である。

## 学びを分解するための五つの効率

今後は「GPU algorithmが速いか」を少なくとも次の五つへ分解する。

1. **Semantic expressiveness**
   - algorithmをprimitiveの有限合成として正しく表現できるか。
2. **Work efficiency**
   - global sort、全edge scan、inactive laneなど不要な仕事が少ないか。
3. **Launch efficiency**
   - 同じ仕事を何dispatch、何submissionで実行するか。
4. **Observation efficiency**
   - GPUが生成したscalar/lengthを何回hostが待つか。
5. **Materialization efficiency**
   - temporary allocation、copy、gather、scratch再構築がどれだけあるか。

segmentation algebraは主に1と2を改善する。今回残った支配項は3、4、5である。

## Massivelyに提案する抽象

### P0: 標準trace contract

最初に追加すべきものは、意味論を変えない計測基盤である。

一つのtraceから少なくとも次を取得できるようにする。

- primitive名とlowering名。
- kernel dispatch数。
- queue submission数。
- host observation数。
- observationごとの待ち時間。
- temporary allocation数とbytes。
- scratch pool hit/miss。
- CPU encode/submit時間。
- GPU timestamp。
- wall-clock時間。

最適化は「wall timeが下がった」だけでなく、狙ったcounterが因果的に下がったことを
示す。今回のようにsort passを消して別のscan passを増やした場合も、すぐ判別できる。

### P1: Internal `ExecutionDomain`

Massively v0.87内部には既に次がある。

- `MVal<R, T>`。
- `LogicalExtent::{Fixed, Device}`。
- device countを保持する`SelectionControl`。
- device extentを伝播できる内部map/selection lowering。

不足しているのは、logical extentをkernelの実行範囲へ変換する共通loweringである。

概念的には次の内部viewを導入する。

```rust,ignore
enum ExecutionDomain<R> {
    Fixed {
        len: u32,
    },
    Device {
        count: PrivateDeviceScalar<R, u32>,
        upper_bound: u32,
    },
}
```

`ExecutionDomain`を`LogicalExtent`と並ぶ第二の保存carrierにはしない。既存
`LogicalExtent`からlaunch時に一時的に得るlowering view/policyとし、extent identityと
capacityの正本は増やさない。

`ExecutionDomain::Device`は次を行う。

1. device countから`[x, y, z]` indirect argumentsを生成する。
2. 同じcount storageとblock sizeについてarguments allocation/bindingを再利用する。
3. count 0を安全なno-opとして扱う。
4. 最終workgroupではcountによるOOB guardも残す。
5. indirect dispatch非対応backendではupper-bound dispatchへfallbackする。

0-groupのdata dispatchだけでは、次outputのlength/statusを0へ更新できない。前roundの
値が残ると空frontierが復活するため、各stepはdata dispatchの有無にかかわらず
next extentとstep statusを初期化するalways-run control passを持つか、producerが
0を必ず書く。empty stateが次stepでもemptyである「absorbing state」を実行契約にする。
countの値はroundごとに変わるため、indirect argumentsの内容はproducer完了後に毎回
再生成する。再利用できるのはbuffer、binding、lowering planであり、古いargumentsの
中身ではない。

これは公開`MVal`を要求しない。既存のprivate logical extentを、実際のdispatch policyへ
接続するexecutor内部の抽象である。

indirect dispatchはinactive laneを減らすが、kernel launch数そのものは減らさない。
したがって後述のfinite compositionとfusionが同時に必要である。

scan/reductionは一つのindirect dispatchへ置き換えるだけではない。hostはcapacityから
最大multi-stage DAGを構築し、各stageのdevice group count、zero-stageのcontrol初期化、
scratchの有効範囲を伝播する必要がある。最初はselectionと単純なindexed copyで
ExecutionDomainを検証し、その後にhierarchical collectiveへ広げる。

WGPU backendについては、producer kernelが書いた0／非0 countから次kernelを起動する
実機smoke testをRadeon 680Mで先に行う。backend実装の存在だけでportableな動作を
仮定しない。

### P3: 条件付き有限`CompositionScope`

今回もっとも検討価値がある新しい境界は、公開device scalarや無制限loopより限定した、
有限でnon-escapingなcomposition scopeである。ただし、これを公開すればscope内に
device scalar、dynamic sequence、別の`map`／`select` interfaceを持つことになる。
したがって、これはv0.87方針と自然に両立する追加ではなく、採用済み判断を限定的に
再検討する設計変更である。

概念APIは次のようになる。

```rust,ignore
let status = exec.compose_epoch(8, |scope, state| {
    // Rust上では有限DAGを構築する。device依存whileは許可しない。
    let candidates = scope.map(state.frontier(), BuildCandidates);
    let winners = scope.select(candidates, IsWinner);

    scope.scatter_selected(
        &winners,
        BuildTopologyUpdates,
        state.topology_mut(),
    );

    let next = scope.expand_fixed::<4>(&winners, EmitAffectedEdges);
    state.set_frontier(next);

    scope.pack_status(PackEpochStatus)
})?;
```

これは確定APIではなく、必要な契約を示す。

- scopeへ入るstorageのphysical capacityはhost-known。
- scope内の選択数、長さ、真偽値、statusはopaqueなdevice value。
- scope内の動的sequenceはscope外へescapeできない。
- DAGは有限で、最大round数またはepoch fuelがhost-known。
- scope終了時の外部`MVec`は従来どおりhost-exact。
- 次epochへ渡すsequenceは、terminalのpacked observationに含めた最終lengthで一度
  resolveするか、recorded planが所有するprivate stateとしてscope内に留める。
- host observationはterminalのpacked statusまたはexact result確定時に限定する。
- active countが0になった後の残りroundはindirect count 0またはguard付きno-op。
- next extent/statusはactive countが0でも必ず0へ更新し、emptyをabsorbing stateにする。
- capacityは可能な限りhostで構造的に証明する。device依存validationが必要な更新は
  `validate -> commit`の二相にし、error判定前にstateを部分更新しない。
- すべてのdevice indexはload/store前にguardする。sticky `DeviceStatus`は診断と
  terminal報告に使い、memory safetyやtransactionalityの代用にしない。
- mutable state operationはeffect tokenを入出力し、read/write・write/write dependencyを
  DAGへ明示する。optimizerはaliasを証明できないstate accessをreorderしない。
- 「childをmaterializeしてからparent slotをoverwrite」のような順序は、SSA versioned
  stateまたは明示barrierで保持する。
- stable order、tie-break、foreign executor、overflow契約を既存APIと共有する。

`pack_status`はdevice上にterminal statusを生成するだけで、closureの途中ではhost readを
行わない。実際の観測は`compose_epoch`のterminalに一度だけ行う。

scopeのtarget machineにはcountを保持するだけでなく、privateな
`add`／`mul_const`／`min`／`ceil_div`と、device-produced baseを使う
`base + counting`、prefix gather/scatterが必要である。Delaunayではwinner countから
2倍／4倍のchild長とappend baseを作るため、現在のようにsliceのstart/limitがhost値の
ままではindirect dispatchだけを追加しても同期は消えない。これらはraw scalarとして
公開せず、scope内のshape/index expressionとして扱う。

この制約は過去の汎用`Iteration`案よりescape/lifetime上の危険を小さくするが、
scopedな動的制御APIである事実と必要なoperation matrixの大きさは変わらない。差は
次の限定にある。

- unknown回数のdevice-side `while`を公開しない。
- GPU allocatorを要求しない。
- 新しい永続storage型をscope外へ追加しない。
- 任意closureをpersistent kernelへ入れない。
- backendはbounded unroll、batched commands、indirect dispatchへloweringできる。

まずMassively内部またはunstable feature-gated prototypeとして作る。16K profileの
exact-length round境界はinsertion約19回とlegalization約27回の計46回であり、
8-round epochなら`ceil(19 / 8) + ceil(27 / 8) = 7`回まで減らせる。selection由来の
前後約2回を残すと、exact-length観測数の目安は48回から約9回である。さらに前処理の
scalar観測が少なくとも2回あるため、全host observationは少なくとも約11回になる。
前後処理も同じscopeへ統合できた場合だけさらに減る。

既存のstable APIはhost-exactのまま維持できるが、experimental scope内部は
host-exactではない。Power Diagram、graph frontier、CSR refinementなど第二の用途で
効果と安全性を再現し、v0.87方針を変更する合意が得られるまでstable public APIには
しない。non-escaping scopeはlifetime/escape契約では公開`MVal`／`BoundedVec`より
限定されるが、必要なmap、selection、scan、scatter、segmented operation、shape演算、
borrow/effect loweringのmatrixは小さくない。この実装規模も採否判断へ含める。

### P1: Caller-provided outputの一貫化

固定shapeまたはhost-known capacityを持つoperationには、事前確保outputへ書く経路を
一貫して提供する。

特に必要なのは次である。

- ordinary scan。
- scan by key。
- length-preserving segmented algorithm。
- per-segment summary。

`map`はlazy expressionを`vector::copy`で既存outputへ書けるが、collectiveの一部は
owned result allocationしか公開されていない。内部には既に`run_into`相当の経路が
あるため、公開surfaceを増やす場合も新しい意味論は必要ない。

gather/materializeは`lazy::permute + vector::copy`でpreallocated sinkへ既に表現できる
ため、専用`gather_into`を公開する提案には含めない。

目的は利用者に全scratchを管理させることではない。長寿命のapplication state/outputと
短寿命のexecutor scratchを分け、前者だけをcallerが再利用できるようにする。

### P2: Proven indexed/predicated sink lowering

read側には`permute(values, indices)`というindexed viewがある。一方、write側の
scatterは一つのindices列をtuple全体で共有する。

Delaunayのtopology updateでは、同じsource rowから複数の配列へ、それぞれ異なる
destination indexで書く。現在は複数scatterへ分かれやすい。

write側にもindexed sink表現があれば複数scatterを一つのlowering候補にできる。しかし
read `permute`との単純な対称ではない。duplicate destinationはwrite raceになり、
predicated leafは既存値の保持、sink間alias、read/write aliasも扱う必要がある。

```rust,ignore
// Public API案ではなく、sealedな内部IRの概念。
let sinks = (
    Sink::UniqueIndexed(edge_a, proven_unique_a),
    Sink::UniqueIndexed(edge_b, proven_unique_b),
    Sink::UniqueIndexed(triangle, proven_unique_triangle),
);
```

初期提案はpublic viewではなく、Massively自身が一意性を証明したindices、または
既存scatterのpreconditionを保持できるsealed provenanceを入力とする内部multi-sink
loweringに限定する。次の契約を解決する。

- destination範囲の検査方針。
- 同一sinkへの重複indexの扱い。
- zipされたsink間のalias制約。
- false predicateで既存destinationを保持する方法。
- inputとoutputがaliasする場合のsnapshot/order契約。
- stable/deterministic writeが必要な場合のprecondition。
- input read slotとoutput index slotを合わせたarity上限。

異なるdestinationを持つ複数scatterを実際に一dispatchへloweringできることを
microbenchmarkで確認し、二つ以上のpublic use caseと安全な検証方法が得られるまで
公開APIへ昇格しない。

### P1: Dependency-aware scratch/workspace

current runtimeのmemory poolがallocation自体を再利用しても、各primitiveはhandle、
logical metadata、small parameter buffer、binding、scan scratchを再構築する。

executor内部に、shape/layoutとqueue lifetimeを理解するscratch leaseを追加する。

- scan positions/prefixes。
- selection positions/indices。
- segmented heads/IDs。
- temporary SoA columns。
- indirect dispatch arguments。
- fixed-size status buffer。

単純な`HashMap<shape, buffer>`では不十分である。同一WGPU queue上の順序付きcommandは
後続利用との実行順を保証できるため、常にGPU完了fenceが必要なわけではない。必要なのは
DAG上のliveness/alias、encoderとsubmissionの所有権、CPU map、将来のcross-queue利用を
含むdependency-aware leaseである。

同じ有限DAGを繰り返す場合は、さらに`RecordedPlan`相当の内部表現で次を再利用する。

- pipeline。
- bind layout。
- static parameters。
- scratch liveness。
- ping-pong assignment。
- exact arity。

これはWGPU command bufferを無条件に再submitする意味ではない。pipeline、bind/static
metadata、scratch liveness planをcacheし、commandsはbackend契約に従って必要な時に
再encodeする。

公開`Iteration`ではなく、executorのexecution policyとして実装する。

### P1: Library-derived segmentation provenance

公開`Segmentation`の検証は維持する。公開unchecked constructorや
caller-provided「trusted」flagは追加しない。

一方、Massively自身のscan、segmented filter、stable routeが生成したoffsetsには、
次の不変条件をproducerが既に知っている場合がある。

- 先頭0。
- 非減少。
- terminal countが既知のprivate extentと同一。
- offset capacityがsegment count + 1。

finite scopeまたはcompound lowering内部では、このprovenanceをopaque tokenとして
引き継ぎ、次のsegmented operationで再materialize・再検証しない経路を持てる。

tokenだけをmutable offsetsへ付けてはならない。planはoffset storageを所有または凍結し、
外部mutable aliasを禁止する。更新可能なstorageを使う場合はgenerationを持ち、headsと
IDsのcacheも同じgenerationへ結び付けてstale metadataを拒否する。

同じpartitionを複数operationで使う場合は、heads、必要ならIDsを一度だけ生成して
`SegmentationPlan`内部で再利用する。derived representationを常にcacheするのではなく、
profileで複数回利用が確認された場合だけ、利用回数と分布に基づいてcacheする。

scope外へ出す時は通常の安全な`Segmentation`へresolveする。

del2dがraw offsetsを使う間は、debug/test buildで先頭、単調性、terminal count、
offset countを独立に検査し、application側のproof obligationを実行可能なtestにする。

### P1: 既存selection loweringの改善

現在のstable selectionは概ね次の形になる。

```text
flags
  -> inclusive scan
  -> selected indices materialization
  -> indexed copy
  -> last/count observation
```

一つのpayloadだけをcopyする場合、true rowをscan positionへ直接stable scatterすれば、
selected indices bufferと一つのindexed-copy stageを省ける可能性がある。

同じselectionを複数terminalが使う場合は、現在の`SelectionControl`を一度作って共有
した方がよい。tupleの複数columnを一つのterminalへ書く場合はsingle consumerとして
数える。

したがってselection loweringは次をcost modelで選ぶ。

- single consumer: direct stable scatter。
- multi consumer: indices/controlをmaterializeして共有。
- dense stencil: predicated capacity pass。
- sparse stencil: compact control + indirect dispatch。

公開`copy_where`のstable orderとhost-exact resultは変えない。

### P2: Barrier-aware fusionとCSE

現在のlazy `map`、`zip`、`permute`は一つのconsumer kernel内で有効にfusionされる。
一方、scan、selection materialization、gather結果などのterminalでfusionが切れる。

finite planが得られた場合、次を検討する。

- map/permuteを次のscan・scatter・materializeのprologueへ融合。
- scan最終stageの安全なpointwise epilogue。
- 同一kernel内の同じpermutation/index loadをload-CSE。
- kernelを跨ぐ場合は、再計算とmaterialize/reuseをcost modelで比較。
- 一つのselection controlを複数consumerで共有。
- 複数のnon-aliasing sinkを一つのdispatchへ統合。
- 実際のleaf数に合わせたexact-arity kernel。

global barrierを必要とするscan/reductionを無理に一kernelへ畳まない。
過剰fusionはregister pressure、shader variant、compile timeを増やすため、
profile-guided cost modelと反証benchmarkを必須にする。

### P2: Segmented stable rebucketの内部specialization

今回のCSR transitionは一般化すると次である。

```text
classify into fixed K buckets
  -> one-hot tuple
  -> segmented scan
  -> segment tail counts
  -> destination length scan
  -> stable scatter
```

これはCSR refinement、radix split、mesh refinementでも現れる。

しかし現在のprimitiveで意味は完全に表現できるため、直ちに
`seg::stable_route<K>`を新しい公開primitiveとして追加しない。まずfinite plan上で
このpatternを認識し、

- flag生成とscan inputの融合、
- tail countの同時生成、
- scratch再利用、
- fixed `K` specialization、

を内部loweringとして実装する。

第二のapplicationで同じ意味と契約が必要になり、内部pattern matchingでは不安定に
なる場合だけ、公開route abstractionを再検討する。

## 現時点で提案しないもの

### 公開`MVal`／`MExtent`／`BoundedVec`

v0.87内部には実装部品があり、技術的には可能である。しかし公開すると、

- dynamic slice base、
- capacityとlogical lengthの関係、
- device scalar演算、
- overflow/status観測、
- zipのextent identity、
- 対応algorithmの範囲、

まで一度に公開契約になる。

まずnon-escaping `CompositionScope`で性能効果と安全性を検証する。scopeでは不足し、
複数applicationがdynamic sequence自体を交換する必要を示した場合だけ再検討する。

### 汎用`Worklist`／`Iteration`／`fixed_point`

名前だけのrunnerはlaunch、allocation、submissionを減らさない。無制限device loopは
WebGPUのglobal synchronization、watchdog、portabilityとも相性が悪い。

有限scopeをbackendがbounded unroll/epochへloweringするところまでを今回の提案範囲と
する。

### Delaunay固有primitive

次はdel2d側に置く。

- edge frontier。
- triangle/edge claim。
- edge flip。
- topology generation。
- stale task rejection。
- Delaunay certificate。

`claim`がmatching、mesh、coloringなど第二の用途でも同じ契約を持つことが実証された
場合は、`atomic scatter arg-min + all resources claimed`という小さいprimitiveへ
分解して再検討する。

### Public unchecked `Segmentation`

hot loopの同期を避けるためにvalidation責任を一般利用者へ移してはならない。
library-derived provenanceは内部最適化として扱う。

### Persistent kernelとGPU allocator

今回の容量は入力`N`から上限を計算できる。persistent kernelはgrid全体のbarrierと
watchdog問題を持つ。どちらも有限compositionより先に導入しない。

### 常時capacity over-dispatch

正しさのfallbackとしては有用だが、疎なfrontierの標準strategyにはしない。

## del2d側で続ける実装上の工夫

Massively側の変更を待たず、次を優先する。

### P0: Losing illegal edgeをfrontierへ持ち越す

次frontierを概念的に

```text
changed quadrilateralのboundary edges
  union
illegalだったがclaimに負けたedges
```

とする。

stable edge IDとepoch dedupを使い、loserを落とさずsparse waveを継続できれば、
full certificateの頻度を下げられる可能性がある。

ただし、全内部edgeのlocal Delaunay検査で同値性を証明するまで現在のfull certificateを
残す。

### P0: 一つのgeometry snapshotでmatchingを深くする

現在はtriangleごとの第一ownerを一段だけ選ぶ。同じlegality snapshot上で、
既にclaimされたtriangleを除外しながら2〜数段のdeterministic matchingを作れば、
高価なgeometry再評価とhost roundを減らせる可能性がある。

round数だけでなく、追加claim passと減ったgeometry passの合計を測る。

### P1: Winnerが触るsegmentだけを再分類する

現在はwinnerのないsegmentを含む全pending pointについてorientationとsegmented scanを
実行する。

- non-empty segmentのfrontierを持つ。
- winnerが触る1〜2 segmentだけbucket分類する。
- untouched segmentはblock/segment単位で引き継ぐ。

というsparse segmented rewriteを検討する。

### P1: Persistent application workspace

triangle/edge容量が固定されたlegalization phaseでは、geometry、owner、children、
active flagsなどをA/B bufferとして事前確保し、公開済みの`vector::copy`、scatter系APIで
可能な範囲から再利用する。

allocation数とbytesを計測し、runtime poolが既に吸収している部分と区別する。

### P1: 実frontier密度によるschedule選択

現在のthresholdは主に入力点数で決めている。既存のhost observationで得たfrontier/
winner countを追加同期なしで利用し、

- frontier density、
- winner density、
- edge count、
- 直近roundの縮小率、

からdense/full/sparseを切り替える。

Radeon 680Mの4,096／131,072 thresholdを他hardwareへ固定しない。

### P1: f32 filter + robust fallback

launch/observation問題を改善した後は、Radeon 680Mで高価なf64 predicateを減らす。

1. 元のf32座標に対してf32 determinantと保守的な誤差境界を計算。
2. 符号が確実なcaseを即決。
3. ambiguous caseだけ既存のf64 robust predicateへ送る。

filterの誤差境界は数学的に証明し、ambiguous compactionが新しい同期を作らないことを
確認する。証明にはWGSL/backendのFMA contraction、subnormal/FTZ、NaN/Infの契約も
含める。現時点ではkernel時間よりwall overheadが大きいため第一優先ではない。

### P2: 複数triangulationの外側batch

一つのtriangulationは初期roundのactive itemが少ない。独立した複数point cloudを
外側のSegmentationでflattenし、一つのGPU batchとして実行すれば、初期occupancyと
固定launch費を改善できる可能性がある。

単一巨大meshだけでなく、GPUが得意なbatch workloadでも仮説を評価する。

### P2: Spatial tile

Morton順tileごとの局所triangulationと境界mergeは、global mutable topologyのround数を
減らす可能性がある。ただし正当性、degenerate case、merge complexityが大きいため、
上記の実行系改善後に検討する。

## 検証条件

### Correctness

少なくとも次を検査する。

- 全triangleの正向き。
- triangle重複なし。
- edge incidenceが高々2。
- edge crossingなし。
- 全内部edgeのlocal Delaunay条件。
- 非退化入力では、すべてのcanonical unique siteが参照されること。
- duplicateは最初のoriginal IDへ正しく写像されること。
- 全点collinear入力ではempty triangulationを返すこと。
- 同一backend・同一build・同一入力に対するdeterministic output。
- dense/full/frontierを強制した全schedule。
- 0点、1点、2点、3点。
- duplicate後にunique siteが3点未満になる入力。
- collinear、cocircular、duplicate。
- NaN、Inf、`-0.0`／`+0.0`、`f32::MAX`近傍、subnormal。
- grid、cluster、uniform random。
- edge上への挿入。
- epoch counter wraparound。
- 大規模16K以上。

cocircular入力では正しいtriangulationが一意でないため、CPUとのtriangle集合一致だけを
oracleにしない。

frontier最適化はmanifold検査だけで合格にしてはならない。
local Delaunay、orientation、incircleのtest oracleはGPUの`UnaryOp`実装を共有せず、
CPU側の独立predicateまたは高精度property oracleを使い、自己検証を避ける。

### Execution contract

finite composition prototypeでは次を確認する。

- scope内部のhost readが0。
- terminalでpacked statusを一度だけ読む。
- device count 0でnext extent/statusが必ず0になり、emptyがabsorbing stateである。
- capacity上限をhostで証明するか、deviceの`validate -> commit`で部分更新を防ぐ。
- invalid indexはload/store前にguardされ、sticky statusだけへ安全性を依存しない。
- state effect tokenがread/write・write/write順序を保持する。
- foreign executorを拒否する。
- dynamic extentを持つ値がscope外へescapeしない。
- indirect非対応backendのfallbackが同じ結果を返す。

### Performance

同一入力、同一build、同一warm-up条件で次を比較する。

1. v0.87 host-exact baseline。
2. internal selection lowering改善。
3. scratch/workspace reuse。
4. indirect execution domain。
5. finite composition 4／8／16 round epoch。
6. proven indexed multi-sink lowering / fusion。
7. del2d algorithm側のfrontier/matching改善。

各段階で次を記録する。

- wall time。
- GPU kernel aggregate。
- dispatch。
- submission。
- observation回数と待ち時間。
- temporary allocation数とbytes。
- active item数とsegment長分布。
- round数。

16K prototypeの最初の目標は、selection由来のexact-length observationを48回から
約9回以下へ減らすこととする。前処理のscalar観測を残す場合、総host observationの
目安は少なくとも約11回である。正確な総数は標準traceで確認する。dispatch削減は
別の目標として追跡し、readback削減と混同しない。

CPU超えをMassively API単体の受入条件にはしない。CPUとGPUでalgorithmが異なり、
hardwareにも依存するためである。ただしapplicationとしては常にCPU比較を公開し、
GPUが遅い間は成功と表現しない。

### Public APIへ昇格する条件

新しい公開抽象は次を満たす場合だけ採用する。

1. 既存primitiveでは意味を表せない、または有限scopeなしでは制御値が必ずhostへ
   escapeすること。
2. del2d以外に最低一つの実applicationがあること。
3. stable order、host-exact terminal、error契約を維持できること。
4. 狙ったdispatch/observation/allocation counterが因果的に減ること。
5. application wall timeで有意な改善が再現すること。
6. WGPU以外のbackendに安全なfallbackがあること。

## 実装順序

### Phase 0: 計測

- 標準trace counterと再現可能な集計script/raw artifact。
- del2dのuniform/cluster/grid/cocircular benchmark。
- 全schedule correctness suite。
- Radeon 680Mでの基準値固定。

### Phase 1: APIを変えない内部改善

- `copy_where` single-consumer direct stable scatter。
- selection controlの共有。
- proven-unique indicesに限定したsealed multi-sink lowering prototype。
- lazy consumer fusionと同一permutationのCSE候補。
- dependency-aware scratch reuse。
- exact arityのmicrobenchmark。

### Phase 2: 小さい直交API

- fixed-shape collectiveのcaller-provided output。
- derived segmentation metadataの内部継承。

### Phase 3: Device execution domain

- WGPU indirect-dispatch smoke test。
- `LogicalExtent -> ExecutionDomain` lowering。
- zero-count、upper-bound、fallback検証。
- selectionと単純なindexed copyの限定pathへ適用。
- upper-bound multi-stage DAGを持つscan/reductionは別stepで検証。

### Phase 4: Feature-gated finite composition

- opaque scoped scalar/extent。
- fixed-capacity scoped sequence。
- packed status。
- 4／8／16 round bounded epoch。
- del2d insertionとlegalizationでA/B比較。

### Phase 5: 再評価

- Power Diagramまたはgraph frontierへ適用。
- public化、internal-only継続、または不採用を判断。
- 効果が小さい場合は抽象を増やさず、del2d algorithm側のspatial/batch設計へ戻る。

## 結論

今回の実装は、Massively v0.87の中心仮説の半分を強く支持した。

`map`、`permute`、`zip`、scan、selection、scatter、`SegmentIterator`を組み合わせれば、
Delaunayのような複雑なmutable topology algorithmもGPU上の有限並列passとして
記述できる。triangle-major CSRによってglobal sortを消せたことは、segmentation
algebra、CSR、`SegmentIterator`の価値を具体的に示している。公開`Segmentation`
concrete typeのhot-loop適合性を示したものではない。

一方、表現できることと速く実行できることは同じではない。現在はprimitive間の
materialization、1,219回のdispatch、48回のexact-length観測に加えたscalar観測を持ち、
524K点でもCPUより3.67倍遅い。各overheadの厳密な寄与率は標準traceで分離する必要が
ある。

次のMassivelyに必要なのは、より大きなdomain algorithmではなく、有限primitive合成を
そのまま保ちながら、

- private device extent、
- indirect execution domain、
- selection control、
- segmentation provenance、
- proven indexed multi-sink lowering、
- scratch lifetime、
- packed observation、
- barrier-aware fusion、

を一つの有限execution boundaryで最適化できる仕組みである。

まず既存API内部のloweringとworkspaceを改善し、それでもhost-exact境界が支配することを
再確認した後、non-escapingな`CompositionScope`をfeature-gatedで実証する。この順序なら
v0.87の少数primitiveという哲学を保ちつつ、性能仮説を反証可能な形で次へ進められる。
