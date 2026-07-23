# 提案: Massivelyにおける合成可能なGPU常駐動的シーケンス

状態: P0の公開bounded sequence案はv0.86で不採用。以下は実験結果と将来の
最適化候補の記録である。

2026-07-23の最終判断と実装順は
[`proposal-compound-operations.md`](proposal-compound-operations.md)を正本とする。
公開device lengthや汎用`Iteration`は追加しない。polygonのlengths／0-based cell
IDs／offsetsを`seg::Segmentation`で同じ分割として扱い、per-edge contextは
`segment_ids()`と`lazy::permute`でbroadcastする。whole-segmentとper-entryの
clippingは外部Power Diagram sourceで比較してアプリ側の構成を選ぶ。性能差だけを
理由にcontext専用APIを追加せず、最適化する場合も既存合成を内部fusionする。

v0.86では`MVal`とdevice logical extentを内部実装に限定し、公開APIは同期
インターフェイス一種類に統一する。可変長結果は公開return boundaryで正確な長さへ
解決し、固定長処理では長さreadbackを行わない。公開`BoundedVec`、公開
`DeviceSize`、bounded版との二重API、およびindirect dispatchは導入しない。
記録・再実行、temporary reuse、kernel fusion、汎用flat-mapの提案は、このAPI判断
とは独立した将来候補として残す。

## 背景と目的

`power-diagram`の実験では、専用のPower diagramカーネルを使用せず、
公開されているMassivelyプリミティブの合成によってPower cellの
クリッピングを実装している。

半平面クリッピングでは、処理ラウンドごとに各CSRセグメントの長さが変化する。
これは、Massivelyの動的データ処理能力を検証するうえで有用なケースである。
Radeon 680M（RADV/Vulkan）上でランダムな256サイトをプロファイルした結果、
次のことが分かった。

- ウォールクロック時間は約708 ms
- カーネル起動回数は11,008回
- GPUタイムスタンプで計測した実行時間は約23.9 ms
- セグメント単位のフィルタリングとunique／compactionにより、クリッピングの
  各ラウンドで少なくとも2回、結果長を取得するためのGPUからCPUへの同期が発生

ラウンド途中の結果長取得を避けるため、固定容量バッファを使った合成も試した。
しかし、この方式ではパディングを含む全スロットを走査する必要がある。
256サイトではカーネル起動が8,708回、GPU実行時間が約133 ms、
ウォールクロック時間が約1.09秒となった。

この結果から、次の2つが独立した問題として確認できる。

1. 動的な出力長を扱うと、現在はホスト同期が必要になる。
2. ホスト同期を避けるために固定容量化すると、過剰なパディング処理が発生する。

さらに、どちらの方式でも個別のカーネル起動回数が多すぎる。

## P0: GPU上に保持される論理長

ホスト側では確保容量が既知だが、実際の論理長はGPU上のスカラーとして保持される
bounded device sequenceを追加する。

```rust,ignore
pub struct DeviceSize<R> { /* GPU上のu32スカラー */ }

pub struct BoundedVec<R, T> {
    storage: DeviceVec<R, T>,
    capacity: usize,
    len: DeviceSize<R>,
}

pub struct BoundedSegmented<R, T> {
    values: BoundedVec<R, T>,
    offsets: DeviceVec<R, u32>,
}
```

selectionおよびセグメント単位のcompactionには、scanの最終値をCPUへ読み戻さず、
これらの型を返す遅延評価版を追加する。

```rust,ignore
copy_if_bounded(exec, input, pred) -> Result<BoundedVec<R, T>, Error>
unique_bounded(exec, input, equal) -> Result<BoundedVec<R, T>, Error>

ForEachSegment(Filter(pred))
    .run_bounded(exec, segments)
    -> Result<BoundedSegmented<R, T>, Error>
```

後続のMassivelyアルゴリズムもbounded形式を入力として受け取れるようにする。
対応可能なバックエンドではCubeCLの間接ディスパッチ
（`CubeCount::Dynamic`）を利用し、各カーネルにもGPU上の論理長を渡して
境界判定を行う。確保容量は引き続きアロケーションと検証に使用する。
論理長をCPUへ同期するのは、明示的な`to_host`、`materialize_exact`、
`shrink_to_fit`だけとする。

この機能は現在の即時評価APIと共存できる。既存関数は正確な長さを持つ
`DeviceVec`を返し続け、新しいbounded／deferred関数によって同期境界を
明示できるようにする。

### 内部実装の変更

現在の`SelectionControl`は`count: u32`を保持している。scan結果から
`SelectionControl`を構築するときは、scatter用のメモリ確保とカーネル起動の前に
`last_u32`を呼び出し、結果長をホストへ読み戻している。

遅延評価版のcontrolでは、代わりに次のような型を保持する。

```rust,ignore
enum LogicalSize<R> {
    Host(u32),
    Device(DeviceSize<R>),
}
```

同じ変更を、現在`run_into`からホスト上の`u32`を返している
セグメント単位のcompaction実行パスにも適用する。

## P0: 記録・再実行可能なアルゴリズムグラフ

プリミティブの合成を記録し、一時アロケーションを再利用しながら、依存関係を含む
グラフ全体をまとめてサブミットできるExecutor機能を追加する。

```rust,ignore
let graph = exec.record(|graph| {
    // ping-pong用の一時領域を含む通常のMassivelyアルゴリズム
})?;

graph.run(&parameters)?;
```

必要な要件は次のとおり。

- raw CubeCLカーネルではなく、通常の公開プリミティブを記録できる
- コンパイル済みパイプラインとバインディングレイアウトをキャッシュする
- 実行間で中間Device column用のメモリアリーナを再利用する
- 依存する複数のディスパッチを、可能な限り少ないバックエンドへの
  サブミットにまとめる
- クリッピングのrankなどのスカラー引数を、グラフの再構築や再コンパイルなしで
  変更できる
- ホストへのreadbackが要求されるまで同期しない

この機能は、GPU実行時間が約23.9 msであるのに対して、ウォールクロック時間が
約708 msかかっている差を縮めるために必要である。また、反復scan、
グラフアルゴリズム、疎行列ソルバー、ジオメトリ処理などにも再利用できる。

## P1: セグメント単位のadjacent flat-map（不採用の記録）

このP1は採用しない。cyclic predecessorは`counting`したentry indexと、
`segment_ids()`から`permute`したsegment start／endにより計算できる。previous
vertexをもう一度`permute`し、current vertexとcell contextをzipすれば全edgeを
並列処理できる。以下は当初案を保存したものであり、実装指示ではない。

「1つの入力要素から0〜K個の出力要素を生成する」という一般的な処理を表す、
セグメント単位のプリミティブを追加する。各入力要素の処理では、同じセグメント内の
直前の要素も参照できるようにする。

```rust,ignore
ForEachSegment(AdjacentFlatMap::<2, _>(op))
    .run_bounded(exec, segments)
    -> Result<BoundedSegmented<R, Output>, Error>
```

operationは固定数の出力スロットと、有効なスロット数またはvalidity maskを返す。

```rust,ignore
trait AdjacentExpandOp<Input, Output, const K: usize> {
    fn apply(previous: Input, current: Input) -> ([Output; K], u32);
}
```

さらに、各セグメントにつき1つのcontext値を渡せるoverloadを用意する。
これにより、クリッピング平面などのセグメント固有パラメータを、すべてのflat要素へ
手動でgatherする必要がなくなる。

Massivelyは、この操作をemission、segmented scan、offset再構築、scatterへ
loweringできる。その際、Structure of Arraysの各columnをまとめて融合する。

Power diagramでは`K = 2`とすることで、Sutherland–Hodgman法における4つの
辺判定を直接表現できる。これにより、各クリッピングラウンドで現在必要になっている
多数のmap、gather、一時column、カーネル起動を削減できる。

このプリミティブはPower diagram専用ではなく、次の用途にも適用できる。

- tokenizerによる要素展開
- 疎行の書き換え
- polyline clipping
- run expansion
- 辺を書き換えながら行う隣接リストのfiltering

## 受け入れ条件

256サイトのPower diagramベンチマークにおいて、次の条件を満たすこと。

1. クリッピングループ内でGPUからCPUへの同期が発生しない。
2. アルゴリズムが公開Massivelyプリミティブとスカラーoperation objectのみで
   記述されている。
3. クリッピングの各rankで一時領域が再利用される。
4. ウォールクロック時間がホストのカーネル起動処理に支配されず、
   GPUタイムスタンプ時間に近づく。
5. cellの面積と重心が逐次実行版`power-point-cpu`の結果と一致する。
6. 同じAPIについて、selection、segmented expansion、空の出力、長さの異なる
   セグメント、グラフ再実行を対象とした非ジオメトリのテストが用意されている。

## 推奨する実装順序

1. `DeviceSize`、`BoundedVec`、遅延評価版selection／segmented compactionを
   導入する。
2. 間接ディスパッチを利用し、map、gather、scatter、scan、
   segmented algorithmがbounded inputを扱えるようにする。
3. 記録可能なグラフと一時アロケーションの再利用を追加する。
4. adjacent処理は既存の`counting`／`segment_ids`／`permute`／`zip`／`flat_map`で
   記述する。最適化する場合もこの公開合成を内部fusionし、新しいAPIは追加しない。
