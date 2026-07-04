# ソフトウェアのコンセプト

このファイルは修正するな。

## アーキテクチャ

CubeCL (https://github.com/tracel-ai/cubecl) の上に
NVIDIA/Thrust (https://github.com/NVIDIA/thrust) のようなものを構築する。

## データ

### DeviceVec

GPU上のデータ領域を表す。
転送はexplicitに行う。

- to_device: CPU->GPU転送
- to_vec: GPU->CPU転送

### DeviceSlice / DeviceSliceMut

- 不変スライス: DeviceVec::slice<R: RangeBounds>(&self, range: R) -> DeviceSlice
- 可変スライス: DeviceVec::slice_mut<R: RangeBounds>(&self, range: R) -> DeviceSliceMut
- スライスはスライスを作れる
  - DeviceSlice::slice -> DeviceSlice
  - DeviceSliceMut::slice -> DeviceSlice
  - DeviceSliceMut::slice_mut -> DeviceSliceMut

### SoA (Structure of Array)

GPUで計算を行うに当たって、AoSよりSoAの方が性能上有利。
DeviceSliceをTupleでまとめ、MIterにした上で計算に使う。

MIter<n> = SoAn(DeviceSlice, DeviceSlice, ...)
MIterMut<n> = SoAn(DeviceSliceMut, DeviceSliceMut, ...)

## アルゴリズム

全関数を抽象的な形で表す。

### 記法

- &[]aはMIter<a>を表す。
- &mut[]aはMIterMut<a>を表す。
- T?は、Option<T>を表す。
- []aはOwnedなDeviceVecのTupleを表す。MVec<a>と呼ぶ。主にallocation APIの出力を表す。
- &[T]はDeviceSlice<T>を表す。
- [T]はDeviceVec<T>を表す。
- idxはMIndexを表す。

### リスト

以下の場合を実装すればOK
- k <= 3
- a,b,c <= 7

- adjacent_difference(xs: &[]a, sum: a->a->a, out: &mut[]a)
- adjacent_find(xs: &[]a, eq: a->a->bool) -> idx?
- all_of(xs: &[]a, p: a->bool) -> bool
- any_of(xs: &[]a, p: a->bool) -> bool
- copy_where(xs: &[]a, stencil: &[u32], out: &mut[]a) -> idx
- count_if(xs: &[]a, p: a->bool) -> idx
- equal(xs: &[]a, ys: &[]a, eq: a->a->bool) -> bool
- exclusive_scan(xs: &[]a, zero: a, sum: a->a->a, out: &mut[]a)
- exclusive_scan_by_key(keys: &[]k, values: &[]a, eq: k->k->bool, zero: a, sum: a->a->a, out: &mut[]a)
- fill(v: a, out: &mut[]a)
- find_first_of(xs: &[]a, needles: &[]a, eq: a->a->bool) -> idx?
- find_if(xs: &[]a, p: a->bool) -> idx?
- gather(xs: &[]a, indices: &[idx], out: &mut[]a)
- gather_where(xs: &[]a, indices: &[idx], stencil: &[u32], out: &mut[]a)
- inclusive_scan(xs: &[]a, op: a->a->a, out: &mut[]a)
- inclusive_scan_by_key(keys: &[]k, values: &[]a, eq: k->k->bool, sum: a->a->a, out: &mut[]a)
- is_partitioned(xs: &[]a, p: a->bool) -> bool
- is_sorted_until(xs: &[]a, cmp: a->a->bool) -> idx
- is_sorted(xs: &[]a, cmp: a->a->bool) -> bool
- lexicographical_compare(xs: &[]a, ys: &[]a, cmp: a->a->bool) -> bool
- lower_bound(xs: &[]a, vs: &[]a, cmp: a->a->bool, out: &mut[idx])
- max_element(xs: &[]a, cmp: a->a->bool) -> idx?
- merge(xs: &[]a, ys: &[]a, cmp: a->a->bool, out: &mut[]a)
- merge_by_key(keys1: &[]k, values1: &[]a, keys2: &[]k, values2: &[]a, cmp: k->k->bool, out_k: &mut[]k, out_v: &mut[]a)
- min_element(xs: &[]a, cmp: a->a->bool) -> idx?
- minmax_element(xs: &[]a, cmp: a->a->bool) -> (idx, idx)?
- mismatch(xs: &[]a, ys: &[]a, eq: a->a->bool) -> idx?
- none_of(xs: &[]a, p: a->bool) -> bool
- partition(xs: &[]a, p: a->bool, out: &mut[]a) -> idx
- reduce(xs: &[]a, zero: a, sum: a->a->a) -> a
- reduce_by_key(keys: &[]k, values: &[]a, eq: k->k->bool, zero: a, sum: a->a->a, out_k: &mut[]k, out_v: &mut[]a) -> idx
- remove_where(xs: &[]a, stencil: &[u32], out: &mut[]a) -> idx
- replace_where(v: a, stencil: &[u32], out: &mut[]a)
- reverse(xs: &[]a, out: &mut[]a)
- scatter(xs: &[]a, indices: &[idx], out: &mut[]a)
- scatter_where(xs: &[]a, indices: &[idx], stencil: &[u32], out: &mut[]a)
- set_difference(xs: &[]a, ys: &[]a, cmp: a->a->bool, out: &mut[]a) -> idx
- set_intersection(xs: &[]a, ys: &[]a, cmp: a->a->bool, out: &mut[]a) -> idx
- set_union(xs: &[]a, ys: &[]a, cmp: a->a->bool, out: &mut[]a) -> idx
- sort(xs: &[]a, cmp: a->a->bool, out: &mut[]a)
- sort_by_key(keys: &[]k, values: &[]a, cmp: k->k->bool, out_k: &mut[]k, out_v: &mut[]a)
- transform(xs: &[]a, op: a->b, out: &mut[]b)
- transform_where(xs: &[]a, op: a->b, stencil: &[u32], out: &mut[]b)
- unique(xs: &[]a, eq: a->a->bool, out: &mut[]a) -> idx
- unique_by_key(keys: &[]k, values: &[]a, cmp: k->k->bool, out_k: &mut[]k, out_v: &mut[]a) -> idx
- upper_bound(xs: &[]a, vs: &[]a, cmp: a->a->bool, out: &mut[idx])
