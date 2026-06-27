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

- 不変スライス: DeviceVec::slice<R: RangeBounds>(range: R) -> DeviceSlice
- 可変スライス: DeviceVec::slice_mut<R: RangeBounds>(range: R) -> DeviceSliceMut
- スライスはスライスを作れる
  - DeviceSlice::slice -> DeviceSlice
  - DeviceSliceMut::slice -> DeviceSlice
  - DeviceSliceMut::slice_mut -> DeviceSliceMut

### SoA (Structure of Array)

GPUで計算を行うに当たって、AoSよりSoAの方が性能上有利。
DeviceSliceをTupleでまとめ、MIterにした上で計算に使う。

MIter<n> = SoAn(MSlice, MSlice, ...)
MIterMut<n> = SoAn(DeviceSliceMut, DeviceSliceMut, ...)

## アルゴリズム

全関数を抽象的な形で表す。

### 記法

- &[]nはMIter<n>を表す。
- &mut[T]はMIterMut<n>を表す。
- T?は、Option<T>を表す。
- []nはOwnedなDeviceVecのTupleを表す。MVec<n>と呼ぶ。
- &[T]はMSlice<Item=T>を表す。

### リスト

以下の場合を実装すればOK
- k <= 1
- a,b,c <= 3

- adjacent_difference(xs: &[]a, sum: a->a->a) -> []a
- adjacent_find(xs: &[]a, eq: a->a->bool) -> int?
- all_of(xs: &[]a, p: a->bool) -> bool
- any_of(xs: &[]a, p: a->bool) -> bool
- copy_where(xs: &[]a, stencil: &[u32]) -> []a
- count_if(xs: &[]a, p: a->bool) -> int
- equal(xs: &[]a, ys: &[]a, eq: a->a->bool) -> bool
- equal_range(xs: &[]a, v: a, cmp: a->a->bool) -> int
- exclusive_scan(xs: &[]a, zero: a, sum: a->a->a) -> []a
- exclusive_scan_by_key(keys: &[]k, values: &[]a, eq: k->k->bool, zero: a, sum: a->a->a) -> []a
- find_first_of(xs: &[]a, needles: &[]a, eq: a->a->bool) -> int?
- find_if(xs: &[]a, p: a->bool) -> int?
- gather(xs: &[]a, indices: &[u32], out: &mut[]a)
- gather_where(xs: &[]a, indices: &[u32], stencil: &[u32], out: &mut[]a)
- inclusive_scan(xs: &[]a, op: a->a->a) -> []a
- inclusive_scan_by_key(keys: &[]k, values: &[]a, eq: k->k->bool, sum: a->a->a) -> []a
- inner_product(xs: &[]a, ys: &[]b, zipper: a->b->c, zero: c, sum: c->c->c) -> c
- is_partitioned(xs: &[]a, p: a->bool) -> bool
- is_sorted_until(xs: &[]a, cmp: a->a->bool) -> int
- is_sorted(xs: &[]a, cmp: a->a->bool) -> bool
- lexicographical_compare(xs: &[]a, ys: &[]a, cmp: a->a->bool) -> bool
- lower_bound(xs: &[]a, v: a, cmp: a->a->bool) -> int
- max_element(xs: &[]a, cmp: a->a->bool) -> int?
- merge(xs: &[]a, ys: &[]a, cmp: a->a->bool) -> []a
- merge_by_key(keys1: &[]k, values1: &[]a, keys2: &[]k, values2: &[]a, cmp: k->k->bool) -> ([]k, []a)
- min_element(xs: &[]a, cmp: a->a->bool) -> int?
- minmax_element(xs: &[]a, cmp: a->a->bool) -> (int,int)?
- mismatch(xs: &[]a, ys: &[]a, eq: a->a->bool) -> int?
- none_of(xs: &[]a, p: a->bool) -> bool
- partition(xs: &[]a, p: a->bool) -> ([]a, []a)
- reduce(xs: &[]a, zero: a, sum: a->a->a) -> a
- reduce_by_key(keys: &[]k, values: &[]a, eq: k->k->bool, zero: a, sum: a->a->a) -> ([]k, []a)
- remove_where(xs: &[]a, stencil: &[u32]) -> []a
- replace_where(v: a, stencil: &[u32], out: &mut[]a)
- reverse(xs: &[]a) -> []a
- scatter(xs: &[]a, indices: &[u32], out: &mut[]a)
- scatter_where(xs: &[]a, indices: &[u32], stencil: &[u32], out: &mut[]a)
- set_difference(xs: &[]a, ys: &[]a, cmp: a->a->bool) -> []a
- set_intersection(xs: &[]a, ys: &[]a, cmp: a->a->bool) -> []a
- set_union(xs: &[]a, ys: &[]a, cmp: a->a->bool) -> []a
- sort(xs: &[]a, cmp: a->a->bool) -> []a
- sort_by_key(keys: &[]k, values: &[]a, cmp: k->k->bool) -> ([]k, []a)
- stable_sort(xs: &[]a, cmp: a->a->bool) -> []a
- stable_sort_by_key(keys: &[]k, values: &[]a, cmp: k->k->bool) -> ([]k, []a)
- transform(xs: &[]a, op: a->b, out: &mut[]b)
- transform_where(xs: &[]a, op: a->b, stencil: &[u32], out: &mut[]b)
- unique(xs: &[]a, eq: a->a->bool) -> []a
- unique_by_key(keys: &[]k, values: &[]a, cmp: k->k->bool) -> ([]k, []a)
- upper_bound(xs: &[]a, v: a, cmp: a->a->bool) -> int