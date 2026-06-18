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

### SoA (Structure of Array)

GPUで計算を行うに当たって、AoSよりSoAの方が性能上有利。
SoA1（主に&DeviceVec）をzipしてまとめてSoAnにした上で計算に使う。

- SoA1
  - &DeviceVec
- SoAn (n=2..)
  - zipして作る

### ユーティリティ

- zip(xs: &[]n, ys: &[]m) -> &[]n+m

## アルゴリズム

全関数を抽象的な形で表す。

### 記法

- &[]nはSoAnを表す。
- &[T]はSoA1を表す。&[int]は特に要素が1のものを表す。
- T?は、Option<T>を表す。
- []nはOwnedなDeviceVecのn-tupleを表す。
  - ただし、1-tupleの場合はDeviceVecをそのまま返す（理由: それがSoA1の定義だから）。

### リスト

- adjacent_difference(xs: &[]n, sum: n->n->n) -> []n
- adjacent_find: (xs: &[]n, eq: n->n->bool) -> int?
- all_of(xs: &[]n, p: n->bool) -> bool
- any_of(xs: &[]n, p: n->bool) -> bool
- copy_if(xs: &[]n, stencil: &[]k, p: k->bool) -> []n
- count_if(xs: &[]n, p: n->bool) -> int
- equal(xs: &[]n, ys: &[]n, eq: n->n->bool) -> bool
- equal_range(xs: &[]n, v: n, cmp: n->n->bool) -> int
- exclusive_scan(xs: &[]n, zero: n, sum: n->n->n) -> []n
- exclusive_scan_by_key(keys: &[]m, values: &[]n, eq: m->m->bool, zero: n, sum: n->n->n) -> []n
- find_first_of(xs: &[]n, needles: &[]n, eq: n->n->bool) -> int?
- find_if(xs: &[]n, p: n->bool) -> int?
- gather(xs: &[]n, indices: &[int]) -> []n
- gather_if(xs: &[]n, indices: &[int], stencil: &[]k, p: k->bool) -> []n
- inclusive_scan(xs: &[]n, op: n->n->n) -> []n
- inclusive_scan_by_key(keys: &[]m, values: &[]n, eq: m->m->bool, sum: n->n->n) -> []n
- inner_product(xs: &[]n, ys: &[]m, zipper: n->m->l, zero: l, sum: l->l->l) -> l
- is_partitioned(xs: &[]n, p: n->bool) -> bool
- is_sorted_until(xs: &[]n, cmp: n->n->bool) -> int
- is_sorted(xs: &[]n, cmp: n->n->bool) -> bool
- lexicographical_compare(xs: &[]n, ys: &[]n, cmp: n->n->bool) -> bool
- lower_bound(xs: &[]n, v: n, cmp: n->n->bool) -> int
- max_element(xs: &[]n, cmp: n->n->bool) -> int?
- merge(xs: &[]n, ys: &[]n, cmp: n->n->bool) -> []n
- merge_by_key(keys1: &[]m, values1: &[]n, keys2: &[]m, values2: &[]n, cmp: m->m->bool) -> ([]m, []n)
- min_element(xs: &[]n, cmp: n->n->bool) -> int?
- minmax_element(xs: &[]n, cmp: n->n->bool) -> (int,int)?
- mismatch(xs: &[]n, ys: &[]n, eq: n->n->bool) -> int?
- none_of(xs: &[]n, p: n->bool) -> bool
- partition(xs: &[]n, p: n->bool) -> ([]n, []n)
- reduce(xs: &[]n, zero: n, sum: n->n->n) -> n
- reduce_by_key(keys: &[]m, values: &[]n, eq: m->m->bool, zero: n, sum: n->n->n) -> ([]m, []n)
- remove_if(xs: &[]n, p: n->bool) -> []n
- replace_if(xs: &[]n, v: n, stencil: &[]k, p: k->bool) -> []n
- reverse(xs: &[]n) -> []n
- scatter(xs: &[]n, indices: &[int], len: int, default: n) -> []n
- scatter_if(xs: &[]n, indices: &[int], len: int, default: n, stencil: &[]k, p: k->bool) -> []n
- set_difference(xs: &[]n, ys: &[]n, cmp: n->n->bool) -> []n
- set_intersection(xs: &[]n, ys: &[]n, cmp: n->n->bool) -> []n
- set_union(xs: &[]n, ys: &[]n, cmp: n->n->bool) -> []n
- sort(xs: &[]n, cmp: n->n->bool) -> []n
- sort_by_key(keys: &[]m, values: &[]n, cmp: m->m->bool) -> ([]m, []n)
- stable_sort(xs: &[]n, cmp: n->n->bool) -> []n
- stable_sort_by_key(keys: &[]m, values: &[]n, cmp: m->m->bool) -> ([]m, []n)
- transform(xs: &[]n, op: n->m) -> []m
- unique(xs: &[]n, eq: n->n->bool) -> []n
- unique_by_key(keys: &[]m, values: &[]n, cmp: m->m->bool) -> ([]m, []n)
- upper_bound(xs: &[]n, v: n, cmp: n->n->bool) -> int