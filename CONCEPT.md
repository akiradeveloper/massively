# 設計コンセプト

Massivelyの基本的な設計コンセプトについて記述する。

このファイルは修正するな。

## データ

### DeviceVec

- to_device: CPU-to-GPU転送
- to_host: GPU-to-CPU転送

### イテレータ / Zip

- DeviceSlice<T>: MIter<Item = T>
- DeviceSliceMut<T>: MIterMut<Item = T>
- DeviceVec<T>: MVec<Item = T>
- Zip(MIter<A>, MIter<B>): MIter<Item = (A,B)>
- Zip(MIterMut<A>, MIterMut<B>): MIterMut<Item = (A,B)>
- Zip(MVec<A>, MVec<B>): MVec<Item = (A,B)>

### スライス

- MIter::slice(range) -> MIter
- MIterMut::slice(range) -> MIter
- MIterMut::slice_mut(range) -> MIterMut
- MVec::slice(range) -> MIter
- MVec::slice_mut(range) -> MIterMut

## アルゴリズム

### 記法

- &[a] = MIter<Item = a>
- &mut[a] = MIterMut<Item = a>
- [a] = MVec<Item = a>
- T? = Option<T>
- idx = MIndex
- flag = bool

### vectorアルゴリズム

- k <= 3
- a,b,c <= 12

- adjacent_difference(xs: &[a], sum: a->a->a) -> [a]
- adjacent_find(xs: &[a], eq: a->a->flag) -> idx?
- all_of(xs: &[a], p: a->flag) -> flag
- any_of(xs: &[a], p: a->flag) -> flag
- copy(xs: &[a], out: &mut[a])
- copy_where(xs: &[a], stencil: &[flag]) -> [a]
- count_if(xs: &[a], p: a->flag) -> idx
- equal(xs: &[a], ys: &[a], eq: a->a->flag) -> flag
- exclusive_scan(xs: &[a], zero: a, sum: a->a->a) -> [a]
- exclusive_scan_by_key(keys: &[k], values: &[a], eq: k->k->flag, zero: a, sum: a->a->a) -> [a]
- fill(v: a, out: &mut[a])
- find_first_of(xs: &[a], needles: &[a], eq: a->a->flag) -> idx?
- find_if(xs: &[a], p: a->flag) -> idx?
- gather(xs: &[a], indices: &[idx]) -> [a]
- gather_where(xs: &[a], indices: &[idx], stencil: &[flag], out: &mut[a])
- inclusive_scan(xs: &[a], op: a->a->a) -> [a]
- inclusive_scan_by_key(keys: &[k], values: &[a], eq: k->k->flag, sum: a->a->a) -> [a]
- is_partitioned(xs: &[a], p: a->flag) -> flag
- is_sorted_until(xs: &[a], cmp: a->a->flag) -> idx
- is_sorted(xs: &[a], cmp: a->a->flag) -> flag
- lexicographical_compare(xs: &[a], ys: &[a], cmp: a->a->flag) -> flag
- lower_bound(xs: &[a], vs: &[a], cmp: a->a->flag) -> [idx]
- max_element(xs: &[a], cmp: a->a->flag) -> idx?
- merge(xs: &[a], ys: &[a], cmp: a->a->flag) -> [a]
- merge_by_key(keys1: &[k], values1: &[a], keys2: &[k], values2: &[a], cmp: k->k->flag) -> [a]
- min_element(xs: &[a], cmp: a->a->flag) -> idx?
- minmax_element(xs: &[a], cmp: a->a->flag) -> (idx, idx)?
- mismatch(xs: &[a], ys: &[a], eq: a->a->flag) -> idx?
- none_of(xs: &[a], p: a->flag) -> flag
- partition(xs: &[a], p: a->flag) -> ([a], idx)
- reduce(xs: &[a], zero: a, sum: a->a->a) -> a
- reduce_by_key(keys: &[k], values: &[a], eq: k->k->flag, zero: a, sum: a->a->a) -> ([k], [a])
- radix_sort_by_key(keys: &[radix k], values: &[a]) -> [a]
- remove_where(xs: &[a], stencil: &[flag]) -> [a]
- replace_where(v: a, stencil: &[flag], out: &mut[a])
- reverse(xs: &[a]) -> [a]
- scatter(xs: &[a], indices: &[idx], out: &mut[a])
- scatter_where(xs: &[a], indices: &[idx], stencil: &[flag], out: &mut[a])
- scatter_reduce(xs: &[a], indices: &[idx], init: a, sum: a->a->a, out: &mut[a])
- set_difference(xs: &[a], ys: &[a], cmp: a->a->flag) -> [a]
- set_intersection(xs: &[a], ys: &[a], cmp: a->a->flag) -> [a]
- set_union(xs: &[a], ys: &[a], cmp: a->a->flag) -> [a]
- sort(xs: &[a], cmp: a->a->flag) -> [a]
- sort_by_key(keys: &[k], values: &[a], cmp: k->k->flag) -> [a]
- map(xs: &[a], op: a->b) -> [b]
- transform_where(xs: &[a], op: a->b, stencil: &[flag], out: &mut[b])
- unique(xs: &[a], eq: a->a->flag) -> [a]
- unique_by_key(keys: &[k], values: &[a], cmp: k->k->flag) -> [a]
- upper_bound(xs: &[a], vs: &[a], cmp: a->a->flag) -> [idx]
