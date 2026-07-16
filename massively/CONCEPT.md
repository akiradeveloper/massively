# 設計コンセプト

Massivelyの基本的な設計コンセプトについて記述する。

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

- MIter::slice -> MIter
- MIterMut::slice -> MIter
- MIterMut::slice_mut -> MIterMut
- MVec::slice -> MIter
- MVec::slice_mut -> MIterMut

## アルゴリズム

### 記法

- &[a] = MIter<Item = a>
- &mut[a] = MIterMut<Item = a>
- [a] = MVec<Item = a>
- T? = Option<T>
- idx = usize

### vectorアルゴリズム

- k <= 3
- a,b,c <= 12

- adjacent_difference(xs: &[a], sum: a->a->a) -> [a]
- adjacent_find(xs: &[a], eq: a->a->bool) -> idx?
- all_of(xs: &[a], p: a->bool) -> bool
- any_of(xs: &[a], p: a->bool) -> bool
- copy(xs: &[a], out: &mut[a])
- copy_where(xs: &[a], stencil: &[bool]) -> [a]
- count_if(xs: &[a], p: a->bool) -> idx
- equal(xs: &[a], ys: &[a], eq: a->a->bool) -> bool
- exclusive_scan(xs: &[a], zero: a, sum: a->a->a) -> [a]
- exclusive_scan_by_key(keys: &[k], values: &[a], eq: k->k->bool, zero: a, sum: a->a->a) -> [a]
- fill(v: a, out: &mut[a])
- find_first_of(xs: &[a], needles: &[a], eq: a->a->bool) -> idx?
- find_if(xs: &[a], p: a->bool) -> idx?
- gather(xs: &[a], indices: &[idx]) -> [a]
- gather_where(xs: &[a], indices: &[idx], stencil: &[bool], out: &mut[a])
- inclusive_scan(xs: &[a], op: a->a->a) -> [a]
- inclusive_scan_by_key(keys: &[k], values: &[a], eq: k->k->bool, sum: a->a->a) -> [a]
- is_partitioned(xs: &[a], p: a->bool) -> bool
- is_sorted_until(xs: &[a], cmp: a->a->bool) -> idx
- is_sorted(xs: &[a], cmp: a->a->bool) -> bool
- lexicographical_compare(xs: &[a], ys: &[a], cmp: a->a->bool) -> bool
- lower_bound(xs: &[a], vs: &[a], cmp: a->a->bool) -> [idx]
- max_element(xs: &[a], cmp: a->a->bool) -> idx?
- merge(xs: &[a], ys: &[a], cmp: a->a->bool) -> [a]
- merge_by_key(keys1: &[k], values1: &[a], keys2: &[k], values2: &[a], cmp: k->k->bool) -> [a]
- min_element(xs: &[a], cmp: a->a->bool) -> idx?
- minmax_element(xs: &[a], cmp: a->a->bool) -> (idx, idx)?
- mismatch(xs: &[a], ys: &[a], eq: a->a->bool) -> idx?
- none_of(xs: &[a], p: a->bool) -> bool
- partition(xs: &[a], p: a->bool) -> ([a], idx)
- reduce(xs: &[a], zero: a, sum: a->a->a) -> a
- reduce_by_key(keys: &[k], values: &[a], eq: k->k->bool, zero: a, sum: a->a->a) -> ([k], [a])
- remove_where(xs: &[a], stencil: &[bool]) -> [a]
- replace_where(v: a, stencil: &[bool], out: &mut[a])
- reverse(xs: &[a]) -> [a]
- scatter(xs: &[a], indices: &[idx], out: &mut[a])
- scatter_where(xs: &[a], indices: &[idx], stencil: &[bool], out: &mut[a])
- scatter_reduce(xs: &[a], indices: &[idx], init: a, sum: a->a->a, out: &mut[a])
- set_difference(xs: &[a], ys: &[a], cmp: a->a->bool) -> [a]
- set_intersection(xs: &[a], ys: &[a], cmp: a->a->bool) -> [a]
- set_union(xs: &[a], ys: &[a], cmp: a->a->bool) -> [a]
- sort(xs: &[a], cmp: a->a->bool) -> [a]
- sort_by_key(keys: &[k], values: &[a], cmp: k->k->bool) -> [a]
- transform(xs: &[a], op: a->b) -> [b]
- transform_where(xs: &[a], op: a->b, stencil: &[bool], out: &mut[b])
- unique(xs: &[a], eq: a->a->bool) -> [a]
- unique_by_key(keys: &[k], values: &[a], cmp: k->k->bool) -> [a]
- upper_bound(xs: &[a], vs: &[a], cmp: a->a->bool) -> [idx]