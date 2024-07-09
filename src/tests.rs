use super::*;

use super::CowHashMap;
use super::Entry::{Occupied, Vacant};
use std::cell::RefCell;
use std::hash::RandomState;

#[test]
fn test_zero_capacities() {
    type HM = CowHashMap<i32, i32>;

    let m = HM::new();
    assert_eq!(m.capacity(), 0);

    let m = HM::default();
    assert_eq!(m.capacity(), 0);

    let m = HM::with_hasher(RandomState::new());
    assert_eq!(m.capacity(), 0);

    let m = HM::with_capacity(0);
    assert_eq!(m.capacity(), 0);

    let m = HM::with_capacity_and_hasher(0, RandomState::new());
    assert_eq!(m.capacity(), 0);

    let m = HM::new();
    m.insert(1, 1);
    m.insert(2, 2);
    m.remove(&1);
    m.remove(&2);
    m.shrink_to_fit();
    assert_eq!(m.capacity(), 0);

    let m = HM::new();
    m.reserve(0);
    assert_eq!(m.capacity(), 0);
}

#[test]
fn test_create_capacity_zero() {
    let m = CowHashMap::with_capacity(0);

    assert!(m.insert(1, 1).is_none());

    assert!(m.contains_key(&1));
    assert!(!m.contains_key(&0));
}

#[test]
fn test_insert() {
    let m = CowHashMap::new();
    assert_eq!(m.len(), 0);
    assert!(m.insert(1, 2).is_none());
    assert_eq!(m.len(), 1);
    assert!(m.insert(2, 4).is_none());
    assert_eq!(m.len(), 2);
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert_eq!(*m.get(&2).unwrap(), 4);
}

#[test]
fn test_clone() {
    let m = CowHashMap::new();
    assert_eq!(m.len(), 0);
    assert!(m.insert(1, 2).is_none());
    assert_eq!(m.len(), 1);
    assert!(m.insert(2, 4).is_none());
    assert_eq!(m.len(), 2);
    let m2 = m.clone();
    assert_eq!(*m2.get(&1).unwrap(), 2);
    assert_eq!(*m2.get(&2).unwrap(), 4);
    assert_eq!(m2.len(), 2);
}

thread_local! { static DROP_VECTOR: RefCell<Vec<i32>> = RefCell::new(Vec::new()) }

#[derive(Hash, PartialEq, Eq)]
struct Droppable {
    k: usize,
}

impl Droppable {
    fn new(k: usize) -> Droppable {
        DROP_VECTOR.with(|slot| {
            slot.borrow_mut()[k] += 1;
        });

        Droppable { k }
    }
}

impl Drop for Droppable {
    fn drop(&mut self) {
        DROP_VECTOR.with(|slot| {
            slot.borrow_mut()[self.k] -= 1;
        });
    }
}

impl Clone for Droppable {
    fn clone(&self) -> Droppable {
        Droppable::new(self.k)
    }
}

#[test]
fn test_drops() {
    DROP_VECTOR.with(|slot| {
        *slot.borrow_mut() = vec![0; 200];
    });

    {
        let m = CowHashMap::new();

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });

        for i in 0..100 {
            let d1 = Droppable::new(i);
            let d2 = Droppable::new(i + 100);
            m.insert(d1, d2);
        }

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 1);
            }
        });

        for i in 0..50 {
            let k = Droppable::new(i);
            let v = m.remove(&k);

            assert!(v.is_some());

            DROP_VECTOR.with(|v| {
                assert_eq!(v.borrow()[i], 1);
                assert_eq!(v.borrow()[i + 100], 1);
            });
        }

        DROP_VECTOR.with(|v| {
            for i in 0..50 {
                assert_eq!(v.borrow()[i], 0);
                assert_eq!(v.borrow()[i + 100], 0);
            }

            for i in 50..100 {
                assert_eq!(v.borrow()[i], 1);
                assert_eq!(v.borrow()[i + 100], 1);
            }
        });
    }

    DROP_VECTOR.with(|v| {
        for i in 0..200 {
            assert_eq!(v.borrow()[i], 0);
        }
    });
}

#[test]
fn test_into_iter_drops() {
    DROP_VECTOR.with(|v| {
        *v.borrow_mut() = vec![0; 200];
    });

    let hm = {
        let hm = CowHashMap::new();

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });

        for i in 0..100 {
            let d1 = Droppable::new(i);
            let d2 = Droppable::new(i + 100);
            hm.insert(d1, d2);
        }

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 1);
            }
        });

        hm
    };

    // By the way, ensure that cloning doesn't screw up the dropping.
    drop(hm.clone());

    {
        let mut half = hm.into_iter().take(50);

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 1);
            }
        });

        for _ in half.by_ref() {}
    };

    DROP_VECTOR.with(|v| {
        for i in 0..200 {
            assert_eq!(v.borrow()[i], 0);
        }
    });
}

#[test]
fn test_empty_remove() {
    let m: CowHashMap<i32, bool> = CowHashMap::new();
    assert_eq!(m.remove(&0), None);
}

#[test]
fn test_empty_entry() {
    let m: CowHashMap<i32, bool> = CowHashMap::new();
    match m.entry(0) {
        Occupied(_) => panic!(),
        Vacant(_) => {}
    }
    assert!(*m.entry(0).or_insert(true));
    assert_eq!(m.len(), 1);
}

#[test]
fn test_empty_iter() {
    let m: CowHashMap<i32, bool> = CowHashMap::new();
    assert_eq!(m.drain().next(), None);
    assert_eq!(m.keys().next(), None);
    assert_eq!(m.values().next(), None);
    assert_eq!(m.values_mut().next(), None);
    assert_eq!(m.iter().next(), None);
    assert_eq!(m.iter_mut().next(), None);
    assert_eq!(m.len(), 0);
    assert!(m.is_empty());
    assert_eq!(m.into_iter().next(), None);
}

#[test]
fn test_lots_of_insertions() {
    let m = CowHashMap::new();

    // Try this a few times to make sure we never screw up the hashmap's
    // internal state.
    let loops = if cfg!(miri) { 2 } else { 10 };
    for _ in 0..loops {
        assert!(m.is_empty());

        let count = if cfg!(miri) { 101 } else { 1001 };

        for i in 1..count {
            assert!(m.insert(i, i).is_none());

            for j in 1..=i {
                let r = m.get(&j);
                assert_eq!(r, Some(Arc::new(j)));
            }

            for j in i + 1..count {
                let r = m.get(&j);
                assert_eq!(r, None);
            }
        }

        for i in count..(2 * count) {
            assert!(!m.contains_key(&i));
        }

        // remove forwards
        for i in 1..count {
            assert!(m.remove(&i).is_some());

            for j in 1..=i {
                assert!(!m.contains_key(&j));
            }

            for j in i + 1..count {
                assert!(m.contains_key(&j));
            }
        }

        for i in 1..count {
            assert!(!m.contains_key(&i));
        }

        for i in 1..count {
            assert!(m.insert(i, i).is_none());
        }

        // remove backwards
        for i in (1..count).rev() {
            assert!(m.remove(&i).is_some());

            for j in i..count {
                assert!(!m.contains_key(&j));
            }

            for j in 1..i {
                assert!(m.contains_key(&j));
            }
        }
    }
}

#[test]
fn test_find_mut() {
    let m = CowHashMap::new();
    assert!(m.insert(1, 12).is_none());
    assert!(m.insert(2, 8).is_none());
    assert!(m.insert(5, 14).is_none());
    let new = 100;
    match m.get_mut(&5) {
        None => panic!(),
        Some(mut x) => *x = new,
    }
    assert_eq!(m.get(&5), Some(Arc::new(new)));
}

#[test]
fn test_insert_overwrite() {
    let m = CowHashMap::new();
    assert!(m.insert(1, 2).is_none());
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert!(!m.insert(1, 3).is_none());
    assert_eq!(*m.get(&1).unwrap(), 3);
}

#[test]
fn test_insert_conflicts() {
    let m = CowHashMap::with_capacity(4);
    assert!(m.insert(1, 2).is_none());
    assert!(m.insert(5, 3).is_none());
    assert!(m.insert(9, 4).is_none());
    assert_eq!(*m.get(&9).unwrap(), 4);
    assert_eq!(*m.get(&5).unwrap(), 3);
    assert_eq!(*m.get(&1).unwrap(), 2);
}

#[test]
fn test_conflict_remove() {
    let m = CowHashMap::with_capacity(4);
    assert!(m.insert(1, 2).is_none());
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert!(m.insert(5, 3).is_none());
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert_eq!(*m.get(&5).unwrap(), 3);
    assert!(m.insert(9, 4).is_none());
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert_eq!(*m.get(&5).unwrap(), 3);
    assert_eq!(*m.get(&9).unwrap(), 4);
    assert!(m.remove(&1).is_some());
    assert_eq!(*m.get(&9).unwrap(), 4);
    assert_eq!(*m.get(&5).unwrap(), 3);
}

#[test]
fn test_is_empty() {
    let m = CowHashMap::with_capacity(4);
    assert!(m.insert(1, 2).is_none());
    assert!(!m.is_empty());
    assert!(m.remove(&1).is_some());
    assert!(m.is_empty());
}

#[test]
fn test_remove() {
    let m = CowHashMap::new();
    m.insert(1, 2);
    assert_eq!(m.remove(&1), Some(Arc::new(2)));
    assert_eq!(m.remove(&1), None);
}

#[test]
fn test_remove_entry() {
    let m = CowHashMap::new();
    m.insert(1, 2);
    assert_eq!(m.remove_entry(&1), Some((1, Arc::new(2))));
    assert_eq!(m.remove(&1), None);
}

#[test]
fn test_iterate() {
    let m = CowHashMap::with_capacity(4);
    for i in 0..32 {
        assert!(m.insert(i, i * 2).is_none());
    }
    assert_eq!(m.len(), 32);

    let mut observed: u32 = 0;

    for (k, v) in &m {
        assert_eq!(*v, k * 2);
        observed |= 1 << k;
    }
    assert_eq!(observed, 0xFFFF_FFFF);
}

#[test]
fn test_keys() {
    let pairs = [(1, 'a'), (2, 'b'), (3, 'c')];
    let map: CowHashMap<_, _> = pairs.into_iter().collect();
    let keys: Vec<_> = map.keys().collect();
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&1));
    assert!(keys.contains(&2));
    assert!(keys.contains(&3));
}

#[test]
fn test_values() {
    let pairs = [(1, 'a'), (2, 'b'), (3, 'c')];
    let map: CowHashMap<_, _> = pairs.into_iter().collect();
    let values: Vec<_> = map.values().collect();
    assert_eq!(values.len(), 3);
    assert!(values.contains(&Arc::new('a')));
    assert!(values.contains(&Arc::new('b')));
    assert!(values.contains(&Arc::new('c')));
}

#[test]
fn test_values_mut() {
    let pairs = [(1, 1), (2, 2), (3, 3)];
    let map: CowHashMap<_, _> = pairs.into_iter().collect();
    for mut value in map.values_mut() {
        *value = (*value) * 2
    }
    let values: Vec<_> = map.values().collect();
    assert_eq!(values.len(), 3);
    assert!(values.contains(&Arc::new(2)));
    assert!(values.contains(&Arc::new(4)));
    assert!(values.contains(&Arc::new(6)));
}

#[test]
fn test_into_keys() {
    let pairs = [(1, 'a'), (2, 'b'), (3, 'c')];
    let map: CowHashMap<_, _> = pairs.into_iter().collect();
    let keys: Vec<_> = map.into_keys().collect();

    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&1));
    assert!(keys.contains(&2));
    assert!(keys.contains(&3));
}

#[test]
fn test_into_values() {
    let pairs = [(1, 'a'), (2, 'b'), (3, 'c')];
    let map: CowHashMap<_, _> = pairs.into_iter().collect();
    let values: Vec<_> = map.into_values().collect();

    assert_eq!(values.len(), 3);
    assert!(values.contains(&Arc::new('a')));
    assert!(values.contains(&Arc::new('b')));
    assert!(values.contains(&Arc::new('c')));
}

#[test]
fn test_find() {
    let m = CowHashMap::new();
    assert!(m.get(&1).is_none());
    m.insert(1, 2);
    match m.get(&1) {
        None => panic!(),
        Some(v) => assert_eq!(*v, 2),
    }
}

#[test]
fn test_eq() {
    let m1 = CowHashMap::new();
    m1.insert(1, 2);
    m1.insert(2, 3);
    m1.insert(3, 4);

    let m2 = CowHashMap::new();
    m2.insert(1, 2);
    m2.insert(2, 3);

    assert!(m1 != m2);

    m2.insert(3, 4);

    assert_eq!(m1, m2);
}

#[test]
fn test_show() {
    let map = CowHashMap::new();
    let empty: CowHashMap<i32, i32> = CowHashMap::new();

    map.insert(1, 2);
    map.insert(3, 4);

    let map_str = format!("{map:?}");

    assert!(map_str == "{1: 2, 3: 4}" || map_str == "{3: 4, 1: 2}");
    assert_eq!(format!("{empty:?}"), "{}");
}

#[test]
fn test_reserve_shrink_to_fit() {
    let m = CowHashMap::new();
    m.insert(0, 0);
    m.remove(&0);
    assert!(m.capacity() >= m.len());
    for i in 0..128 {
        m.insert(i, i);
    }
    m.reserve(256);

    let usable_cap = m.capacity();
    for i in 128..(128 + 256) {
        m.insert(i, i);
        assert_eq!(m.capacity(), usable_cap);
    }

    for i in 100..(128 + 256) {
        assert_eq!(m.remove(&i), Some(Arc::new(i)));
    }
    m.shrink_to_fit();

    assert_eq!(m.len(), 100);
    assert!(!m.is_empty());
    assert!(m.capacity() >= m.len());

    for i in 0..100 {
        assert_eq!(m.remove(&i), Some(Arc::new(i)));
    }
    m.shrink_to_fit();
    m.insert(0, 0);

    assert_eq!(m.len(), 1);
    assert!(m.capacity() >= m.len());
    assert_eq!(m.remove(&0), Some(Arc::new(0)));
}

#[test]
fn test_from_iter() {
    let xs = [(1, 1), (2, 2), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let map: CowHashMap<_, _> = xs.iter().cloned().collect();

    for &(k, v) in &xs {
        assert_eq!(map.get(&k), Some(Arc::new(v)));
    }

    assert_eq!(map.iter().len(), xs.len() - 1);
}

#[test]
fn test_size_hint() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let map: CowHashMap<_, _> = xs.iter().cloned().collect();

    let mut iter = map.iter();

    for _ in iter.by_ref().take(3) {}

    assert_eq!(iter.size_hint(), (3, Some(3)));
}

#[test]
fn test_iter_len() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let map: CowHashMap<_, _> = xs.iter().cloned().collect();

    let mut iter = map.iter();

    for _ in iter.by_ref().take(3) {}

    assert_eq!(iter.len(), 3);
}

#[test]
fn test_mut_size_hint() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let map: CowHashMap<_, _> = xs.iter().cloned().collect();

    let mut iter = map.iter_mut();

    for _ in iter.by_ref().take(3) {}

    assert_eq!(iter.size_hint(), (3, Some(3)));
}

#[test]
fn test_iter_mut_len() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let map: CowHashMap<_, _> = xs.iter().cloned().collect();

    let mut iter = map.iter_mut();

    for _ in iter.by_ref().take(3) {}

    assert_eq!(iter.len(), 3);
}

#[test]
fn test_entry() {
    let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

    let map: CowHashMap<_, _> = xs.iter().cloned().collect();

    // Existing key (insert)
    match map.entry(1) {
        Vacant(_) => unreachable!(),
        Occupied(view) => {
            assert_eq!(view.get(), Arc::new(10));
            assert_eq!(view.insert(100), Arc::new(10));
        }
    }
    assert_eq!(map.get(&1).unwrap(), Arc::new(100));
    assert_eq!(map.len(), 6);

    // Existing key (update)
    match map.entry(2) {
        Vacant(_) => unreachable!(),
        Occupied(view) => {
            let mut v = view.get_mut();
            let new_v = (*v) * 10;
            *v = new_v;
        }
    }
    assert_eq!(map.get(&2).unwrap(), Arc::new(200));
    assert_eq!(map.len(), 6);

    // Existing key (take)
    match map.entry(3) {
        Vacant(_) => unreachable!(),
        Occupied(view) => {
            assert_eq!(view.remove(), Arc::new(30));
        }
    }
    assert_eq!(map.get(&3), None);
    assert_eq!(map.len(), 5);

    // Inexistent key (insert)
    match map.entry(10) {
        Occupied(_) => unreachable!(),
        Vacant(view) => {
            assert_eq!(*view.insert(1000), 1000);
        }
    }
    assert_eq!(map.get(&10).unwrap(), Arc::new(1000));
    assert_eq!(map.len(), 6);
}

#[test]
fn test_capacity_not_less_than_len() {
    let a = CowHashMap::new();
    let mut item = 0;

    for _ in 0..116 {
        a.insert(item, 0);
        item += 1;
    }

    assert!(a.capacity() > a.len());

    let free = a.capacity() - a.len();
    for _ in 0..free {
        a.insert(item, 0);
        item += 1;
    }

    assert_eq!(a.len(), a.capacity());

    // Insert at capacity should cause allocation.
    a.insert(item, 0);
    assert!(a.capacity() > a.len());
}

#[test]
fn test_occupied_entry_key() {
    let a = CowHashMap::new();
    let key = "hello there";
    let value = "value goes here";
    assert!(a.is_empty());
    a.insert(key, value);
    assert_eq!(a.len(), 1);
    assert_eq!(a.get(key).unwrap(), Arc::new(value));

    match a.entry(key) {
        Vacant(_) => panic!(),
        Occupied(e) => assert_eq!(key, *e.key()),
    }
    assert_eq!(a.len(), 1);
    assert_eq!(a.get(key).unwrap(), Arc::new(value));
}

#[test]
fn test_vacant_entry_key() {
    let a: CowHashMap<&str, &str> = CowHashMap::new();
    let key = "hello there";
    let value = "value goes here";

    assert!(a.is_empty());
    match a.entry(key) {
        Occupied(_) => panic!(),
        Vacant(e) => {
            assert_eq!(key, *e.key());
            e.insert(value);
        }
    }
    assert_eq!(a.len(), 1);
    assert_eq!(a.get(key).unwrap(), Arc::new(value));
}

#[test]
fn test_retain() {
    let map: CowHashMap<i32, i32> = (0..100).map(|x| (x, x * 10)).collect();

    map.retain(|&k, _| k % 2 == 0);
    assert_eq!(map.len(), 50);
    assert_eq!(map.get(&2).unwrap(), Arc::new(20));
    assert_eq!(map.get(&4).unwrap(), Arc::new(40));
    assert_eq!(map.get(&6).unwrap(), Arc::new(60));
}

#[test]
fn from_array() {
    let map = CowHashMap::from([(1, 2), (3, 4)]);
    let unordered_duplicates = CowHashMap::from([(3, 4), (1, 2), (1, 2)]);
    assert_eq!(map, unordered_duplicates);

    // This next line must infer the hasher type parameter.
    // If you make a change that causes this line to no longer infer,
    // that's a problem!
    let _must_not_require_type_annotation = CowHashMap::from([(1, 2)]);
}
