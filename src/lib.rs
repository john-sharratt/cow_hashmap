#[cfg(test)]
mod tests;

use self::Entry::*;
use cow_hashbrown::{self as base, Equivalent};

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::error::Error;
use std::fmt::{self, Debug};
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::FusedIterator;
use std::sync::{Arc, Mutex, MutexGuard};

pub use cow_hashbrown::hash_map::CowValueGuard;

// The first shard level size will infrequently copy itself
// after the hashmap has been populated with a decent amount of elements.
const DEFAULT_SHARD_LEVEL1_SIZE: u64 = 256;
// The second shard level will more frequently copy itself
// during insert operations given a hash collision is much
// less likely however on very large populations the copy
// rate will drop off
const DEFAULT_SHARD_LEVEL2_SIZE: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ShardIndex {
    hash: u16,
}

fn key_shards<K: Hash + ?Sized, S: BuildHasher, const N1: u64, const N2: u64>(
    key: &K,
    hash_builder: &S,
) -> (ShardIndex, ShardIndex) {
    let mut state = hash_builder.build_hasher();
    key.hash(&mut state);
    let hash1 = state.finish();

    key.hash(&mut state);
    let hash2 = state.finish();

    (
        ShardIndex {
            hash: (hash1 % DEFAULT_SHARD_LEVEL1_SIZE) as u16,
        },
        ShardIndex {
            hash: (hash2 % DEFAULT_SHARD_LEVEL2_SIZE) as u16,
        },
    )
}

type ShardMap<K, V, S> = base::CowHashMap<ShardIndex, base::CowHashMap<K, V, S>, S>;

/// A [hash map] implemented with quadratic probing and SIMD lookup.
///
/// By default, `HashMap` uses a hashing algorithm selected to provide
/// resistance against HashDoS attacks. The algorithm is randomly seeded, and a
/// reasonable best-effort is made to generate this seed from a high quality,
/// secure source of randomness provided by the host without blocking the
/// program. Because of this, the randomness of the seed depends on the output
/// quality of the system's random number coroutine when the seed is created.
/// In particular, seeds generated when the system's entropy pool is abnormally
/// low such as during system boot may be of a lower quality.
///
/// The default hashing algorithm is currently SipHash 1-3, though this is
/// subject to change at any point in the future. While its performance is very
/// competitive for medium sized keys, other hashing algorithms will outperform
/// it for small keys such as integers as well as large keys such as long
/// strings, though those algorithms will typically *not* protect against
/// attacks such as HashDoS.
///
/// The hashing algorithm can be replaced on a per-`HashMap` basis using the
/// [`default`], [`with_hasher`], and [`with_capacity_and_hasher`] methods.
/// There are many alternative [hashing algorithms available on crates.io].
///
/// It is required that the keys implement the [`Eq`] and [`Hash`] traits, although
/// this can frequently be achieved by using `#[derive(PartialEq, Eq, Hash)]`.
/// If you implement these yourself, it is important that the following
/// property holds:
///
/// ```text
/// k1 == k2 -> hash(k1) == hash(k2)
/// ```
///
/// In other words, if two keys are equal, their hashes must be equal.
/// Violating this property is a logic error.
///
/// It is also a logic error for a key to be modified in such a way that the key's
/// hash, as determined by the [`Hash`] trait, or its equality, as determined by
/// the [`Eq`] trait, changes while it is in the map. This is normally only
/// possible through [`Cell`], [`RefCell`], global state, I/O, or unsafe code.
///
/// The behavior resulting from either logic error is not specified, but will
/// be encapsulated to the `HashMap` that observed the logic error and not
/// result in undefined behavior. This could include panics, incorrect results,
/// aborts, memory leaks, and non-termination.
///
/// The hash table implementation is a Rust port of Google's [SwissTable].
/// The original C++ version of SwissTable can be found [here], and this
/// [CppCon talk] gives an overview of how the algorithm works.
///
/// [hash map]: crate::collections#use-a-hashmap-when
/// [hashing algorithms available on crates.io]: https://crates.io/keywords/hasher
/// [SwissTable]: https://abseil.io/blog/20180927-swisstables
/// [here]: https://github.com/abseil/abseil-cpp/blob/master/absl/container/internal/raw_hash_set.h
/// [CppCon talk]: https://www.youtube.com/watch?v=ncHmEUmJZf4
///
/// # Examples
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// // Type inference lets us omit an explicit type signature (which
/// // would be `HashMap<String, String>` in this example).
/// let mut book_reviews = HashMap::new();
///
/// // Review some books.
/// book_reviews.insert(
///     "Adventures of Huckleberry Finn".to_string(),
///     "My favorite book.".to_string(),
/// );
/// book_reviews.insert(
///     "Grimms' Fairy Tales".to_string(),
///     "Masterpiece.".to_string(),
/// );
/// book_reviews.insert(
///     "Pride and Prejudice".to_string(),
///     "Very enjoyable.".to_string(),
/// );
/// book_reviews.insert(
///     "The Adventures of Sherlock Holmes".to_string(),
///     "Eye lyked it alot.".to_string(),
/// );
///
/// // Check for a specific one.
/// // When collections store owned values (String), they can still be
/// // queried using references (&str).
/// if !book_reviews.contains_key("Les Misérables") {
///     println!("We've got {} reviews, but Les Misérables ain't one.",
///              book_reviews.len());
/// }
///
/// // oops, this review has a lot of spelling mistakes, let's delete it.
/// book_reviews.remove("The Adventures of Sherlock Holmes");
///
/// // Look up the values associated with some keys.
/// let to_find = ["Pride and Prejudice", "Alice's Adventure in Wonderland"];
/// for &book in &to_find {
///     match book_reviews.get(book) {
///         Some(review) => println!("{book}: {review}"),
///         None => println!("{book} is unreviewed.")
///     }
/// }
///
/// // Look up the value for a key (will panic if the key is not found).
/// println!("Review for Jane: {}", book_reviews.get("Pride and Prejudice").unwrap());
///
/// // Iterate over everything.
/// for (book, review) in &book_reviews {
///     println!("{book}: \"{review}\"");
/// }
/// ```
///
/// A `HashMap` with a known list of items can be initialized from an array:
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// let solar_distance = HashMap::from([
///     ("Mercury", 0.4),
///     ("Venus", 0.7),
///     ("Earth", 1.0),
///     ("Mars", 1.5),
/// ]);
/// ```
///
/// `HashMap` implements an [`Entry` API](#method.entry), which allows
/// for complex methods of getting, setting, updating and removing keys and
/// their values:
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// // type inference lets us omit an explicit type signature (which
/// // would be `HashMap<&str, u8>` in this example).
/// let mut player_stats = HashMap::new();
///
/// fn random_stat_buff() -> u8 {
///     // could actually return some random value here - let's just return
///     // some fixed value for now
///     42
/// }
///
/// // insert a key only if it doesn't already exist
/// player_stats.entry("health").or_insert(100);
///
/// // insert a key using a function that provides a new value only if it
/// // doesn't already exist
/// player_stats.entry("defence").or_insert_with(random_stat_buff);
///
/// // update a key, guarding against the key possibly not being set
/// let mut stat = player_stats.entry("attack").or_insert(100);
/// *stat += random_stat_buff();
///
/// // modify an entry before an insert with in-place mutation
/// player_stats.entry("mana").and_modify(|mut mana| *mana += 200).or_insert(100);
/// ```
///
/// The easiest way to use `HashMap` with a custom key type is to derive [`Eq`] and [`Hash`].
/// We must also derive [`PartialEq`].
///
/// [`RefCell`]: crate::cell::RefCell
/// [`Cell`]: crate::cell::Cell
/// [`default`]: Default::default
/// [`with_hasher`]: Self::with_hasher
/// [`with_capacity_and_hasher`]: Self::with_capacity_and_hasher
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// #[derive(Clone, Hash, Eq, PartialEq, Debug)]
/// struct Viking {
///     name: String,
///     country: String,
/// }
///
/// impl Viking {
///     /// Creates a new Viking.
///     fn new(name: &str, country: &str) -> Viking {
///         Viking { name: name.to_string(), country: country.to_string() }
///     }
/// }
///
/// // Use a HashMap to store the vikings' health points.
/// let vikings = HashMap::from([
///     (Viking::new("Einar", "Norway"), 25),
///     (Viking::new("Olaf", "Denmark"), 24),
///     (Viking::new("Harald", "Iceland"), 12),
/// ]);
///
/// // Use derived implementation to print the status of the vikings.
/// for (viking, health) in &vikings {
///     println!("{viking:?} has {health} hp");
/// }
/// ```

pub struct CowHashMap<K: Clone, V, S = RandomState> {
    base: base::CowHashMap<ShardIndex, ShardMap<K, V, S>, S>,
    lock: Arc<Mutex<()>>,
}

impl<K: Clone, V> CowHashMap<K, V, RandomState> {
    /// Creates an empty `HashMap`.
    ///
    /// The hash map is initially created with a capacity of 0, so it will not allocate until it
    /// is first inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// let mut map: HashMap<&str, i32> = HashMap::new();
    /// ```
    #[inline]
    #[must_use]
    pub fn new() -> CowHashMap<K, V, RandomState> {
        Default::default()
    }
}

impl<K: Clone, V, S> CowHashMap<K, V, S> {
    /// Creates an empty `HashMap` which will use the given hash builder to hash
    /// keys.
    ///
    /// The created map has the default initial capacity.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for
    /// the HashMap to be useful, see its documentation for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::hash::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut map = HashMap::with_hasher(s);
    /// map.insert(1, 2);
    /// ```
    #[inline]
    pub fn with_hasher(hash_builder: S) -> CowHashMap<K, V, S> {
        CowHashMap {
            base: base::CowHashMap::with_hasher(hash_builder),
            lock: Default::default(),
        }
    }

    /// An iterator visiting all keys in arbitrary order.
    /// The iterator element type is `&'a K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let map = HashMap::from([
    ///     ("a", 1),
    ///     ("b", 2),
    ///     ("c", 3),
    /// ]);
    ///
    /// for key in map.keys() {
    ///     println!("{key}");
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// In the current implementation, iterating over keys takes O(capacity) time
    /// instead of O(len) because it internally visits empty buckets too.
    pub fn keys(&self) -> Keys<K, V, S> {
        Keys { inner: self.iter() }
    }

    /// Creates a consuming iterator visiting all the keys in arbitrary order.
    /// The map cannot be used after calling this.
    /// The iterator element type is `K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let map = HashMap::from([
    ///     ("a", 1),
    ///     ("b", 2),
    ///     ("c", 3),
    /// ]);
    ///
    /// let mut vec: Vec<&str> = map.into_keys().collect();
    /// // The `IntoKeys` iterator produces keys in arbitrary order, so the
    /// // keys must be sorted to test them against a sorted array.
    /// vec.sort_unstable();
    /// assert_eq!(vec, ["a", "b", "c"]);
    /// ```
    ///
    /// # Performance
    ///
    /// In the current implementation, iterating over keys takes O(capacity) time
    /// instead of O(len) because it internally visits empty buckets too.
    #[inline]
    pub fn into_keys(self) -> IntoKeys<K, V, S>
    where
        S: Clone,
    {
        IntoKeys {
            inner: self.into_iter(),
        }
    }

    /// An iterator visiting all values in arbitrary order.
    /// The iterator element type is `&'a V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let map = HashMap::from([
    ///     ("a", 1),
    ///     ("b", 2),
    ///     ("c", 3),
    /// ]);
    ///
    /// for val in map.values() {
    ///     println!("{val}");
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// In the current implementation, iterating over values takes O(capacity) time
    /// instead of O(len) because it internally visits empty buckets too.
    pub fn values(&self) -> Values<K, V, S> {
        Values { inner: self.iter() }
    }

    /// An iterator visiting all values mutably in arbitrary order.
    /// The iterator element type is `&'a mut V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map = HashMap::from([
    ///     ("a", 1),
    ///     ("b", 2),
    ///     ("c", 3),
    /// ]);
    ///
    /// for mut val in map.values_mut() {
    ///     *val = *val + 10;
    /// }
    ///
    /// for val in map.values() {
    ///     println!("{val}");
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// In the current implementation, iterating over values takes O(capacity) time
    /// instead of O(len) because it internally visits empty buckets too.
    pub fn values_mut(&self) -> ValuesMut<K, V, S> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }

    /// Creates a consuming iterator visiting all the values in arbitrary order.
    /// The map cannot be used after calling this.
    /// The iterator element type is `V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let map = HashMap::from([
    ///     ("a", 1),
    ///     ("b", 2),
    ///     ("c", 3),
    /// ]);
    ///
    /// let mut vec: Vec<Arc<i32>> = map.into_values().collect();
    /// // The `IntoValues` iterator produces values in arbitrary order, so
    /// // the values must be sorted to test them against a sorted array.
    /// vec.sort_unstable();
    /// assert_eq!(vec, [Arc::new(1), Arc::new(2), Arc::new(3)]);
    /// ```
    ///
    /// # Performance
    ///
    /// In the current implementation, iterating over values takes O(capacity) time
    /// instead of O(len) because it internally visits empty buckets too.
    #[inline]
    pub fn into_values(self) -> IntoValues<K, V, S>
    where
        S: Clone,
    {
        IntoValues {
            inner: self.into_iter(),
        }
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// The iterator element type is `(&'a K, &'a V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let map = HashMap::from([
    ///     ("a", 1),
    ///     ("b", 2),
    ///     ("c", 3),
    /// ]);
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {key} val: {val}");
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// In the current implementation, iterating over map takes O(capacity) time
    /// instead of O(len) because it internally visits empty buckets too.
    pub fn iter(&self) -> Iter<K, V, S> {
        Iter {
            base: self.base.iter(),
            shard1: None,
            shard2: None,
        }
    }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    /// The iterator element type is `(&'a K, &'a mut V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map = HashMap::from([
    ///     ("a", 1),
    ///     ("b", 2),
    ///     ("c", 3),
    /// ]);
    ///
    /// // Update all values
    /// for (_, mut val) in map.iter_mut() {
    ///     *val *= 2;
    /// }
    ///
    /// for (key, val) in &map {
    ///     println!("key: {key} val: {val}");
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// In the current implementation, iterating over map takes O(capacity) time
    /// instead of O(len) because it internally visits empty buckets too.
    pub fn iter_mut(&self) -> IterMut<K, V, S> {
        IterMut {
            base: self.base.iter(),
            shard1: None,
            shard2: None,
        }
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut a = HashMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.base
            .iter()
            .map(|(_, t)| t.iter().map(|(_, t)| t.len()).sum::<usize>())
            .sum()
    }

    /// Returns `true` if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut a = HashMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        if self.base.is_empty() {
            return true;
        }
        self.base.iter().all(|(_, t)| t.is_empty())
    }

    /// Clears the map, returning all key-value pairs as an iterator. Keeps the
    /// allocated memory for reuse.
    ///
    /// If the returned iterator is dropped before being fully consumed, it
    /// drops the remaining key-value pairs. The returned iterator keeps a
    /// mutable borrow on the map to optimize its implementation.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut a = HashMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    ///
    /// for (k, v) in a.drain().take(1) {
    ///     assert!(k == 1 || k == 2);
    ///     assert!(v == Arc::new("a") || v == Arc::new("b"));
    /// }
    ///
    /// assert!(a.is_empty());
    /// ```
    #[inline]
    pub fn drain(&self) -> Drain<K, V, S> {
        let ret = Drain {
            base: self.base.drain(),
            shard1: None,
            shard2: None,
        };
        ret
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all pairs `(k, v)` for which `f(&k, &mut v)` returns `false`.
    /// The elements are visited in unsorted (and unspecified) order.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map: HashMap<i32, i32> = (0..8).map(|x| (x, x*10)).collect();
    /// map.retain(|&k, _| k % 2 == 0);
    /// assert_eq!(map.len(), 4);
    /// ```
    ///
    /// # Performance
    ///
    /// In the current implementation, this operation takes O(capacity) time
    /// instead of O(len) because it internally visits empty buckets too.
    #[inline]
    pub fn retain<F>(&self, mut f: F)
    where
        F: FnMut(&K, &V) -> bool,
        V: Clone,
        S: Clone,
    {
        let mut f = move |k: &K, v: &V| -> bool { f(k, v) };
        for shard1 in self.base.values() {
            for shard2 in shard1.values() {
                shard2.retain(&mut f);
            }
            shard1.retain(|_, t| !t.is_empty());
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all pairs `(k, v)` for which `f(&k, &mut v)` returns `false`.
    /// The elements are visited in unsorted (and unspecified) order.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map: HashMap<i32, i32> = (0..8).map(|x| (x, x*10)).collect();
    /// map.retain(|&k, _| k % 2 == 0);
    /// assert_eq!(map.len(), 4);
    /// ```
    ///
    /// # Performance
    ///
    /// In the current implementation, this operation takes O(capacity) time
    /// instead of O(len) because it internally visits empty buckets too.
    #[inline]
    pub fn retain_mut<F>(&self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
        V: Clone,
        S: Clone,
    {
        let mut f = move |k: &K, v: &mut V| -> bool { f(k, v) };
        for shard1 in self.base.values() {
            for shard2 in shard1.values() {
                shard2.retain_mut(&mut f);
            }
            shard1.retain(|_, t| !t.is_empty());
        }
    }

    /// Clears the map, removing all key-value pairs. Keeps the allocated memory
    /// for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut a = HashMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    #[inline]
    pub fn clear(&self) {
        self.base.clear();
    }

    /// Returns a reference to the map's [`BuildHasher`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::hash::RandomState;
    ///
    /// let hasher = RandomState::new();
    /// let map: HashMap<i32, i32> = HashMap::with_hasher(hasher);
    /// let hasher: &RandomState = map.hasher();
    /// ```
    #[inline]
    pub fn hasher(&self) -> &S {
        self.base.hasher()
    }
}

impl<K: Clone, V, S> CowHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Grabs the shard that is relevant for this particular key
    fn shard<Q: ?Sized>(&self, key: &Q) -> Option<Arc<base::CowHashMap<K, V, S>>>
    where
        Q: Hash + Eq,
    {
        let (shard1, shard2) =
            key_shards::<Q, S, DEFAULT_SHARD_LEVEL1_SIZE, DEFAULT_SHARD_LEVEL2_SIZE>(
                key,
                self.base.hasher(),
            );
        self.base.get(&shard1)?.get(&shard2)
    }

    fn shard_and_parent<Q: ?Sized>(
        &self,
        key: &Q,
    ) -> Option<(
        Arc<base::CowHashMap<K, V, S>>,
        Arc<cow_hashbrown::CowHashMap<ShardIndex, cow_hashbrown::CowHashMap<K, V, S>, S>>,
    )>
    where
        Q: Hash + Eq,
    {
        let (shard1, shard2) =
            key_shards::<Q, S, DEFAULT_SHARD_LEVEL1_SIZE, DEFAULT_SHARD_LEVEL2_SIZE>(
                key,
                self.base.hasher(),
            );
        let parent = self.base.get(&shard1)?;

        let ret = parent.get(&shard2)?;

        Some((ret, parent))
    }

    /// Grabs the shard that is relevant for this particular key
    fn shard_mut<Q: ?Sized>(&self, key: &Q) -> CowValueGuard<base::CowHashMap<K, V, S>>
    where
        Q: Hash + Eq,
        S: Clone,
    {
        let (shard1, shard2) =
            key_shards::<Q, S, DEFAULT_SHARD_LEVEL1_SIZE, DEFAULT_SHARD_LEVEL2_SIZE>(
                key,
                self.base.hasher(),
            );

        self.base
            .entry(shard1)
            .or_insert_with_mut(|| base::CowHashMap::with_hasher(self.base.hasher().clone()))
            .entry(shard2)
            .or_insert_with_mut(|| base::CowHashMap::with_hasher(self.base.hasher().clone()))
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut letters = HashMap::new();
    ///
    /// for ch in "a short treatise on fungi".chars() {
    ///     letters.entry(ch).and_modify(|mut counter| *counter += 1).or_insert(1);
    /// }
    ///
    /// assert_eq!(letters.get(&'s').unwrap(), Arc::new(2));
    /// assert_eq!(letters.get(&'t').unwrap(), Arc::new(3));
    /// assert_eq!(letters.get(&'u').unwrap(), Arc::new(1));
    /// assert_eq!(letters.get(&'y'), None);
    /// ```
    #[inline]
    pub fn entry(&self, key: K) -> Entry<K, V, S>
    where
        S: Clone,
    {
        map_entry(self.shard_mut(&key).entry(key))
    }

    /// Locks the hashmap for exclusive access (all other accessors
    /// must also use this lock method otherwise they can still
    /// access the hashmap)
    pub fn lock<'a>(&'a self) -> MutexGuard<'a, ()> {
        self.lock.lock().unwrap()
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    /// If the entry does not exist then the act of creating the entry will be done
    /// under a lock to prevent concurrent inserts
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut letters = HashMap::new();
    ///
    /// for ch in "a short treatise on fungi".chars() {
    ///     letters.entry(ch).and_modify(|mut counter| *counter += 1).or_insert(1);
    /// }
    ///
    /// assert_eq!(letters.get(&'s').unwrap(), Arc::new(2));
    /// assert_eq!(letters.get(&'t').unwrap(), Arc::new(3));
    /// assert_eq!(letters.get(&'u').unwrap(), Arc::new(1));
    /// assert_eq!(letters.get(&'y'), None);
    /// ```
    #[inline]
    pub fn entry_partial_lock(&self, key: K) -> LockableEntry<K, V, S>
    where
        S: Clone,
    {
        LockableEntry(map_entry(self.shard_mut(&key).entry(key)), &self.lock)
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(Arc::new("a")));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[inline]
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<Arc<V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.shard(k)?.get(k)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// The supplied key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get_key_value(&1), Some((1, Arc::new("a"))));
    /// assert_eq!(map.get_key_value(&2), None);
    /// ```
    #[inline]
    pub fn get_key_value<Q: ?Sized>(&self, k: &Q) -> Option<(K, Arc<V>)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.shard(k)?.get_key_value(k)
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let map = self.shard(k);
        match map {
            Some(map) => map.contains_key(k),
            None => false,
        }
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// if let Some(mut x) = map.get_mut(&1) {
    ///     *x = "b";
    /// }
    /// assert_eq!(map.get(&1).unwrap(), Arc::new("b"));
    /// ```
    #[inline]
    pub fn get_mut<Q: ?Sized>(&self, k: &Q) -> Option<CowValueGuard<V>>
    where
        S: Clone,
        K: Borrow<Q>,
        Q: Hash + Eq,
        V: Clone,
    {
        self.shard_mut(k).get_mut(k)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the [module-level
    /// documentation] for more.
    ///
    /// [module-level documentation]: crate::collections#insert-and-complex-keys
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(*map.insert(37, "c").unwrap(), "b");
    /// assert_eq!(*map.get(&37).unwrap(), "c");
    /// ```
    #[inline]
    pub fn insert(&self, k: K, v: V) -> Option<Arc<V>>
    where
        S: Clone,
    {
        self.shard_mut(&k).insert(k, v)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the [module-level
    /// documentation] for more.
    ///
    /// [module-level documentation]: crate::collections#insert-and-complex-keys
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(*map.insert(37, "c").unwrap(), "b");
    /// assert_eq!(*map.get(&37).unwrap(), "c");
    /// ```
    #[inline]
    pub fn insert_mut(&self, k: K, v: V) -> Option<CowValueGuard<V>>
    where
        S: Clone,
        V: Clone,
    {
        self.shard_mut(&k).insert_mut(k, v)
    }

    /// Tries to insert a key-value pair into the map, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// If the map already had this key present, nothing is updated, and
    /// an error containing the occupied entry and the value is returned.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(**map.try_insert(37, "a").unwrap(), *"a");
    ///
    /// let err = map.try_insert(37, "b").unwrap_err();
    /// assert_eq!(err.entry.key(), &37);
    /// assert_eq!(err.entry.get(), Arc::new("a"));
    /// assert_eq!(err.value, "b");
    /// ```
    pub fn try_insert(&self, key: K, value: V) -> Result<Arc<V>, OccupiedError<K, V, S>>
    where
        S: Clone,
    {
        match self.shard_mut(&key).entry(key) {
            cow_hashbrown::hash_map::Entry::Occupied(entry) => Err(OccupiedError {
                entry: OccupiedEntry { base: entry },
                value,
            }),
            cow_hashbrown::hash_map::Entry::Vacant(entry) => Ok(entry.insert(value)),
        }
    }

    /// Tries to insert a key-value pair into the map, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// If the map already had this key present, nothing is updated, and
    /// an error containing the occupied entry and the value is returned.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(**map.try_insert(37, "a").unwrap(), *"a");
    ///
    /// let err = map.try_insert(37, "b").unwrap_err();
    /// assert_eq!(err.entry.key(), &37);
    /// assert_eq!(err.entry.get(), Arc::new("a"));
    /// assert_eq!(err.value, "b");
    /// ```
    pub fn try_insert_mut(
        &self,
        key: K,
        value: V,
    ) -> Result<CowValueGuard<V>, OccupiedError<K, V, S>>
    where
        S: Clone,
        V: Clone,
    {
        match self.shard_mut(&key).entry(key) {
            cow_hashbrown::hash_map::Entry::Occupied(entry) => Err(OccupiedError {
                entry: OccupiedEntry { base: entry },
                value,
            }),
            cow_hashbrown::hash_map::Entry::Vacant(entry) => Ok(entry.insert_mut(value)),
        }
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some(Arc::new("a")));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[inline]
    pub fn remove<Q: ?Sized>(&self, k: &Q) -> Option<Arc<V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
        S: Clone,
    {
        let (shard, parent) = self.shard_and_parent(k)?;
        let ret = shard.remove(k);

        if shard.is_empty() {
            parent.retain(|_, t| !t.is_empty());
        }

        ret
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// # fn main() {
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove_entry(&1), Some((1, Arc::new("a"))));
    /// assert_eq!(map.remove(&1), None);
    /// # }
    /// ```
    #[inline]
    pub fn remove_entry<Q: ?Sized>(&self, k: &Q) -> Option<(K, Arc<V>)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
        S: Clone,
    {
        let (shard, parent) = self.shard_and_parent(k)?;

        let ret = shard.remove_entry(k);
        if shard.is_empty() {
            parent.retain(|_, t| !t.is_empty());
        }

        ret
    }
}

impl<K, V, S> Clone for CowHashMap<K, V, S>
where
    K: Clone,
    V: Clone,
    S: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            lock: self.lock.clone(),
        }
    }

    #[inline]
    fn clone_from(&mut self, other: &Self) {
        self.base.clone_from(&other.base);
        self.lock.clone_from(&other.lock);
    }
}

impl<K: Clone, V, S> PartialEq for CowHashMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &CowHashMap<K, V, S>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter()
            .all(|(key, value)| other.get(&key).map_or(false, |v| *value == *v))
    }
}

impl<K: Clone, V, S> Eq for CowHashMap<K, V, S>
where
    K: Eq + Hash,
    V: Eq,
    S: BuildHasher,
{
}

impl<K: Clone, V, S> Debug for CowHashMap<K, V, S>
where
    K: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Clone, V, S> Default for CowHashMap<K, V, S>
where
    S: Default,
{
    /// Creates an empty `HashMap<K, V, S>`, with the `Default` value for the hasher.
    #[inline]
    fn default() -> CowHashMap<K, V, S> {
        CowHashMap::<K, V, S>::with_hasher(Default::default())
    }
}

// Note: as what is currently the most convenient built-in way to construct
// a HashMap, a simple usage of this function must not *require* the user
// to provide a type annotation in order to infer the third type parameter
// (the hasher parameter, conventionally "S").
// To that end, this impl is defined using RandomState as the concrete
// type of S, rather than being generic over `S: BuildHasher + Default`.
// If type parameter defaults worked on impls, and if type parameter
// defaults could be mixed with const generics, then perhaps
// this could be generalized.
// See also the equivalent impl on HashSet.
impl<K: Clone, V, const N: usize> From<[(K, V); N]> for CowHashMap<K, V, RandomState>
where
    K: Eq + Hash,
{
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let map1 = HashMap::from([(1, 2), (3, 4)]);
    /// let map2: HashMap<_, _> = [(1, 2), (3, 4)].into();
    /// assert_eq!(map1, map2);
    /// ```
    fn from(arr: [(K, V); N]) -> Self {
        Self::from_iter(arr)
    }
}

/// An iterator over the entries of a `HashMap`.
///
/// This `struct` is created by the [`iter`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`iter`]: HashMap::iter
///
/// # Example
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// let map = HashMap::from([
///     ("a", 1),
/// ]);
/// let iter = map.iter();
/// ```
pub struct Iter<K: Clone, V, S> {
    base: cow_hashbrown::hash_map::Iter<ShardIndex, ShardMap<K, V, S>>,
    shard1: Option<cow_hashbrown::hash_map::Iter<ShardIndex, base::CowHashMap<K, V, S>>>,
    shard2: Option<cow_hashbrown::hash_map::Iter<K, V>>,
}

impl<K: Clone, V, S> Clone for Iter<K, V, S> {
    #[inline]
    fn clone(&self) -> Self {
        Iter {
            base: self.base.clone(),
            shard1: None,
            shard2: None,
        }
    }
}

impl<K: Clone + Debug, V: Debug, S> fmt::Debug for Iter<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// A mutable iterator over the entries of a `HashMap`.
///
/// This `struct` is created by the [`iter_mut`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`iter_mut`]: HashMap::iter_mut
///
/// # Example
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// let mut map = HashMap::from([
///     ("a", 1),
/// ]);
/// let iter = map.iter_mut();
/// ```
pub struct IterMut<K: Clone, V, S> {
    base: cow_hashbrown::hash_map::Iter<ShardIndex, ShardMap<K, V, S>>,
    shard1: Option<cow_hashbrown::hash_map::Iter<ShardIndex, base::CowHashMap<K, V, S>>>,
    shard2: Option<cow_hashbrown::hash_map::IterMut<K, V>>,
}

impl<K: Clone, V, S> IterMut<K, V, S> {
    /// Returns an iterator of references over the remaining items.
    #[inline]
    pub fn iter(&self) -> Iter<K, V, S> {
        Iter {
            base: self.base.clone(),
            shard1: None,
            shard2: None,
        }
    }
}

/// An owning iterator over the entries of a `HashMap`.
///
/// This `struct` is created by the [`into_iter`] method on [`HashMap`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: IntoIterator::into_iter
///
/// # Example
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// let map = HashMap::from([
///     ("a", 1),
/// ]);
/// let iter = map.into_iter();
/// ```
pub struct IntoIter<K: Clone, V, S> {
    base: cow_hashbrown::hash_map::Iter<ShardIndex, ShardMap<K, V, S>>,
    shard1: Option<cow_hashbrown::hash_map::Iter<ShardIndex, base::CowHashMap<K, V, S>>>,
    shard2: Option<cow_hashbrown::hash_map::Iter<K, V>>,
}

impl<K: Clone, V, S> IntoIter<K, V, S> {
    /// Returns an iterator of references over the remaining items.
    #[inline]
    pub fn iter(&self) -> Iter<K, V, S> {
        Iter {
            base: self.base.clone(),
            shard1: None,
            shard2: None,
        }
    }
}

/// An iterator over the keys of a `HashMap`.
///
/// This `struct` is created by the [`keys`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`keys`]: HashMap::keys
///
/// # Example
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// let map = HashMap::from([
///     ("a", 1),
/// ]);
/// let iter_keys = map.keys();
/// ```
pub struct Keys<K: Clone, V, S> {
    inner: Iter<K, V, S>,
}

impl<K: Clone, V, S> Clone for Keys<K, V, S> {
    #[inline]
    fn clone(&self) -> Self {
        Keys {
            inner: self.inner.clone(),
        }
    }
}

impl<K: Clone + Debug, V, S> fmt::Debug for Keys<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// An iterator over the values of a `HashMap`.
///
/// This `struct` is created by the [`values`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`values`]: HashMap::values
///
/// # Example
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// let map = HashMap::from([
///     ("a", 1),
/// ]);
/// let iter_values = map.values();
/// ```
pub struct Values<K: Clone, V, S> {
    inner: Iter<K, V, S>,
}

impl<K: Clone, V, S> Clone for Values<K, V, S> {
    #[inline]
    fn clone(&self) -> Self {
        Values {
            inner: self.inner.clone(),
        }
    }
}

impl<K: Clone, V: Debug, S> fmt::Debug for Values<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// A draining iterator over the entries of a `HashMap`.
///
/// This `struct` is created by the [`drain`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`drain`]: HashMap::drain
///
/// # Example
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// let mut map = HashMap::from([
///     ("a", 1),
/// ]);
/// let iter = map.drain();
/// ```
pub struct Drain<K: Clone, V, S> {
    base: cow_hashbrown::hash_map::Iter<ShardIndex, ShardMap<K, V, S>>,
    shard1: Option<cow_hashbrown::hash_map::Iter<ShardIndex, base::CowHashMap<K, V, S>>>,
    shard2: Option<cow_hashbrown::hash_map::Iter<K, V>>,
}

impl<'a, K: Clone, V, S> Drain<K, V, S> {
    /// Returns an iterator of references over the remaining items.
    #[inline]
    pub fn iter(&self) -> Iter<K, V, S> {
        Iter {
            base: self.base.clone(),
            shard1: None,
            shard2: None,
        }
    }
}

/// A mutable iterator over the values of a `HashMap`.
///
/// This `struct` is created by the [`values_mut`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`values_mut`]: HashMap::values_mut
///
/// # Example
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// let mut map = HashMap::from([
///     ("a", 1),
/// ]);
/// let iter_values = map.values_mut();
/// ```
pub struct ValuesMut<K: Clone, V, S> {
    inner: IterMut<K, V, S>,
}

/// An owning iterator over the keys of a `HashMap`.
///
/// This `struct` is created by the [`into_keys`] method on [`HashMap`].
/// See its documentation for more.
///
/// [`into_keys`]: HashMap::into_keys
///
/// # Example
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// let map = HashMap::from([
///     ("a", 1),
/// ]);
/// let iter_keys = map.into_keys();
/// ```
pub struct IntoKeys<K: Clone, V, S> {
    inner: IntoIter<K, V, S>,
}

/// An owning iterator over the values of a `HashMap`.
///
/// This `struct` is created by the [`into_values`] method on [`HashMap`].
/// See its documentation for more.
///
/// [`into_values`]: HashMap::into_values
///
/// # Example
///
/// ```
/// use cow_hashmap::CowHashMap as HashMap;
///
/// let map = HashMap::from([
///     ("a", 1),
/// ]);
/// let iter_keys = map.into_values();
/// ```
pub struct IntoValues<K: Clone, V, S> {
    inner: IntoIter<K, V, S>,
}

/// A view into a single entry in a map, which may either be vacant or occupied.
///
/// This `enum` is constructed from the [`entry`] method on [`HashMap`].
///
/// [`entry`]: HashMap::entry
pub enum Entry<K: Clone, V, S> {
    /// An occupied entry.
    Occupied(OccupiedEntry<K, V, S>),

    /// A vacant entry.
    Vacant(VacantEntry<K, V, S>),
}

impl<K: Clone, V, S> Entry<K, V, S> {
    fn is_occupied(&self) -> bool {
        matches!(self, Entry::Occupied(_))
    }
}

impl<K: Clone + Debug, V: Debug, S> Debug for Entry<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Vacant(ref v) => f.debug_tuple("Entry").field(v).finish(),
            Occupied(ref o) => f.debug_tuple("Entry").field(o).finish(),
        }
    }
}

/// A view into a single entry in a map, which may either be vacant or occupied.
///
/// This `enum` is constructed from the [`entry`] method on [`HashMap`].
///
/// [`entry`]: HashMap::entry
pub struct LockableEntry<'a, K: Clone, V, S>(Entry<K, V, S>, &'a Arc<Mutex<()>>);

impl<K: Clone + Debug, V: Debug, S> Debug for LockableEntry<'_, K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Vacant(ref v) => f.debug_tuple("Entry").field(v).finish(),
            Occupied(ref o) => f.debug_tuple("Entry").field(o).finish(),
        }
    }
}

/// A view into an occupied entry in a `HashMap`.
/// It is part of the [`Entry`] enum.
pub struct OccupiedEntry<K: Clone, V, S> {
    base: cow_hashbrown::hash_map::OccupiedEntry<K, V, S>,
}

impl<K: Clone + Debug, V: Debug, S> Debug for OccupiedEntry<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", self.key())
            .field("value", &self.get())
            .finish_non_exhaustive()
    }
}

/// A view into a vacant entry in a `HashMap`.
/// It is part of the [`Entry`] enum.
pub struct VacantEntry<K: Clone, V, S> {
    base: cow_hashbrown::hash_map::VacantEntry<K, V, S>,
}

impl<K: Clone + Debug, V, S> Debug for VacantEntry<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VacantEntry").field(self.key()).finish()
    }
}

/// The error returned by [`try_insert`](HashMap::try_insert) when the key already exists.
///
/// Contains the occupied entry, and the value that was not inserted.
pub struct OccupiedError<K: Clone, V, S> {
    /// The entry in the map that was already occupied.
    pub entry: OccupiedEntry<K, V, S>,
    /// The value which was not inserted, because the entry was already occupied.
    pub value: V,
}

impl<K: Clone + Debug, V: Debug, S> Debug for OccupiedError<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedError")
            .field("key", self.entry.key())
            .field("old_value", &self.entry.get())
            .field("new_value", &self.value)
            .finish_non_exhaustive()
    }
}

impl<K: Clone + Debug, V: Debug, S> fmt::Display for OccupiedError<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to insert {:?}, key {:?} already exists with value {:?}",
            self.value,
            self.entry.key(),
            self.entry.get(),
        )
    }
}

impl<K: Clone + fmt::Debug, V: fmt::Debug, S> Error for OccupiedError<K, V, S> {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "key already exists"
    }
}

impl<K: Clone, V, S> IntoIterator for &CowHashMap<K, V, S>
where
    S: Clone,
{
    type Item = (K, Arc<V>);
    type IntoIter = IntoIter<K, V, S>;

    #[inline]
    fn into_iter(self) -> IntoIter<K, V, S> {
        IntoIter {
            base: self.base.iter(),
            shard1: None,
            shard2: None,
        }
    }
}

impl<K: Clone, V, S> IntoIterator for &mut CowHashMap<K, V, S>
where
    S: Clone,
{
    type Item = (K, Arc<V>);
    type IntoIter = IntoIter<K, V, S>;

    #[inline]
    fn into_iter(self) -> IntoIter<K, V, S> {
        IntoIter {
            base: self.base.iter(),
            shard1: None,
            shard2: None,
        }
    }
}

impl<K: Clone, V, S> IntoIterator for CowHashMap<K, V, S>
where
    S: Clone,
{
    type Item = (K, Arc<V>);
    type IntoIter = IntoIter<K, V, S>;

    /// Creates a consuming iterator, that is, one that moves each key-value
    /// pair out of the map in arbitrary order. The map cannot be used after
    /// calling this.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let map = HashMap::from([
    ///     ("a", 1),
    ///     ("b", 2),
    ///     ("c", 3),
    /// ]);
    ///
    /// // Not possible with .iter()
    /// let vec: Vec<(&str, Arc<i32>)> = map.into_iter().collect();
    /// ```
    #[inline]
    fn into_iter(self) -> IntoIter<K, V, S> {
        IntoIter {
            base: self.base.iter(),
            shard1: None,
            shard2: None,
        }
    }
}

impl<K: Clone, V, S> Iterator for Iter<K, V, S> {
    type Item = (K, Arc<V>);

    #[inline]
    fn next(&mut self) -> Option<(K, Arc<V>)> {
        loop {
            if let Some(ref mut shard2) = self.shard2 {
                if let Some(r) = shard2.next() {
                    return Some(r);
                }
                self.shard2.take();
            }
            if let Some(ref mut shard1) = self.shard1 {
                if let Some(r) = shard1.next() {
                    self.shard2.replace(r.1.iter());
                    continue;
                }
                self.shard1.take();
            }
            if let Some(next) = self.base.next() {
                self.shard1.replace(next.1.iter());
                continue;
            }
            return None;
        }
    }
    #[inline]
    fn count(self) -> usize {
        self.base
            .clone()
            .map(|(_, t)| t.iter().map(|(_, t)| t.len()).sum::<usize>())
            .sum()
    }
}
impl<K: Clone, V, S> ExactSizeIterator for Iter<K, V, S> {
    #[inline]
    fn len(&self) -> usize {
        self.base
            .clone()
            .map(|(_, t)| t.iter().map(|(_, t)| t.len()).sum::<usize>())
            .sum()
    }
}

impl<K: Clone, V, S> FusedIterator for Iter<K, V, S> {}

impl<K: Clone, V, S> Iterator for IterMut<K, V, S>
where
    V: Clone,
{
    type Item = (K, CowValueGuard<V>);

    #[inline]
    fn next(&mut self) -> Option<(K, CowValueGuard<V>)> {
        loop {
            if let Some(ref mut shard2) = self.shard2 {
                if let Some(r) = shard2.next() {
                    return Some(r);
                }
                self.shard2.take();
            }
            if let Some(ref mut shard1) = self.shard1 {
                if let Some(r) = shard1.next() {
                    self.shard2.replace(r.1.iter_mut());
                    continue;
                }
                self.shard1.take();
            }
            if let Some(next) = self.base.next() {
                self.shard1.replace(next.1.iter());
                continue;
            }
            return None;
        }
    }
    #[inline]
    fn count(self) -> usize {
        self.base
            .clone()
            .map(|(_, t)| t.iter().map(|(_, t)| t.len()).sum::<usize>())
            .sum()
    }
}
impl<K: Clone, V, S> ExactSizeIterator for IterMut<K, V, S>
where
    V: Clone,
{
    #[inline]
    fn len(&self) -> usize {
        self.base
            .clone()
            .map(|(_, t)| t.iter().map(|(_, t)| t.len()).sum::<usize>())
            .sum()
    }
}
impl<K: Clone, V, S> FusedIterator for IterMut<K, V, S> where V: Clone {}

impl<K: Clone, V, S> fmt::Debug for IterMut<K, V, S>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<K: Clone, V, S> Iterator for IntoIter<K, V, S>
where
    S: Clone,
{
    type Item = (K, Arc<V>);

    #[inline]
    fn next(&mut self) -> Option<(K, Arc<V>)> {
        loop {
            if let Some(ref mut shard2) = self.shard2 {
                if let Some(r) = shard2.next() {
                    return Some(r);
                }
                self.shard2.take();
            }
            if let Some(ref mut shard1) = self.shard1 {
                if let Some(r) = shard1.next() {
                    self.shard2.replace(r.1.iter());
                    continue;
                }
                self.shard1.take();
            }
            if let Some(next) = self.base.next() {
                self.shard1.replace(next.1.iter());
                continue;
            }
            return None;
        }
    }
    #[inline]
    fn count(self) -> usize {
        self.base
            .clone()
            .map(|(_, t)| t.iter().map(|(_, t)| t.len()).sum::<usize>())
            .sum()
    }
}
impl<K: Clone, V, S> ExactSizeIterator for IntoIter<K, V, S>
where
    S: Clone,
{
    #[inline]
    fn len(&self) -> usize {
        self.base
            .clone()
            .map(|(_, t)| t.iter().map(|(_, t)| t.len()).sum::<usize>())
            .sum()
    }
}
impl<K: Clone, V, S> FusedIterator for IntoIter<K, V, S> where S: Clone {}

impl<K: Clone + Debug, V: Debug, S> fmt::Debug for IntoIter<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<K: Clone, V, S> Iterator for Keys<K, V, S> {
    type Item = K;

    #[inline]
    fn next(&mut self) -> Option<K> {
        self.inner.next().map(|(k, _)| k)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    #[inline]
    fn count(self) -> usize {
        self.inner.len()
    }
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.inner.fold(init, |acc, (k, _)| f(acc, k))
    }
}
impl<K: Clone, V, S> ExactSizeIterator for Keys<K, V, S> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<K: Clone, V, S> FusedIterator for Keys<K, V, S> {}

impl<K: Clone, V, S> Iterator for Values<K, V, S> {
    type Item = Arc<V>;

    #[inline]
    fn next(&mut self) -> Option<Arc<V>> {
        self.inner.next().map(|(_, v)| v)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    #[inline]
    fn count(self) -> usize {
        self.inner.len()
    }
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.inner.fold(init, |acc, (_, v)| f(acc, v))
    }
}
impl<K: Clone, V, S> ExactSizeIterator for Values<K, V, S> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<K: Clone, V, S> FusedIterator for Values<K, V, S> {}

impl<K: Clone, V, S> Iterator for ValuesMut<K, V, S>
where
    V: Clone,
{
    type Item = CowValueGuard<V>;

    #[inline]
    fn next(&mut self) -> Option<CowValueGuard<V>> {
        self.inner.next().map(|(_, v)| v)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    #[inline]
    fn count(self) -> usize {
        self.inner.len()
    }
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.inner.fold(init, |acc, (_, v)| f(acc, v))
    }
}
impl<K: Clone, V, S> ExactSizeIterator for ValuesMut<K, V, S>
where
    V: Clone,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<K: Clone, V, S> FusedIterator for ValuesMut<K, V, S> where V: Clone {}

impl<K: Clone, V: fmt::Debug, S> fmt::Debug for ValuesMut<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.inner.iter().map(|(_, val)| val))
            .finish()
    }
}

impl<K: Clone, V, S> Iterator for IntoKeys<K, V, S>
where
    S: Clone,
{
    type Item = K;

    #[inline]
    fn next(&mut self) -> Option<K> {
        self.inner.next().map(|(k, _)| k)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    #[inline]
    fn count(self) -> usize {
        self.inner.len()
    }
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.inner.fold(init, |acc, (k, _)| f(acc, k))
    }
}
impl<K: Clone, V, S> ExactSizeIterator for IntoKeys<K, V, S>
where
    S: Clone,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<K: Clone, V, S> FusedIterator for IntoKeys<K, V, S> where S: Clone {}

impl<K: Clone + Debug, V, S> fmt::Debug for IntoKeys<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.inner.iter().map(|(k, _)| k))
            .finish()
    }
}

impl<K: Clone, V, S> Iterator for IntoValues<K, V, S>
where
    S: Clone,
{
    type Item = Arc<V>;

    #[inline]
    fn next(&mut self) -> Option<Arc<V>> {
        self.inner.next().map(|(_, v)| v)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    #[inline]
    fn count(self) -> usize {
        self.inner.len()
    }
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.inner.fold(init, |acc, (_, v)| f(acc, v))
    }
}
impl<K: Clone, V, S> ExactSizeIterator for IntoValues<K, V, S>
where
    S: Clone,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<K: Clone, V, S> FusedIterator for IntoValues<K, V, S> where S: Clone {}

impl<K: Clone, V: Debug, S> fmt::Debug for IntoValues<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.inner.iter().map(|(_, v)| v))
            .finish()
    }
}

impl<'a, K: Clone, V, S> Iterator for Drain<K, V, S> {
    type Item = (K, Arc<V>);

    #[inline]
    fn next(&mut self) -> Option<(K, Arc<V>)> {
        loop {
            if let Some(ref mut shard2) = self.shard2 {
                if let Some(r) = shard2.next() {
                    return Some(r);
                }
                self.shard2.take();
            }
            if let Some(ref mut shard1) = self.shard1 {
                if let Some(r) = shard1.next() {
                    self.shard2.replace(r.1.iter());
                    continue;
                }
                self.shard1.take();
            }
            if let Some(next) = self.base.next() {
                self.shard1.replace(next.1.iter());
                continue;
            }
            return None;
        }
    }
}
impl<K: Clone, V, S> ExactSizeIterator for Drain<K, V, S> {
    #[inline]
    fn len(&self) -> usize {
        self.base
            .clone()
            .map(|(_, t)| t.iter().map(|(_, t)| t.len()).sum::<usize>())
            .sum()
    }
}
impl<K: Clone, V, S> FusedIterator for Drain<K, V, S> {}

impl<K: Clone, V, S> fmt::Debug for Drain<K, V, S>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<K: Clone, V, S> Entry<K, V, S> {
    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// map.entry("poneyland").or_insert(3);
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(3));
    ///
    /// *map.entry("poneyland").or_insert(10) *= 2;
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(6));
    /// ```
    #[inline]
    pub fn or_insert(self, default: V) -> Arc<V>
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            Occupied(entry) => entry.get(),
            Vacant(entry) => entry.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// map.entry("poneyland").or_insert(3);
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(3));
    ///
    /// *map.entry("poneyland").or_insert(10) *= 2;
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(6));
    /// ```
    #[inline]
    pub fn or_insert_mut(self, default: V) -> CowValueGuard<V>
    where
        K: Hash,
        S: BuildHasher,
        V: Clone,
    {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert_mut(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// let value = "hoho";
    ///
    /// map.entry("poneyland").or_insert_with(|| value);
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new("hoho"));
    /// ```
    #[inline]
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> Arc<V>
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            Occupied(entry) => entry.get(),
            Vacant(entry) => entry.insert(default()),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// let value = "hoho";
    ///
    /// map.entry("poneyland").or_insert_with(|| value);
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new("hoho"));
    /// ```
    #[inline]
    pub fn or_insert_with_mut<F: FnOnce() -> V>(self, default: F) -> CowValueGuard<V>
    where
        K: Hash,
        S: BuildHasher,
        V: Clone,
    {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert_mut(default()),
        }
    }

    /// Ensures a value is in the entry by trying to insert the result of function if empty,
    /// and returns a mutable reference to the value in the entry if that function was
    /// successful.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// let value = "hoho";
    ///
    /// map.entry("poneyland").or_try_insert_with(|| Some(value)).unwrap();
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new("hoho"));
    /// ```
    #[inline]
    pub fn or_try_insert_with<F: FnOnce() -> Option<V>>(self, f: F) -> Option<Arc<V>>
    where
        K: Hash,
        S: BuildHasher,
    {
        Some(match self {
            Occupied(entry) => entry.get(),
            Vacant(entry) => entry.insert(f()?),
        })
    }

    /// Ensures a value is in the entry by trying to insert the result of function if empty,
    /// and returns a mutable reference to the value in the entry if that function was
    /// successful.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// let value = "hoho";
    ///
    /// map.entry("poneyland").or_try_insert_with(|| Some(value)).unwrap();
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new("hoho"));
    /// ```
    #[inline]
    pub fn or_try_insert_with_mut<F: FnOnce() -> Option<V>>(self, f: F) -> Option<CowValueGuard<V>>
    where
        K: Hash,
        S: BuildHasher,
        V: Clone,
    {
        Some(match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert_mut(f()?),
        })
    }

    /// Ensures a value is in the entry by inserting, if empty, the result of the default function.
    /// This method allows for generating key-derived values for insertion by providing the default
    /// function a reference to the key that was moved during the `.entry(key)` method call.
    ///
    /// The reference to the moved key is provided so that cloning or copying the key is
    /// unnecessary, unlike with `.or_insert_with(|| ... )`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, usize> = HashMap::new();
    ///
    /// map.entry("poneyland").or_insert_with_key(|key| key.chars().count());
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(9));
    /// ```
    #[inline]
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> Arc<V>
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            Occupied(entry) => entry.get(),
            Vacant(entry) => {
                let value = default(entry.key());
                entry.insert(value)
            }
        }
    }

    /// Ensures a value is in the entry by inserting, if empty, the result of the default function.
    /// This method allows for generating key-derived values for insertion by providing the default
    /// function a reference to the key that was moved during the `.entry(key)` method call.
    ///
    /// The reference to the moved key is provided so that cloning or copying the key is
    /// unnecessary, unlike with `.or_insert_with(|| ... )`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, usize> = HashMap::new();
    ///
    /// map.entry("poneyland").or_insert_with_key(|key| key.chars().count());
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(9));
    /// ```
    #[inline]
    pub fn or_insert_with_key_mut<F: FnOnce(&K) -> V>(self, default: F) -> CowValueGuard<V>
    where
        K: Hash,
        S: BuildHasher,
        V: Clone,
    {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => {
                let value = default(entry.key());
                entry.insert_mut(value)
            }
        }
    }

    /// Returns a reference to this entry's key.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    #[inline]
    pub fn key(&self) -> &K {
        match *self {
            Occupied(ref entry) => entry.key(),
            Vacant(ref entry) => entry.key(),
        }
    }

    /// Provides in-place access to an occupied entry before any
    /// potential inserts into the map.
    #[inline]
    pub fn and<F>(self, f: F) -> Self
    where
        F: FnOnce(Arc<V>),
        V: Clone,
    {
        match self {
            Occupied(entry) => {
                f(entry.get());
                Occupied(entry)
            }
            Vacant(entry) => Vacant(entry),
        }
    }

    /// Provides in-place mutable access to an occupied entry before any
    /// potential inserts into the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// map.entry("poneyland")
    ///    .and_modify(|mut e| { *e += 1 })
    ///    .or_insert(42);
    /// assert_eq!(*map.get("poneyland").unwrap(), 42);
    ///
    /// map.entry("poneyland")
    ///    .and_modify(|mut e| { *e += 1 })
    ///    .or_insert(42);
    /// assert_eq!(*map.get("poneyland").unwrap(), 43);
    /// ```
    #[inline]
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(CowValueGuard<V>),
        V: Clone,
    {
        match self {
            Occupied(entry) => {
                f(entry.get_mut());
                Occupied(entry)
            }
            Vacant(entry) => Vacant(entry),
        }
    }
}

impl<K: Clone, V, S> LockableEntry<'_, K, V, S> {
    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// map.entry("poneyland").or_insert(3);
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(3));
    ///
    /// *map.entry("poneyland").or_insert(10) *= 2;
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(6));
    /// ```
    #[inline]
    pub fn or_insert(self, default: V) -> Arc<V>
    where
        K: Hash,
        S: BuildHasher,
    {
        if self.0.is_occupied() {
            self.0.or_insert(default)
        } else {
            let _guard = self.1.lock();
            self.0.or_insert(default)
        }
    }

    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// map.entry("poneyland").or_insert(3);
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(3));
    ///
    /// *map.entry("poneyland").or_insert(10) *= 2;
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(6));
    /// ```
    #[inline]
    pub fn or_insert_mut(self, default: V) -> CowValueGuard<V>
    where
        K: Hash,
        S: BuildHasher,
        V: Clone,
    {
        if self.0.is_occupied() {
            self.0.or_insert_mut(default)
        } else {
            let _guard = self.1.lock();
            self.0.or_insert_mut(default)
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// let value = "hoho";
    ///
    /// map.entry("poneyland").or_insert_with(|| value);
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new("hoho"));
    /// ```
    #[inline]
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> Arc<V>
    where
        K: Hash,
        S: BuildHasher,
    {
        if self.0.is_occupied() {
            self.0.or_insert_with(default)
        } else {
            let _guard = self.1.lock();
            self.0.or_insert_with(default)
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// let value = "hoho";
    ///
    /// map.entry("poneyland").or_insert_with(|| value);
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new("hoho"));
    /// ```
    #[inline]
    pub fn or_insert_with_mut<F: FnOnce() -> V>(self, default: F) -> CowValueGuard<V>
    where
        K: Hash,
        S: BuildHasher,
        V: Clone,
    {
        if self.0.is_occupied() {
            self.0.or_insert_with_mut(default)
        } else {
            let _guard = self.1.lock();
            self.0.or_insert_with_mut(default)
        }
    }

    /// Ensures a value is in the entry by trying to insert the result of function if empty,
    /// and returns a mutable reference to the value in the entry if that function was
    /// successful.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// let value = "hoho";
    ///
    /// map.entry("poneyland").or_try_insert_with(|| Some(value)).unwrap();
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new("hoho"));
    /// ```
    #[inline]
    pub fn or_try_insert_with<F: FnOnce() -> Option<V>>(self, f: F) -> Option<Arc<V>>
    where
        K: Hash,
        S: BuildHasher,
    {
        if self.0.is_occupied() {
            self.0.or_try_insert_with(f)
        } else {
            let _guard = self.1.lock();
            self.0.or_try_insert_with(f)
        }
    }

    /// Ensures a value is in the entry by trying to insert the result of function if empty,
    /// and returns a mutable reference to the value in the entry if that function was
    /// successful.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map = HashMap::new();
    /// let value = "hoho";
    ///
    /// map.entry("poneyland").or_try_insert_with(|| Some(value)).unwrap();
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new("hoho"));
    /// ```
    #[inline]
    pub fn or_try_insert_with_mut<F: FnOnce() -> Option<V>>(self, f: F) -> Option<CowValueGuard<V>>
    where
        K: Hash,
        S: BuildHasher,
        V: Clone,
    {
        if self.0.is_occupied() {
            self.0.or_try_insert_with_mut(f)
        } else {
            let _guard = self.1.lock();
            self.0.or_try_insert_with_mut(f)
        }
    }

    /// Ensures a value is in the entry by inserting, if empty, the result of the default function.
    /// This method allows for generating key-derived values for insertion by providing the default
    /// function a reference to the key that was moved during the `.entry(key)` method call.
    ///
    /// The reference to the moved key is provided so that cloning or copying the key is
    /// unnecessary, unlike with `.or_insert_with(|| ... )`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, usize> = HashMap::new();
    ///
    /// map.entry("poneyland").or_insert_with_key(|key| key.chars().count());
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(9));
    /// ```
    #[inline]
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> Arc<V>
    where
        K: Hash,
        S: BuildHasher,
    {
        if self.0.is_occupied() {
            self.0.or_insert_with_key(default)
        } else {
            let _guard = self.1.lock();
            self.0.or_insert_with_key(default)
        }
    }

    /// Ensures a value is in the entry by inserting, if empty, the result of the default function.
    /// This method allows for generating key-derived values for insertion by providing the default
    /// function a reference to the key that was moved during the `.entry(key)` method call.
    ///
    /// The reference to the moved key is provided so that cloning or copying the key is
    /// unnecessary, unlike with `.or_insert_with(|| ... )`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, usize> = HashMap::new();
    ///
    /// map.entry("poneyland").or_insert_with_key(|key| key.chars().count());
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(9));
    /// ```
    #[inline]
    pub fn or_insert_with_key_mut<F: FnOnce(&K) -> V>(self, default: F) -> CowValueGuard<V>
    where
        K: Hash,
        S: BuildHasher,
        V: Clone,
    {
        if self.0.is_occupied() {
            self.0.or_insert_with_key_mut(default)
        } else {
            let _guard = self.1.lock();
            self.0.or_insert_with_key_mut(default)
        }
    }

    /// Returns a reference to this entry's key.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    #[inline]
    pub fn key(&self) -> &K {
        self.0.key()
    }
    
    /// Provides in-place access to an occupied entry before any
    /// potential inserts into the map.
    #[inline]
    pub fn and<F>(self, f: F) -> Self
    where
        F: FnOnce(Arc<V>),
        V: Clone,
    {
        match self.0 {
            Occupied(entry) => {
                f(entry.get());
                Self(Occupied(entry), self.1)
            }
            Vacant(entry) => Self(Vacant(entry), self.1),
        }
    }

    /// Provides in-place mutable access to an occupied entry before any
    /// potential inserts into the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// map.entry("poneyland")
    ///    .and_modify(|mut e| { *e += 1 })
    ///    .or_insert(42);
    /// assert_eq!(*map.get("poneyland").unwrap(), 42);
    ///
    /// map.entry("poneyland")
    ///    .and_modify(|mut e| { *e += 1 })
    ///    .or_insert(42);
    /// assert_eq!(*map.get("poneyland").unwrap(), 43);
    /// ```
    #[inline]
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(CowValueGuard<V>),
        V: Clone,
    {
        if self.0.is_occupied() {
            Self(self.0.and_modify(f), self.1)
        } else {
            let _guard = self.1.lock();
            Self(self.0.and_modify(f), self.1)
        }
    }
}

impl<K: Clone, V: Default, S> Entry<K, V, S> {
    /// Ensures a value is in the entry by inserting the default value if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() {
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, Option<u32>> = HashMap::new();
    /// map.entry("poneyland").or_default();
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(None));
    /// # }
    /// ```
    #[inline]
    pub fn or_default(self) -> Arc<V>
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            Occupied(entry) => entry.get(),
            Vacant(entry) => entry.insert(Default::default()),
        }
    }

    /// Ensures a value is in the entry by inserting the default value if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() {
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, Option<u32>> = HashMap::new();
    /// map.entry("poneyland").or_default();
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(None));
    /// # }
    /// ```
    #[inline]
    pub fn or_default_mut(self) -> CowValueGuard<V>
    where
        K: Hash,
        S: BuildHasher,
        V: Clone,
    {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert_mut(Default::default()),
        }
    }
}

impl<K: Clone, V: Default, S> LockableEntry<'_, K, V, S> {
    /// Ensures a value is in the entry by inserting the default value if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() {
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, Option<u32>> = HashMap::new();
    /// map.entry("poneyland").or_default();
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(None));
    /// # }
    /// ```
    #[inline]
    pub fn or_default(self) -> Arc<V>
    where
        K: Hash,
        S: BuildHasher,
    {
        if self.0.is_occupied() {
            self.0.or_default()
        } else {
            let _guard = self.1.lock();
            self.0.or_default()
        }
    }

    /// Ensures a value is in the entry by inserting the default value if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() {
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, Option<u32>> = HashMap::new();
    /// map.entry("poneyland").or_default();
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(None));
    /// # }
    /// ```
    #[inline]
    pub fn or_default_mut(self) -> CowValueGuard<V>
    where
        K: Hash,
        S: BuildHasher,
        V: Clone,
    {
        if self.0.is_occupied() {
            self.0.or_default_mut()
        } else {
            let _guard = self.1.lock();
            self.0.or_default_mut()
        }
    }
}

impl<K: Clone, V, S> OccupiedEntry<K, V, S> {
    /// Gets a reference to the key in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// map.entry("poneyland").or_insert(12);
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    #[inline]
    pub fn key(&self) -> &K {
        self.base.key()
    }

    /// Take the ownership of the key and value from the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use cow_hashmap::Entry;
    ///
    /// let mut map: HashMap<String, u32> = HashMap::new();
    /// let key = "ponyland".to_string();
    /// map.entry(key.clone()).or_insert(12);
    ///
    /// if let Entry::Occupied(o) = map.entry(key.clone()) {
    ///     // We delete the entry from the map.
    ///     o.remove_entry();
    /// }
    ///
    /// assert_eq!(map.contains_key(&key), false);
    /// ```
    #[inline]
    pub fn remove_entry(self) -> (K, Arc<V>)
    where
        K: Equivalent<K>,
    {
        self.base.remove_entry()
    }

    /// Gets a reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use cow_hashmap::Entry;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<String, u32> = HashMap::new();
    /// let key = "ponyland".to_string();
    /// map.entry(key.clone()).or_insert(12);
    ///
    /// if let Entry::Occupied(o) = map.entry(key.clone()) {
    ///     assert_eq!(o.get(), Arc::new(12));
    /// }
    /// ```
    #[inline]
    pub fn get(&self) -> Arc<V> {
        self.base.get()
    }

    /// Gets a mutable reference to the value in the entry.
    ///
    /// If you need a reference to the `OccupiedEntry` which may outlive the
    /// destruction of the `Entry` value, see [`into_mut`].
    ///
    /// [`into_mut`]: Self::into_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use cow_hashmap::Entry;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(12));
    /// if let Entry::Occupied(mut o) = map.entry("poneyland") {
    ///     *o.get_mut() += 10;
    ///     assert_eq!(*o.get(), 22);
    ///
    ///     // We can use the same Entry multiple times.
    ///     *o.get_mut() += 2;
    /// }
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(24));
    /// ```
    #[inline]
    pub fn get_mut(&self) -> CowValueGuard<V>
    where
        V: Clone,
    {
        self.base.get_mut()
    }

    /// Converts the `OccupiedEntry` into a mutable reference to the value in the entry
    /// with a lifetime bound to the map itself.
    ///
    /// If you need multiple references to the `OccupiedEntry`, see [`get_mut`].
    ///
    /// [`get_mut`]: Self::get_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use cow_hashmap::Entry;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<String, u32> = HashMap::new();
    /// let key = "poneyland".to_string();
    /// map.entry(key.clone()).or_insert(12);
    ///
    /// assert_eq!(map.get(&key).unwrap(), Arc::new(12));
    /// if let Entry::Occupied(o) = map.entry(key.clone()) {
    ///     *o.into_mut() += 10;
    /// }
    ///
    /// assert_eq!(map.get(&key).unwrap(), Arc::new(22));
    /// ```
    #[inline]
    pub fn into_mut(self) -> CowValueGuard<V>
    where
        V: Clone,
    {
        self.base.into_mut()
    }

    /// Sets the value of the entry, and returns the entry's old value.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use cow_hashmap::Entry;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(mut o) = map.entry("poneyland") {
    ///     assert_eq!(o.insert(15), Arc::new(12));
    /// }
    ///
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(15));
    /// ```
    #[inline]
    pub fn insert(&self, value: V) -> Arc<V> {
        self.base.insert(value)
    }

    /// Takes the value out of the entry, and returns it.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use cow_hashmap::Entry;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<String, u32> = HashMap::new();
    /// let key = "poneyland".to_string();
    /// map.entry(key.clone()).or_insert(12);
    ///
    /// if let Entry::Occupied(o) = map.entry(key.clone()) {
    ///     assert_eq!(o.remove(), Arc::new(12));
    /// }
    ///
    /// assert_eq!(map.contains_key(&key), false);
    /// ```
    #[inline]
    pub fn remove(self) -> Arc<V>
    where
        K: Equivalent<K>,
    {
        self.base.remove()
    }

    /// Replaces the entry, returning the old key and value. The new key in the hash map will be
    /// the key used to create this entry.
    #[inline]
    pub fn replace_entry(self, value: V) -> (K, Arc<V>)
    where
        V: Clone,
    {
        self.base.replace_entry(value)
    }

    /// Replaces the entry, returning the old key and value. The new key in the hash map will be
    /// the key used to create this entry.
    #[inline]
    pub fn replace_entry_mut(self, value: V) -> (K, CowValueGuard<V>)
    where
        V: Clone,
    {
        self.base.replace_entry_mut(value)
    }

    /// Replaces the key in the hash map with the key used to create this entry.
    /// ```
    #[inline]
    pub fn replace_key(self) -> K {
        self.base.replace_key()
    }
}

impl<K: Clone, V, S> VacantEntry<K, V, S> {
    /// Gets a reference to the key that would be used when inserting a value
    /// through the `VacantEntry`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    #[inline]
    pub fn key(&self) -> &K {
        self.base.key()
    }

    /// Take ownership of the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use cow_hashmap::Entry;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// if let Entry::Vacant(v) = map.entry("poneyland") {
    ///     v.into_key();
    /// }
    /// ```
    #[inline]
    pub fn into_key(self) -> K {
        self.base.into_key()
    }

    /// Sets the value of the entry with the `VacantEntry`'s key,
    /// and returns a mutable reference to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use cow_hashmap::Entry;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// if let Entry::Vacant(o) = map.entry("poneyland") {
    ///     o.insert(37);
    /// }
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(37));
    /// ```
    #[inline]
    pub fn insert(self, value: V) -> Arc<V>
    where
        K: Hash,
        S: BuildHasher,
    {
        self.base.insert(value)
    }

    /// Sets the value of the entry with the `VacantEntry`'s key,
    /// and returns a mutable reference to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use cow_hashmap::CowHashMap as HashMap;
    /// use cow_hashmap::Entry;
    /// use std::sync::Arc;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// if let Entry::Vacant(o) = map.entry("poneyland") {
    ///     o.insert(37);
    /// }
    /// assert_eq!(map.get("poneyland").unwrap(), Arc::new(37));
    /// ```
    #[inline]
    pub fn insert_mut(self, value: V) -> CowValueGuard<V>
    where
        K: Hash,
        S: BuildHasher,
        V: Clone,
    {
        self.base.insert_mut(value)
    }
}

impl<K: Clone, V, S> FromIterator<(K, V)> for CowHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Clone + Default,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> CowHashMap<K, V, S> {
        let mut map = CowHashMap::with_hasher(Default::default());
        map.extend(iter);
        map
    }
}

/// Inserts all new key-values from the iterator and replaces values with existing
/// keys with new values returned from the iterator.
impl<K: Clone, V, S> Extend<(K, V)> for CowHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Clone,
{
    #[inline]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.shard_mut(&k).insert(k, v);
        }
    }
}

impl<'a, K: Clone, V, S> Extend<(&'a K, &'a V)> for CowHashMap<K, V, S>
where
    K: Eq + Hash,
    V: Clone,
    S: BuildHasher + Clone,
{
    #[inline]
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.shard_mut(k).insert(k.clone(), v.clone());
        }
    }
}

#[inline]
fn map_entry<K: Clone, V, S>(raw: cow_hashbrown::hash_map::Entry<K, V, S>) -> Entry<K, V, S> {
    match raw {
        cow_hashbrown::hash_map::Entry::Occupied(base) => Entry::Occupied(OccupiedEntry { base }),
        cow_hashbrown::hash_map::Entry::Vacant(base) => Entry::Vacant(VacantEntry { base }),
    }
}
