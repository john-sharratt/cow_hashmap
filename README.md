COW HashMap
===========

[![Crates.io](https://img.shields.io/crates/v/cow_hashbrown.svg)](https://crates.io/crates/cow_hashbrown)
[![Documentation](https://docs.rs/cow_hashbrown/badge.svg)](https://docs.rs/cow_hashbrown)

This crate takes the original hashmap implementation that was ported from
Google's high-performance [SwissTable] hash map and wraps it in AtomicPtr
compare and replace operations which give it copy-on-write sementics.

Note: That inserting values into the hashmap will copy shards that make
up the hash map thus there are performance considerations to take into
account here.

This hashmap will perform reasonable well for inserts and very fast for reads
thus it is highly suited to read intensive operations.

Access the values at the leafs of the hashmap are also copy-on-write
thus readonly access is very fast however writes will copy the original
value and perform a compare-and-swap operation.

Many of the constructs that use lamda functions to perform write
operations have been implemented inside the compare-and-swap loop
thus they allow for concurrent writes without losing data however
when accessing a value using `get_mut` the value you be entirely
replaced when it falls out of scope.

## Internals

This HashMap works as follows:

- The hashmap is built using 3 layers of different sizes that are optimized
  to minimize the amount of copying.
- Inserting or removing a value from the hashmap copies a particular shard
  and then performs a compare and swap operation to safely replace it in the
  main map.
- Each entry in the HashMap is cloned whenever it is edited.
- Mutation functions with lamda's will compare and swap the mutated value
  thus ensuring concurrent writes are safe
- At the leaf of the layers is a normal hashbrown table that holds the
  real value.

## Performance

Not the best performance testing but it's better than nothing!

1 million entries inserted and removed from the HashMap.

```
PS C:\Users\johna\prog\cow_hashmap> cargo test --release test_millions_of_inserts_and_removes
    Finished `release` profile [optimized] target(s) in 0.01s
     Running unittests src\lib.rs (target\release\deps\cow_hashmap-1df93076c4962117.exe)

running 1 test
test tests::test_millions_of_inserts_and_removes ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 32 filtered out; finished in 3.79s

PS C:\Users\johna\prog\cow_hashmap>
```

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
cow_hashbrown = "0.1"
```

Then:

```rust
use cow_hashmap::CowHashMap;

let map = CowHashMap::new();
map.insert(1, "one");
```

## License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
