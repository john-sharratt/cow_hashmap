COW HashMap
===========

[![Crates.io](https://img.shields.io/crates/v/cow_hashbrown.svg)](https://crates.io/crates/cow_hashbrown)
[![Documentation](https://docs.rs/cow_hashbrown/badge.svg)](https://docs.rs/cow_hashbrown)

This crate takes the original hashmap implementation that was ported from
Google's high-performance [SwissTable] hash map and wraps it in AtomicPtr
compare and replace operations which give it copy-on-write sementics.

Note: That inserting values into the hashmap will copy the entire hashmap
every time thus inserts are no where near as fast as the original swishtable
implementation however they do execute the operation lock-free while
concurrent reads can happen in parallel. Essentially this construct is
really good for read intensive operations.

Access the values at the leafs of the hashmap are also copy-on-write
thus readonly access is very fast however writes will copy the original
value and perform a compare-and-swap operation.

Many of the constructs that use lamda functions to perform write
operations have been implemented inside the compare-and-swap loop
thus they allow for concurrent writes without losing data however
when accessing a value using `get_mut` the value you be entirely
replaced when it falls out of scope.

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
