// Lightweight shim: map absl::flat_hash_map to std::unordered_map.
// Only the ILS algorithm in Bmatching uses absl::flat_hash_map.
#pragma once
#include <unordered_map>

namespace absl {
template <typename K, typename V,
          typename Hash = std::hash<K>,
          typename Eq = std::equal_to<K>,
          typename Alloc = std::allocator<std::pair<const K, V>>>
using flat_hash_map = std::unordered_map<K, V, Hash, Eq, Alloc>;
}  // namespace absl
