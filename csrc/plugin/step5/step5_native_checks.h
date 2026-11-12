// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>
#include <climits>
#include <initializer_list>
#include <optional>
#include <cstring>
#include <cstdint>
namespace step5_native_checks {
inline void tensor(const at::Tensor& t, const at::Tensor& q, at::ScalarType dtype, int rank) {
  TORCH_CHECK(t.is_cuda() && t.device() == q.device() && t.layout() == at::kStrided &&
              t.scalar_type() == dtype && t.dim() == rank, "新版 DSA tensor device/dtype/rank 不支持");
  int64_t span = 1;
  for (int axis = 0; axis < rank; ++axis) {
    const int64_t n = t.size(axis), s = t.stride(axis);
    TORCH_CHECK(n <= INT_MAX && s >= 0 && s <= INT64_MAX / 2 && (n <= 1 || s > 0), "新版 DSA geometry/stride 越界");
    if (n > 1) {
      TORCH_CHECK(s <= (INT64_MAX / 8 - span) / (n - 1), "新版 DSA 地址计算溢出");
      span += (n - 1) * s;
    }
  }
}
inline void no_alias(const at::Tensor& out, std::initializer_list<at::Tensor> inputs) {
  for (const auto& t : inputs) TORCH_CHECK(!out.is_alias_of(t), "新版 DSA output aliases input");
}
inline void common(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& regions,
    const at::Tensor& table, const at::Tensor& pos, const at::Tensor& out, double scale) {
  tensor(q, q, at::kBFloat16, 3); tensor(out, q, at::kBFloat16, 3);
  tensor(cache, q, at::kBFloat16, 4); tensor(regions, q, at::kLong, 2);
  tensor(table, q, at::kInt, 2); tensor(pos, q, at::kLong, 1);
  const auto* prop = at::cuda::getDeviceProperties(q.get_device());
  TORCH_CHECK(prop->major == 8 && prop->minor == 9 && std::strstr(prop->name, "890P"), "新版 DSA 仅支持 PPU 1.5/890P");
  TORCH_CHECK(q.size(2) == 192 && (q.size(1) == 8 || q.size(1) == 16) && out.sizes() == q.sizes(), "新版 DSA head/output geometry 不支持");
  TORCH_CHECK(cache.size(0) > 0 && cache.size(1) > 0 && cache.size(3) == 384 &&
    (cache.size(2) == 16 || cache.size(2) == 32 || cache.size(2) == 64 || cache.size(2) == 128), "新版 DSA cache geometry 不支持");
  TORCH_CHECK(regions.size(0) == q.size(0) && regions.size(1) > 0 && regions.size(1) <= 513 &&
    regions.stride(1) == 1 && pos.numel() == q.size(0) && pos.is_contiguous(), "新版 DSA 选区/position 不支持");
  TORCH_CHECK(table.stride(1) == 1 && table.size(0) > 0 && table.size(1) > 0 &&
    table.size(1) * cache.size(2) <= INT_MAX - 8 && table.stride(0) >= table.size(1), "新版 DSA table 不支持");
  TORCH_CHECK(cache.stride(3) == 1 && reinterpret_cast<uintptr_t>(cache.data_ptr()) % 128 == 0,
    "新版 DSA cache alignment 不支持");
  for (int i = 0; i < 3; ++i) TORCH_CHECK(cache.stride(i) % 64 == 0, "新版 DSA cache stride 不支持");
  // 单 group 的 stride 不参与寻址；与 Python 的 LBNHC/BLNHC 门禁保持一致。
  TORCH_CHECK(cache.stride(2) >= 384 &&
    (cache.size(1) == 1 || cache.stride(1) >= cache.size(2) * cache.stride(2)) &&
    cache.stride(0) >= (cache.size(1) - 1) * cache.stride(1) + cache.size(2) * cache.stride(2), "新版 DSA cache overlap 不支持");
  // 标签只编码 28 位物理 region；地址跨度由 tensor() 的 int64 溢出检查保护。
  TORCH_CHECK(cache.size(0) * cache.size(2) / 8 < (int64_t(1) << 28), "新版 DSA offset encoding 容量不足（physical region）");
  for (const auto& t : {q, out}) TORCH_CHECK(t.stride(2) == 1 && t.stride(0) % 2 == 0 &&
    t.stride(1) % 2 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0, "新版 DSA query/output alignment 不支持");
  TORCH_CHECK(!out.numel() || (out.stride(1) >= 192 && out.stride(0) >= (q.size(1) - 1) * out.stride(1) + 192), "新版 DSA output overlap 不支持");
  TORCH_CHECK(q.size(0) * q.size(1) * 32 <= INT_MAX && std::isfinite(scale) && float(scale) > 0 && scale <= 3.4028234663852886e38, "新版 DSA launch geometry/scale 不支持");
  no_alias(out, {q, cache, regions, table, pos});
}
inline void prefill(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& regions,
    const at::Tensor& table, const at::Tensor& owners, const at::Tensor& pos,
    const at::Tensor& tiles, const at::Tensor& out, int64_t logical_regions, double scale, int64_t query_tile) {
  common(q, cache, regions, table, pos, out, scale);
  tensor(owners, q, at::kLong, 1); tensor(tiles, q, at::kInt, 2);
  TORCH_CHECK(cache.size(1) == 1 && q.stride(1) == 192 && q.stride(0) % 64 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 128 == 0,
    "新版 prefill group/query 不支持");
  TORCH_CHECK(owners.numel() == q.size(0) && owners.is_contiguous() && tiles.size(0) == 2 &&
    tiles.is_contiguous() && (q.size(0) == 0 || tiles.size(1) > 0) && tiles.size(1) <= q.size(0), "新版 prefill owner/tile 不支持");
  TORCH_CHECK(logical_regions > 0 && logical_regions <= INT_MAX / 8 &&
    (query_tile == 0 || query_tile == 1 || query_tile == 2 || query_tile == 4 || query_tile == 8), "新版 prefill capacity/tile 不支持");
  no_alias(out, {owners, tiles});
}
inline void decode(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& regions,
    const at::Tensor& table, const at::Tensor& pos, const at::Tensor& active, const at::Tensor& out,
    const std::optional<at::Tensor>& gate, int64_t query_len, double scale) {
  common(q, cache, regions, table, pos, out, scale); tensor(active, q, at::kBool, 1);
  TORCH_CHECK(query_len > 0 && query_len <= INT_MAX / cache.size(1), "新版 decode query_len 不支持");
  const int64_t per = query_len * cache.size(1);
  TORCH_CHECK(q.size(0) % per == 0 && table.size(0) >= q.size(0) / per &&
    active.numel() == q.size(0) && active.is_contiguous(), "新版 decode 请求布局不支持");
  no_alias(out, {active});
  if (gate) {
    tensor(*gate, q, at::kBFloat16, 2);
    TORCH_CHECK(gate->size(0) == q.size(0) / cache.size(1) && gate->size(1) == cache.size(1) * q.size(1), "新版 decode gate shape 不支持");
    no_alias(out, {*gate});
  }
}
}  // namespace step5_native_checks
