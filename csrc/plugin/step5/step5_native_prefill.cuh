// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hggc_runtime.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <type_traits>
#include "step5_native_device.cuh"
#include "step5_runtime.h"
namespace STEP5_NATIVE_PREFILL_NAMESPACE {
using namespace step5_dsa_native;
constexpr int kHeads = STEP5_PREFILL_HEADS;
static_assert(kHeads == 8 || kHeads == 16);
constexpr int kWidth = 513, kGroupRows = 16, kWindowRegions = 8192;
constexpr int kWords = kWindowRegions / 32;
constexpr int kValidationBlocks = sizeof(int4) / sizeof(int32_t);
constexpr int kSplitBoundarySlot = kGroupRows / 8;
struct Params {
  PagedKV kv;
  const __ppu_bfloat16* q;
  __ppu_bfloat16* out;
  const int64_t *regions, *pos, *owners;
  const int32_t *starts, *ends;
  uint64_t* slots;
  int32_t *tile_counts, *tiles_valid;
  int rows, groups, width, logical_regions;
  int64_t region_stride, qs, qh, os, oh;
  float scale;
  __device__ int64_t tile_index(int group, int tile) const { return int64_t(group) * kGroupRows + tile; }
  // Reject invalid partitions before reading Q/KV or writing output.
  template <bool Partition = false>
  __device__ bool valid_tiles() const {
    if constexpr (Partition) {
      const int4 flags = *reinterpret_cast<const int4*>(tiles_valid);
      return (flags.x & flags.y & flags.z & flags.w) != 0;
    }
    return *tiles_valid != 0;
  }
  __device__ bool well_formed_group(int g) const {
    const int first = starts[g], last = ends[g];
    if (first < 0 || last <= first || last > rows || last - first > kGroupRows) return false;
    // Overlapping tiles would race on output and union scratch.
    if ((g == 0 ? first != 0 : first != ends[g - 1]) ||
        (g + 1 == groups ? last != rows : last != starts[g + 1])) return false;
    const int64_t owner = owners[first];
    for (int row = first + 1; row < last; ++row) if (owners[row] != owner) return false;
    return true;
  }
  __device__ int position(int row) const { return checked_position(pos[row]); }
  __device__ int live(int row) const {
    const int p = position(row);
    return p < 0 ? 0 : min(width, p / 8 + 1);
  }
  __device__ uint64_t selection(int row, int slot) const {
    const int64_t region = regions[int64_t(row) * region_stride + width - live(row) + slot];
    const int p = position(row);
    if (region < 0 || region > p / 8 || p < 0) return 0;
    const int phys = kv.physical_region(owners[row], region);
    if (phys < 0) return 0;
    const uint32_t n = min(int64_t(8), int64_t(p) - region * 8 + 1);
    return (uint64_t(phys) << 32) | n;
  }
  __device__ void write_slot(int row, int slot, uint32_t tag, uint32_t visible, uint32_t max_visible) const {
    const int64_t idx = int64_t(row) * width + slot;
    slots[idx] = (uint64_t((tag << 4) | max_visible) << 32) | visible;
  }
  __device__ uint64_t read_slot(int row, int slot) const {
    const int64_t idx = int64_t(row) * width + slot;
    return slots[idx];
  }
  template <bool Scaled> __device__ int64_t offset(uint32_t tag) const {
    if constexpr (Scaled) return int64_t(tag) << 9;
    return kv.offset(tag);
  }
  template <int BQ, bool Scaled = false> __device__ void copy(__ppu_bfloat16* dst, uint64_t entry, bool value) const {
    const uint32_t lookup = entry >> 32;
    const uint32_t physical_region = BQ == 1 ? lookup : lookup >> 4;
    const auto* src = kv.data + offset<Scaled>(physical_region) + (value ? kHeadDim : 0);
    const int n = BQ == 1 ? int(uint32_t(entry)) : int(lookup & 15);
    const int start = n == 0 ? 8 : 0;
    aiu_copy_group(dst, src, n + start, start, kRegionTokens, kv.row_bytes());
  }
};
// Kernel definitions must stay in both arch views so HGGC does not emit an
// unregistered host stub.
template <int Heads> __global__ void step5_validate_tiles_kernel(Params p) {
  bool valid = true;
  for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < p.groups;
       g += blockDim.x * gridDim.x)
    valid = p.well_formed_group(g) && valid;
  const int all_valid = __syncthreads_and(valid);
  if (threadIdx.x == 0) p.tiles_valid[blockIdx.x] = all_valid != 0;
}
template <int Words> struct WindowState { int warp_prefix[Words / 32]; int count; unsigned next_base; };
template <int BQ, bool BalanceWords, int Words, int Threads = Words, bool Partition = false, bool Scaled = false>
__device__ __forceinline__ void collect_window(const Params& p, int first, int last,
                                               unsigned base, int written,
                                               WindowState<Threads>& state,
                                               int* tile_count) {
  extern __shared__ __align__(128) unsigned char temp[];
  auto* bitmap = reinterpret_cast<uint32_t (*)[Words]>(temp);
  const int tid = threadIdx.x;
  const int query = first + tid % BQ;
  int last_pos = query < last ? p.position(query) : -1;
#pragma unroll
  for (int stride = BQ / 2; stride; stride >>= 1)
    last_pos = max(last_pos, __shfl_xor_sync(0xffffffffu, last_pos, stride));
  const unsigned final_region = unsigned(max(0, last_pos)) / 8;
  const int used_words = final_region >= base
                             ? min(Words, int((final_region - base) / 32) + 1)
                             : 1;
  const int clear_log = 32 - __clz(unsigned(used_words - 1));
  const int clear_words = 1 << clear_log;
  for (int i = tid; i < BQ * clear_words; i += blockDim.x)
    bitmap[i >> clear_log][i & (clear_words - 1)] = 0;
  if (tid == 0) state.next_base = UINT_MAX;
  __syncthreads();
  unsigned next = UINT_MAX;
  const int warp = __builtin_ppu_to_uniform_b32(tid / 32);
  const int row = warp % BQ;
  const int query_row = first + row;
  const int position = query_row < last ? p.position(query_row) : -1;
  const int live = position < 0 ? 0 : min(p.width, position / 8 + 1);
  const auto* regions =
      p.regions + int64_t(query_row < last ? query_row : first) * p.region_stride + p.width - live;
  for (int slot = (warp / BQ) * 32 + (tid & 31); slot < live;
       slot += Threads / BQ) {
    const int64_t id = regions[slot];
    if (uint64_t(id) > unsigned(position / 8)) continue;
    const unsigned region = id;
    if (region < base) continue;
    const unsigned local = region - base;
    if (local >= (Words * 32)) {
      next = min(next, region & ~((Words * 32) - 1u));
      continue;
    }
    atomicOr(&bitmap[row][local / 32], 1u << (local % 32));
  }
  if (next != UINT_MAX) atomicMin(&state.next_base, next);
  __syncthreads();
  int word = tid;
  uint32_t slice = ~0u;
  if constexpr (BalanceWords) {
    // Divide a short causal prefix among the whole CTA, in logical order.
    const int log_bits =
        min(5, 32 - __clz(unsigned(max(0, last_pos)) / (8 * Threads)));
    const int bits = 1 << log_bits;
    word = tid >> (5 - log_bits);
    slice = (~0u >> (32 - bits)) << ((tid * bits) & 31);
  }
  uint32_t members[BQ], mask = 0;
#pragma unroll
  for (int r = 0; r < BQ; ++r) {
    members[r] = word < clear_words ? bitmap[r][word] & slice : 0;
    mask |= members[r];
  }
  const int count = __popc(mask);
  int inclusive = count;
#pragma unroll
  for (int step = 1; step < 32; step *= 2) {
    const int prev = __shfl_up_sync(0xffffffffu, inclusive, step);
    if ((tid & 31) >= step) inclusive += prev;
  }
  if ((tid & 31) == 31) state.warp_prefix[tid / 32] = inclusive;
  __syncthreads();
  if (tid == 0) {
    int sum = 0;
    for (int warp = 0; warp < int(blockDim.x) / 32; ++warp) {
      const int n = state.warp_prefix[warp];
      state.warp_prefix[warp] = sum;
      sum += n;
    }
    state.count = sum;
    if (state.next_base == UINT_MAX) *tile_count = written + sum;
  }
  __syncthreads();
  int offset = written + state.warp_prefix[tid / 32] + inclusive - count;
  uint32_t positions[BQ];
  uint32_t common = mask;
  int earliest = INT_MAX;
#pragma unroll
  for (int r = 0; r < BQ; ++r) {
    const bool active = first + r < last;
    const int pos = active ? p.position(first + r) : (kHeads == 8 ? 0 : INT_MAX);
    positions[r] = uint32_t(pos);
    if constexpr (kHeads == 16 && BalanceWords) {
      if (active) common &= members[r];
      earliest = min(earliest, pos);
    }
  }
  // The tail pairs four regions exclusive to each half of the CTA.
  uint32_t left_bits = 0, right_bits = 0;
  int left_rank = 0, right_rank = 0, boundary = 0, paired_count = 0, left_excess = 0,
      right_excess = 0;
  if constexpr (Partition && BalanceWords && BQ == 8) {
    if (last_pos / 8 >= p.width && first + 8 <= last && !written &&
        state.next_base == UINT_MAX && state.count <= 4 * kWidth) {
      uint32_t left = members[0] | members[1] | members[2] | members[3];
      uint32_t right = members[4] | members[5] | members[6] | members[7];
      uint32_t left_only = left & ~right, right_only = right & ~left;
      uint32_t lane_counts = __popc(left_only) | (__popc(right_only) << 16),
               inclusive_sides = lane_counts;
#pragma unroll
      for (int step = 1; step < 32; step *= 2) {
        uint32_t previous = __shfl_up_sync(0xffffffffu, inclusive_sides, step);
        if ((tid & 31) >= step) inclusive_sides += previous;
      }
      __shared__ uint32_t sums[Threads / 32], totals;
      if ((tid & 31) == 31) sums[warp] = inclusive_sides;
      __syncthreads();
      if (!tid) {
        uint32_t total = 0;
        for (int w = 0; w < Threads / 32; ++w) {
          auto n = sums[w];
          sums[w] = total;
          total += n;
        }
        totals = total;
      }
      __syncthreads();
      const int left_count = totals & 65535, right_count = totals >> 16,
                shared_count = state.count - left_count - right_count;
      const int paired = (min(left_count, right_count) / 4) * 4;
      const int common_count = state.count - 2 * paired;
      const int split_start = (common_count + 7) & ~7,
                padded_count = split_start + 2 * paired;
      if (paired * 20 >= state.count && paired && padded_count <= 8 * p.width) {
        const uint32_t before = sums[warp] + inclusive_sides - lane_counts;
        left_bits = left_only;
        right_bits = right_only;
        left_rank = before & 65535;
        right_rank = before >> 16;
        offset -= left_rank + right_rank;
        boundary = split_start;
        paired_count = paired;
        left_excess = shared_count;
        right_excess = shared_count + left_count - paired;
        if (!tid) {
          *tile_count = padded_count;
          tile_count[kSplitBoundarySlot] = split_start + 1;
        }
        for (int i = common_count + tid; i < split_start; i += Threads)
          p.write_slot(first, i, 0, 0, 0);
      }
    }
  }
  const int active_rows = min(BQ, last - first);
  const uint32_t full = 0x88888888u >> (4 * (8 - active_rows));
  const int64_t request = p.owners[first];
  const int32_t* pages = uint64_t(request) < unsigned(p.kv.requests)
      ? p.kv.table + request * p.kv.table_stride : nullptr;
  const unsigned shift = p.kv.region_shift;
  const unsigned page_mask = (1u << shift) - 1;
  while (mask) {
    const int bit = __ffs(mask) - 1, region = word * 32 + bit;
    const uint32_t start = (base + region) * 8;
    uint32_t lengths = full, visible = 8;
    if (kHeads == 16 && BalanceWords && earliest >= 0 &&
        uint32_t(earliest) >= start + 7) {
      if (!(common & (1u << bit))) {
        lengths = 0;
#pragma unroll
        for (int r = 0; r < BQ; ++r)
          lengths |= ((members[r] >> bit) & 1u) << (4 * r + 3);
      }
    } else {
      lengths = visible = 0;
#pragma unroll
      for (int r = 0; r < BQ; ++r) {
        if (members[r] & (1u << bit)) {
          const uint32_t n = min(8u, positions[r] - start + 1);
          visible = max(visible, n);
          lengths |= n << (4 * r);
        }
      }
    }
    const unsigned id = base + region;
    const unsigned page_id = id >> shift;
    const int page = pages && page_id < unsigned(p.kv.table_width) ? pages[page_id] : -1;
    const bool valid = unsigned(page) < unsigned(p.kv.page_count);

    int dst;
    if ((left_bits >> bit) & 1u) {
      int i = left_rank++;
      dst = i < paired_count ? boundary + (i / 4) * 8 + (i & 3)
                             : left_excess + i - paired_count;
    } else if ((right_bits >> bit) & 1u) {
      int i = right_rank++;
      dst = i < paired_count ? boundary + (i / 4) * 8 + 4 + (i & 3)
                             : right_excess + i - paired_count;
    } else
      dst = offset++;
    const uint32_t in_page = id & page_mask;
    uint32_t tag;
    if constexpr (Scaled) {
      tag = uint32_t(page) * uint32_t(p.kv.block_stride >> 9) +
            in_page * uint32_t(p.kv.token_stride >> 6);
    } else {
      tag = (uint32_t(page) << shift) | in_page;
    }
    p.write_slot(first, dst, valid ? tag : 0, valid ? lengths : 0, valid ? visible : 0);
    mask &= mask - 1;
  }
}

template <int BQ, bool BalanceWords, int Words, bool Partition = false, bool Scaled = false>
__global__ void step5_prepare_union_kernel(Params p) {
  constexpr int Tiles = kGroupRows / BQ;
  const int g = BalanceWords ? blockIdx.x : blockIdx.x / Tiles;
  if (!p.valid_tiles<Partition>()) return;
  const int last = min(p.ends[g], p.rows);
  const int start_tile = BalanceWords ? 0 : blockIdx.x % Tiles;
  const int end_tile = BalanceWords ? Tiles : start_tile + 1;
  __shared__ WindowState<Words> state;
  for (int tile = start_tile; tile < end_tile; ++tile) {
    const int first = p.starts[g] + tile * BQ;
    if (first >= last) break;
    if (BalanceWords && tile) __syncthreads();
    int* tile_count = p.tile_counts + p.tile_index(g, tile);
    if constexpr (Partition && BalanceWords && BQ == 8)
      if (threadIdx.x == 0) tile_count[kSplitBoundarySlot] = 0;
    if constexpr (kHeads == 8 && BQ == 8 && !BalanceWords && Words == kWords) {
      constexpr int LocalRegions = 1024;
      if (p.logical_regions <= LocalRegions) {
        const int r = first + threadIdx.x;
        const bool fits =
            threadIdx.x >= BQ || r >= last || p.position(r) < 8 * LocalRegions;
        if (__syncthreads_and(fits)) {
          collect_window<BQ, true, LocalRegions / 32, Words, false>(
              p, first, last, 0, 0, state, tile_count);
          continue;
        }
      }
    }
    collect_window<BQ, BalanceWords, Words, Words, Partition, Scaled>(
        p, first, last, 0, 0, state, tile_count);
    unsigned base = state.next_base;
    int written = state.count;
    while (base != UINT_MAX) {
      __syncthreads();
      collect_window<BQ, false, Words, Words, false, Scaled>(p, first, last, base,
                                                            written, state, tile_count);
      base = state.next_base;
      written += state.count;
    }
  }
}

template <int BQ, int HT, int BS, int PARTS>
__global__ __launch_bounds__((BQ == 1 ? 1 : BQ / 2) * HT * 32 * PARTS)
void step5_paged_attention_kernel(Params p) {
#if __HGGC_ARCH__ >= 150
  constexpr int TG = BS / 8, NT = BS / 16;
  static_assert(BQ == 1 || BQ == 2 || BQ == 4);
  static_assert((BQ == 1) == (PARTS > 1));
  extern __shared__ __align__(128) unsigned char scratch[];
  constexpr int warps_per_part = (BQ == 1 ? 1 : BQ / 2) * HT;
  const int part = threadIdx.x / (32 * warps_per_part);
  auto* stage = scratch + part * BS * 192 * 4;
  auto& sk = *reinterpret_cast<KvStage<TG>*>(stage);
  auto& sv = *reinterpret_cast<KvStage<TG>*>(stage + BS * 192 * 2);
  constexpr bool warp_local_stage = PARTS > 1 && warps_per_part == 1;
  const int g = blockIdx.x / (kGroupRows / BQ), tile = blockIdx.x % (kGroupRows / BQ);
  if (!p.valid_tiles()) return;
  const int first = p.starts[g] + tile * BQ, end_row = min(p.rows, p.ends[g]);
  if (first >= end_row) return;
  const int warp = threadIdx.x / 32 % warps_per_part, lane = threadIdx.x % 32;
  constexpr int pairs = BQ == 1 ? 1 : BQ / 2;
  const int pair = warp % pairs, head_tile = warp / pairs, query_offset = pair * 2;
  const int row = first + query_offset, head = head_tile * 8 + (lane >> 2);
  const int t0 = lane & 3, lr = lane & 7, lu = lane >> 3;
  const int a_col = 8 * (lu >> 1), b_col = 8 * (lu & 1);
  const int total = BQ == 1 ? p.live(first) : p.tile_counts[p.tile_index(g, tile)];
  const int partition_stride = (total + PARTS - 1) / PARTS, offset = part * partition_stride;
  const int partition_size = BQ == 1 ? max(0, min(partition_stride, total - offset)) : total;
  const int walk = warp_local_stage ? partition_size : partition_stride;
  MmaOperand qfrag[kHeadMmaTiles]; MmaAccumulator out[kHeadMmaTiles];
#pragma unroll
  for (int k = 0; k < kHeadMmaTiles; ++k) {
    clear_fragment(qfrag[k]); clear_fragment(out[k]);
#pragma unroll
    for (int tau = 0; tau < 2; ++tau) {
      if (row + tau < end_row && (BQ > 1 || tau == 0)) {
        const auto* src = p.q + (row + tau) * p.qs + head * p.qh + k * 16 + 2 * t0;
        qfrag[k].x[tau] = *reinterpret_cast<const uint32_t*>(src); qfrag[k].x[tau + 2] = *reinterpret_cast<const uint32_t*>(src + 8);
      }
    }
  }
  float maximum[2] = {-INFINITY, -INFINITY}, denominator[2] = {0, 0};
  for (int base = 0; base < walk; base += TG) {
    uint64_t entries[TG];
#pragma unroll
    for (int r = 0; r < TG; ++r) {
      uint64_t entry = 0;
      if (base + r < partition_size) { if constexpr (BQ == 1) entry = p.selection(first, offset + base + r); else entry = p.read_slot(first, base + r); }
      entries[r] = entry;
    }
    if (warp == 0) {
#pragma unroll
      for (int r = 0; r < TG; ++r) p.copy<BQ>(&sk[r][0][0][0], entries[r], false);
      asm volatile("ppu.cp.async.commit_group;" ::: "memory");
#pragma unroll
      for (int r = 0; r < TG; ++r) p.copy<BQ>(&sv[r][0][0][0], entries[r], true);
      asm volatile("ppu.cp.async.commit_group;" ::: "memory"); asm volatile("ppu.cp.async.wait_group 1;" ::: "memory");
    }
    if constexpr (warp_local_stage) __syncwarp(); else __syncthreads();
    MmaAccumulator score[NT]; MmaOperand kfrag;
#pragma unroll
    for (int t = 0; t < NT; ++t) clear_fragment(score[t]);
#pragma unroll
    for (int k = 0; k < kHeadMmaTiles; ++k) {
      const int chunk = (((k % 4) * 16 + b_col) >> 3) ^ lr;
#pragma unroll
      for (int t = 0; t < NT; ++t) { ldm_x4(kfrag, &sk[t * 2 + (lu >> 1)][k / 4][lr][chunk << 3]); mma_bf16(score[t], qfrag[k], kfrag, score[t]); }
    }
    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int tau = (i >> 1) & 1, half = i >> 2;
        const uint32_t limits = entries[t * 2 + half];
        const int visible = (limits >> (4 * (query_offset + tau))) & 15;
        score[t].x[i] = 2 * t0 + (i & 1) < visible ? score[t].x[i] : -INFINITY;
        tile_max[tau] = fmaxf(tile_max[tau], score[t].x[i]);
      }
    }
    float rescale[2];
#pragma unroll
    for (int tau = 0; tau < 2; ++tau) {
      const float tile_peak = head_max(tile_max[tau]), next = fmaxf(maximum[tau], tile_peak);
      rescale[tau] = next == -INFINITY ? 1.f : __expf((maximum[tau] - next) * p.scale);
      maximum[tau] = next; denominator[tau] *= rescale[tau];
    }
    MmaOperand prob[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int tau = (i >> 1) & 1;
        const float offset = maximum[tau] == -INFINITY ? 0.f : maximum[tau];
        const float a = __expf((score[t].x[i] - offset) * p.scale), b = __expf((score[t].x[i + 1] - offset) * p.scale);
        *reinterpret_cast<__ppu_bfloat162*>(&prob[t].x[i / 2]) = __floats2bfloat162_rn(a, b); denominator[tau] += a + b;
      }
    }
    if (!__all_sync(0xffffffffu, rescale[0] == 1.f && rescale[1] == 1.f)) {
#pragma unroll
      for (int j = 0; j < kHeadMmaTiles; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) out[j].x[i] *= rescale[(i >> 1) & 1];
      }
    }
    if (warp == 0) asm volatile("ppu.cp.async.wait_group 0;" ::: "memory");
    if constexpr (warp_local_stage) __syncwarp(); else __syncthreads();
    MmaOperand vfrag;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int j = 0; j < kHeadMmaTiles; ++j) {
        const int chunk = (((j % 4) * 16 + a_col) >> 3) ^ lr;
        ldm_x4_trans(vfrag, &sv[t * 2 + (lu & 1)][j / 4][lr][chunk << 3]); mma_bf16(out[j], prob[t], vfrag, out[j]);
      }
    }
    if constexpr (warp_local_stage) __syncwarp(); else __syncthreads();
  }
  if constexpr (PARTS > 1) {
    constexpr int nh = 8 * HT, width = nh * 192;
    const float denom = head_sum(denominator[0]);
    const float local = denom > 0 ? maximum[0] * p.scale + __logf(denom) : -INFINITY;
    float* partial = reinterpret_cast<float*>(scratch); float* stats = partial + (PARTS - 1) * width;
    __syncthreads();
    if (t0 == 0) stats[part * nh + head] = local;
    if (part) {
#pragma unroll
      for (int j = 0; j < kHeadMmaTiles; ++j) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int i = half * 4, d = j * 16 + 2 * t0 + half * 8; const float inv = denom > 0 ? 1.f / denom : 0.f;
          *reinterpret_cast<float2*>(partial + (part - 1) * width + head * 192 + d) = make_float2(out[j].x[i] * inv, out[j].x[i + 1] * inv);
        }
      }
    }
    __syncthreads();
    if (part == 0) {
      float m = -INFINITY, weights[PARTS], sum = 0.f;
#pragma unroll
      for (int t = 0; t < PARTS; ++t) m = fmaxf(m, stats[t * nh + head]);
#pragma unroll
      for (int t = 0; t < PARTS; ++t) { weights[t] = m == -INFINITY ? 0.f : __expf(stats[t * nh + head] - m); sum += weights[t]; }
#pragma unroll
      for (int j = 0; j < kHeadMmaTiles; ++j) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int i = half * 4, d = j * 16 + 2 * t0 + half * 8;
          const float w0 = denom > 0 ? weights[0] / denom : 0.f;
          float a = out[j].x[i] * w0, b = out[j].x[i + 1] * w0;
#pragma unroll
          for (int t = 1; t < PARTS; ++t) {
            float2 x = *reinterpret_cast<const float2*>(partial + (t - 1) * width + head * 192 + d); a += weights[t] * x.x; b += weights[t] * x.y;
          }
          const float inv = sum > 0 ? 1.f / sum : 0.f;
          *reinterpret_cast<__ppu_bfloat162*>(p.out + row * p.os + head * p.oh + d) = __floats2bfloat162_rn(a * inv, b * inv);
        }
      }
    }
    return;
  }
#pragma unroll
  for (int tau = 0; tau < 2; ++tau) {
    if (row + tau >= end_row) continue;
    const float denom = head_sum(denominator[tau]), inv = denom > 0 ? 1.f / denom : 0.f;
#pragma unroll
    for (int j = 0; j < kHeadMmaTiles; ++j) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int i = 2 * tau + 4 * half, d = j * 16 + 2 * t0 + 8 * half;
        const float a = out[j].x[i] * inv, b = out[j].x[i + 1] * inv;
        *reinterpret_cast<__ppu_bfloat162*>(p.out + (row + tau) * p.os + head * p.oh + d) = __floats2bfloat162_rn(a, b);
      }
    }
  }
#else
  __builtin_trap();
#endif
}
template <int Heads> __device__ __forceinline__ __ppu_bfloat16* paired_cube(__ppu_bfloat16* base, int region, int cube) {
  if constexpr (Heads == 8) return base + (region * 3 + cube) * 512;
  const int offset = cube < 2 ? region * 1024 + cube * 512 : 8192 + region * 512;
  return base + offset;
}
template <int Cubes> __device__ __forceinline__ void stage_region_cubes(__ppu_bfloat16* dst,
    const __ppu_bfloat16* src, int extent, int start, int64_t row_bytes) {
#if __HGGC_ARCH__ >= 150
  const unsigned addr = __cvta_generic_to_shared(dst);
  asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.LLC::128B.padz.swzl.b16 "
    "[%0], [%1], {%2, %3, %4}, {%2, %5, %4}, {%6, %7}, {%8, %9, %8}, %8;"
    :: "r"(addr), "l"(src), "r"(kCubeWidth), "r"(extent), "r"(Cubes), "r"(kRegionTokens),
       "l"(row_bytes), "l"(kCubeRowBytes), "r"(0), "r"(start) : "memory");
#endif
}
template <int Heads> __device__ __forceinline__ void stage_shared_region(__ppu_bfloat16* dst,
    const __ppu_bfloat16* src, int region, int extent, int start, int64_t row_bytes) {
  if constexpr (Heads == 8) aiu_copy_group(paired_cube<Heads>(dst, region, 0), src, extent, start, kRegionTokens, row_bytes);
  else {
    stage_region_cubes<2>(paired_cube<Heads>(dst, region, 0), src, extent, start, row_bytes);
    stage_region_cubes<1>(paired_cube<Heads>(dst, region, 2), src + 2 * kCubeWidth, extent, start, row_bytes);
  }
}
template <bool Transpose> __device__ __forceinline__ void load_paired(MmaOperand& fragment,
    const __ppu_bfloat16* src, int leading_offset, int secondary_offset) {
#if __HGGC_ARCH__ >= 150
  auto* words = fragment.x; const uint32_t addr = uint32_t(__cvta_generic_to_shared(src)) / 16;
  if constexpr (Transpose)
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0, %1, %2, %3}, [%4], %5, %6, %7;"
      : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3]) : "r"(addr), "r"(leading_offset), "r"(secondary_offset), "r"(0));
  else
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0, %1, %2, %3}, [%4], %5, %6, %7;"
      : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3]) : "r"(addr), "r"(leading_offset), "r"(secondary_offset), "r"(0));
#endif
}
template <bool Private, int Heads, bool SplitRows = false,
          bool IssueValueEarly = false, bool Partition = false, bool Scaled = false>
__device__ __forceinline__ void native_tile(const Params& p, int first, int end_row,
                                            int walk, int row_offset = 0,
                                            int split_start = 0) {
#if __HGGC_ARCH__ >= 150
  constexpr bool value_before_key_wait = !Partition || SplitRows || IssueValueEarly;
  constexpr bool TwoRows = Heads == 8 && !Private;
  constexpr int Halves = Heads == 16 || TwoRows ? 2 : 1;
  constexpr int StageWarps = Heads == 8 ? 4 : 8;
  constexpr int BS = Private ? 16 : 64, TG = BS / 8;
  constexpr int key_batch = !Private && Heads == 8 ? 2 : 1;
  constexpr int value_batch = Private ? 1 : (Heads == 8 ? 2 : kHeadMmaTiles);
  const int warp = __builtin_ppu_to_uniform_b32(threadIdx.x / 32);
  const int lane = threadIdx.x % 32;
  const int row = first + warp + row_offset, head = lane >> 2;
  if constexpr (Private) {
    if (row >= end_row) return;
  }
  const int t0 = lane & 3, lr = lane & 7, lu = lane >> 3;
  const int a_col = 8 * (lu >> 1), b_col = 8 * (lu & 1);
  extern __shared__ __align__(128) unsigned char scratch[];
  // Each private warp reuses its K tile for V once QK has consumed it.
  auto* stage = scratch + (Private ? warp * BS * 192 * 2 : 0);
  auto& sk = *reinterpret_cast<KvStage<TG>*>(stage);
  auto& sv =
      *reinterpret_cast<KvStage<TG>*>(stage + (Private ? 0 : BS * 192 * 2));

  auto* key_base = reinterpret_cast<__ppu_bfloat16*>(&sk);
  auto* value_base = reinterpret_cast<__ppu_bfloat16*>(&sv);

  MmaOperand qfrag[kHeadMmaTiles];
  MmaAccumulator out[kHeadMmaTiles];
  if constexpr (!Private) {
    constexpr int64_t kQueryHeadBytes = kHeadDim * sizeof(__ppu_bfloat16);
    auto* qstage = reinterpret_cast<__ppu_bfloat16*>(scratch) + warp * 16 * 192;
    if constexpr (Heads == 8) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int query = row + half * StageWarps;
        const bool present = query < end_row;
        const auto* source = p.q + (present ? query : first) * p.qs;
        aiu_copy_group(qstage + half * 8 * 192, source, 8, present ? 0 : 8, 8,
                       kQueryHeadBytes);
      }
    } else if (row < end_row) {
      aiu_copy_group(qstage, p.q + row * p.qs, 16, 0, 16, kQueryHeadBytes);
    }
    asm volatile("ppu.cp.async.commit_group;" ::: "memory");
    asm volatile("ppu.cp.async.wait_group 0;" ::: "memory");
    __syncwarp();
#pragma unroll
    for (int k = 0; k < kHeadMmaTiles; ++k) {
      clear_fragment(out[k]);
      if (TwoRows || row < end_row) {
        const int offset =
            k < 8 ? (k % 4) * 16 + (k / 4) * 8 : Heads * 128 + (k % 4) * 16;
        const int leading = Heads == 8 ? 192 : 64;
        const int secondary = Heads == 8 ? 65 : 129;
        load_paired<false>(qfrag[k], qstage + offset, leading,
                           k < 8 ? secondary - 2 * (k / 4) : 1);
        // The MMA A operand orders row halves before column halves.
        const uint32_t second = qfrag[k].x[1];
        qfrag[k].x[1] = qfrag[k].x[2];
        qfrag[k].x[2] = second;
      } else {
        clear_fragment(qfrag[k]);
      }
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int k = 0; k < kHeadMmaTiles; ++k) {
      clear_fragment(qfrag[k]);
      clear_fragment(out[k]);
#pragma unroll
      for (int tau = 0; tau < Halves; ++tau) {
        const int qhead = head + (Heads == 16 ? tau * 8 : 0);
        const auto* src = p.q + row * p.qs + qhead * p.qh + k * 16 + 2 * t0;
        *reinterpret_cast<__ppu_bfloat162*>(&qfrag[k].x[tau]) =
            *reinterpret_cast<const __ppu_bfloat162*>(src);
        *reinterpret_cast<__ppu_bfloat162*>(&qfrag[k].x[tau + 2]) =
            *reinterpret_cast<const __ppu_bfloat162*>(src + 8);
      }
    }
  }
  constexpr float log2e = 1.4426950408889634f;
  constexpr float initial_max = -FLT_MAX;
  float maximum[2] = {initial_max, initial_max}, denominator[2] = {0, 0};
  const float scale2 = p.scale * log2e;
  int staged_visible = 0;
  const __ppu_bfloat16* staged_src = nullptr;
  auto prefetch_key = [&](int next_base, uint32_t cached = 0, uint32_t future = 0) {
#pragma unroll
    for (int r = 0; r < TG; r += StageWarps) {
      const int slot = warp + r;
      uint32_t tag = 0;
      if (next_base + slot < walk) {
        if constexpr (Heads == 16) {
          if (next_base & 31)
            tag = __shfl_sync(0xffffffffu, cached, (next_base & 31) + slot);
          else if (next_base)
            tag = future;
          else
            tag = uint32_t(p.read_slot(first, slot) >> 32);
        } else {
          tag = uint32_t(p.read_slot(first, next_base + slot) >> 32);
        }
      }
      tag = __builtin_ppu_to_uniform_b32(tag);
      const int64_t offset = p.offset<Scaled>(tag >> 4);
      if constexpr (Partition && Heads == 16) {
        staged_visible = tag & 15;
        staged_src = p.kv.data + offset;
      }
      stage_shared_region<Heads>(key_base, p.kv.data + offset, slot, 8, 0,
                                 p.kv.row_bytes());
    }
    asm volatile("ppu.cp.async.commit_group;" ::: "memory");
  };
  if constexpr (!Private) {
    if (walk) prefetch_key(0);
  }
  uint64_t next_entry = 0;
  if constexpr (!Private && Heads == 16)
    next_entry = lane < walk ? p.read_slot(first, lane) : 0;
  uint32_t cached_tag = 0, cached_limits[TwoRows ? 2 : 1]{};
  int scan = 0;
  uint32_t remaining = 0, lane_entry = 0;
  auto run_section = [&](auto tail_tag, int begin, int end) {
    constexpr bool Tail = decltype(tail_tag)::value;
    // Each half reads and overwrites only its own KV slots in the tail.
    auto sync_stage = [&] {
      if constexpr (Tail) {
        if (warp < 4) __ppu_barrier_sync(2, 128, 15u);
        else __ppu_barrier_sync(3, 128, 15u);
      } else {
        __syncthreads();
      }
    };

    for (int base = begin; Private ? (scan < walk || remaining != 0) : base < end;
         base += TG) {
      uint32_t future_tag = 0;
      if constexpr (!Private && Heads == 16) {
        // Fetch the next 32 entries halfway through the current window.
        if ((base & 31) == 16) {
          const int next_slot = base + 16 + lane;
          next_entry = next_slot < walk ? p.read_slot(first, next_slot) : 0;
        }
        if ((base & 31) + TG == 32)
          future_tag = __shfl_sync(0xffffffffu, uint32_t(next_entry >> 32), warp);
      }
      uint64_t entries[Private ? TG : 1];
      uint32_t row_limits[TwoRows ? 2 : 1]{};
      if constexpr (Private) {
#pragma unroll
        for (int r = 0; r < TG; ++r) {
          while (!remaining && scan < walk) {
            const auto entry =
                scan + lane < walk ? p.read_slot(first, scan + lane) : 0;
            const auto n = (uint32_t(entry) >> (4 * (warp + row_offset))) & 15u;
            lane_entry = (uint32_t(entry >> 32) & ~15u) | n;
            remaining = __ballot_sync(0xffffffffu, n != 0);
            scan += 32;
          }
          entries[r] = 0;
          if (remaining) {
            const uint32_t id =
                __shfl_sync(0xffffffffu, lane_entry, __ffs(remaining) - 1);
            entries[r] = (uint64_t(id) << 32) | (id & 15u);
            remaining &= remaining - 1;
          }
        }
      } else {
        static_assert(32 % TG == 0);
        if ((base & 31) == 0) {
          const uint64_t entry = Heads == 16 ? next_entry :
              (base + lane < walk ? p.read_slot(first, base + lane) : 0);
          cached_tag = uint32_t(entry >> 32);
#pragma unroll
          for (int half = 0; half < (TwoRows ? 2 : 1); ++half) {
            const int shift = 4 * (warp + StageWarps * half);
            uint32_t bits = ((uint32_t(entry) >> shift) & 15u)
                            << (4 * (lane & 7));
#pragma unroll
            for (int step = 1; step < TG; step *= 2)
              bits |= __shfl_xor_sync(0xffffffffu, bits, step);
            cached_limits[half] = bits;
          }
        }
#pragma unroll
        for (int half = 0; half < (TwoRows ? 2 : 1); ++half)
          row_limits[half] = __builtin_ppu_to_uniform_b32(
              __shfl_sync(0xffffffffu, cached_limits[half], base & 31));
      }
      if constexpr (Private) {
#pragma unroll
        for (int r = 0; r < TG; ++r)
          p.copy<8, Scaled>(&sk[r][0][0][0], entries[r], false);
        asm volatile("ppu.cp.async.commit_group;" ::: "memory");
        asm volatile("ppu.cp.async.wait_group 0;" ::: "memory");
      } else {
        if constexpr (!value_before_key_wait) {
          asm volatile("ppu.cp.async.wait_group 0;" ::: "memory");
          sync_stage();
        }
        if constexpr (Heads == 8) {
          if (base) sync_stage();
        }
#pragma unroll
        for (int r = 0; r < TG; r += StageWarps) {
          const int slot = warp + r;
          const __ppu_bfloat16* src;
          int visible;
          if constexpr (Partition && Heads == 16) {
            src = staged_src;
            visible = staged_visible;
          } else {
            const uint32_t tag = __builtin_ppu_to_uniform_b32(
                __shfl_sync(0xffffffffu, cached_tag, (base & 31) + slot));
            src = p.kv.data + p.offset<Scaled>(tag >> 4);
            visible = tag & 15;
          }
          const int start = visible == 0 ? 8 : 0;
          stage_shared_region<Heads>(value_base, src + kHeadDim, slot,
                                     visible + start, start, p.kv.row_bytes());
        }
        asm volatile("ppu.cp.async.commit_group;" ::: "memory");
        if constexpr (value_before_key_wait)
          asm volatile("ppu.cp.async.wait_group 1;" ::: "memory");
      }
      if constexpr (Private)
        __syncwarp();
      else if constexpr (value_before_key_wait)
        sync_stage();
      auto wait_for_value_and_prefetch_key = [&] {
        asm volatile("ppu.cp.async.wait_group 0;" ::: "memory");
        if constexpr (Private) {
          __syncwarp();
        } else {
          sync_stage();
          if (base + TG < walk) prefetch_key(base + TG, cached_tag, future_tag);
        }
      };
      constexpr bool reschedule = Partition && !IssueValueEarly;
      constexpr bool Half = Tail;
      constexpr int NT = Half ? 2 : BS / 16;
      const int start_t = Half ? (warp / 4) * 2 : 0;
      if constexpr (Half) row_limits[0] >>= (warp / 4) * 16;
      MmaAccumulator score[NT];
#pragma unroll
      for (int t = 0; t < NT; ++t) clear_fragment(score[t]);
      auto load_key = [&](MmaOperand& key, int k, int t) {
        const int chunk = (((k % 4) * 16 + b_col) >> 3) ^ lr;
        if constexpr (Private)
          ldm_x4(key, &sk[t * 2 + (lu >> 1)][k / 4][lr][chunk << 3]);
        else
          load_paired<false>(
              key,
              paired_cube<Heads>(key_base, (t + start_t) * 2, k < 8 ? 0 : k / 4) +
                  (k % 4) * 16 + (k < 8 ? (k / 4) * 8 : 0),
              Heads == 8 ? 192 : (k < 8 ? 128 : 64), k < 8 ? 65 - 2 * (k / 4) : 1);
      };
      if constexpr (reschedule) {
        // Keep two QK steps of matrix operands in flight.
        constexpr int lookahead = 2 * NT;
        MmaOperand keys[lookahead];
#pragma unroll
        for (int op = 0; op < lookahead; ++op) load_key(keys[op], op / NT, op % NT);
#pragma unroll
        for (int op = 0; op < kHeadMmaTiles * NT; ++op) {
          MmaOperand key = keys[op % lookahead];
          constexpr int operations = kHeadMmaTiles * NT;
          if (op + lookahead < operations)
            load_key(keys[op % lookahead], (op + lookahead) / NT, op % NT);
          mma_bf16(score[op % NT], qfrag[op / NT], key, score[op % NT]);
        }
      } else {
#pragma unroll
        for (int k = 0; k < kHeadMmaTiles; ++k) {
#pragma unroll
          for (int t = 0; t < NT; t += key_batch) {
            MmaOperand keys[key_batch];
#pragma unroll
            for (int i = 0; i < key_batch; ++i) load_key(keys[i], k, t + i);
#pragma unroll
            for (int i = 0; i < key_batch; ++i)
              mma_bf16(score[t + i], qfrag[k], keys[i], score[t + i]);
          }
        }
      }
      if constexpr (Private) {
        __syncwarp();
#pragma unroll
        for (int r = 0; r < TG; ++r)
          p.copy<8, Scaled>(&sv[r][0][0][0], entries[r], true);
        asm volatile("ppu.cp.async.commit_group;" ::: "memory");
      }
      float tile_max[2] = {-INFINITY, -INFINITY};
      if (!Private &&
          (row_limits[0] & (Half ? 0xffffu : ~0u)) == (Half ? 0x8888u : 0x88888888u) &&
          (!TwoRows || row_limits[TwoRows ? 1 : 0] == 0x88888888u)) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
          for (int i = 0; i < 8; ++i)
            tile_max[(i >> 1) & 1] = fmaxf(tile_max[(i >> 1) & 1], score[t].x[i]);
        }
      } else if constexpr (!Private && Partition) {
        const bool partial = (row_limits[0] & 0x77777777u) != 0;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            int half = i >> 2, tau = (i >> 1) & 1;
            int visible = (row_limits[0] >> (4 * (t * 2 + half))) & 15;
            bool ok = partial ? 2 * t0 + (i & 1) < visible : visible != 0;
            float z = ok ? score[t].x[i] : -INFINITY;
            score[t].x[i] = z;
            tile_max[tau] = fmaxf(tile_max[tau], z);
          }
        }
      } else {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int tau = (i >> 1) & 1, half = i >> 2;
            const int visible =
                Private ? int(entries[t * 2 + half])
                        : (row_limits[TwoRows ? tau : 0] >> (4 * (t * 2 + half))) & 15;
            const bool valid = 2 * t0 + (i & 1) < visible;
            score[t].x[i] = valid ? score[t].x[i] : -INFINITY;
            tile_max[tau] = fmaxf(tile_max[tau], score[t].x[i]);
          }
        }
      }
      if (!__all_sync(0xffffffffu, tile_max[0] * scale2 <= maximum[0] &&
                                       tile_max[1] * scale2 <= maximum[1])) {
        float rescale[2];
#pragma unroll
        for (int tau = 0; tau < 2; ++tau) {
          const float tile_peak = head_max(tile_max[tau]);
          const float bound = ceilf(tile_peak * (scale2 * 0.125f)) * 8.f;
          const bool first_nonempty =
              Heads == 8 && maximum[tau] == initial_max && tile_peak != -INFINITY;
          const float next =
              first_nonempty ? tile_peak * scale2 : fmaxf(maximum[tau], bound);
          rescale[tau] = __exp2f(maximum[tau] - next);
          maximum[tau] = next;
          denominator[tau] *= rescale[tau];
        }
#pragma unroll
        for (int j = 0; j < kHeadMmaTiles; ++j) {
#pragma unroll
          for (int i = 0; i < 8; ++i) out[j].x[i] *= rescale[(i >> 1) & 1];
        }
      }
      if constexpr (reschedule) wait_for_value_and_prefetch_key();
      MmaOperand prob[NT];
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          const int tau = (i >> 1) & 1;
          const float a = __exp2f(fmaf(score[t].x[i], scale2, -maximum[tau]));
          const float b = __exp2f(fmaf(score[t].x[i + 1], scale2, -maximum[tau]));
          *reinterpret_cast<__ppu_bfloat162*>(&prob[t].x[i / 2]) =
              __floats2bfloat162_rn(a, b);
          // Use unrounded probabilities for the FP32 denominator.
          denominator[tau] += a + b;
        }
      }
      if constexpr (!reschedule) wait_for_value_and_prefetch_key();
      constexpr int batch = reschedule ? (Half ? 6 : 3) : value_batch;
      constexpr int groups = kHeadMmaTiles / batch;
#pragma unroll
      for (int step = 0; step < NT * groups; ++step) {
        const int t = reschedule ? step % NT : step / groups;
        const int j = batch * (reschedule ? step / NT : step % groups);
        MmaOperand values[batch];
#pragma unroll
        for (int i = 0; i < batch; ++i) {
          const int d = j + i;
          const int chunk = (((d % 4) * 16 + a_col) >> 3) ^ lr;
          if constexpr (Private)
            ldm_x4_trans(values[i], &sv[t * 2 + (lu & 1)][d / 4][lr][chunk << 3]);
          else
            load_paired<true>(
                values[i],
                paired_cube<Heads>(value_base, (t + start_t) * 2, d < 8 ? 0 : d / 4) +
                    (d % 4) * 16 + (d < 8 ? (d / 4) * 8 : 0),
                d < 8 ? 64 : 1, Heads == 8 ? 192 : (d < 8 ? 128 : 64));
        }
#pragma unroll
        for (int i = 0; i < batch; ++i)
          mma_bf16(out[j + i], prob[t], values[i], out[j + i]);
      }
      if constexpr (Private) __syncwarp();
      else if constexpr (Heads == 16 && value_before_key_wait)
        sync_stage();
    }
  };
  if constexpr (SplitRows) {
    run_section(std::false_type{}, 0, split_start);
    run_section(std::true_type{}, split_start, walk);
  } else {
    run_section(std::false_type{}, 0, walk);
  }
#pragma unroll
  for (int tau = 0; tau < Halves; ++tau) {
    const int qrow = row + (TwoRows ? StageWarps * tau : 0);
    if (qrow >= end_row) continue;
    const int h = head + (Heads == 16 ? tau * 8 : 0);
    const float denom = head_sum(denominator[tau]);
    const float inv = denom > 0 ? 1.f / denom : 0.f;
#pragma unroll
    for (int j = 0; j < kHeadMmaTiles; ++j) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int i = 2 * tau + 4 * half;
        const int d = !Private && j < 8
                          ? (j % 4) * 16 + (j / 4) * 8 + half * 64 + 2 * t0
                          : j * 16 + 2 * t0 + 8 * half;
        const float a = out[j].x[i] * inv, b = out[j].x[i + 1] * inv;
        *reinterpret_cast<__ppu_bfloat162*>(p.out + qrow * p.os + h * p.oh + d) =
            __floats2bfloat162_rn(a, b);
      }
    }
  }
#endif
}

template <int Heads, bool Partition, bool Scaled>
__global__ __launch_bounds__(Heads * 16) void step5_native_attention_kernel(Params p) {
#if __HGGC_ARCH__ >= 150
  if (!p.valid_tiles<Partition>()) return;
  constexpr int tiles = kGroupRows / 8;
  const int block = blockIdx.x;
  int group;
  if constexpr (Heads == 16 && !Partition) {
    group = (block & 1) ? block / 2 : p.groups - 1 - block / 2;
  } else if constexpr (Heads == 16) {
    auto group_count = [&](int g) { return p.tile_counts[p.tile_index(g, 0)]; };
    const int first_count = group_count(0);
    const int mid_count = group_count(p.groups / 2);
    const int last_count = group_count(p.groups - 1);
    const bool irregular = max(first_count, max(mid_count, last_count)) -
                               min(first_count, min(mid_count, last_count)) >
                           kWidth / 4;
    if (p.kv.requests == 1) group = block;
    else if (!irregular)
      group = (block & 1) ? block / 2 : p.groups - 1 - block / 2;
    else {
      constexpr int stripe = 32;
      const int pair = block / (2 * stripe);
      const int offset = block % (2 * stripe);
      const int high = min(stripe, p.groups - 2 * pair * stripe);
      group = offset < high ? p.groups - 1 - pair * stripe - offset
                            : pair * stripe + offset - high;
    }
  } else {
    group = p.groups - 1 - block / tiles;
  }
  const int end_row = min(p.rows, p.ends[group]);
  const int start_tile = Heads == 16 ? 0 : blockIdx.x % tiles;
  const int end_tile = Heads == 16 ? tiles : start_tile + 1;
  for (int tile = start_tile; tile < end_tile; ++tile) {
    const int first = p.starts[group] + 8 * tile;
    if (first >= end_row) break;
    if (Heads == 16 && tile) __syncthreads();
    const int count = p.tile_counts[p.tile_index(group, tile)];
    const int cut_code =
        Partition ? p.tile_counts[p.tile_index(group, tile) + kSplitBoundarySlot] : 0;
    if (count > 4 * kWidth && !cut_code) {
      if constexpr (Heads == 8) {
#pragma unroll 1
        for (int offset = 0; offset < 8; offset += 4)
          native_tile<true, 8>(p, first, end_row, count, offset);
      } else {
        native_tile<true, 16, false, false, false, Scaled>(p, first, end_row, count);
      }
    } else {
      if constexpr (Partition) {
        if (count == p.width && p.live(first) == p.width && !cut_code) {
          native_tile<false, Heads, false, true, Partition, Scaled>(p, first, end_row,
                                                                   count);
          continue;
        }
        if (cut_code) {
          native_tile<false, Heads, true, false, Partition, Scaled>(
              p, first, end_row, count, 0, cut_code - 1);
          continue;
        }
      }
      native_tile<false, Heads, false, false, Partition, Scaled>(p, first, end_row,
                                                                count);
    }
  }
#endif
}
template <int BQ, bool Partition = false, bool Scaled = false> void launch(Params p, hggcStream_t stream) {
  constexpr int HT = kHeads / 8, BS = BQ == 1 ? 16 : 64, PARTS = BQ == 1 ? 8 : 1;
  constexpr bool group_cta = BQ == 8 && kHeads == 16;
  const int blocks = p.groups * (group_cta ? 1 : kGroupRows / BQ);
  if constexpr (BQ > 1) {
    constexpr int bytes = BQ * kWords * sizeof(uint32_t);
    if (p.logical_regions > 2 * kWindowRegions) {
      step5_runtime::check(hggcFuncSetAttribute(step5_prepare_union_kernel<BQ, group_cta, 2 * kWords, Partition, Scaled>, hggcFuncAttributeMaxDynamicSharedMemorySize, 2 * bytes));
      step5_prepare_union_kernel<BQ, group_cta, 2 * kWords, Partition, Scaled><<<blocks, 2 * kWords, 2 * bytes, stream>>>(p);
    } else step5_prepare_union_kernel<BQ, group_cta, kWords, Partition, Scaled><<<blocks, kWords, bytes, stream>>>(p);
    step5_runtime::check(hggcGetLastError());
  }
  constexpr int threads = (BQ == 1 ? 1 : BQ / 2) * HT * 32, bytes = BS * 192 * 4 * PARTS;
  static_assert(PARTS == 1 || ((PARTS - 1) * HT * 8 * 192 + PARTS * HT * 8) * 4 <= bytes);
  if constexpr (BQ == 8) {
    step5_runtime::check(hggcFuncSetAttribute(step5_native_attention_kernel<kHeads, Partition, Scaled>, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    step5_native_attention_kernel<kHeads, Partition, Scaled><<<blocks, threads, bytes, stream>>>(p);
  } else {
    step5_runtime::check(hggcFuncSetAttribute(step5_paged_attention_kernel<BQ, HT, BS, PARTS>, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    step5_paged_attention_kernel<BQ, HT, BS, PARTS><<<blocks, threads * PARTS, bytes, stream>>>(p);
  }
}
void run(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& regions,
    const at::Tensor& table, const at::Tensor& owners, const at::Tensor& pos,
    const at::Tensor& starts, const at::Tensor& ends, const at::Tensor& out,
    int64_t logical_regions, double scale, int64_t query_tile) {
  if (!q.size(0)) return;
  const c10::cuda::CUDAGuard guard(q.device());
  const auto options = q.options().dtype(at::kInt);
  auto slots = at::empty(regions.sizes(), options.dtype(at::kLong));
  auto tile_counts = at::empty({starts.numel(), kGroupRows}, options);
  const int tile = query_tile ? query_tile : q.size(0) <= 32 ? 1 : q.size(0) <= 256 ? 4 : 8;
  const int validation_blocks = kHeads == 16 && tile == 8 && starts.numel() > 64
                                    ? kValidationBlocks : 1;
  auto tiles_valid = at::empty({validation_blocks}, options);
  Params p{};
  p.kv = paged_kv(cache, table); p.q = reinterpret_cast<const __ppu_bfloat16*>(q.data_ptr());
  p.out = reinterpret_cast<__ppu_bfloat16*>(out.data_ptr()); p.pos = pos.data_ptr<int64_t>();
  p.regions = regions.data_ptr<int64_t>(); p.owners = owners.data_ptr<int64_t>();
  p.starts = starts.data_ptr<int32_t>(); p.ends = ends.data_ptr<int32_t>();
  p.slots = reinterpret_cast<uint64_t*>(slots.data_ptr<int64_t>());
  p.tile_counts = tile_counts.data_ptr<int32_t>(); p.tiles_valid = tiles_valid.data_ptr<int32_t>();
  p.rows = q.size(0); p.groups = starts.numel();
  p.width = regions.size(1); p.logical_regions = logical_regions; p.region_stride = regions.stride(0); p.scale = scale;
  p.qs = q.stride(0); p.qh = q.stride(1); p.os = out.stride(0); p.oh = out.stride(1);
  const auto stream = get_current_cuda_stream(q.get_device());
  step5_validate_tiles_kernel<kHeads><<<validation_blocks, 256, 0, stream>>>(p);
  step5_runtime::check(hggcGetLastError());
  switch (tile) {
    case 1: launch<1>(p, stream); break;
    case 2: launch<2>(p, stream); break;
    case 4: launch<4>(p, stream); break;
    case 8:
      if constexpr (kHeads == 16) {
        if (p.groups > 64) {
          // A 28-bit tag in 512-element units avoids repeated page-stride
          // multiplies in the main loop. Wider or unaligned views use page IDs.
          const int64_t last = (cache.size(0) - 1) * cache.stride(0) +
                               (cache.size(2) - kRegionTokens) * cache.stride(2);
          if ((cache.stride(0) & 511) == 0 && last < (int64_t(1) << 37))
            launch<8, true, true>(p, stream);
          else
            launch<8, true>(p, stream);
          break;
        }
      }
      launch<8>(p, stream);
      break;
  }
  step5_runtime::check(hggcGetLastError());
}
}  // namespace STEP5_NATIVE_PREFILL_NAMESPACE
