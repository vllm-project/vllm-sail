// SPDX-License-Identifier: Apache-2.0
//
// C++ implementation of the Step4 DSA fixed-decode top-k region selector.
// The streaming radix dataflow keeps shared memory independent of the context
// length, so one kernel covers short contexts and the 1M-token target.
//
// Radix scheme: [11, 11, 10] (3 rounds) replacing the original [8,8,8,8].
// Rounds 0-1 use 2048-bin histograms (11-bit extraction); round 2 uses
// 1024 bins (10-bit).  Only standard CUDA and ATen APIs are used.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "../step5_runtime.h"
#include <torch/all.h>

#ifndef STEP4_DSA_SELECTOR_THREADS
#define STEP4_DSA_SELECTOR_THREADS 1024
#endif

// Each compilation of this file produces a warp-size specialization.
// The namespace encodes the thread count so that two specializations can be
// linked into the same shared library without symbol collisions.
#if STEP4_DSA_SELECTOR_THREADS == 256
namespace step4_dsa_w256 {
#elif STEP4_DSA_SELECTOR_THREADS == 1024
namespace step4_dsa_w1024 {
#else
#error "STEP4_DSA_SELECTOR_THREADS must be 256 or 1024"
#endif

constexpr int kThreads = STEP4_DSA_SELECTOR_THREADS;
constexpr int kRadix = 1024;              // 2^10 for last round
constexpr int kCoarseRadix = 2048;        // 2^11 for first two rounds
constexpr int kNumSelectorRounds = 3;     // [11, 11, 10]
constexpr int kStreamingChunkRegions = 16384;
#ifndef STEP5_DSA_EXACT_FP32
#define STEP5_DSA_EXACT_FP32 0
#endif
constexpr float kRegionEpsilon = 1.0e-9f;

// Hybrid score cache for the single-CTA selector path.
// We keep the total cache at 8192 floats (r <= 8192) but split it between
// registers and shared memory to reduce smem footprint and improve occupancy.
// The per-thread register budget is tuned per warp size so that both 256 and
// 1024 variants stay within the SM register file and keep a non-zero smem
// buffer (zero-length arrays are avoided).
//   - warp-256: 8 regs/thread  -> 2048 register + 6144 smem
//   - warp-1024: 4 regs/thread -> 4096 register + 4096 smem
constexpr int kRegisterCachePerThread = (kThreads == 1024) ? 4 : 8;
constexpr int kRegisterCacheRegions = kRegisterCachePerThread * kThreads;
constexpr int kSmemBufferRegions = 8192 - kRegisterCacheRegions;
constexpr int kMaxBufferedRegions = 8192;  // register + smem cache capacity
static_assert(kRegisterCacheRegions <= kMaxBufferedRegions,
              "Register cache cannot exceed total buffered regions");
static_assert(kSmemBufferRegions > 0,
              "Smem buffer must be positive for all supported warp sizes");

static_assert(kThreads >= 256 && kThreads <= 1024 &&
                  (kThreads & (kThreads - 1)) == 0,
              "Step4 DSA selector threads must be a power of two in [256,1024]");

// Threshold scans read histogram bins while publishing warp subtotals.
struct Step4DsaRadixShared {
  int histogram[kCoarseRadix];
  int warp_sum[kThreads / 32];
  uint32_t prefix_value;
  uint32_t prefix_mask;
  int threshold;
  int remain;
  int selected_base;
  int output_counter;
  int count_greater;
};

// Chunk CTAs stream scores and only allocate the radix scratch.
struct Step4DsaCachedRadixShared : Step4DsaRadixShared {
  // Hybrid score cache for the single-CTA selector path.
  // Regions [0, kRegisterCacheRegions) are held in per-thread registers;
  // regions [kRegisterCacheRegions, kMaxBufferedRegions) are cached here.
  float score_buffer[kSmemBufferRegions];
};

struct Step4DsaMergeRadixShared {
  int histogram[kCoarseRadix];
  int warp_sum[kThreads / 32];
  uint32_t score_prefix_value;
  uint32_t score_prefix_mask;
  uint32_t region_prefix_value;
  uint32_t region_prefix_mask;
  int threshold;
  int remain;
  int selected_base;
  int output_counter;
  int count_greater;
};

__device__ __forceinline__ uint32_t ordered_score(float score, int region) {
  float const encoded = STEP5_DSA_EXACT_FP32
                            ? score : score + float(region) * kRegionEpsilon;
  uint32_t const raw = __float_as_uint(encoded);
  return raw ^ ((raw >> 31) != 0 ? 0xffffffffu : 0x80000000u);
}

__device__ __forceinline__ void store_region(
    int region, int slot, int request, int head, int heads, int output_width,
    int const* block_table, int block_table_stride, int block_table_length,
    int regions_per_page, int region_block, bool packed, int* region_output,
    int* start_output) {
  int selected = region;
  int start = region * region_block;
  if (packed) {
    int const logical_page = region / regions_per_page;
    int const fragment = region - logical_page * regions_per_page;
    int const physical_page =
        logical_page >= 0 && logical_page < block_table_length
            ? block_table[request * block_table_stride + logical_page]
            : -1;
    bool const valid = physical_page >= 0;
    selected = valid ? physical_page * regions_per_page + fragment : -1;
    start = valid ? start : -1;
  }
  int const offset = (request * heads + head) * output_width + slot;
  region_output[offset] = selected;
  start_output[offset] = start;
}

// Per-thread staging size for the scatter phase.  Each thread buffers up to
// this many selected region indices in registers; when the buffer fills, the
// warp performs a single batched atomicAdd and flushes the buffered regions
// to contiguous output slots, reducing atomic contention and improving
// coalescing of the output writes.
constexpr int kThreadStaging = 8;

// Compute the contiguous output slot range for this warp's staged selected
// regions and claim it with a single global atomicAdd.  The per-thread counts
// are prefix-summed via warp shuffle (always executed with the full 32-lane
// mask so the SIMT mask stays consistent on PPU).  Returns the base slot
// claimed for the warp and writes this lane's exclusive offset into
// warp_exclusive.
__device__ __forceinline__ int claim_warp_output_slots(
    int staged_count, int thread, Step4DsaRadixShared& storage,
    int& warp_exclusive) {
  int const lane_id = thread & 31;
  int warp_inclusive = staged_count;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    int const n = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, offset);
    if (lane_id >= offset) warp_inclusive += n;
  }
  int const warp_total = __shfl_sync(0xFFFFFFFF, warp_inclusive, 31);
  warp_exclusive = warp_inclusive - staged_count;
  int base_slot = 0;
  if (lane_id == 0) {
    base_slot = atomicAdd(&storage.output_counter, warp_total);
  }
  base_slot = __shfl_sync(0xFFFFFFFF, base_slot, 0);
  return base_slot;
}

// Hierarchical warp-shuffle suffix sum for equal-count reduction.
// Returns the number of "equal" regions held by threads with a larger global
// index than this thread, using two warp-level shuffle passes.
__device__ __forceinline__ int equal_count_higher_equal(
    int local_equal, int thread, Step4DsaRadixShared& storage) {
  int const lane_id = thread & 31;
  int const warp_id = thread >> 5;

  // Step 1: Inclusive suffix sum within each warp.
  int warp_suffix = local_equal;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    int const n = __shfl_down_sync(0xFFFFFFFF, warp_suffix, offset);
    if (lane_id + offset < 32) warp_suffix += n;
  }
  int const warp_total = __shfl_sync(0xFFFFFFFF, warp_suffix, 0);
  if (lane_id == 0) {
    storage.warp_sum[warp_id] = warp_total;
  }
  __syncthreads();

  // Step 2: Suffix sum of per-warp totals (one warp's worth of data).
  if (thread < 32) {
    int const val = (thread < kThreads / 32) ? storage.warp_sum[thread] : 0;
    int suffix = val;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      int const n = __shfl_down_sync(0xFFFFFFFF, suffix, offset);
      if (lane_id + offset < 32) suffix += n;
    }
    if (thread < kThreads / 32) {
      storage.warp_sum[thread] = suffix;
    }
  }
  __syncthreads();

  // Step 3: Combine intra-warp suffix (excluding this lane) with suffix of
  // all later warps.
  return (warp_suffix - local_equal) +
         ((warp_id + 1 < kThreads / 32) ? storage.warp_sum[warp_id + 1] : 0);
}

// Warp-based threshold bin selection parameterised on histogram size.
//
// Phase 1: Warp-level inclusive prefix sum via __shfl_up_sync; warp totals
//          published to shared memory.
// Phase 2: Warp-0 computes the serial exclusive prefix of warp totals and
//          publishes the total (via output_counter) and per-warp exclusive
//          offsets back to warp_sum[].
// Phase 3: All bin-owning threads combine the warp exclusive offset with
//          their intra-warp prefix to obtain the per-bin exclusive prefix
//          sum, then search for the threshold in parallel.
//
// Only two __syncthreads (vs 2*log2(n_bins) in the tree-based original).
template <int n_bins, typename SharedStorage>
__device__ __forceinline__ void choose_threshold_parallel(
    SharedStorage& storage, int thread) {
  constexpr int kWarps = kThreads / 32;
  constexpr int kOwningThreads = (kThreads < n_bins) ? kThreads : n_bins;
  constexpr int kItemsPerThread =
      (kThreads <= n_bins) ? (n_bins / kThreads) : 0;

  int const lane_id = thread & 31;
  int const warp_id = thread >> 5;
  bool const owns_bin = (thread < kOwningThreads);

  // Step 1: Load owned histogram items into registers.
  // Max items per thread occurs at kThreads=256, n_bins=4096 => 16.
  int local_sum = 0;
  int orig[16];
  static_assert(kThreads >= 256, "orig[] sized for kThreads >= 256");
  if constexpr (kThreads <= n_bins) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      orig[i] = storage.histogram[thread * kItemsPerThread + i];
      local_sum += orig[i];
    }
  } else {
    orig[0] = owns_bin ? storage.histogram[thread] : 0;
    local_sum = orig[0];
  }

  // Step 2: Warp-level inclusive prefix sum via __shfl_up_sync.
  int warp_inc = local_sum;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    int n = __shfl_up_sync(0xFFFFFFFF, warp_inc, offset);
    if (lane_id >= offset) warp_inc += n;
  }
  int const warp_exc = warp_inc - local_sum;

  // Step 3: Last lane of each warp publishes the warp total.
  if (lane_id == 31) {
    storage.warp_sum[warp_id] = warp_inc;
  }
  __syncthreads();

  // Step 4: Warp 0 computes the exclusive prefix of warp totals in parallel
  // using warp shuffle. Each lane loads one warp total, performs a Kogge-Stone
  // inclusive prefix sum, then writes back the exclusive value. This reduces
  // the serial latency of the original loop and helps warps arrive at the next
  // barrier more closely together.
  if (thread < 32) {
    int const val = (lane_id < kWarps) ? storage.warp_sum[lane_id] : 0;
    int prefix = val;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      int const n = __shfl_up_sync(0xFFFFFFFF, prefix, offset);
      if (lane_id >= offset) prefix += n;
    }
    int const exclusive = prefix - val;
    if (lane_id < kWarps) {
      storage.warp_sum[lane_id] = exclusive;
    }
    if (lane_id == kWarps - 1) {
      storage.output_counter = prefix;  // total
      storage.remain = min(storage.remain, prefix);
    }
  }
  __syncthreads();

  // Step 5: Read total and per-warp exclusive prefix from shared memory.
  int const total = storage.output_counter;
  int const warp_exclusive = owns_bin ? storage.warp_sum[warp_id] : 0;

  // Step 6: Combine warp_exclusive + warp_exc for per-bin exclusive prefix.
  int prefix_sum = warp_exclusive + warp_exc;
  int local_threshold = -1;
  int local_count_greater = 0;

  if constexpr (kThreads <= n_bins) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      prefix_sum += orig[i];
      int const count_greater = total - prefix_sum;
      if (count_greater < storage.remain &&
          count_greater + orig[i] >= storage.remain) {
        local_threshold = thread * kItemsPerThread + i;
        local_count_greater = count_greater;
      }
    }
  } else {
    if (owns_bin) {
      prefix_sum += orig[0];
      int const count_greater = total - prefix_sum;
      if (count_greater < storage.remain &&
          count_greater + orig[0] >= storage.remain) {
        local_threshold = thread;
        local_count_greater = count_greater;
      }
    }
  }

  // Step 7: Publish result.
  if (local_threshold >= 0) {
    storage.threshold = local_threshold;
    storage.count_greater = local_count_greater;
  }
  __syncthreads();
}

// Stable-sort epilogue fused into the single-CTA selector, reproducing the
// Triton _stable_topk_regions SORT=True contract: ignore-NaN cutoff, ties filled
// by smallest region ID, emitted as [-1 pad] + ascending IDs + [current].
struct Step4DsaSortShared {
  int sort_buf[2048];
  float warp_min[kThreads / 32];
  int warp_sums[kThreads / 32];
  int warp_offsets[kThreads / 32];
  float threshold_f;
  int counter;
  int found;
};
static_assert(sizeof(Step4DsaSortShared) <= sizeof(Step4DsaCachedRadixShared),
              "sort epilogue shared memory must fit the selector allocation");

__device__ __forceinline__ float step4_dsa_warp_fminf(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fminf(v, __shfl_down_sync(0xFFFFFFFFu, v, off));
  }
  return v;
}

__device__ __forceinline__ void step4_dsa_stable_sort_epilogue(
    char* raw_shared, float const* score_row, int current, int topk,
    int const* region_output, int64_t* sorted_output, int work,
    int output_width, int thread) {
  auto& sh = *reinterpret_cast<Step4DsaSortShared*>(raw_shared);
  int const nwarps = kThreads / 32;
  int const lane = thread & 31;
  int const warp = thread >> 5;
  int block_sort = 1;
  while (block_sort < topk + 1) block_sort <<= 1;

  for (int i = thread; i < block_sort; i += kThreads) {
    sh.sort_buf[i] = (i < topk) ? -1 : (i == topk ? current : 2147483647);
  }
  if (thread == 0) sh.counter = 0;
  __syncthreads();

  // threshold = fminf over candidate scores (invalid slots = +inf, NaN ignored).
  float local_min = INFINITY;
  for (int slot = thread; slot < topk; slot += kThreads) {
    int const id = region_output[work * output_width + slot];
    bool const valid = (id >= 0 && id < current);
    float const val = valid ? score_row[id] : INFINITY;
    local_min = fminf(local_min, val);
  }
  float const wmin = step4_dsa_warp_fminf(local_min);
  if (lane == 0) sh.warp_min[warp] = wmin;
  __syncthreads();
  if (thread < 32) {
    float v = (thread < nwarps) ? sh.warp_min[thread] : INFINITY;
    v = step4_dsa_warp_fminf(v);
    if (thread == 0) sh.threshold_f = v;
  }
  __syncthreads();
  float const threshold = sh.threshold_f;

  // Keep every candidate strictly above the cutoff (order irrelevant: sorted).
  for (int slot = thread; slot < topk; slot += kThreads) {
    int const id = region_output[work * output_width + slot];
    bool const valid = (id >= 0 && id < current);
    float const val = valid ? score_row[id] : INFINITY;
    if (valid && val > threshold) {
      int const pos = atomicAdd(&sh.counter, 1);
      sh.sort_buf[pos] = id;
    }
  }
  __syncthreads();
  int const num_above = sh.counter;
  int const needed = min(current, topk) - num_above;

  // Fill ties with the smallest region IDs whose score equals the cutoff.
  if (thread == 0) sh.found = 0;
  __syncthreads();
  int start = 0;
  while (sh.found < needed && start < current) {
    int const region = start + thread;
    int const eq =
        (region < current && score_row[region] == threshold) ? 1 : 0;
    int incl = eq;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int const n = __shfl_up_sync(0xFFFFFFFFu, incl, off);
      if (lane >= off) incl += n;
    }
    if (lane == 31) sh.warp_sums[warp] = incl;
    __syncthreads();
    if (thread < 32) {
      int const w = (thread < nwarps) ? sh.warp_sums[thread] : 0;
      int wincl = w;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        int const n = __shfl_up_sync(0xFFFFFFFFu, wincl, off);
        if (thread >= off) wincl += n;
      }
      if (thread < nwarps) sh.warp_offsets[thread] = wincl - w;
    }
    __syncthreads();
    int const excl = sh.warp_offsets[warp] + (incl - eq);
    int const gpos = sh.found + excl;
    if (eq && gpos < needed) {
      sh.sort_buf[num_above + gpos] = region;
    }
    int const chunk_total = sh.warp_offsets[nwarps - 1] + sh.warp_sums[nwarps - 1];
    __syncthreads();
    if (thread == 0) sh.found += chunk_total;
    __syncthreads();
    start += kThreads;
  }

  // Ascending bitonic sort over block_sort entries (-1 pad first, current last
  // among real values, INT_MAX padding beyond topk).
  __syncthreads();
  for (int k = 2; k <= block_sort; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = thread; i < block_sort; i += kThreads) {
        int const ixj = i ^ j;
        if (ixj > i) {
          bool const asc = ((i & k) == 0);
          int const a = sh.sort_buf[i];
          int const b = sh.sort_buf[ixj];
          if ((a > b) == asc) {
            sh.sort_buf[i] = b;
            sh.sort_buf[ixj] = a;
          }
        }
      }
      __syncthreads();
    }
  }
  for (int col = thread; col <= topk; col += kThreads) {
    sorted_output[work * (topk + 1) + col] =
        static_cast<int64_t>(sh.sort_buf[col]);
  }
}

// Selector kernel: 3 rounds [11, 11, 10].
__global__ void step4_dsa_radix_selector_kernel(
    float const* scores, int64_t const* query_current,
    int const* block_table, int* region_output, int* start_output, int heads,
    int regions, int topk, int region_block, int page_size,
    int block_table_stride, int block_table_length, bool packed,
    int64_t* sorted_output) {
  extern __shared__ char raw_shared[];
  auto& storage = *reinterpret_cast<Step4DsaCachedRadixShared*>(raw_shared);
  int const work = blockIdx.x;
  int const request = work / heads;
  int const head = work - request * heads;
  int const thread = threadIdx.x;
  int const output_width = topk + 1;
  int64_t const current_in = query_current[request];
  int const current = static_cast<int>(
      current_in <= 0 ? 0 : min(current_in, int64_t(regions - 1)));
  int const target = min(topk, current);
  // Match the Triton stabilization's region count clamped to regions (not
  // regions-1) so the fused sort agrees when context fills the score row.
  int const sort_length = static_cast<int>(
      current_in <= 0 ? 0 : min(current_in, int64_t(regions)));
  float const* score_row = scores + (request * heads + head) * regions;

  if (current <= topk) {
    for (int slot = thread; slot < output_width; slot += kThreads) {
      if (slot < current || slot == topk) {
        store_region(slot == topk ? current : slot, slot, request, head, heads,
                     output_width, block_table, block_table_stride,
                     block_table_length, page_size / region_block, region_block,
                     packed, region_output, start_output);
      } else {
        int const offset = work * output_width + slot;
        region_output[offset] = -1;
        start_output[offset] = -1;
      }
    }
    if (sorted_output != nullptr) {
      __syncthreads();
      step4_dsa_stable_sort_epilogue(raw_shared, score_row, sort_length, topk,
                                     region_output, sorted_output, work,
                                     output_width, thread);
    }
    return;
  }

  for (int slot = thread; slot < output_width; slot += kThreads) {
    int const offset = work * output_width + slot;
    region_output[offset] = -1;
    start_output[offset] = -1;
  }
  if (thread == 0) {
    storage.prefix_value = 0;
    storage.prefix_mask = 0;
    storage.remain = target;
    storage.selected_base = 0;
  }
  __syncthreads();

  // Layer 1: hybrid score cache for small contexts.
  // Split the 8192-element cache between registers (2048 floats) and shared
  // memory (6144 floats) to reduce smem footprint while retaining the benefits
  // of a one-time load.  Regions >= kMaxBufferedRegions fall back to streaming
  // from global memory.
  bool const use_hybrid_cache = (current <= kMaxBufferedRegions);
  int const register_count = use_hybrid_cache ? min(current, kRegisterCacheRegions) : 0;
  int const smem_count = use_hybrid_cache ? max(0, current - kRegisterCacheRegions) : 0;
  float register_cache[kRegisterCachePerThread];
#pragma unroll
  for (int i = 0; i < kRegisterCachePerThread; ++i) {
    register_cache[i] = 0.0f;
  }

  if (register_count > 0) {
#pragma unroll
    for (int i = 0; i < kRegisterCachePerThread; ++i) {
      int const region = thread * kRegisterCachePerThread + i;
      if (region < register_count) {
        register_cache[i] = score_row[region];
      }
    }
  }
  if (smem_count > 0) {
    // Simple scalar load path (instead of cp.async) to verify correctness
    // independent of async copy alignment / behavior.
    for (int smem_region = thread; smem_region < smem_count;
         smem_region += kThreads) {
      int const global_region = kRegisterCacheRegions + smem_region;
      storage.score_buffer[smem_region] = score_row[global_region];
    }
    __syncthreads();
  }
  float const* score_src = use_hybrid_cache ? nullptr : score_row;

  // [11, 11, 10]: shifts = {21, 10, 0}, bins = {2048, 2048, 1024}
#pragma unroll
  for (int round = 0; round < kNumSelectorRounds; ++round) {
    int const n_bins = (round < 2) ? kCoarseRadix : kRadix;
    int const bits   = (round < 2) ? 11 : 10;
    int const shift  = (round < 2) ? (32 - (round + 1) * 11) : 0;
    uint32_t const bin_mask = (1u << bits) - 1u;

    for (int bin = thread; bin < n_bins; bin += kThreads) {
      storage.histogram[bin] = 0;
    }
    __syncthreads();

    uint32_t const prefix_value = storage.prefix_value;
    uint32_t const prefix_mask = storage.prefix_mask;
    if (use_hybrid_cache) {
      // Histogram from register cache (8 elements per thread, unrolled).
#pragma unroll
      for (int i = 0; i < kRegisterCachePerThread; ++i) {
        int const region = thread * kRegisterCachePerThread + i;
        if (region < register_count) {
          uint32_t const key = ordered_score(register_cache[i], region);
          if ((key & prefix_mask) == prefix_value) {
            atomicAdd(&storage.histogram[int((key >> shift) & bin_mask)], 1);
          }
        }
      }
      // Histogram from smem cache (strided).
      for (int smem_region = thread; smem_region < smem_count;
           smem_region += kThreads) {
        int const region = kRegisterCacheRegions + smem_region;
        uint32_t const key = ordered_score(
            storage.score_buffer[smem_region], region);
        if ((key & prefix_mask) == prefix_value) {
          atomicAdd(&storage.histogram[int((key >> shift) & bin_mask)], 1);
        }
      }
    } else {
      for (int region = thread; region < current; region += kThreads) {
        uint32_t const key = ordered_score(score_src[region], region);
        if ((key & prefix_mask) == prefix_value) {
          atomicAdd(&storage.histogram[int((key >> shift) & bin_mask)], 1);
        }
      }
    }
    __syncthreads();

    if (n_bins == kCoarseRadix) {
      choose_threshold_parallel<kCoarseRadix>(storage, thread);
    } else {
      choose_threshold_parallel<kRadix>(storage, thread);
    }

    if (thread == 0 && storage.remain > 0) {
      uint32_t const round_mask = bin_mask << shift;
      storage.prefix_mask |= round_mask;
      storage.prefix_value =
          (storage.prefix_value & ~round_mask) |
          (uint32_t(storage.threshold) << shift);
      storage.selected_base += storage.count_greater;
      storage.remain -= storage.count_greater;
      if (storage.remain == storage.histogram[storage.threshold]) {
        storage.selected_base += storage.remain;
        storage.remain = 0;
      }
    }
    __syncthreads();
    if (storage.remain == 0) break;
  }

  // Scatter phase: emit selected regions with warp-local staging.
  uint32_t const threshold_key = storage.prefix_value;
  bool const include_threshold = storage.remain == 0;
  int local_equal = 0;
  if (thread == 0) {
    storage.output_counter = 0;
  }
  __syncthreads();

  int staged_regions[kThreadStaging];
  int staged_count = 0;

  auto process_region_for_scatter = [&](int region, float raw, bool valid) {
    uint32_t key = 0;
    if (valid) {
      key = ordered_score(raw, region);
    }
    bool const greater =
        valid && (key > threshold_key || (include_threshold && key == threshold_key));
    bool const equal = valid && !include_threshold && (key == threshold_key);
    if (greater) {
      staged_regions[staged_count++] = region;
    } else if (equal) {
      ++local_equal;
    }
    if (__any_sync(0xFFFFFFFF, staged_count == kThreadStaging)) {
      int warp_exclusive = 0;
      int const base_slot =
          claim_warp_output_slots(staged_count, thread, storage,
                                  warp_exclusive);
#pragma unroll
      for (int i = 0; i < kThreadStaging; ++i) {
        if (i < staged_count) {
          int const slot = base_slot + warp_exclusive + i;
          if (slot < storage.selected_base) {
            store_region(staged_regions[i], slot, request, head, heads,
                         output_width, block_table, block_table_stride,
                         block_table_length, page_size / region_block,
                         region_block, packed, region_output, start_output);
          }
        }
      }
      staged_count = 0;
    }
  };

  auto flush_staged = [&]() {
    int warp_exclusive = 0;
    int const base_slot =
        claim_warp_output_slots(staged_count, thread, storage, warp_exclusive);
#pragma unroll
    for (int i = 0; i < kThreadStaging; ++i) {
      if (i < staged_count) {
        int const slot = base_slot + warp_exclusive + i;
        if (slot < storage.selected_base) {
          store_region(staged_regions[i], slot, request, head, heads,
                       output_width, block_table, block_table_stride,
                       block_table_length, page_size / region_block,
                       region_block, packed, region_output, start_output);
        }
      }
    }
  };

  if (use_hybrid_cache) {
    // Phase A: scatter from register cache (8 fixed iterations per thread).
    // Always call process_region_for_scatter to keep __any_sync convergent.
#pragma unroll
    for (int i = 0; i < kRegisterCachePerThread; ++i) {
      int const region = thread * kRegisterCachePerThread + i;
      bool const valid = (region < register_count);
      process_region_for_scatter(region, register_cache[i], valid);
    }
    // Flush before switching to smem to keep staged_count bounded.
    flush_staged();
    staged_count = 0;

    // Phase B: scatter from smem cache (fixed iterations to keep warp
    // convergent; invalid lanes simply pass valid=false).
    if (smem_count > 0) {
      int const smem_iters = (smem_count + kThreads - 1) / kThreads;
#pragma unroll
      for (int iter = 0; iter < smem_iters; ++iter) {
        int const smem_region = thread + iter * kThreads;
        bool const valid = (smem_region < smem_count);
        int const region = kRegisterCacheRegions + smem_region;
        float const raw =
            valid ? storage.score_buffer[smem_region] : 0.0f;
        process_region_for_scatter(region, raw, valid);
      }
    }
  } else {
    int const begin = int((int64_t(current) * thread) / kThreads);
    int const end = int((int64_t(current) * (thread + 1)) / kThreads);
    // A fixed trip count keeps full-mask warp ops convergent when the tail is not
    // evenly divisible.
    int const iterations = (current + kThreads - 1) / kThreads;
    for (int i = 0; i < iterations; ++i) {
      int const region = begin + i;
      bool const valid = region < end;
      process_region_for_scatter(region, valid ? score_src[region] : 0.0f, valid);
    }
  }

  flush_staged();
  int const higher_equal =
      include_threshold ? 0 : equal_count_higher_equal(local_equal, thread, storage);
  int const local_take =
      min(local_equal, max(storage.remain - higher_equal, 0));
  int emitted = 0;

  if (use_hybrid_cache) {
    // Tie-breaking from register cache (reverse order, deterministic).
    for (int i = kRegisterCachePerThread - 1; i >= 0 && emitted < local_take;
         --i) {
      int const region = thread * kRegisterCachePerThread + i;
      if (region < register_count &&
          ordered_score(register_cache[i], region) == threshold_key) {
        int const slot = storage.selected_base + higher_equal + emitted;
        if (slot < topk) {
          store_region(region, slot, request, head, heads, output_width,
                       block_table, block_table_stride, block_table_length,
                       page_size / region_block, region_block, packed,
                       region_output, start_output);
        }
        ++emitted;
      }
    }
    // Tie-breaking from smem cache (reverse strided order, matching the
    // scatter partition so local_take is consumed from the same regions that
    // contributed to local_equal).
    if (thread < smem_count) {
      int const max_n = (smem_count - 1 - thread) / kThreads;
      for (int n = max_n; n >= 0 && emitted < local_take; --n) {
        int const smem_region = thread + n * kThreads;
        int const region = kRegisterCacheRegions + smem_region;
        if (ordered_score(storage.score_buffer[smem_region], region) ==
            threshold_key) {
          int const slot = storage.selected_base + higher_equal + emitted;
          if (slot < topk) {
            store_region(region, slot, request, head, heads, output_width,
                         block_table, block_table_stride, block_table_length,
                         page_size / region_block, region_block, packed,
                         region_output, start_output);
          }
          ++emitted;
        }
      }
    }
  } else {
    int const begin = int((int64_t(current) * thread) / kThreads);
    int const end = int((int64_t(current) * (thread + 1)) / kThreads);
    for (int region = end - 1; region >= begin && emitted < local_take;
         --region) {
      if (ordered_score(score_src[region], region) == threshold_key) {
        int const slot = storage.selected_base + higher_equal + emitted;
        if (slot < topk) {
          store_region(region, slot, request, head, heads, output_width,
                       block_table, block_table_stride, block_table_length,
                       page_size / region_block, region_block, packed,
                       region_output, start_output);
        }
        ++emitted;
      }
    }
  }

  if (thread == 0) {
    store_region(current, topk, request, head, heads, output_width,
                 block_table, block_table_stride, block_table_length,
                 page_size / region_block, region_block, packed,
                 region_output, start_output);
  }
  if (sorted_output != nullptr) {
    __syncthreads();
    step4_dsa_stable_sort_epilogue(raw_shared, score_row, sort_length, topk,
                                   region_output, sorted_output, work,
                                   output_width, thread);
  }
}

// Chunk kernel: 3 rounds [11, 11, 10] (same as selector).
__global__ void step4_dsa_radix_chunk_kernel(
    float const* scores, int64_t const* query_current,
    int* candidate_output, int heads, int regions, int topk, int region_block,
    int num_chunks, int candidate_stride) {
  extern __shared__ char raw_shared[];
  auto& storage = *reinterpret_cast<Step4DsaRadixShared*>(raw_shared);
  int const chunk_work = blockIdx.x;
  int const work = chunk_work / num_chunks;
  int const chunk = chunk_work - work * num_chunks;
  int const request = work / heads;
  int const head = work - request * heads;
  int const thread = threadIdx.x;
  int64_t const current_in = query_current[request];
  int const current = static_cast<int>(
      current_in <= 0 ? 0 : min(current_in, int64_t(regions - 1)));
  int const chunk_begin = chunk * kStreamingChunkRegions;
  int const chunk_end = min(chunk_begin + kStreamingChunkRegions, current);
  int const valid_length = max(chunk_end - chunk_begin, 0);
  int const target = min(topk, valid_length);
  float const* score_row = scores + work * regions;
  int* output_row =
      candidate_output + work * candidate_stride + chunk * topk;

  // Merge only reads the chunks intersecting the request's history.
  if (valid_length == 0) return;
  if (valid_length <= topk) {
    for (int slot = thread; slot < topk; slot += kThreads) {
      output_row[slot] = slot < valid_length ? chunk_begin + slot : -1;
    }
    return;
  }

  for (int slot = thread; slot < topk; slot += kThreads) {
    output_row[slot] = -1;
  }
  if (thread == 0) {
    storage.prefix_value = 0;
    storage.prefix_mask = 0;
    storage.remain = target;
    storage.selected_base = 0;
  }
  __syncthreads();

#pragma unroll
  for (int round = 0; round < kNumSelectorRounds; ++round) {
    int const n_bins = (round < 2) ? kCoarseRadix : kRadix;
    int const bits   = (round < 2) ? 11 : 10;
    int const shift  = (round < 2) ? (32 - (round + 1) * 11) : 0;
    uint32_t const bin_mask = (1u << bits) - 1u;

    for (int bin = thread; bin < n_bins; bin += kThreads) {
      storage.histogram[bin] = 0;
    }
    __syncthreads();

    uint32_t const prefix_value = storage.prefix_value;
    uint32_t const prefix_mask = storage.prefix_mask;
    for (int region = chunk_begin + thread; region < chunk_end;
         region += kThreads) {
      uint32_t const key = ordered_score(score_row[region], region);
      if ((key & prefix_mask) == prefix_value) {
        atomicAdd(&storage.histogram[int((key >> shift) & bin_mask)], 1);
      }
    }
    __syncthreads();

    if (n_bins == kCoarseRadix) {
      choose_threshold_parallel<kCoarseRadix>(storage, thread);
    } else {
      choose_threshold_parallel<kRadix>(storage, thread);
    }

    if (thread == 0 && storage.remain > 0) {
      uint32_t const round_mask = bin_mask << shift;
      storage.prefix_mask |= round_mask;
      storage.prefix_value =
          (storage.prefix_value & ~round_mask) |
          (uint32_t(storage.threshold) << shift);
      storage.selected_base += storage.count_greater;
      storage.remain -= storage.count_greater;
      if (storage.remain == storage.histogram[storage.threshold]) {
        storage.selected_base += storage.remain;
        storage.remain = 0;
      }
    }
    __syncthreads();
    if (storage.remain == 0) break;
  }

  int const begin =
      chunk_begin + int((int64_t(valid_length) * thread) / kThreads);
  int const end =
      chunk_begin + int((int64_t(valid_length) * (thread + 1)) / kThreads);
  uint32_t const threshold_key = storage.prefix_value;
  bool const include_threshold = storage.remain == 0;
  int local_equal = 0;
  if (thread == 0) {
    storage.output_counter = 0;
  }
  __syncthreads();

  int staged_regions[kThreadStaging];
  int staged_count = 0;

  int const iterations = (valid_length + kThreads - 1) / kThreads;
  for (int i = 0; i < iterations; ++i) {
    int const region = begin + i;
    bool const valid = region < end;
    uint32_t const key = valid ? ordered_score(score_row[region], region) : 0;
    if (valid &&
        (key > threshold_key || (include_threshold && key == threshold_key))) {
      staged_regions[staged_count++] = region;
    } else if (valid && !include_threshold && key == threshold_key) {
      ++local_equal;
    }
    // Always call __any_sync with the full warp so the SIMT mask stays
    // convergent even when only a subset of lanes found a greater region.
    if (__any_sync(0xFFFFFFFF, staged_count == kThreadStaging)) {
      int warp_exclusive = 0;
      int const base_slot =
          claim_warp_output_slots(staged_count, thread, storage,
                                  warp_exclusive);
#pragma unroll
      for (int i = 0; i < kThreadStaging; ++i) {
        if (i < staged_count) {
          int const slot = base_slot + warp_exclusive + i;
          if (slot < storage.selected_base) {
            output_row[slot] = staged_regions[i];
          }
        }
      }
      staged_count = 0;
    }
  }
  {
    int warp_exclusive = 0;
    int const base_slot =
        claim_warp_output_slots(staged_count, thread, storage, warp_exclusive);
#pragma unroll
    for (int i = 0; i < kThreadStaging; ++i) {
      if (i < staged_count) {
        int const slot = base_slot + warp_exclusive + i;
        if (slot < storage.selected_base) {
          output_row[slot] = staged_regions[i];
        }
      }
    }
  }
  int const higher_equal =
      include_threshold ? 0 : equal_count_higher_equal(local_equal, thread, storage);
  int const local_take =
      min(local_equal, max(storage.remain - higher_equal, 0));
  int emitted = 0;
  for (int region = end - 1; region >= begin && emitted < local_take;
       --region) {
    if (ordered_score(score_row[region], region) == threshold_key) {
      int const slot = storage.selected_base + higher_equal + emitted;
      if (slot < topk) {
        output_row[slot] = region;
      }
      ++emitted;
    }
  }
}

// Merge kernel: 4 rounds [11 score, 11 score, 10 score, 10 region].
__global__ void step4_dsa_radix_merge_kernel(
    float const* scores, int64_t const* query_current,
    int const* candidate_input, int candidate_stride,
    int const* block_table, int* region_output, int* start_output, int heads,
    int regions, int topk, int region_block, int page_size,
    int block_table_stride, int block_table_length, int num_chunks,
    bool packed) {
  extern __shared__ char raw_shared[];
  auto& storage = *reinterpret_cast<Step4DsaMergeRadixShared*>(raw_shared);
  int const work = blockIdx.x;
  int const request = work / heads;
  int const head = work - request * heads;
  int const thread = threadIdx.x;
  int const output_width = topk + 1;
  int64_t const current_in = query_current[request];
  int const current = static_cast<int>(
      current_in <= 0 ? 0 : min(current_in, int64_t(regions - 1)));
  int const target = min(topk, current);
  int const useful_chunks = min(
      num_chunks, (current + kStreamingChunkRegions - 1) / kStreamingChunkRegions);
  int const candidate_count = useful_chunks * topk;
  float const* score_row = scores + work * regions;
  int const* candidate_row = candidate_input + work * candidate_stride;

  if (useful_chunks <= 1) {
    for (int slot = thread; slot < output_width; slot += kThreads) {
      int const region = slot == topk ? current
                         : useful_chunks == 1 ? candidate_row[slot] : -1;
      if (slot == topk || (region >= 0 && region < current)) {
        store_region(region, slot, request, head, heads, output_width,
                     block_table, block_table_stride, block_table_length,
                     page_size / region_block, region_block, packed,
                     region_output, start_output);
      } else {
        int const offset = work * output_width + slot;
        region_output[offset] = -1;
        start_output[offset] = -1;
      }
    }
    return;
  }

  for (int slot = thread; slot < output_width; slot += kThreads) {
    int const offset = work * output_width + slot;
    region_output[offset] = -1;
    start_output[offset] = -1;
  }
  if (thread == 0) {
    storage.score_prefix_value = 0;
    storage.score_prefix_mask = 0;
    storage.region_prefix_value = 0;
    storage.region_prefix_mask = 0;
    storage.remain = target;
    storage.selected_base = 0;
  }
  __syncthreads();

  // Rounds 0-2: score bits [11, 11, 10]; Round 3: region bits [10].
  constexpr int kMergeRounds = STEP5_DSA_EXACT_FP32 ? 3 : 4;
#pragma unroll
  for (int round = 0; round < kMergeRounds; ++round) {
    bool const score_round = round < 3;
    int n_bins, bits, shift;
    if (round == 0) {
      n_bins = kCoarseRadix; bits = 11; shift = 21;  // score [31:21]
    } else if (round == 1) {
      n_bins = kCoarseRadix; bits = 11; shift = 10;  // score [20:10]
    } else if (round == 2) {
      n_bins = kRadix; bits = 10; shift = 0;         // score [9:0]
    } else {
      n_bins = kRadix; bits = 10; shift = 22;        // region [31:22]
    }
    uint32_t const bin_mask = (1u << bits) - 1u;

    for (int bin = thread; bin < n_bins; bin += kThreads) {
      storage.histogram[bin] = 0;
    }
    __syncthreads();

    uint32_t const score_prefix_value = storage.score_prefix_value;
    uint32_t const score_prefix_mask = storage.score_prefix_mask;
    uint32_t const region_prefix_value = storage.region_prefix_value;
    uint32_t const region_prefix_mask = storage.region_prefix_mask;
    for (int index = thread; index < candidate_count; index += kThreads) {
      int const region = candidate_row[index];
      if (region < 0 || region >= current) continue;
      uint32_t const score_key = ordered_score(score_row[region], region);
      uint32_t const region_key = static_cast<uint32_t>(region);
      if ((score_key & score_prefix_mask) == score_prefix_value &&
          (region_key & region_prefix_mask) == region_prefix_value) {
        uint32_t const key = score_round ? score_key : region_key;
        atomicAdd(&storage.histogram[int((key >> shift) & bin_mask)], 1);
      }
    }
    __syncthreads();

    if (n_bins == kCoarseRadix) {
      choose_threshold_parallel<kCoarseRadix>(storage, thread);
    } else {
      choose_threshold_parallel<kRadix>(storage, thread);
    }

    if (thread == 0 && storage.remain > 0) {
      uint32_t const round_mask = bin_mask << shift;
      if (score_round) {
        storage.score_prefix_mask |= round_mask;
        storage.score_prefix_value =
            (storage.score_prefix_value & ~round_mask) |
            (uint32_t(storage.threshold) << shift);
      } else {
        storage.region_prefix_mask |= round_mask;
        storage.region_prefix_value =
            (storage.region_prefix_value & ~round_mask) |
            (uint32_t(storage.threshold) << shift);
      }
      storage.selected_base += storage.count_greater;
      storage.remain -= storage.count_greater;
    }
    __syncthreads();
  }

  uint32_t const threshold_score = storage.score_prefix_value;
  uint32_t const threshold_region = storage.region_prefix_value;
  if (thread == 0) {
    storage.output_counter = 0;
    storage.threshold = 0;
  }
  __syncthreads();
  for (int index = thread; index < candidate_count; index += kThreads) {
    int const region = candidate_row[index];
    if (region < 0 || region >= current) continue;
    uint32_t const score_key = ordered_score(score_row[region], region);
    bool const selected =
        score_key > threshold_score ||
        (score_key == threshold_score &&
         static_cast<uint32_t>(region) >= threshold_region);
    if (selected) {
      // Exact-score mode first keeps every item strictly above the cutoff so ties
      // cannot take their slots. The plugin orders tied IDs stably by reusing
      // upstream's post-processing.
      int const slot = STEP5_DSA_EXACT_FP32 && score_key == threshold_score
                           ? storage.selected_base + atomicAdd(&storage.threshold, 1)
                           : atomicAdd(&storage.output_counter, 1);
      bool const keep = STEP5_DSA_EXACT_FP32 && score_key > threshold_score
                            ? slot < storage.selected_base : slot < target;
      if (keep) {
        store_region(region, slot, request, head, heads, output_width,
                     block_table, block_table_stride, block_table_length,
                     page_size / region_block, region_block, packed,
                     region_output, start_output);
      }
    }
  }

  if (thread == 0) {
    store_region(current, topk, request, head, heads, output_width,
                 block_table, block_table_stride, block_table_length,
                 page_size / region_block, region_block, packed,
                 region_output, start_output);
  }
}

void step4_dsa_radix_select(
    torch::Tensor const& scores, torch::Tensor const& query_current,
    torch::Tensor const& block_table, torch::Tensor& candidate_workspace,
    torch::Tensor& region_output, torch::Tensor& start_output,
    int64_t region_block, int64_t page_size, int64_t topk, bool packed,
    torch::Tensor const& sorted_output) {
  TORCH_CHECK(scores.is_cuda() && query_current.is_cuda() &&
                  block_table.is_cuda() && candidate_workspace.is_cuda() &&
                  region_output.is_cuda() && start_output.is_cuda(),
              "Step4 DSA radix tensors must be on the CUDA device");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && scores.dim() == 3 &&
                  scores.is_contiguous(),
              "scores must be contiguous float32 [query,heads,regions]");
  TORCH_CHECK(query_current.scalar_type() == torch::kInt64 &&
                  query_current.dim() == 1 &&
                  query_current.size(0) == scores.size(0) &&
                  query_current.is_contiguous(),
              "query_current must be contiguous int64 region counts [query]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 &&
                  (block_table.dim() == 1 || block_table.dim() == 2) &&
                  block_table.is_contiguous(),
              "block_table must be contiguous int32 [pages] or [query,pages]");
  TORCH_CHECK(block_table.dim() == 1 || block_table.size(0) == scores.size(0),
              "batched block_table must have one row per query");
  TORCH_CHECK(candidate_workspace.scalar_type() == torch::kInt32 &&
                  candidate_workspace.dim() == 3 &&
                  candidate_workspace.size(0) == scores.size(0) &&
                  candidate_workspace.size(1) == scores.size(1) &&
                  candidate_workspace.is_contiguous(),
              "candidate workspace must be contiguous int32 "
              "[query,heads,capacity]");
  TORCH_CHECK(region_block > 0 && page_size >= region_block &&
                  page_size % region_block == 0,
              "invalid Step4 DSA region/page geometry");
  TORCH_CHECK(topk > 0 && topk <= 1024 && scores.size(2) > 0 &&
                  scores.size(2) <= std::numeric_limits<int>::max(),
              "invalid Step4 DSA radix top-k geometry");
  auto expected =
      std::vector<int64_t>{scores.size(0), scores.size(1), topk + 1};
  TORCH_CHECK(region_output.sizes() == expected &&
                  start_output.sizes() == expected &&
                  region_output.scalar_type() == torch::kInt32 &&
                  start_output.scalar_type() == torch::kInt32 &&
                  region_output.is_contiguous() && start_output.is_contiguous(),
              "radix outputs must be contiguous int32 [query,heads,topk+1]");

  bool const want_sorted = sorted_output.numel() > 0;
  if (want_sorted) {
    auto sorted_expected =
        std::vector<int64_t>{scores.size(0) * scores.size(1), topk + 1};
    TORCH_CHECK(sorted_output.scalar_type() == torch::kInt64 &&
                    sorted_output.sizes() == sorted_expected &&
                    sorted_output.is_contiguous() && sorted_output.is_cuda() &&
                    sorted_output.device() == scores.device(),
                "sorted output must be contiguous int64 [query*heads,topk+1]");
    TORCH_CHECK(scores.size(2) <= kStreamingChunkRegions,
                "stable-sort epilogue only supports the single-CTA path");
    at::assert_no_internal_overlap(sorted_output);
    at::assert_no_overlap(sorted_output, region_output);
    at::assert_no_overlap(sorted_output, start_output);
    at::assert_no_overlap(sorted_output, scores);
  }
  int64_t* sorted_ptr =
      want_sorted ? sorted_output.data_ptr<int64_t>() : nullptr;

  int const query_rows = static_cast<int>(scores.size(0));
  int const heads = static_cast<int>(scores.size(1));
  int const regions = static_cast<int>(scores.size(2));
  int const candidate_stride =
      static_cast<int>(candidate_workspace.size(2));
  // A rank-2 block table holds one row of pages per query token (MTP draft
  // batching); rank-1 shares a single row across the batch, hence stride 0.
  int const block_table_stride =
      block_table.dim() == 2 ? static_cast<int>(block_table.stride(0)) : 0;
  int const block_table_length = static_cast<int>(block_table.size(-1));
  auto stream = get_current_cuda_stream(scores.device().index());

  constexpr size_t kSelectorSmem = sizeof(Step4DsaCachedRadixShared);
  constexpr size_t kChunkSmem = sizeof(Step4DsaRadixShared);
  constexpr size_t kMergeSmem = sizeof(Step4DsaMergeRadixShared);
  static_assert(kSelectorSmem <= 48 * 1024,
                "selector shared memory exceeds default 48KB limit");
  static_assert(kMergeSmem <= 48 * 1024,
                "merge shared memory exceeds default 48KB limit");
  // Both structs fit within the default 48KB dynamic shared memory limit,
  // so no cudaFuncSetAttribute is needed.

  if (regions <= kStreamingChunkRegions) {
    step4_dsa_radix_selector_kernel<<<query_rows * heads, kThreads,
                                      kSelectorSmem, stream>>>(
        scores.data_ptr<float>(), query_current.data_ptr<int64_t>(),
        block_table.data_ptr<int>(), region_output.data_ptr<int>(),
        start_output.data_ptr<int>(), heads, regions, static_cast<int>(topk),
        static_cast<int>(region_block), static_cast<int>(page_size),
        block_table_stride, block_table_length, packed, sorted_ptr);
  } else {
    int const num_chunks =
        (regions + kStreamingChunkRegions - 1) / kStreamingChunkRegions;
    TORCH_CHECK(
        int64_t(num_chunks) * topk <= candidate_stride,
        "candidate workspace is too small for Step4 DSA multi-CTA radix: need ",
        int64_t(num_chunks) * topk, ", got ", candidate_stride);
    step4_dsa_radix_chunk_kernel<<<
        query_rows * heads * num_chunks, kThreads,
        kChunkSmem, stream>>>(
        scores.data_ptr<float>(), query_current.data_ptr<int64_t>(),
        candidate_workspace.data_ptr<int>(), heads, regions,
        static_cast<int>(topk), static_cast<int>(region_block), num_chunks,
        candidate_stride);
    step4_dsa_radix_merge_kernel<<<
        query_rows * heads, kThreads, kMergeSmem, stream>>>(
        scores.data_ptr<float>(), query_current.data_ptr<int64_t>(),
        candidate_workspace.data_ptr<int>(), candidate_stride,
        block_table.data_ptr<int>(), region_output.data_ptr<int>(),
        start_output.data_ptr<int>(), heads, regions, static_cast<int>(topk),
        static_cast<int>(region_block), static_cast<int>(page_size),
        block_table_stride, block_table_length, num_chunks, packed);
  }
  hggcError_t const error = hggcGetLastError();
  TORCH_CHECK(error == hggcSuccess, "Step4 DSA radix launch failed: ",
              hggcGetErrorString(error));
}

// Expose the launcher so bindings.cpp can register the v1 operator and
// dispatch internally between the two thread-count specializations.
void step4_dsa_radix_select(
    torch::Tensor const& scores, torch::Tensor const& query_current,
    torch::Tensor const& block_table, torch::Tensor& candidate_workspace,
    torch::Tensor& region_output, torch::Tensor& start_output,
    int64_t region_block, int64_t page_size, int64_t topk, bool packed,
    torch::Tensor const& sorted_output);

}  // namespace step4_dsa_w256 / step4_dsa_w1024
