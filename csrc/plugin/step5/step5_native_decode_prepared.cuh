// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Included in step5_native_decode after Decode is defined.
constexpr int kStreamSplits = 7;

enum class RegionLayout { Prepared, Compact, Paged };

template <RegionLayout Layout>
__global__ __launch_bounds__(128) void stream_attention(Decode p) {
#if __HGGC_ARCH__ >= 150
  constexpr int Heads = 16, Halves = 2, Warps = 4, TG = 2, NT = 1;
  const int warp = __builtin_ppu_to_uniform_b32(threadIdx.x / 32);
  const int work = blockIdx.x * Warps + warp;
  const int rows = Layout == RegionLayout::Prepared ? p.rows : 64;
  const int row = work % rows, split = work / rows;
  const int pos = checked_position(p.positions[row]);
  const int live = p.active[row] && pos >= 0 ? min(p.slots, pos / 8 + 1) : 0;
  const int tiles = (live + TG - 1) / TG;
  const int begin = (split * tiles / kStreamSplits) * TG;
  const int end = min(live, ((split + 1) * tiles / kStreamSplits) * TG);
  const int width = end - begin;
  const int lane = threadIdx.x & 31, t0 = lane & 3, lr = lane & 7, lu = lane >> 3;
  const int head = lane >> 2, a_col = 8 * (lu >> 1), b_col = 8 * (lu & 1);
  const int64_t* regions = p.regions + int64_t(row) * p.rs + p.slots - live;
  const auto* kv = p.kv.data;
  __shared__ __align__(128) __ppu_bfloat16 staged_kv[Warps][3][TG][3][8][64];
  MmaOperand qfrag[kHeadMmaTiles];
  MmaAccumulator output[kHeadMmaTiles];
  auto load_query = [&]<int Begin, int End>() {
#pragma unroll
    for (int d = Begin; d < End; ++d) {
      clear_fragment(qfrag[d]);
      clear_fragment(output[d]);
#pragma unroll
      for (int tau = 0; tau < Halves; ++tau) {
        const auto* src = p.q + int64_t(row) * p.qs + (head + tau * 8) * p.qh;
        qfrag[d].x[tau] = *reinterpret_cast<const uint32_t*>(src + 16 * d + 2 * t0);
        qfrag[d].x[tau + 2] = *reinterpret_cast<const uint32_t*>(src + 16 * d + 2 * t0 + 8);
      }
    }
  };
  float maximum[Halves], denom[Halves];
#pragma unroll
  for (int tau = 0; tau < Halves; ++tau) {
    maximum[tau] = -FLT_MAX;
    denom[tau] = 0.f;
  }
  const float scale2 = p.scale * 1.4426950408889634f;
  uint32_t region = 0;
  int visible = 0;
  int tokens[2][TG], physical[2][TG];
  auto resolve = [&](int off, int st) {
    if ((off & 31) == 0) {
      const int index = begin + off + lane;
      if constexpr (Layout == RegionLayout::Prepared) {
        const uint64_t entry = off + lane < width ? regions[index] : 0;
        region = uint32_t(entry >> 4);
        visible = int(entry & 15);
      } else {
        const int64_t id = off + lane < width ? regions[index] : -1;
        const int phys = pos >= 0 && id >= 0 && id <= pos / kRegionTokens
                             ? p.kv.physical_region(row / p.rows_per_request, id)
                             : -1;
        region = uint32_t(max(phys, 0));
        if constexpr (Layout == RegionLayout::Compact) region *= kRegionTokens * 2 * kHeadDim / 64;
        visible = phys < 0 ? 0 : min(int64_t(kRegionTokens), int64_t(pos) - id * 8 + 1);
      }
    }
#pragma unroll
    for (int r = 0; r < TG; ++r) {
      const int source = (off & 31) + r;
      tokens[st][r] = __builtin_ppu_to_uniform_b32(__shfl_sync(0xffffffffu, visible, source));
      physical[st][r] = __builtin_ppu_to_uniform_b32(__shfl_sync(0xffffffffu, region, source));
    }
  };
  auto copy_plane = [&]<bool Value>(int st) {
#pragma unroll
    for (int r = 0; r < TG; r++) {
      int n = tokens[st][r], start = n ? 0 : kRegionTokens;
      const unsigned address = physical[st][r];
      const int64_t offset =
          Layout == RegionLayout::Paged ? p.kv.offset(address) : int64_t(address) * 64;
      const auto* src = kv + offset + (Value ? kHeadDim : 0);
      unsigned dst =
          static_cast<unsigned>(__cvta_generic_to_shared(staged_kv[warp][Value ? 2 : st][r]));
      // Stage 64 channels at a time and zero-pad invisible tokens.
      if constexpr (Layout == RegionLayout::Prepared) {
        asm volatile(
            "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.LLC::128B."
            "padz.swzl.b16 [%0], [%1], {%2,%3,%4}, {%2,%5,%4}, "
            "{%6,%7}, {%8,%9,%8}, %8;" ::"r"(dst),
            "l"(src), "r"(64), "r"(n + start), "r"(3), "r"(8), "l"(int64_t(768)), "l"(int64_t(128)),
            "r"(0), "r"(start)
            : "memory");
      } else {
        asm volatile(
            "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.LLC::256B."
            "padz.swzl.b16 [%0], [%1], {%2,%3,%4}, {%2,%5,%4}, "
            "{%6,%7}, {%8,%9,%8}, %8;" ::"r"(dst),
            "l"(src), "r"(64), "r"(n + start), "r"(3), "r"(8), "l"(int64_t(768)), "l"(int64_t(128)),
            "r"(0), "r"(start)
            : "memory");
      }
    }
    asm volatile("ppu.cp.async.commit_group;" ::: "memory");
  };
  load_query.template operator()<0, 1>();
  if (width > 0) {
    resolve(0, 0);
    copy_plane.template operator()<false>(0);
  }
  if (width > 0) copy_plane.template operator()<true>(0);
  load_query.template operator()<1, 6>();
  if (width > TG) {
    resolve(TG, 1);
    copy_plane.template operator()<false>(1);
  }
  load_query.template operator()<6, 12>();
  for (int base = 0; base < width; base += TG) {
    int counts[TG];
    bool full = true;
    const int stage = (base / TG) & 1;
    const bool next = base + TG < width;
    // K for the following tile may already be in flight behind this tile's V.
    if (next)
      asm volatile("ppu.cp.async.wait_group 2;" ::: "memory");
    else
      asm volatile("ppu.cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
#pragma unroll
    for (int r = 0; r < TG; ++r) {
      counts[r] = tokens[stage][r];
      full &= counts[r] == kRegionTokens;
    }
    MmaAccumulator scores[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) clear_fragment(scores[t]);
#pragma unroll
    for (int d = 0; d < kHeadMmaTiles; ++d) {
      const int chunk = (((d % 4) * 16 + b_col) >> 3) ^ lr;
      MmaOperand key[NT];
#pragma unroll
      for (int t = 0; t < NT; ++t)
        ldm_x4(key[t], &staged_kv[warp][stage][2 * t + (lu >> 1)][d / 4][lr][chunk * 8]);
#pragma unroll
      for (int t = 0; t < NT; ++t) mma_bf16(scores[t], qfrag[d], key[t], scores[t]);
    }
    __syncwarp();
    float peak[Halves];
#pragma unroll
    for (int tau = 0; tau < Halves; ++tau) peak[tau] = -INFINITY;
    if (full) {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int tau = (i >> 1) & 1;
          peak[tau] = fmaxf(peak[tau], scores[t].x[i]);
        }
      }
    } else {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int tau = (i >> 1) & 1;
          const bool valid = 2 * t0 + (i & 1) < counts[2 * t + (i >> 2)];
          scores[t].x[i] = valid ? scores[t].x[i] : -INFINITY;
          peak[tau] = fmaxf(peak[tau], scores[t].x[i]);
        }
      }
    }
    bool increase = false;
#pragma unroll
    for (int tau = 0; tau < Halves; ++tau) increase |= peak[tau] * scale2 > maximum[tau];
    if (__any_sync(0xffffffffu, increase)) {
      float rescale[Halves];
#pragma unroll
      for (int tau = 0; tau < Halves; ++tau) {
        const float local = head_max(peak[tau]) * scale2;
        const float bound = maximum[tau] == -FLT_MAX ? local : ceilf(local * 0.125f) * 8.f;
        const float next = local > maximum[tau] ? bound : maximum[tau];
        rescale[tau] = __exp2f(maximum[tau] - next);
        denom[tau] *= rescale[tau];
        maximum[tau] = next;
      }
#pragma unroll
      for (int d = 0; d < kHeadMmaTiles; ++d) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int tau = (i >> 1) & 1;
          output[d].x[i] *= rescale[tau];
        }
      }
    }
    MmaOperand prob[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int tau = (i >> 1) & 1;
        const float a = __exp2f(fmaf(scores[t].x[i], scale2, -maximum[tau]));
        const float b = __exp2f(fmaf(scores[t].x[i + 1], scale2, -maximum[tau]));
        denom[tau] += a + b;
        *reinterpret_cast<__ppu_bfloat162*>(&prob[t].x[i / 2]) = __floats2bfloat162_rn(a, b);
      }
    }
    if (next)
      asm volatile("ppu.cp.async.wait_group 1;" ::: "memory");
    else
      asm volatile("ppu.cp.async.wait_group 0;" ::: "memory");
    __syncwarp();
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int d = 0; d < kHeadMmaTiles; ++d) {
        MmaOperand value;
        const int chunk = (((d % 4) * 16 + a_col) >> 3) ^ lr;
        ldm_x4_trans(value, &staged_kv[warp][2][2 * t + (lu & 1)][d / 4][lr][chunk * 8]);
        mma_bf16(output[d], prob[t], value, output[d]);
      }
    }
    __syncwarp();
    if (next) copy_plane.template operator()<true>(stage ^ 1);
    if (base + 2 * TG < width) {
      resolve(base + 2 * TG, stage);
      copy_plane.template operator()<false>(stage);
    }
  }
  // Store MMA fragments as contiguous float4s for the merge kernel.
  const int row_split = row * kStreamSplits + split;
#pragma unroll
  for (int part = 0; part < Halves; ++part) {
    const float norm = head_sum(denom[part]);
    if (t0 == 0) {
      auto* stats = reinterpret_cast<float2*>(p.partial_lse);
      stats[row_split * Heads + 8 * part + head] =
          make_float2(norm > 0.f ? maximum[part] : -INFINITY, norm);
    }
  }
#pragma unroll
  for (int d = 0; d < kHeadMmaTiles; ++d) {
    const int tile = (row * kHeadMmaTiles + d) * kStreamSplits + split;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float* v = &output[d].x[half * 4];
      reinterpret_cast<float4*>(p.partial)[(tile * 2 + half) * 32 + lane] =
          make_float4(v[0], v[1], v[2], v[3]);
    }
  }
#endif
}

__global__ void merge_attention_fragments(Decode p) {
#if __HGGC_ARCH__ >= 150
  constexpr int kHeads = 16;
  constexpr int kGroup = 8;  // Seven split lanes and one padding lane.
  constexpr int kTiles = kHeadMmaTiles / kMergeWarps;
  static_assert(kStreamSplits == kGroup - 1 && kHeads * kGroup == kMergeWarps * 32);
  const int row = blockIdx.x / kTiles;
  if (!p.active[row] && !p.clear_inactive) return;
  const int d = (blockIdx.x % kTiles) * kMergeWarps + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int split = threadIdx.x % kGroup;
  const int split_head = threadIdx.x / kGroup;
  __shared__ float weights[kStreamSplits][kHeads], gate[kHeads];
  float2 stats = make_float2(-INFINITY, 0.f);
  if (split < kStreamSplits) {
    const int index = (row * kStreamSplits + split) * kHeads + split_head;
    stats = reinterpret_cast<const float2*>(p.partial_lse)[index];
  }
  float peak = stats.x;
#pragma unroll
  for (int offset = 1; offset < kGroup; offset *= 2)
    peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, offset, kGroup));
  const float weight = peak > -INFINITY ? __exp2f(stats.x - peak) : 0.f;
  float norm = weight == 0.f ? 0.f : weight * stats.y;
#pragma unroll
  for (int offset = 1; offset < kGroup; offset *= 2)
    norm += __shfl_xor_sync(0xffffffffu, norm, offset, kGroup);
  if (split < kStreamSplits) weights[split][split_head] = norm > 0.f ? weight / norm : 0.f;
  if (split == 0) gate[split_head] = p.gate_value(row, split_head);
  __syncthreads();

  const int head = lane / 4;
  const auto* partial = reinterpret_cast<const float4*>(p.partial);
  float4 values[kStreamSplits][2], result[2] = {};
#pragma unroll
  for (int s = 0; s < kStreamSplits; ++s) {
    const int tile = (row * kHeadMmaTiles + d) * kStreamSplits + s;
#pragma unroll
    for (int half = 0; half < 2; ++half) values[s][half] = partial[(tile * 2 + half) * 32 + lane];
  }
  auto combine = [&]<bool SkipZero>() {
    result[0] = result[1] = {};
#pragma unroll
    for (int s = 0; s < kStreamSplits; ++s) {
      const float w0 = weights[s][head], w1 = weights[s][8 + head];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float4 v = values[s][half];
        float4& r = result[half];
        if (!SkipZero || w0 != 0.f) {
          r.x = fmaf(w0, v.x, r.x);
          r.y = fmaf(w0, v.y, r.y);
        }
        if (!SkipZero || w1 != 0.f) {
          r.z = fmaf(w1, v.z, r.z);
          r.w = fmaf(w1, v.w, r.w);
        }
      }
    }
  };
  combine.template operator()<false>();
  // Recheck nonfinite results so a zero-weight split cannot propagate NaN.
  const float4 a = result[0], b = result[1];
  if (!isfinite((a.x + a.y) + (a.z + a.w) + (b.x + b.y) + (b.z + b.w)))
    combine.template operator()<true>();
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int dim = d * 16 + half * 8 + 2 * (lane & 3);
    auto* dst = p.out + int64_t(row) * p.os + head * p.oh + dim;
    const float4 r = result[half];
    *reinterpret_cast<__ppu_bfloat162*>(dst) = p.output_pair(r.x, r.y, gate[head]);
    *reinterpret_cast<__ppu_bfloat162*>(dst + 8 * p.oh) = p.output_pair(r.z, r.w, gate[8 + head]);
  }
#endif
}

// The low four bits carry the causal token count. The remaining bits count
// 128-byte units from the start of this KV view.
__device__ __forceinline__ int64_t encode_region(const Decode& p, int row, int pos, int64_t id) {
  if (id < 0 || id > pos / kRegionTokens) return 0;
  const int physical = p.kv.physical_region(row / p.rows_per_request, int(id));
  if (physical < 0) return 0;
  const int64_t visible = min(int64_t(kRegionTokens), pos - id * kRegionTokens + 1);
  return ((p.kv.offset(physical) >> 6) << 4) | visible;
}

__global__ void prepare_decode_regions(Decode p, int64_t* packed) {
#if __HGGC_ARCH__ >= 150
  constexpr int kThreads = 256;
  constexpr int kItems = 2;
  constexpr uint32_t kMultiplier = 2654435761u;
  constexpr uint32_t kInverse = 244002641u;
  constexpr uint32_t kMask = 0x7fffffffu;
  using Sort = cub::BlockRadixSort<uint32_t, kThreads, kItems>;
  __shared__ typename Sort::TempStorage scratch;
  const int row = blockIdx.x, tid = threadIdx.x;
  const int pos = checked_position(p.positions[row]);
  const int live = p.active[row] && pos >= 0 ? min(p.slots, pos / kRegionTokens + 1) : 0;
  const int64_t* ids = p.regions + int64_t(row) * p.rs;
  int64_t* dest = packed + int64_t(row) * p.slots;

  // Preserve slot order when at least half the sampled IDs agree across the paired rows.
  const int slot = tid * p.slots / kThreads;
  const int64_t mine = ids[slot];
  const int64_t other = p.regions[int64_t(row ^ 1) * p.rs + slot];
  const bool aligned = __syncthreads_count(mine >= 0 && mine == other) >= kThreads / 2;
  if (aligned || live < 128) {
    for (int j = tid; j < p.slots; j += kThreads)
      dest[j] = j >= p.slots - live ? encode_region(p, row, pos, ids[j]) : 0;
    return;
  }

  uint32_t keys[kItems];
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    const int j = tid + i * kThreads;
    const int64_t id = j >= p.slots - live && j < p.slots - 1 ? ids[j] : -1;
    keys[i] =
        id >= 0 && id <= pos / kRegionTokens ? (uint32_t(id) * kMultiplier) & kMask : UINT32_MAX;
  }
  // Bucket by logical ID so adjacent rows read shared regions close together.
  // Reverse the sort to put invalid entries first, and keep the last slot in place.
  Sort(scratch).SortBlockedToStriped(keys, 21, 32);
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    const int j = tid + i * kThreads;
    if (j < p.slots - 1) {
      const int64_t id = keys[i] == UINT32_MAX ? -1 : int((keys[i] * kInverse) & kMask);
      dest[p.slots - 2 - j] = encode_region(p, row, pos, id);
    }
  }
  if (tid == 0) dest[p.slots - 1] = live ? encode_region(p, row, pos, ids[p.slots - 1]) : 0;
#endif
}

inline bool can_prepare(const Decode& p, int64_t query_len) {
  const auto& kv = p.kv;
  const int64_t last =
      (kv.page_count - 1) * kv.block_stride + ((8 << kv.region_shift) - 1) * kv.token_stride;
  return p.heads == 16 && p.groups == 1 && query_len % 2 == 0 && p.rows >= 32 && p.rows <= 128 &&
         p.rows % 4 == 0 && p.slots >= 128 && kv.token_stride == 2 * kHeadDim &&
         kv.block_stride >= 2 * (8 << kv.region_shift) * kv.token_stride &&
         (last >> 6) <= UINT32_MAX;
}

inline bool can_stream_compact(const Decode& p) {
  const auto& kv = p.kv;
  const int64_t stride = (8 << kv.region_shift) * (2 * kHeadDim);
  return kv.block_stride == stride && kv.page_count * (stride / 64) <= (int64_t(1) << 32);
}

template <RegionLayout Layout>
inline void launch_stream(Decode p, const at::TensorOptions& options, hggcStream_t stream) {
  const int partial_rows = p.rows * p.heads * kStreamSplits;
  auto partial = at::empty({partial_rows, kHeadDim}, options);
  auto stats = at::empty({partial_rows, 2}, options);
  p.partial = partial.data_ptr<float>();
  p.partial_lse = stats.data_ptr<float>();
  p.splits = kStreamSplits;
  at::Tensor plan;
  if constexpr (Layout == RegionLayout::Prepared) {
    plan = at::empty({p.rows, p.slots}, options.dtype(at::kLong));
    auto* entries = plan.data_ptr<int64_t>();
    prepare_decode_regions<<<p.rows, 256, 0, stream>>>(p, entries);
    p.regions = entries;
    p.rs = p.slots;
  }
  stream_attention<Layout><<<p.rows * kStreamSplits / 4, 128, 0, stream>>>(p);
  merge_attention_fragments<<<p.rows*(kHeadMmaTiles / kMergeWarps), 32 * kMergeWarps, 0, stream>>>(
      p);
}
