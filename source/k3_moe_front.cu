// A DCP decode step's MoE front in one launch: everything between the front
// GEMM (router | lat_down | this rank's shared gate | up, one f32 row of LDS
// columns) and the batched GEMMs.
//
//   kern_k3_moe_front(const f32* S, const f32* bias, const bf16* rs, i32* idx, f32* wts,
//       u8* q, u8* sf, bf16* act, i32* done, i32* cta_batch, i32* cta_limit,
//       i32* num_non_exiting, i32* total_padded, i32* route_map, i32* exp2perm, int first, int T)
//   grid (T, 1, 1)   block (1024, 1, 1)   T <= 64
//
// Row b's block:
//   * the router top-k of columns [0, EXPERTS) (k3_router_argmax.cu's picks and
//     order): warp w < EXPERTS / 32 sorts its 32 biased scores by a bitonic
//     network, then warps 0..13 merge the warps' top 16 pairwise (below);
//   * meanwhile warps 28..31 run k3_land.cu's land_situ on the shared expert's
//     gate | up (the SH columns after the latent), and once the sorts are done
//     warps 14..27 land the latent (columns EXPERTS + [0, LATENT)) to bf16 and
//     quantise it to mxfp8 as k3_moe_prefill.cu's kern_k3_moe_quant (four
//     threads per 32-element group);
// then the last block to finish (`done`, a zeroed carry it re-zeroes) builds
// the routing tables of this rank's experts `first + j * STRIDE` (j < LOCAL)
// over all T rows: what FlashInfer's routing writes for the batched GEMMs'
// TILE-row CTAs, every expert's rows in the order of shared-memory atomics
// (as FlashInfer's are; the GEMMs' rows do not depend on it), padding rows
// routing token 0, exp2perm -1 for another rank's expert.
//
//   nvcc -cubin -arch=sm_103a -DEXPERTS=896 -DLDS=6016 -DSH=768 -DSTRIDE=8 k3_moe_front.cu
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#ifndef EXPERTS
#define EXPERTS 896
#endif
#ifndef LDS
#define LDS 6016
#endif
#ifndef SH
#define SH 768
#endif
#ifndef STRIDE
#define STRIDE 8
#endif
#define LATENT 3584
#define TOPK 16
#define TILE 16
#define LOCAL (EXPERTS / STRIDE)
#define THREADS 1024
#define QUANT_THREADS (LATENT / 8)
#define SITU_THREADS (SH / 4)
static_assert(QUANT_THREADS % 32 == 0 && SH % 4 == 0, "whole warps");
static_assert(LOCAL <= THREADS && TOPK * 64 <= THREADS, "one expanded id per thread");

typedef unsigned int u32;

__device__ __forceinline__ u32 ord_f32(float f) {
  u32 b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ float tanh_approx(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float situ_f(float pg, float pu) {
  float g = __bfloat162float(__float2bfloat16(pg));
  float u = __bfloat162float(__float2bfloat16(pu));
  float a = 4.0f * tanh_approx(g * 0.25f);
  float s = __frcp_rn(1.0f + __expf(-g));
  float c = 25.0f * tanh_approx(u * 0.04f);
  return (a * s) * c;
}

// Eight latent columns of a 32-column group (its four threads are adjacent lanes):
// landed to bf16, the group's UE8M0 scale, e4m3 at nearest.
__device__ __forceinline__ void quant8(const float* __restrict__ x, uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
                                       int u) {
  const float4 a = reinterpret_cast<const float4*>(x)[2 * u], c = reinterpret_cast<const float4*>(x)[2 * u + 1];
  const float raw[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
  float v[8];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    v[j] = __bfloat162float(__float2bfloat16(raw[j]));
    amax = fmaxf(amax, fabsf(v[j]));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  int e = amax > 0.f ? (int)ceilf(log2f(amax / 448.f)) : 0;
  e = min(max(e, -127), 127);
  const float s = exp2f((float)-e);
  if ((u & 3) == 0) sf[u >> 2] = (uint8_t)(e + 127);
  uint8_t o[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = __nv_cvt_float_to_fp8(v[j] * s, __NV_SATFINITE, __NV_E4M3);
  reinterpret_cast<uint2*>(q)[u] = *reinterpret_cast<const uint2*>(o);
}

// The last block's routing tables over T rows' ids, thread i one expanded id: its slot within its
// expert claimed by a shared-memory atomic (an expert's rows in arrival order, as FlashInfer's atomics
// leave them; the GEMMs compute every row on its own and the combine gathers by exp2perm).
__device__ __forceinline__ void route(const int* __restrict__ idx, int first, int T, int* count,
                                      int* __restrict__ cta_batch, int* __restrict__ cta_limit,
                                      int* __restrict__ num_non_exiting, int* __restrict__ total_padded,
                                      int* __restrict__ route_map, int* __restrict__ exp2perm) {
  __shared__ int off[LOCAL + 1];
  const int i = threadIdx.x, lane = i % 32;
  int j = -1;
  if (i < T * TOPK) {
    const int d = __ldcg(idx + i) - first;
    if (d >= 0 && d % STRIDE == 0) j = d / STRIDE;
  }
  const int slot = j >= 0 ? atomicAdd_block(count + j, 1) : 0;
  __syncthreads();
  if (i < 32) {  // off = exclusive scan of the experts' CTA counts
    constexpr int EPL = (LOCAL + 31) / 32;
    int c[EPL], sum = 0;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const int e = lane * EPL + k;
      c[k] = e < LOCAL ? (count[e] + TILE - 1) / TILE : 0;
      sum += c[k];
    }
    int incl = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int n = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += n;
    }
    int run = incl - sum;
#pragma unroll
    for (int k = 0; k < EPL; ++k) {
      const int e = lane * EPL + k;
      if (e < LOCAL) off[e] = run;
      run += c[k];
    }
    if (lane == 31) {
      off[LOCAL] = incl;
      num_non_exiting[0] = incl;
      total_padded[0] = incl * TILE;
    }
  }
  __syncthreads();
  if (i < LOCAL) {
    for (int c = off[i]; c < off[i + 1]; ++c) {
      cta_batch[c] = i;
      cta_limit[c] = min((c + 1) * TILE, off[i] * TILE + count[i]);
    }
    for (int r = off[i] * TILE + count[i]; r < off[i + 1] * TILE; ++r) route_map[r] = 0;
  }
  if (i < T * TOPK) {
    int row = -1;
    if (j >= 0) {
      row = off[j] * TILE + slot;
      route_map[row] = i / TOPK;
    }
    exp2perm[i] = row;
  }
}

// A warp's 32 keys sorted descending by a bitonic network. The scores' keys are 64 bits: the biased
// score's order bits, then the expert complemented (larger key = larger score, ties to the smaller
// expert: k3_router_argmax.cu's order).
template <typename K>
__device__ __forceinline__ K sort32(K v, int lane) {
#pragma unroll
  for (int k = 2; k <= 32; k <<= 1)
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const K o = __shfl_xor_sync(0xffffffffu, v, j);
      v = (((lane & j) == 0) == ((lane & k) == 0)) ? max(v, o) : min(v, o);
    }
  return v;
}

// Two descending lists of 16 (a in lanes 0-15, b reversed in 16-31: a bitonic sequence) merged
// descending: lanes 0-15 hold the top 16 of both.
__device__ __forceinline__ unsigned long long merge32(unsigned long long v, int lane) {
#pragma unroll
  for (int j = 16; j > 0; j >>= 1) {
    const unsigned long long o = __shfl_xor_sync(0xffffffffu, v, j);
    v = (lane & j) == 0 ? max(v, o) : min(v, o);
  }
  return v;
}

#define SORT_WARPS (EXPERTS / 32)
#define QUANT_WARP0 (SORT_WARPS - QUANT_THREADS / 32)
static_assert(EXPERTS % 32 == 0 && TOPK <= SORT_WARPS && SORT_WARPS <= 31 && QUANT_WARP0 > 0,
              "a warp of experts each, at least TOPK of them; a warp left for the SiTU, the latent's after warp 0");

extern "C" __global__ void __launch_bounds__(THREADS, 1) kern_k3_moe_front(
    const float* __restrict__ S, const float* __restrict__ bias, const __nv_bfloat16* __restrict__ rs,
    int* __restrict__ idx, float* __restrict__ wts, uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
    __nv_bfloat16* __restrict__ act, int* __restrict__ done, int* __restrict__ cta_batch,
    int* __restrict__ cta_limit, int* __restrict__ num_non_exiting, int* __restrict__ total_padded,
    int* __restrict__ route_map, int* __restrict__ exp2perm, int first, int T) {
  __shared__ float s_sig[EXPERTS];
  __shared__ unsigned long long s_cand[EXPERTS];
  __shared__ u32 s_max[32], s_tau;
  __shared__ int s_n, s_last, s_count[LOCAL];

  const int b = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32;
  asm volatile("griddepcontrol.launch_dependents;");
  const float* row = S + (long long)b * LDS;
  if (warp < SORT_WARPS) {
    const float sg = 1.0f / (1.0f + expf(-row[t]));
    s_sig[t] = sg;
    const unsigned long long key = (unsigned long long)ord_f32(sg + bias[t]) << 32 | (u32)~t;
    const u32 m = __reduce_max_sync(0xffffffffu, (u32)(key >> 32));
    if (lane == 0) s_max[warp] = m;
    if (t == 0) s_n = 0;
    if (t < LOCAL) s_count[t] = 0;  // the tables' per-expert counts, should this block be the last
    asm volatile("bar.sync 2, %0;" ::"n"(SORT_WARPS * 32));
    // tau, the 16th largest warp maximum of the score bits: at least 16 keys have score bits >= tau, and
    // the keys order by those bits first, so the top 16 are among the keys whose bits are >= tau
    if (warp == 0) {
      const u32 v = sort32(lane < SORT_WARPS ? s_max[lane] : 0u, lane);
      if (lane == TOPK - 1) s_tau = v;
    }
    asm volatile("bar.sync 2, %0;" ::"n"(SORT_WARPS * 32));
    const bool in = (u32)(key >> 32) >= s_tau;
    const unsigned ballot = __ballot_sync(0xffffffffu, in);
    int at = 0;
    if (lane == 0 && ballot) at = atomicAdd(&s_n, __popc(ballot));
    at = __shfl_sync(0xffffffffu, at, 0);
    if (in) s_cand[at + __popc(ballot & ((1u << lane) - 1))] = key;
    asm volatile("bar.sync 2, %0;" ::"n"(SORT_WARPS * 32));
    if (warp == 0) {
      // the candidates (in no order) 32 at a time: each chunk sorted, its top 16 merged into the running top 16
      const int n = s_n;
      unsigned long long top = 0;
      for (int c = 0; c < n; c += 32) {
        const unsigned long long v = sort32(c + lane < n ? s_cand[c + lane] : 0ull, lane);
        const unsigned long long r = __shfl_sync(0xffffffffu, v, 31 - lane);
        top = c == 0 ? v : merge32(lane < TOPK ? top : r, lane);
      }
      if (lane < TOPK) {
        const int e = (int)~(u32)top;
        const float sg = s_sig[e];
        float den = sg;
#pragma unroll
        for (int off = TOPK / 2; off > 0; off >>= 1) den += __shfl_xor_sync(0x0000ffffu, den, off);
        wts[b * TOPK + lane] = (sg / (den + 1e-20f)) * __bfloat162float(rs[0]);
        idx[b * TOPK + lane] = e;
      }
    } else if (warp >= QUANT_WARP0) {
      quant8(row + EXPERTS, q + (long long)b * LATENT, sf + (long long)b * (LATENT / 32), t - QUANT_WARP0 * 32);
    }
  } else {
    for (int k = t - SORT_WARPS * 32; k < SITU_THREADS; k += THREADS - SORT_WARPS * 32) {
      const float4 g = reinterpret_cast<const float4*>(row + EXPERTS + LATENT)[k];
      const float4 u = reinterpret_cast<const float4*>(row + EXPERTS + LATENT + SH)[k];
      __nv_bfloat162 o[2];
      o[0] = __floats2bfloat162_rn(situ_f(g.x, u.x), situ_f(g.y, u.y));
      o[1] = __floats2bfloat162_rn(situ_f(g.z, u.z), situ_f(g.w, u.w));
      reinterpret_cast<uint2*>(act + (long long)b * SH)[k] = *reinterpret_cast<const uint2*>(o);
    }
  }

  __syncthreads();
  if (t == 0) {  // releases the block's ids (ordered before by the barrier), acquires the others'
    int n;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;" : "=r"(n) : "l"(done) : "memory");
    s_last = n == T - 1;
  }
  __syncthreads();
  if (!s_last) return;
  route(idx, first, T, s_count, cta_batch, cta_limit, num_non_exiting, total_padded, route_map, exp2perm);
  if (t == 0) *done = 0;
}
