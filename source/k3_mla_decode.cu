// K3 MLA decode attention over a DCP member's latent pages, bf16: the DSL decode's split kernel and its contract
// (gen_stage.py mla_attn_op), a CTA pair a (row, split), heads as M.
//
//   extern "C" __global__ void kern_k3_mla_decode(
//       CUtensorMap q_map,     // q_abs [B][96][576] bf16: box 64 dims x 64 heads x 1, 128 B swizzle
//       CUtensorMap kv_map,    // latent pages [pages][64][576] bf16: box 64 dims x 64 positions x 1, 128 B swizzle
//       const bf16* kv,        // the pages kv_map maps
//       const int* table,      // [B][max_pages]
//       const int* seq_lens,   // [B]  positions this member holds
//       const int* bsk,        // [B]  splits asked for
//       float* acc_o,          // [B][128][split_max][512]  the splits' normalized o
//       float* acc_lse,        // [B][128][split_max]       their log2-sum-exp
//       int split_max, int max_pages, float scale_log2);
//
//   grid (2, B, split_max)   cluster (2, 1, 1)   block THREADS   smem SMEM_BYTES dynamic
//
// Row b's t = ceil(n / 128) tiles of 128 positions run as S = ceil(t / per) splits of per = ceil(t / bsk) tiles;
// clusters past S, and rows with n = 0, exit. CTA c holds heads 64c .. 64c + 63 (the last 32 of CTA 1 are the map's
// zero fill) and page 2i + c of tile i. Per tile:
//   S = Q K^T    M 128 (pair) x N 128 positions (each CTA its page) x K 576, into TMEM (two buffers)
//   P = exp2(S scale - m)   softmax warps: warps w and w ^ 2 share a head, positions 0-63 / 64-127
//   O += P V     M 128 x N 256 (each CTA 128 latent dims of each 256) x K 128, two MMAs into TMEM
// m is lazily raised: O and the sum are rescaled only when the tile's max passes m + 8.
// Two rings, each in its MMA's order and freed as soon as its MMAs complete: K, 8 KB boxes of each tile's own page
// (nine: 8 latent, rope) from HBM; V, 16 KB pair slots (2g + p: page 2i + p's dims 256g + 128c .. +128 in CTA c,
// the P V operand of 128 dims) from L2 once the tile's Q K^T is done. PF > 0 also prefetches the own page PF tiles
// ahead into L2 (measured slower; off).
#include <cuda.h>
#include <cstdint>

#define HEADS 96
#define ROW 576
#define LAT 512
#define TILE 128
#define PAGE 64
#define M_TILE 128
#define THREADS 224
#define SOFTMAX_WARPS 4
#define LOAD_WARP 4
#define MMA_WARP 5
#define V_WARP 6
#define BOX 8192                 // 64 dims x 64 rows
#define PAIR (2 * BOX)
#define NK 9                     // K ring: 8 KB boxes, a tile's own page
#define NVP 4                    // V ring: pair slots, a tile's V
#define RESCALE 8.0f
#ifndef PF
#define PF 0
#endif

#define Q_OFF 0
#define P_OFF (Q_OFF + 9 * BOX)
#define RING_OFF (P_OFF + 2 * BOX)
#define V_OFF (RING_OFF + NK * BOX)
#define XCH_OFF (V_OFF + NVP * PAIR)              // [3][2][64] f32: the halves' max (two tiles), then sum
#define BAR_OFF (XCH_OFF + 3 * 2 * 64 * 4)
#define SMEM_USED (BAR_OFF + 384)
#define SMEM_BYTES (SMEM_USED + 1024)

// barriers, 8 B each; the pair's full ones live in CTA 0 and count both CTAs' bytes
#define B_QFULL 0
#define B_FULL (B_QFULL + 1)        // [NK]
#define B_EMPTY (B_FULL + NK)       // [NK]
#define B_VFULL (B_EMPTY + NK)      // [NVP]
#define B_VEMPTY (B_VFULL + NVP)    // [NVP]
#define B_QKDONE (B_VEMPTY + NVP)   // [2]
#define B_SFULL (B_QKDONE + 2)      // [2]
#define B_SEMPTY (B_SFULL + 2)      // [2]
#define B_PFULL (B_SEMPTY + 2)
#define B_ODONE (B_PFULL + 1)
#define B_COUNT (B_ODONE + 1)

#define TM_O 0
#define TM_S 256

__device__ __forceinline__ uint64_t desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr >> 4) & 0x3FFF) | (uint64_t)((lbo >> 4) & 0x3FFF) << 16 | (uint64_t)((sbo >> 4) & 0x3FFF) << 32 |
         (1ull << 46) | (2ull << 61);
}
__device__ __forceinline__ void init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
#ifdef MLA_DEBUG
__device__ __forceinline__ void wait(uint32_t bar, uint32_t parity) {
  for (long long i = 0;; ++i) {
    uint32_t ok;
    asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    if (ok) return;
    if (i == (1ll << 24)) {
      printf("hang: block %d,%d,%d thread %d bar %d parity %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, (bar & 1023) / 8, parity);
      asm volatile("trap;");
    }
  }
}
#define wait_cluster wait
#else
__device__ __forceinline__ void wait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n.reg .pred P;\nW: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}" ::"r"(bar), "r"(parity)
               : "memory");
}
__device__ __forceinline__ void wait_cluster(uint32_t bar, uint32_t parity) {
  asm volatile("{\n.reg .pred P;\nW: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}"
               ::"r"(bar), "r"(parity) : "memory");
}
#endif
__device__ __forceinline__ uint32_t peer0(uint32_t addr) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(r) : "r"(addr));
  return r;
}
__device__ __forceinline__ void expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void arrive_remote(uint32_t cluster_bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(cluster_bar) : "memory");
}
__device__ __forceinline__ void tma(uint32_t dst, const CUtensorMap* map, int x, int y, int z, uint32_t cluster_bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 [%0], [%1, {%2, %3, %4}], [%5];"
               ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(z), "r"(cluster_bar) : "memory");
}
#ifdef MLA_PROF
__device__ unsigned long long mla_prof[32];
#define PT0 const long long _pt = clock64();
#define PT1(i) pacc[i] += clock64() - _pt;
#define PFLUSH if (blockIdx.y == 0 && blockIdx.z == 0 && (threadIdx.x % 32) == 0) for (int _i = 0; _i < 16; ++_i) if (pacc[_i]) atomicAdd(&mla_prof[_i + 16 * blockIdx.x], pacc[_i]);
#else
#define PT0
#define PT1(i) ;
#define PFLUSH
#endif
// a page into L2 by the warp's lanes, 18 lines each (LSU, not the TMA unit the ring's loads queue in)
__device__ __forceinline__ void prefetch(const uint8_t* page, int lane) {
#pragma unroll
  for (int k = 0; k < PAGE * ROW * 2 / 128 / 32; ++k) asm volatile("prefetch.global.L2 [%0];" ::"l"(page + 128 * (lane + 32 * k)));
}
__device__ __forceinline__ void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, int acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
               ::"r"(bar), "h"((uint16_t)3) : "memory");
}
__device__ __forceinline__ void tld32(uint32_t taddr, float* v) {
  uint32_t* u = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
               "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
               : "=r"(u[0]), "=r"(u[1]), "=r"(u[2]), "=r"(u[3]), "=r"(u[4]), "=r"(u[5]), "=r"(u[6]), "=r"(u[7]), "=r"(u[8]),
                 "=r"(u[9]), "=r"(u[10]), "=r"(u[11]), "=r"(u[12]), "=r"(u[13]), "=r"(u[14]), "=r"(u[15]), "=r"(u[16]),
                 "=r"(u[17]), "=r"(u[18]), "=r"(u[19]), "=r"(u[20]), "=r"(u[21]), "=r"(u[22]), "=r"(u[23]), "=r"(u[24]),
                 "=r"(u[25]), "=r"(u[26]), "=r"(u[27]), "=r"(u[28]), "=r"(u[29]), "=r"(u[30]), "=r"(u[31])
               : "r"(taddr));
}
__device__ __forceinline__ void tst32(uint32_t taddr, const float* v) {
  const uint32_t* u = reinterpret_cast<const uint32_t*>(v);
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
               "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
               ::"r"(taddr), "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]), "r"(u[6]), "r"(u[7]), "r"(u[8]),
                 "r"(u[9]), "r"(u[10]), "r"(u[11]), "r"(u[12]), "r"(u[13]), "r"(u[14]), "r"(u[15]), "r"(u[16]), "r"(u[17]),
                 "r"(u[18]), "r"(u[19]), "r"(u[20]), "r"(u[21]), "r"(u[22]), "r"(u[23]), "r"(u[24]), "r"(u[25]), "r"(u[26]),
                 "r"(u[27]), "r"(u[28]), "r"(u[29]), "r"(u[30]), "r"(u[31]));
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t bf2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ bool elect_one() {
  uint32_t p;
  asm volatile("{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\nselp.u32 %0, 1, 0, e;\n}" : "=r"(p));
  return p;
}
__device__ __forceinline__ void pair_sync(int id) { asm volatile("bar.sync %0, 64;" ::"r"(id) : "memory"); }

extern "C" __global__ void __launch_bounds__(THREADS, 1) __cluster_dims__(2, 1, 1)
kern_k3_mla_decode(const __grid_constant__ CUtensorMap q_map, const __grid_constant__ CUtensorMap kv_map,
                   const uint8_t* __restrict__ kv, const int* __restrict__ table, const int* __restrict__ seq_lens, const int* __restrict__ bsk,
                   float* __restrict__ acc_o, float* __restrict__ acc_lse, int split_max, int max_pages, float scale_log2) {
  const int b = blockIdx.y, split = blockIdx.z;
  const int n = seq_lens[b];
  if (n == 0) return;
  const int tiles = (n + TILE - 1) / TILE, asked = bsk[b], per = (tiles + asked - 1) / asked;
  const int t0 = split * per, nt = min(tiles, t0 + per) - t0;
  if (nt <= 0) return;
  const int npages = (n + PAGE - 1) / PAGE;

  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
  uint8_t* const gen = smem_raw + (base - raw);
  const uint32_t bars = base + BAR_OFF;
  auto bar = [&](int i) { return bars + 8 * i; };
  uint32_t c;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(c));
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
#ifdef MLA_PROF
  unsigned long long pacc[16] = {};
#endif

  if (t == 0) {
    for (int i = 0; i < B_COUNT; ++i) init(bar(i), 1);
    init(bar(B_SEMPTY), 2 * SOFTMAX_WARPS);
    init(bar(B_SEMPTY + 1), 2 * SOFTMAX_WARPS);
    init(bar(B_PFULL), 2 * SOFTMAX_WARPS);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(bars + 8 * B_COUNT));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tmem = *reinterpret_cast<volatile uint32_t*>(gen + BAR_OFF + 8 * B_COUNT);

  if (warp == LOAD_WARP || warp == V_WARP) {
    // lane l holds the page ids of tile i + l for the next 32 tiles; lane 0 issues
    const int* const rt = table + (size_t)b * max_pages;
    auto page = [&](int i) { return rt[min(i, npages - 1)]; };
    const bool kw = warp == LOAD_WARP;
    if (kw && lane == 0) {
      if (c == 0) expect(bar(B_QFULL), 2 * 9 * BOX);
      for (int k = 0; k < 9; ++k) tma(base + Q_OFF + k * BOX, &q_map, 64 * k, 64 * c, b, peer0(bar(B_QFULL)));
    }
    int id0 = 0, id1 = 0, ida = 0;
    auto ids = [&](int i) {
      if (i % 32 == 0) {
        const int p = 2 * (t0 + i + lane);
        id0 = page(p + c), id1 = page(p + 1 - c), ida = page(p + 2 * PF + c);
      }
      return i % 32;
    };
    if (kw) {
      for (int i = 0; i < PF && i < nt; ++i) prefetch(kv + (size_t)page(2 * (t0 + i) + c) * PAGE * ROW * 2, lane);
      for (int i = 0; i < nt; ++i) {
        const int l = ids(i), own = __shfl_sync(0xffffffff, id0, l), next = __shfl_sync(0xffffffff, ida, l);
        if (PF && i + PF < nt) prefetch(kv + (size_t)next * PAGE * ROW * 2, lane);
        for (int j = 0; j < 9; ++j) {
          const int k = 9 * i + j, s = k % NK;
          wait(bar(B_EMPTY + s), ((k / NK) & 1) ^ 1);
          if (lane == 0) {
            if (c == 0) expect(bar(B_FULL + s), 2 * BOX);
            tma(base + RING_OFF + s * BOX, &kv_map, 64 * j, 0, own, peer0(bar(B_FULL + s)));
          }
        }
      }
    } else {
      // once a tile's Q K^T is done its pages are in L2: both pages' dims 256g + 128c .. +128, slot 2g + p
      for (int i = 0; i < nt; ++i) {
        const int l = ids(i), own = __shfl_sync(0xffffffff, id0, l), other = __shfl_sync(0xffffffff, id1, l);
        wait(bar(B_QKDONE + (i & 1)), (i >> 1) & 1);
        for (int v = 0; v < 4; ++v) {
          const int k = 4 * i + v, s = k % NVP, pg = (v & 1) == c ? own : other;
          wait(bar(B_VEMPTY + s), ((k / NVP) & 1) ^ 1);
          if (lane == 0) {
            if (c == 0) expect(bar(B_VFULL + s), 2 * PAIR);
            tma(base + V_OFF + s * PAIR, &kv_map, 256 * (v >> 1) + 128 * c, 0, pg, peer0(bar(B_VFULL + s)));
            tma(base + V_OFF + s * PAIR + BOX, &kv_map, 256 * (v >> 1) + 128 * c + 64, 0, pg, peer0(bar(B_VFULL + s)));
          }
        }
      }
    }
  }
  if (warp == MMA_WARP && c == 0) {
    // f32 D, bf16 A and B, K-major A, B K-major (K) / MN-major (V); M 128 (the pair) x N 128 / 256
    const uint32_t id_qk = (1u << 4) | (1u << 7) | (1u << 10) | ((TILE >> 3) << 17) | ((M_TILE >> 4) << 24);
    const uint32_t id_pv = (1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) | ((256 >> 3) << 17) | ((M_TILE >> 4) << 24);
    const bool e = elect_one();
    wait(bar(B_QFULL), 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    auto qk = [&](int i) {
      const int sb = i & 1;
      { PT0 wait(bar(B_SEMPTY + sb), ((i >> 1) & 1) ^ 1); PT1(0) }
      const uint32_t d = tmem + TM_S + 64 * sb;
      for (int j = 0; j < 9; ++j) {
        const int k = 9 * i + j, s = k % NK;
        { PT0 wait(bar(B_FULL + s), (k / NK) & 1); PT1(1) }
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (e) {
#pragma unroll
          for (int q = 0; q < 4; ++q)
            mma(d, desc(base + Q_OFF + j * BOX + q * 32, 16, 1024), desc(base + RING_OFF + s * BOX + q * 32, 16, 1024), id_qk,
                j | q);
          commit(bar(B_EMPTY + s));
        }
        __syncwarp();
      }
      if (e) {
        commit(bar(B_QKDONE + sb));
        commit(bar(B_SFULL + sb));
      }
      __syncwarp();
    };
    auto pv = [&](int i) {
      { PT0 wait(bar(B_PFULL), i & 1); PT1(2) }
      for (int v = 0; v < 4; ++v) {
        const int k = 4 * i + v, s = k % NVP, g = v >> 1, p = v & 1;
        { PT0 wait(bar(B_VFULL + s), (k / NVP) & 1); PT1(3) }
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (e) {
#pragma unroll
          for (int q = 0; q < 4; ++q)
            mma(tmem + TM_O + 128 * g, desc(base + P_OFF + p * BOX + q * 32, 16, 1024),
                desc(base + V_OFF + s * PAIR + q * 2048, BOX, 1024), id_pv, i | p | q);
          commit(bar(B_VEMPTY + s));
        }
        __syncwarp();
      }
      if (e) commit(bar(B_ODONE));
      __syncwarp();
    };
    PT0
    qk(0);
    for (int i = 0; i < nt; ++i) {
      if (i + 1 < nt) qk(i + 1);
      pv(i);
    }
    PT1(7)
  }
  if (warp < SOFTMAX_WARPS) {
    // lane quarter w: head row 32 (w & 1) + lane, positions / O dims half w >> 1
    const int row = 32 * (warp & 1) + lane, half = warp >> 1;
    const uint32_t lanes = (uint32_t)(32 * warp) << 16;
    float* const xch = reinterpret_cast<float*>(gen + XCH_OFF);
    uint8_t* const prow = gen + P_OFF + half * BOX + row * 128;
    const uint32_t sempty = peer0(bar(B_SEMPTY)), pfull = peer0(bar(B_PFULL));
    float m = -INFINITY, sum = 0.f;
    for (int i = 0; i < nt; ++i) {
      const int sb = i & 1;
      float x[64];
      { PT0 wait(bar(B_SFULL + sb), (i >> 1) & 1); if (warp == 0) PT1(4) }
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      PT0
      tld32(tmem + lanes + TM_S + 64 * sb, x);
      tld32(tmem + lanes + TM_S + 64 * sb + 32, x + 32);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncwarp();
      if (lane == 0) arrive_remote(sempty + 8 * sb);
      if (warp == 0) PT1(6)
      { PT0
      const int valid = n - (TILE * (t0 + i) + PAGE * half);
      float mx = -INFINITY;
#pragma unroll
      for (int k = 0; k < 64; ++k) {
        x[k] = k < valid ? x[k] * scale_log2 : -INFINITY;
        mx = fmaxf(mx, x[k]);
      }
      xch[(sb * 2 + half) * 64 + row] = mx;
      pair_sync(1 + (warp & 1));
      mx = fmaxf(mx, xch[(sb * 2 + (half ^ 1)) * 64 + row]);
      const bool raise = mx > m + RESCALE;
      const float scale = raise ? ex2(m - mx) : 1.f;
      if (raise) m = mx;
      float s = 0.f;
      uint32_t pk[32];
#pragma unroll
      for (int k = 0; k < 32; ++k) {
        const float a = ex2(x[2 * k] - m), bb = ex2(x[2 * k + 1] - m);
        s += a + bb;
        pk[k] = bf2(a, bb);
      }
      sum = sum * scale + s;
      if (warp == 0) PT1(8)
      if (i > 0) {
        { PT0 wait(bar(B_ODONE), (i - 1) & 1); if (warp == 0) PT1(5) }
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (__any_sync(0xffffffff, raise)) {
#ifdef MLA_PROF
          pacc[10] += 1940;
#endif
#pragma unroll 1
          for (int q = 0; q < 8; ++q) {
            float o[32];
            tld32(tmem + lanes + TM_O + 32 * q, o);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
            for (int k = 0; k < 32; ++k) o[k] *= scale;
            tst32(tmem + lanes + TM_O + 32 * q, o);
          }
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
      }
#pragma unroll
      for (int u = 0; u < 8; ++u)
        *reinterpret_cast<uint4*>(prow + ((u ^ (row & 7)) << 4)) = make_uint4(pk[4 * u], pk[4 * u + 1], pk[4 * u + 2], pk[4 * u + 3]);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncwarp();
      if (lane == 0) arrive_remote(pfull);
      if (warp == 0) PT1(9) }
    }
    wait(bar(B_ODONE), (nt - 1) & 1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    xch[(4 + half) * 64 + row] = sum;
    pair_sync(1 + (warp & 1));
    const float total = sum + xch[(4 + (half ^ 1)) * 64 + row], inv = 1.f / total;
    const int head = 64 * c + row;
    // TMEM column 32q: dims 256 (q / 4) + 128 half + 32 (q % 4)
    float* const o = acc_o + (((size_t)b * M_TILE + head) * split_max + split) * LAT + 128 * half;
#pragma unroll 1
    for (int q = 0; q < 8; ++q) {
      float v[32];
      tld32(tmem + lanes + TM_O + 32 * q, v);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      if (head < HEADS) {
        float4* dst = reinterpret_cast<float4*>(o + 256 * (q / 4) + 32 * (q % 4));
#pragma unroll
        for (int k = 0; k < 8; ++k) dst[k] = make_float4(v[4 * k] * inv, v[4 * k + 1] * inv, v[4 * k + 2] * inv, v[4 * k + 3] * inv);
      }
    }
    if (half == 0 && head < HEADS) acc_lse[((size_t)b * M_TILE + head) * split_max + split] = m + __log2f(total);
  }
  PFLUSH
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(tmem));
}
