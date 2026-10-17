// The shared expert's down projection of a D member, beside moe_front instead of after FC2:
//   out[t, n] = bf16(sum_k act[t, k] * W[n, k]) for t < T rows, n < N, k < K,
// act being moe_front's SiTU rows, published row by row (`ready`).
//
//   kern_k3_sh_down(const bf16* act, i32* ready, const bf16* W, bf16* out, int T)
//   grid (N / 64, 1, 1)   block (256, 1, 1)   shared SMEM bytes   T <= 64
//
// Launched as a programmatic dependent of moe_front, it does not wait for moe_front's grid: moe_front's
// block b stores act row b and then, for each CTA c of this grid, ready[64 c + b] = 1 (st.release.gpu);
// CTA c's threads acquire its flags of rows [0, T) before reading act and clear them for the next
// layer's moe_front (a zeroed carry). Every moe_front block is resident before this grid launches: the
// trigger fires only once all of them have run it. It reads nothing else of moe_front's and writes only
// its rows of the MoE all-reduce's buffer, which nothing before it touches. FC1, the launch after it,
// waits for this grid and for moe_front's (a dependent's completion does not stand in for its
// primary's). So the projection runs on the SMs moe_front leaves idle while FC1's CTAs wait for the
// routing tables, instead of after FC2.
//
// CTA b takes W's rows [64 b, 64 b + 64): they land in shared memory by cp.async at entry, the T
// activation rows once published (pitch PITCH: an odd number of 16-byte units, so ldmatrix's eight rows
// fall on distinct banks). Up to 24 rows warp w < 2 multiplies rows 64 b + 32 w + [0, 32) by every row
// tile (each fragment read from shared memory once); from 25 rows, where two warps cannot issue the mma
// fast enough, warp w < 4 multiplies rows 64 b + 16 w + [0, 16) by every row tile. Both on the tensor
// cores (mma.sync m16n8k16 bf16, f32 accumulators), each output one
// chain over k = 0, 16, 32, ... in order: the order and the instruction cuBLASLt's kernels for this shape
// sum in (nvjet tst 64 x 8..48 tiles, no split-K), so the result is cuBLASLt's bit for bit (a harness
// compares every output at 1-64 rows on random and real weights; the step's logits are identical). Rounded to bf16 and staged for row-contiguous
// stores.
//
//   nvcc -cubin -arch=sm_103a -DN=7168 -DK=768 k3_sh_down.cu
#include <cuda_bf16.h>
#include <cstdint>

#ifndef N
#define N 7168
#endif
#ifndef K
#define K 768
#endif
#define TMAX 64
#define ROWS 64
#define THREADS 256
#define PITCH (K * 2 + 16)  // an odd number of 16-byte units: ldmatrix's eight rows hit distinct banks
#define SMEM (ROWS * PITCH + TMAX * PITCH + TMAX * ROWS * 2)
static_assert(N % ROWS == 0 && K % 64 == 0, "tiles");
static_assert((PITCH / 16) % 2 == 1, "conflict-free ldmatrix rows");

__device__ __forceinline__ uint32_t smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int TILES, int STEP>
__device__ __forceinline__ void frags16(uint32_t a_at, uint32_t b_at, int p, int tiles, int k, uint32_t (&a)[4],
                                      uint32_t (&b)[TILES][2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(a_at + k * 2));
#pragma unroll
  for (int j = 0; j < TILES; ++j)
    if (p + STEP * j < tiles)
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                   : "=r"(b[j][0]), "=r"(b[j][1]) : "r"(b_at + (p + STEP * j) * 8 * PITCH + k * 2));
}

// Warp (g, p)'s 16 rows times its row tiles j = p, p + 2, ... below `tiles`: one mma chain an output,
// k ascending; the fragments of the next DEPTH steps are in flight while a step multiplies.
template <int TILES, int STEP>
__device__ __forceinline__ void rows16(uint32_t a_at, uint32_t b_at, int p, int tiles, float (&acc)[TILES][4]) {
  constexpr int DEPTH = 3, STEPS = K / 16;
#pragma unroll
  for (int j = 0; j < TILES; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  uint32_t a[DEPTH][4], b[DEPTH][TILES][2];
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) frags16<TILES, STEP>(a_at, b_at, p, tiles, 16 * d, a[d], b[d]);
#pragma unroll
  for (int s = 0; s < STEPS; ++s) {
    const int d = s % DEPTH;
#pragma unroll
    for (int j = 0; j < TILES; ++j)
      if (p + STEP * j < tiles) mma(acc[j], a[d][0], a[d][1], a[d][2], a[d][3], b[d][j][0], b[d][j][1]);
    if (s + DEPTH < STEPS) frags16<TILES, STEP>(a_at, b_at, p, tiles, 16 * (s + DEPTH), a[d], b[d]);
  }
}

template <int TILES>
__device__ __forceinline__ void frags32(uint32_t a_at, uint32_t b_at, int tiles, int k, uint32_t (&a)[2][4],
                                      uint32_t (&b)[TILES][2]) {
#pragma unroll
  for (int m = 0; m < 2; ++m)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a[m][0]), "=r"(a[m][1]), "=r"(a[m][2]), "=r"(a[m][3]) : "r"(a_at + m * 16 * PITCH + k * 2));
#pragma unroll
  for (int j = 0; j < TILES; ++j)
    if (j < tiles)
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                   : "=r"(b[j][0]), "=r"(b[j][1]) : "r"(b_at + j * 8 * PITCH + k * 2));
}

// Warp g's 32 rows (two 16-row halves) times every row tile below `tiles`: one mma chain an output,
// k ascending; the fragments of the next DEPTH steps are in flight while a step multiplies. The sums,
// rounded to bf16, go to the tile [t][64 rows] (row n of this lane's at n0).
template <int TILES>
__device__ __forceinline__ void rows32(uint32_t a_at, uint32_t b_at, int tiles, __nv_bfloat16* tile, int n0, int q) {
  constexpr int DEPTH = 3, STEPS = K / 16;
  float acc[2][TILES][4];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int j = 0; j < TILES; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f;
  uint32_t a[DEPTH][2][4], b[DEPTH][TILES][2];
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) frags32<TILES>(a_at, b_at, tiles, 16 * d, a[d], b[d]);
#pragma unroll
  for (int s = 0; s < STEPS; ++s) {
    const int d = s % DEPTH;
#pragma unroll
    for (int j = 0; j < TILES; ++j)
      if (j < tiles)
#pragma unroll
        for (int m = 0; m < 2; ++m) mma(acc[m][j], a[d][m][0], a[d][m][1], a[d][m][2], a[d][m][3], b[d][j][0], b[d][j][1]);
    if (s + DEPTH < STEPS) frags32<TILES>(a_at, b_at, tiles, 16 * (s + DEPTH), a[d], b[d]);
  }
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int j = 0; j < TILES; ++j)
      if (j < tiles)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          tile[(j * 8 + 2 * q + i % 2) * ROWS + n0 + 16 * m + (i / 2) * 8] = __float2bfloat16(acc[m][j][i]);
}

extern "C" __global__ void __launch_bounds__(THREADS, 1)
    kern_k3_sh_down(const __nv_bfloat16* __restrict__ act, int* __restrict__ ready,
                    const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ out, int T) {
  extern __shared__ __align__(128) uint8_t sm[];  // the weight rows, T activation rows, the bf16 tile [T][64]
  uint8_t* sw = sm;
  uint8_t* sa = sm + ROWS * PITCH;
  __nv_bfloat16* tile = reinterpret_cast<__nv_bfloat16*>(sa + TMAX * PITCH);
  const int t = threadIdx.x, warp = t / 32, lane = t % 32, n0 = blockIdx.x * ROWS, g = lane / 4, q = lane % 4;
  asm volatile("griddepcontrol.launch_dependents;");
  // the CTA's 64 weight rows, 16 bytes a thread at a time (cp.async: every load in flight at once)
  for (int i = t; i < ROWS * (K / 8); i += THREADS)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 ::"r"(smem(sw + (i / (K / 8)) * PITCH + (i % (K / 8)) * 16)), "l"(W + (size_t)n0 * K + i * 8) : "memory");
  asm volatile("cp.async.commit_group;" ::: "memory");
  // the activation rows as moe_front publishes them
  if (t < T) {
    int v;
    int* flag = ready + blockIdx.x * 64 + t;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(flag) : "memory");
    while (v != 1) {
      __nanosleep(64);
      asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(flag) : "memory");
    }
    *flag = 0;  // this CTA's own flag, for the next layer's moe_front
  }
  __syncthreads();
  for (int i = t; i < T * (K / 8); i += THREADS)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 ::"r"(smem(sa + (i / (K / 8)) * PITCH + (i % (K / 8)) * 16)), "l"(act + i * 8) : "memory");
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();

  // up to 24 rows warps 0-1 multiply (32 rows each: every fragment read from shared memory once); from 25
  // rows, where two warps cannot issue the mma fast enough, warps 0-3 take 16 rows each by every row tile
  if (T > 24) {
    if (warp < 4) {
      const int tiles = (T + 7) / 8;
      const uint32_t a_at = smem(sw + (16 * warp + lane % 16) * PITCH + (lane / 16) * 16);
      const uint32_t b_at = smem(sa + (lane % 8) * PITCH + ((lane / 8) % 2) * 16);
      float acc[TMAX / 8][4];
      rows16<TMAX / 8, 1>(a_at, b_at, 0, tiles, acc);
#pragma unroll
      for (int j = 0; j < TMAX / 8; ++j)
        if (j < tiles)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            tile[(j * 8 + 2 * q + i % 2) * ROWS + 16 * warp + g + (i / 2) * 8] = __float2bfloat16(acc[j][i]);
    }
  } else if (warp < 2) {
    // ldmatrix rows: A (16 W rows) lane l -> row l % 16, column half l / 16; B (8 act rows) lane l -> row l % 8, half (l / 8) % 2
    const uint32_t a_at = smem(sw + (32 * warp + lane % 16) * PITCH + (lane / 16) * 16);
    const uint32_t b_at = smem(sa + (lane % 8) * PITCH + ((lane / 8) % 2) * 16);
    const int tiles = (T + 7) / 8;
    if (T <= 8)
      rows32<1>(a_at, b_at, tiles, tile, 32 * warp + g, q);
    else
      rows32<3>(a_at, b_at, tiles, tile, 32 * warp + g, q);
  }
  __syncthreads();
  for (int i = t; i < T * (ROWS / 8); i += THREADS) {
    const int r = i / (ROWS / 8), c = i % (ROWS / 8);
    reinterpret_cast<uint4*>(out + (size_t)r * N + n0)[c] = reinterpret_cast<const uint4*>(tile + r * ROWS)[c];
  }
}
