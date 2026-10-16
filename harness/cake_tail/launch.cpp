#include <cuda.h>
#include <math.h>
#include <stdint.h>
static CUmodule mods[2];
static CUfunction fns[2];
#define CK(x)                                                                  \
  do {                                                                         \
    CUresult r = (x);                                                          \
    if (r != CUDA_SUCCESS)                                                     \
      return (int)r;                                                           \
  } while (0)
extern "C" int setup(int which, const char *path) {
  CK(cuModuleLoad(&mods[which], path));
  CK(cuModuleGetFunction(
      &fns[which], mods[which],
      "kernel_cake_kimi_k3_mla_fp8_paged_attention_ea17187e76fd47ab2653"));
  CK(cuFuncSetAttribute(
      fns[which], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 232448));
  return 0;
}
extern "C" int run(int which, uint64_t qp, uint64_t kvp, uint64_t op,
                   uint64_t mp, uint64_t sp, uint64_t lp, uint64_t cp,
                   uint64_t pp, int rows, int pages, int splits, int batch,
                   uint64_t stream) {
  // Match the published q1 / 96-head wide-kernel ABI. K and V alias the
  // latent cache; rotary K uses its separate 64-byte TMA box and swizzle.
  CUtensorMap tm[5];
  uint64_t stride[1] = {576};
  uint32_t el[2] = {1, 1};
  for (int i = 0; i < 5; i++) {
    bool q = (i == 0 || i == 2);
    bool rope = (i == 2 || i == 3);
    uint64_t dims[2] = {576, (uint64_t)(q ? batch * 96 : rows)};
    uint32_t box[2] = {(uint32_t)(rope ? 64 : 128), 64};
    CK(cuTensorMapEncodeTiled(
        &tm[i], CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void *)(q ? qp : kvp), dims,
        stride, box, el, CU_TENSOR_MAP_INTERLEAVE_NONE,
        rope ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  float scale = 1.f / sqrtf(192.f) * log2f(expf(1.f)), vscale = 1.f;
  int heads = 96, tiles = 1, full = 0;
  void *args[] = {&tm[0], &tm[1],  &tm[2], &tm[3], &tm[4], &op,    &mp,
                  &sp,    &op,     &lp,    &cp,    &pp,    &scale, &vscale,
                  &heads, &splits, &pages, &tiles, &full};
  CUlaunchAttribute attr{};
  attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attr.value.clusterDim = {2, 1, 1};
  CUlaunchConfig cfg{};
  cfg.gridDimX = 2 * batch * splits;
  cfg.gridDimY = cfg.gridDimZ = 1;
  cfg.blockDimX = 384;
  cfg.blockDimY = cfg.blockDimZ = 1;
  cfg.sharedMemBytes = 232448;
  cfg.hStream = (CUstream)stream;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  CK(cuLaunchKernelEx(&cfg, fns[which], args, nullptr));
  return 0;
}
