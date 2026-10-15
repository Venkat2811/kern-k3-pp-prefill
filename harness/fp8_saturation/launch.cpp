#include <cuda.h>
#include <stdint.h>
static CUmodule mods[4];
static CUfunction fns[4];
#define CK(x)                                                                  \
  do {                                                                         \
    CUresult r = (x);                                                          \
    if (r != CUDA_SUCCESS)                                                     \
      return (int)r;                                                           \
  } while (0)
extern "C" int setup(int which, const char *path, const char *entry) {
  CK(cuModuleLoad(&mods[which], path));
  CK(cuModuleGetFunction(&fns[which], mods[which], entry));
  return 0;
}
extern "C" int absorb(int which, uint64_t qp, uint64_t wp, uint64_t op, int b,
                      uint64_t stream) {
  void *args[] = {&qp, &wp, &op, &b};
  CK(cuLaunchKernel(fns[which], 1, 96, 4, 256, 1, 1, 0, (CUstream)stream, args,
                    0));
  return 0;
}
extern "C" int prep(int which, uint64_t p, uint64_t gq, uint64_t gk,
                    uint64_t slots, uint64_t slab, uint64_t q, uint64_t gate,
                    int b, uint64_t stream) {
  long long off = 0, stride = 64 * 576;
  void *args[] = {&p, &gq, &gk, &slots, &slab, &off, &stride, &q, &gate, &b};
  CK(cuLaunchKernel(fns[which], b, 3, 1, 256, 1, 1, 0, (CUstream)stream, args,
                    0));
  return 0;
}
