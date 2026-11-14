// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// trellis_search.cu - the code search of the converter (tools/convert), one CUDA kernel.
//
// One sequence is 256 weights = 128 steps of 2 weights (docs/FORMAT.md).  The stored symbol of a step has K bits (6 or 8) and
// the 12-bit state is
//     s_j = ((s_(j-1) >> K) | (v_j << (12 - K))) & 4095,      s before the first step = 0,
// and the two weights of step j are table[s_j] (times the row scale, which the caller has divided out of the targets).
// The kernel finds, for every sequence, the symbols v_0 .. v_127 that minimise
//     sum over j of | t_j - table[s_j] |^2 ,
// by the Viterbi algorithm.  With `taps` > 0 the target of a step is not the input z_j itself: the errors that the same path
// made at the `taps` steps before are fed back first,
//     t_j = z_j - sum over i = 1 .. taps of  e_(j-i) * F[j][i-1]      (e: 1 x 2 row vector, F: 2 x 2, row-major)
// which is the error feedback of a column-by-column rounding (GPTQ / LDLQ) carried inside the search.
//
// How the search is organised.  The low 12 - K bits of a state are the high bits of the state before it, so all states with the
// same low bits ("class") have the same 2^K predecessors: one best predecessor is found per class and shared.  Each class keeps
// the last `taps` errors of its surviving path.  One thread block of 1,024 threads works on one sequence.
//
// Arithmetic: fp32, compiled with --fmad=false (no fused multiply-add), every sum in the order written here.  The Python
// reference in trellis_search.py repeats the same operations in the same order and gives the same symbols.
#include <cuda_runtime.h>
#include <cfloat>
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#  define YSKN_TRELLIS_API __declspec(dllexport)
#else
#  define YSKN_TRELLIS_API __attribute__((visibility("default")))
#endif

namespace {

constexpr int L = 12, STEPS = 128;

template <int K, int TAPS> struct Cfg {
  static constexpr int S = 1 << L, NLOW = 1 << (L - K), NPRED = 1 << K;
  static constexpr uint32_t LOWMASK = NLOW - 1;
  static constexpr int THREADS = 1024, SPT = S / THREADS;             // states per thread in the metric update
  static constexpr int SL0 = NPRED / SPT;
  static constexpr int SLICES = SL0 > 32 ? 32 : (SL0 < 1 ? 1 : SL0);  // threads that share one class (inside one warp)
  static constexpr int PPT = NPRED / SLICES;                          // predecessors scanned by one thread
  // shared memory of one block: the table, two metric arrays, best metric and target per class, the error rings, best symbol
  static constexpr size_t SMEM = (size_t)S * 8 + 2ull * S * 4 + (size_t)NLOW * 4 + 2ull * NLOW * 8
                               + (TAPS ? 2ull * NLOW * TAPS * 8 : 0) + (size_t)NLOW * 2 + 16;
  static_assert(S % THREADS == 0, "the number of states must be a multiple of 1024");
  static_assert(NLOW * SLICES <= THREADS, "one thread group per class must fit into the block");
  static_assert((SLICES & (SLICES - 1)) == 0, "SLICES must be a power of two");
};

// The best predecessor of every class.  Ties: a thread keeps the first minimum of its own range; the threads of a class are
// then merged in halves (thread i with thread i + o, o = SLICES/2 ... 1) and the lower thread wins a tie.
template <int K, int TAPS>
__device__ __forceinline__ void reduce_best(const float* M, const float2* table, const float2* t_prev, const float2* err_old,
                                            float* best, uint16_t* bestq, float2* err_new, int step) {
  using C = Cfg<K, TAPS>;
  const int tid = threadIdx.x, low = tid / C::SLICES, slice = tid % C::SLICES;
  if (low >= C::NLOW) return;
  float m = FLT_MAX; int q = 0;
  #pragma unroll
  for (int i = 0; i < C::PPT; ++i) {                                  // the predecessors of a class are contiguous
    const int qq = slice * C::PPT + i;
    const float v = M[(low << K) | qq];
    if (v < m) { m = v; q = qq; }
  }
  const unsigned mask = (C::SLICES == 32) ? 0xffffffffu : (((1u << C::SLICES) - 1u) << ((tid & 31) & ~(C::SLICES - 1)));
  #pragma unroll
  for (int o = C::SLICES / 2; o > 0; o >>= 1) {
    const float om = __shfl_down_sync(mask, m, o);
    const int   oq = __shfl_down_sync(mask, q, o);
    if (om < m) { m = om; q = oq; }
  }
  if (slice == 0) {
    best[low] = m; bestq[low] = (uint16_t)q;
    if constexpr (TAPS > 0) {                                         // the class inherits the error ring of its best predecessor
      const uint32_t s_star = ((uint32_t)low << K) | (uint32_t)q, low_prev = s_star & C::LOWMASK;
      float2 e = make_float2(0.f, 0.f);
      if (step > 0 && m < FLT_MAX) { const float2 tp = t_prev[low_prev], c = table[s_star]; e = make_float2(tp.x - c.x, tp.y - c.y); }
      #pragma unroll
      for (int i = TAPS - 1; i > 0; --i) err_new[low * TAPS + i] = err_old[low_prev * TAPS + i - 1];
      err_new[low * TAPS] = e;
    }
  }
}

template <int K, int TAPS>
__global__ void __launch_bounds__(1024)
search_kernel(const float* __restrict__ zs, const float* __restrict__ feedback, const float2* __restrict__ g_table,
              uint16_t* __restrict__ out_sym, uint8_t* __restrict__ bp_g) {
  using C = Cfg<K, TAPS>;
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* p = smem;
  float2* table = reinterpret_cast<float2*>(p); p += (size_t)C::S * 8;
  float*  M     = reinterpret_cast<float*>(p);  p += 2ull * C::S * 4;
  float*  best  = reinterpret_cast<float*>(p);  p += (size_t)C::NLOW * 4;
  float2* tcur  = reinterpret_cast<float2*>(p); p += 2ull * C::NLOW * 8;
  float2* err   = reinterpret_cast<float2*>(p); p += (TAPS ? 2ull * C::NLOW * TAPS * 8 : 0);
  uint16_t* bq  = reinterpret_cast<uint16_t*>(p);

  const int b = blockIdx.x, tid = threadIdx.x;
  const float2* z = reinterpret_cast<const float2*>(zs + (size_t)b * 2 * STEPS);
  for (int i = tid; i < C::S; i += C::THREADS) table[i] = g_table[i];
  for (int i = tid; i < C::S; i += C::THREADS) M[i] = (i == 0) ? 0.f : FLT_MAX;      // every sequence starts in state 0
  if constexpr (TAPS > 0) for (int i = tid; i < 2 * C::NLOW * TAPS; i += C::THREADS) err[i] = make_float2(0.f, 0.f);
  __syncthreads();

  uint8_t* bp = bp_g + (size_t)b * STEPS * C::NLOW;
  int pp = 0;
  for (int j = 0; j < STEPS; ++j) {
    reduce_best<K, TAPS>(M + pp * C::S, table, tcur + pp * C::NLOW, err + pp * C::NLOW * TAPS, best, bq, err + (pp ^ 1) * C::NLOW * TAPS, j);
    __syncthreads();
    if (tid < C::NLOW) {                                              // the target of this step for the class, after the feedback
      float2 t = z[j];
      if constexpr (TAPS > 0) if (feedback) {
        #pragma unroll
        for (int i = 1; i <= TAPS; ++i) {
          if (j - i < 0) break;
          const float2 e = err[((pp ^ 1) * C::NLOW + tid) * TAPS + i - 1];
          const float* F = feedback + ((size_t)j * TAPS + (i - 1)) * 4;   // [e0 e1] * [[F0 F1], [F2 F3]]
          t.x -= e.x * F[0] + e.y * F[2];
          t.y -= e.x * F[1] + e.y * F[3];
        }
      }
      tcur[(pp ^ 1) * C::NLOW + tid] = t;
      bp[(size_t)j * C::NLOW + tid] = (uint8_t)bq[tid];
    }
    __syncthreads();
    float* Mn = M + (pp ^ 1) * C::S;
    #pragma unroll
    for (int i = 0; i < C::SPT; ++i) {
      const int s = tid * C::SPT + i, low = s & C::LOWMASK;
      const float bst = best[low];
      if (bst == FLT_MAX) { Mn[s] = FLT_MAX; continue; }
      const float2 t = tcur[(pp ^ 1) * C::NLOW + low], c = table[s];
      const float e0 = t.x - c.x, e1 = t.y - c.y;
      Mn[s] = bst + e0 * e0 + e1 * e1;
    }
    pp ^= 1;
    __syncthreads();
  }
  if (tid == 0) {                                                     // the best end state (the first one on a tie), then back
    const float* Mf = M + pp * C::S; uint32_t s = 0; float bm = FLT_MAX;
    for (int i = 0; i < C::S; ++i) if (Mf[i] < bm) { bm = Mf[i]; s = i; }
    for (int j = STEPS - 1; j >= 0; --j) {
      out_sym[(size_t)b * STEPS + j] = (uint16_t)(s >> (L - K));
      const uint32_t low = s & C::LOWMASK;
      s = (low << K) | bp[(size_t)j * C::NLOW + low];
    }
  }
}

template <int K, int TAPS>
int launch(const float* z, const float* feedback, const float2* table, uint16_t* sym, uint8_t* bp, int count, cudaStream_t st) {
  const size_t sm = Cfg<K, TAPS>::SMEM;
  cudaFuncSetAttribute(search_kernel<K, TAPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  search_kernel<K, TAPS><<<count, 1024, sm, st>>>(z, feedback, table, sym, bp);
  return (int)cudaGetLastError();
}
} // namespace

extern "C" {

// z: [count][256] fp32 targets (already divided by the row scale).  feedback: [128][taps][4] fp32, or NULL (then taps must be 0).
// table: [4096][2] fp32.  symbols: [count][128] uint16.  scratch: yskn_trellis_scratch_bytes(K, count) bytes.  All device pointers.
// Returns 0, a CUDA error code, or -1 for a combination of K and taps that is not compiled in.
YSKN_TRELLIS_API int yskn_trellis_search(int K, int taps, const float* z, const float* feedback, const float* table,
                                         uint16_t* symbols, uint8_t* scratch, int count, void* cuda_stream) {
  cudaStream_t st = (cudaStream_t)cuda_stream; const float2* t2 = reinterpret_cast<const float2*>(table);
  if (taps == 0) feedback = nullptr;
  if (K == 6 && taps == 0) return launch<6, 0>(z, feedback, t2, symbols, scratch, count, st);
  if (K == 8 && taps == 0) return launch<8, 0>(z, feedback, t2, symbols, scratch, count, st);
  if (K == 6 && taps == 8) return launch<6, 8>(z, feedback, t2, symbols, scratch, count, st);
  if (K == 8 && taps == 8) return launch<8, 8>(z, feedback, t2, symbols, scratch, count, st);
  return -1;
}

YSKN_TRELLIS_API size_t yskn_trellis_scratch_bytes(int K, int count) {
  return (size_t)count * (size_t)STEPS * (size_t)(1u << (L - K));
}

} // extern "C"
