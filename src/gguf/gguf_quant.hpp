// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// GGUF -> engine tensor bridge: the host-side *encoder*.
//
// The engine consumes a quantised stream (see docs/GGUF_DESIGN.md §2 and
// §5.4): a 4096-entry pair codebook, a per-row scale, and a bit-packed trellis
// stream in layout:"warp". This header produces exactly that representation
// from plain float32 weights, so a GGUF file can be turned into the in-memory
// tensors the GEMV kernels already know how to run (src/loader.cpp,
// src/v2_decode.cuh, src/repack.cu).
//
// Pipeline (docs/GGUF_DESIGN.md §5.4 "D. Per-matrix pipeline"):
//
//   W  --rotate_matrix-->  A = diag(so)·(1/√512)H·W·(1/√512)H·diag(si)
//   A  --per-row scale-->  B = A / tau,  tau[r] = max_k |A[r][k]| / 127
//   B  --train_codebook->  q[s] = (q0,q1) in [-127,127]^2, max|q| == 127
//   B  --viterbi-------->  128·bits bits per 256-column sequence
//      --interleave----->  layout:"warp" stream
//      --lut16/scale16-->  f16 codebook + f16 per-row scale
//
// Nothing here includes CUDA: it is pure host code and the test binary runs
// without a GPU. The GPU counterparts of train_codebook() and
// viterbi_quantize() are the only two hot spots that matter at 27B scale; the
// call sites are marked with "GPU:" comments.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace yskn::gguf {

// ---------------------------------------------------------------------------
// Format constants. These mirror the decoder, not the file format:
//   kHadamardBlock  src/a8_prologue.cuh:41-51   (fwht512_blocks)
//   kLutBits        src/v2_decode.cuh:7         (L_BITS = 12)
//   kSeqCols        src/v2_decode.cuh:88-95     (32*K bits of ring per group)
//   kStepsPerSeq    docs/GGUF_DESIGN.md §2.2    (256 cols / d=2)
//   kTileRows       src/loader.cpp:39           (64-row tiles)
// ---------------------------------------------------------------------------
constexpr int kHadamardBlock = 512;
// 1/sqrt(512), the normalisation fwht512_blocks applies (src/a8_prologue.cuh:49).
constexpr float kInvSqrt512 = 0.044194173824159216f;
constexpr int kLutBits = 12;
constexpr int kLutSize = 1 << kLutBits;  // 4096 codebook entries
constexpr int kSeqCols = 256;            // columns per trellis sequence
constexpr int kStepsPerSeq = kSeqCols / 2;  // 128 steps (d = 2)
constexpr int kTileRows = 64;            // rows per fragment tile

// ---------------------------------------------------------------------------
// Encoder knobs.
//
// bits is bits *per trellis step*, so bits/weight = bits/2 (§2.2): bits=6 is
// the 3-bit format, bits=8 the 4-bit one. Only 4, 6 and 8 are accepted because
// lnord_build instantiates exactly those (src/lnord.cu:82-86) and Engine35
// always routes through LNORD (src/engine35.cpp:161-164).
//
// use_rot_in / use_rot_out switch the whole input / output side of the
// rotation (signs *and* Hadamard) on or off. The head uses no rot_out
// (§4.3), and a plain "none" rotation is the identity.
// ---------------------------------------------------------------------------
struct QuantConfig {
  int bits = 6;
  bool use_rot_in = true;
  bool use_rot_out = true;
  uint64_t sign_seed = 7;
  std::string sign_rule = "mulberry32_7919_13";
  // Lloyd rounds over the trellis-reachable states after the initial k-means.
  // 0 keeps the plain k-means codebook (faster, noticeably lossier).
  int refine_iterations = 2;
};

// ---------------------------------------------------------------------------
// Hadamard.
// ---------------------------------------------------------------------------

// In-place, unnormalised fast Walsh-Hadamard transform of one 512-element
// block in natural (Sylvester) order: x <- H x with H[i][j] = (-1)^popcount(i&j).
// H = H^T and H^2 = 512*I, so (1/sqrt(512))H is its own inverse. The same
// transform as fwht512_blocks (src/a8_prologue.cuh:41-51), minus the trailing
// 1/sqrt(512) scale.
void hadamard512(float* x);

// ---------------------------------------------------------------------------
// Step 3: rotation.
//
//   A = diag(signs_out) · (1/sqrt(512))H · W · (1/sqrt(512))H · diag(signs_in)
//
// with W row-major [rows][cols] and A the same shape. This is the inverse of
// the transform the reader applies (§4.4): the output Hadamard runs over each
// 512-*row* block, the input Hadamard over each 512-*column* block.
//
// Signs come from sign_stream_rule() (src/signs.hpp), not from a copy:
//   signs_in  = sign_stream_rule(rule, seed, offset = 0,    cols)
//   signs_out = sign_stream_rule(rule, seed, offset = cols, rows)
// which is the reader's default offset policy (§4.1).
//
// RAGGED TAIL: the reader only transforms whole 512-blocks (fwht512_blocks is
// called with m = cols/512 and m_out = rows/512, both integer divisions), while
// the signs cover every element (X.n = cols / rows). So a trailing partial
// block is left untransformed -- identity -- and still gets its signs. A matrix
// whose rows or cols are below 512 therefore gets signs only on that side.
//
// Throws std::invalid_argument for bits outside {4,6,8} or a bad sign_rule
// (the rule is validated by sign_stream_rule itself).
void rotate_matrix(const float* W, int rows, int cols, const QuantConfig& cfg, float* A);

// ---------------------------------------------------------------------------
// Step 5: codebook.
//
// 4096 entries, each a pair of int8 in [-127,127], plus the scale that maps
// the integer pair back to the float domain it was trained in
// (centroid ~ q * gamma). max|q| == 127 is enforced by rescaling gamma at the
// end; that invariant is what makes build_lut8_tau exactly invertible (§3.2):
// the reader recovers gamma = max|lut16|/127 and q = nearbyint(lut16/gamma),
// and gets back the very integers the Viterbi searched against.
// ---------------------------------------------------------------------------
struct Codebook {
  std::vector<int8_t> q0, q1;  // kLutSize entries each
  float gamma = 0.f;           // centroid scale; max|q| == 127 (see above)

  size_t size() const { return q0.size(); }
};

// k-means over the pairs (B[2t], B[2t+1]) into kLutSize entries.
//
// B holds n floats, i.e. n/2 pairs; n is normally rows*cols of the *normalised*
// matrix (|B| <= 127). Deterministic for a given (B, n, seed): the initial
// centroids are drawn with splitmix64 from the sample, Lloyd runs a fixed
// number of iterations, and empty clusters keep their previous centroid.
//
// GPU: this is a 4096-way nearest-neighbour scan over up to 2^18 pairs -- the
// single hottest loop in the converter. A CUDA port would keep the centroids
// in shared memory (4096*2 floats = 32 KiB) and do one block per point batch,
// or use a 64x64 bucket grid to prune candidates on the host.
Codebook train_codebook(const float* B, size_t n, int seed);

// ---------------------------------------------------------------------------
// Trellis-aware refinement of a codebook.
//
// train_codebook() is plain k-means over the pairs and ignores the trellis: in
// the reader a state s' can only be reached from states whose low NB-K bits
// match. A codebook that is optimal in the unconstrained sense is therefore not
// the best one for the paths the encoder can actually take.
//
// refine_codebook() runs one Viterbi pass, moves each centroid to the mean of
// the pairs that actually landed on it, and repeats. 'iterations' of 0 returns
// 'start' unchanged. The max|q| == 127 invariant is re-applied every round.
//
// When 'stream_out' is non-null it receives the trellis stream for the codebook
// that is returned. The loop's last Viterbi pass ran against the previous
// codebook, so one final pass is needed either way; doing it here lets
// quantize_tensor skip its own, which saves a whole Viterbi pass per tensor.
// ---------------------------------------------------------------------------
Codebook refine_codebook(const float* B, int rows, int cols, const Codebook& start, int bits,
                         int iterations = 2, std::vector<uint32_t>* stream_out = nullptr);

// ---------------------------------------------------------------------------
// Step 6+7: Viterbi and bit packing.
//
// For every row and every 256-column sequence: 128 steps over the 4096-state
// trellis, transition
//
//   state' = (state >> bits) | (b << (12 - bits))        (b in [0, 2^bits))
//
// i.e. the 12-bit window slides up by 'bits' and the new bits enter at the top,
// which is exactly the window v2_state() reads (§2.2, src/v2_decode.cuh:13-19).
// Step j costs (B[2j]-q0[state'])^2 + (B[2j+1]-q1[state'])^2; step 0 starts from
// state 0, which reproduces the zero-padded prefix for free.
//
// 'words' is the *logical* [row][seq][word] stream (wps = 4*bits words per
// sequence), each step's bits packed LSB-first at bit offset bits*j. Run
// interleave_warp() to get the on-disk layout.
//
// The transition min is evaluated per predecessor *group* rather than per
// state: every state s has the same 2^bits predecessors, namely
// (s & lowmask) << bits | t, so one min-reduction of the whole 4096-entry dp
// array serves all states. That keeps the step at O(4096) regardless of bits.
//
// GPU: 4096 states x 128 steps per sequence, one sequence per warp, dp and the
// argmin table in shared memory -- this is the other hot loop.
void viterbi_quantize(const float* B, int rows, int cols, const Codebook& cb, int bits,
                      std::vector<uint32_t>& words);

// ---------------------------------------------------------------------------
// Step 7b: layout:"warp".
//
// Writes the on-disk stream: frag[src(i)] = words[i], with src() from
// src/repack.cu:13-14 (tile = (seq*rows+row)/32, lane = ...%32,
// src = ((tile*(wps/4) + word/4)*32 + lane)*4 + word%4). This is the array that
// belongs at stream_off; fed to yskn_repack_warp_to_frag it produces the mma
// fragment layout the kernels read. src() is a bijection, so the inverse
// permutation is words[i] = frag[src(i)].
//
// Requires rows % 64 == 0 and wps % 4 == 0 (k_repack returns -1 otherwise).
// ---------------------------------------------------------------------------
void interleave_warp(const std::vector<uint32_t>& words, int rows, int seqs, int wps,
                     std::vector<uint32_t>& frag);

// ---------------------------------------------------------------------------
// The finished tensor, in the exact shape the engine uploads.
//
//   lut16    2*kLutSize f16: the f16 codebook (lut16[2s], lut16[2s+1])
//   scale16  physical_rows f16: the per-row scale, tau[r] / gamma
//   stream   physical_rows*seqs*wps words in layout:"warp"
//   gamma    the LUT scale the reader will recover from lut16
//
// rows is the real row count, physical_rows the 64-row padded one (the stream
// and scale16 cover the padded rows; the padding rows are zero).
// ---------------------------------------------------------------------------
struct QuantizedTensor {
  int rows = 0;
  int cols = 0;
  int physical_rows = 0;
  int bits = 0;
  int seqs = 0;
  int wps = 0;
  std::vector<uint16_t> lut16;
  std::vector<uint16_t> scale16;
  std::vector<uint32_t> stream;
  float gamma = 0.f;
};

// Steps 3..8 chained. W is row-major [rows][cols] with cols % 256 == 0; rows is
// padded to a multiple of 64 with zeros. 'shared' reuses an already trained
// codebook (pass nullptr to train one on this matrix's own B).
//
// Throws std::invalid_argument for an unsupported bits value, cols % 256 != 0,
// a non-positive shape, or a shared codebook of the wrong size.
QuantizedTensor quantize_tensor(const float* W, int rows, int cols, const QuantConfig& cfg,
                                const Codebook* shared = nullptr);

// ---------------------------------------------------------------------------
// Step 9 / §5.5: the oracle test.
//
// Decodes 'stream' back with the same 12-bit window arithmetic as
// src/v2_decode.cuh:13-19, recovers the int8 codebook from lut16 exactly the
// way build_lut8_tau does (src/loader.cpp:19-25), and compares the decoded
// values against B -- which must be the *normalised* matrix (A/tau, |B| <= 127)
// for the real rows, rows*cols floats.
//
// max_abs_err is the worst |decoded - B| in B units; rel_err divides it by
// max|B|. Returns false when the tensor is structurally unusable (sizes do not
// match, degenerate LUT); the metrics are then set to +inf.
bool verify_roundtrip(const QuantizedTensor& t, const float* B, float* max_abs_err, float* rel_err);

// Same oracle, but also reports the RMS error over every decoded value.
//
// max|err| is dominated by a handful of outlier pairs and grows with the amount
// of training data (a codebook trained on more data generalises instead of
// overfitting, which raises the worst case while lowering the typical one). RMS
// is the metric that tracks reconstruction quality, so tests and reports should
// prefer it.
bool verify_roundtrip_rms(const QuantizedTensor& t, const float* B, float* rms_err, float* max_abs_err);

// ---------------------------------------------------------------------------
// IEEE binary16 helpers (host side, no cuda_fp16.h). Round-to-nearest-even,
// subnormals and inf/nan handled, so lut16/scale16 match what __float2half_rn
// would produce on the device.
// ---------------------------------------------------------------------------
uint16_t f32_to_f16(float v);
float f16_to_f32(uint16_t h);

}  // namespace yskn::gguf
