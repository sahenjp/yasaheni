// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <stdint.h>
#include <stddef.h>
#if defined(_WIN32)
#  if defined(YSKN_BUILD)
#    define YSKN_API __declspec(dllexport)
#  else
#    define YSKN_API __declspec(dllimport)
#  endif
#else
#  define YSKN_API __attribute__((visibility("default")))
#endif
#define YSKNCUDA_API
#ifdef __cplusplus
extern "C" {
#endif
YSKNCUDA_API int yskn_repack_warp_to_frag(const uint32_t* d_words, uint32_t* d_frag, int rows, int seqs, int wps, void* cuda_stream);
YSKNCUDA_API int yskn_repack_warp_to_frag_host(const uint32_t* h_words, uint32_t* h_frag, int rows, int seqs, int wps);
typedef struct yskn_engine yskn_engine;
struct yskn_park;
 
YSKN_API yskn_engine* yskn_open(const char* yskn_path, int max_ctx, int device, int flags);
YSKN_API void          yskn_close(yskn_engine* e);
// Inspect a .gguf file without loading it. Writes a JSON report into out_json
// (NUL terminated) and returns 0 when the file was parsed as GGUF, non-zero
// otherwise -- the report is still written, with "error" filled in, so a caller
// can always explain the failure. Needs out_cap >= strlen(report) + 1.
YSKN_API int           yskn_gguf_probe(const char* path, char* out_json, int out_cap);
YSKN_API int           yskn_reset(yskn_engine* e);                                        
YSKN_API int           yskn_prefill(yskn_engine* e, const int32_t* ids, int n, int32_t* first_pred);  


YSKN_API int           yskn_set_visual(yskn_engine* e, const uint16_t* features, const int32_t* slots, int n_image,
                                       const int32_t* positions, int n_prompt, int delta);
YSKN_API int           yskn_decode(yskn_engine* e, int32_t* out_ids, int steps);           
YSKN_API int           yskn_nll(yskn_engine* e, const int32_t* ids, int n, float* out_nll);  
YSKN_API int           yskn_logits(yskn_engine* e, float* out_vocab);                      
YSKN_API int           yskn_position(yskn_engine* e);
YSKN_API const char*   yskn_last_error(void);



YSKN_API int           yskn_verify(yskn_engine* e, const int32_t* ids, int M, int use_graph, int32_t* out_argmax, int32_t* top_ids, float* top_vals);
YSKN_API int           yskn_verify_nll(yskn_engine* e, const int32_t* ids, const int32_t* targets, int M, int use_graph, int32_t* out_argmax, float* out_nll);
YSKN_API int           yskn_commit(yskn_engine* e, int n_accept);
YSKN_API int           yskn_prefill_rows(yskn_engine* e, const int32_t* ids, int n, int32_t* first_pred, int32_t* out_argmax, int32_t* top_ids, float* top_vals);
YSKN_API int           yskn_hidden_rows(yskn_engine* e, int n, float* out);               
YSKN_API int           yskn_logits_rows(yskn_engine* e, int row0, int n, float* out);     
 
YSKN_API int           yskn_set_capture(yskn_engine* e, const int32_t* layers, int n);
YSKN_API int           yskn_capture_rows(yskn_engine* e, int k, int n, float* out);




 
YSKN_API int           yskn_load_draft(yskn_engine* e, const char* yskn_path, const char* head_ids_path);
YSKN_API int           yskn_draft_round(yskn_engine* e, int mode, int a, int steps, int use_graph);    
YSKN_API int           yskn_verify_device(yskn_engine* e, int M, int use_graph, int32_t* out_argmax, int32_t* out_toks, int32_t* top_ids);
YSKN_API int           yskn_set_verify_tokens(yskn_engine* e, const int32_t* ids, int first_row, int n);
YSKN_API int           yskn_draft_catchup(yskn_engine* e, int p0, int n, const int32_t* toks, int use_graph);
YSKN_API int           yskn_spec_generate(yskn_engine* e, const int32_t* prompt, int n_prompt, int n_gen, const int32_t* cfg, int n_cfg, int32_t* out_ids, double* stats, double* times_ms);




YSKN_API int           yskn_specx_load_static(const char* path);


YSKN_API int           yskn_set_opt(yskn_engine* e, int key, int value);


YSKN_API int           yskn_lfrow_table(yskn_engine* e, const uint8_t* tab48);
YSKN_API int           yskn_lfrow_promote(yskn_engine* e, int src, int dst);

YSKN_API int           yskn_cancel(yskn_engine* e);
YSKN_API int           yskn_snap_info(yskn_engine* e, int32_t* out4, int32_t* ids, int max_ids);
YSKN_API int           yskn_state_hash(yskn_engine* e, uint64_t* out, int max_out, int* n_out);

// Parked context handles. A handle keeps the committed durable qwen3_5 state in pinned host memory: KV rows [0,pos), the
// linear S and hist arrays, d_toks_[0] and the token ids. The bdraft KV rings and the HCAP capture buffer are not part of the
// handle: resume clears them, so target tokens are still verified but the draft path starts from an empty ring. park refuses
// MTP drafter state ("park: MTP drafter state is not parkable yet"), visual state ("park: visual state is not parkable") and
// a state that is mid-verify or pre-committed ("park: state is not committed").
// Handles are reusable and resume does not consume them. yskn_park_info writes out4 =
// {position, token_id_count, reserved (0), last_token}; bytes receives the pinned byte count of the handle (the same value
// yskn_park_size reports for that position).
// yskn_park_free releases the handle: call it exactly once per handle, never use the pointer afterwards. The engine keeps at
// most one released block (the largest seen) so the next park does not pay cudaMallocHost again; YSKN_PARK_REUSE=0 frees every
// released block instead.
YSKN_API int           yskn_park(yskn_engine* e, const int32_t* ids, int n, struct yskn_park** out);
YSKN_API int           yskn_park_size(yskn_engine* e, int pos, size_t* bytes);
YSKN_API int           yskn_resume(yskn_engine* e, const struct yskn_park* h);
YSKN_API int           yskn_park_info(const struct yskn_park* h, int32_t* out4, size_t* bytes);
YSKN_API void          yskn_park_free(struct yskn_park* h);



#ifdef __cplusplus
}
#endif
