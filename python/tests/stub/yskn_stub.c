/* SPDX-License-Identifier: Apache-2.0 */
/* A CPU-only stand-in for the engine library, used by the Python tests.  It is NOT the engine: it follows scripted outputs and
   mimics the call conventions (output ids written pass by pass, cancel at a pass boundary, thread-local last error, prompt reuse
   reporting) so that the Python layer can be tested without a GPU.  Build: cc -shared -fPIC -O1 -pthread yskn_stub.c -o libyskn.so */
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define EXPORT __attribute__((visibility("default")))
#define EOS_ID 248046

typedef struct { int32_t *ids; int n; int loop_from; } script_t;
typedef struct yskn_engine {
    int max_ctx, pos, draft, tok;
    atomic_int cancel;
    int32_t *hist; int nhist;
    int32_t *snap; int nsnap;
    int opts[64];
} yskn_engine;

static __thread char g_err[256];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static script_t g_queue[64];
static int g_qhead = 0, g_qtail = 0;
static int g_delay_us = 0;
static int g_calls = 0;
static int g_last_n_prompt = 0;
static int32_t g_last_cfg[44];
static int g_last_n_cfg = 0;

static void set_err(const char *m) { strncpy(g_err, m, sizeof g_err - 1); g_err[sizeof g_err - 1] = 0; }
static void sleep_us(int us) { if (us > 0) { struct timespec t = { us / 1000000, (us % 1000000) * 1000L }; nanosleep(&t, 0); } }

EXPORT void stub_push_script(const int32_t *ids, int n, int loop_from) {
    pthread_mutex_lock(&g_mu);
    script_t s; s.ids = malloc(sizeof(int32_t) * (n ? n : 1)); memcpy(s.ids, ids, sizeof(int32_t) * n); s.n = n; s.loop_from = loop_from;
    g_queue[g_qtail++ % 64] = s;
    pthread_mutex_unlock(&g_mu);
}
EXPORT void stub_set_delay_us(int us) { g_delay_us = us; }
EXPORT int stub_get_opt(yskn_engine *e, int key) { return e->opts[key & 63]; }
EXPORT int stub_calls(void) { return g_calls; }
EXPORT int stub_last_n_prompt(void) { return g_last_n_prompt; }
EXPORT int stub_nhist(yskn_engine *e) { return e->nhist; }
EXPORT int stub_last_cfg(int32_t *out) { memcpy(out, g_last_cfg, sizeof g_last_cfg); return g_last_n_cfg; }

EXPORT yskn_engine *yskn_open(const char *path, int max_ctx, int device, int flags) {
    (void)device; (void)flags;
    if (strstr(path, "fail")) { set_err("cudaMalloc failed: out of memory"); return 0; }
    yskn_engine *e = calloc(1, sizeof *e);
    e->max_ctx = max_ctx;
    return e;
}
EXPORT void yskn_close(yskn_engine *e) { if (e) { free(e->hist); free(e->snap); free(e); } }
EXPORT int yskn_reset(yskn_engine *e) { free(e->hist); e->hist = 0; e->nhist = 0; e->pos = 0; free(e->snap); e->snap = 0; e->nsnap = 0; return 0; }
EXPORT int yskn_position(yskn_engine *e) { return e->pos; }
EXPORT const char *yskn_last_error(void) { return g_err; }
EXPORT int yskn_cancel(yskn_engine *e) { atomic_store(&e->cancel, 1); return 0; }
EXPORT int yskn_set_opt(yskn_engine *e, int key, int value) { e->opts[key & 63] = value; return 0; }
EXPORT int yskn_load_bdraft(yskn_engine *e, const char *path) {
    if (strstr(path, "bad")) { set_err("bdraft: bad container"); return 1; }
    e->draft = 1; return 0;
}
EXPORT int yskn_prefill(yskn_engine *e, const int32_t *ids, int n, int32_t *first) { (void)ids; e->pos = n; *first = 1000; return 0; }
EXPORT int yskn_decode(yskn_engine *e, int32_t *out, int steps) { for (int i = 0; i < steps; i++) out[i] = 1000 + i; e->pos += steps; return 0; }
EXPORT int yskn_logits(yskn_engine *e, float *out) { (void)e; (void)out; return 0; }
EXPORT int yskn_nll(yskn_engine *e, const int32_t *ids, int n, float *out) { (void)e; (void)ids; for (int i = 0; i + 1 < n; i++) out[i] = 1.0f; return 0; }
EXPORT int yskn_state_hash(yskn_engine *e, uint64_t *out, int max_out, int *n_out) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < e->nhist; i++) { h ^= (uint64_t)(uint32_t)e->hist[i]; h *= 1099511628211ULL; }
    if (max_out < 2) { set_err("state_hash: buffer too small"); return 1; }
    out[0] = (uint64_t)e->pos; out[1] = h; *n_out = 2; return 0;
}
EXPORT int yskn_snap_info(yskn_engine *e, int32_t *out4, int32_t *ids, int max_ids) {
    g_err[0] = 0;
    out4[0] = e->pos; out4[1] = -1; out4[2] = e->nhist > 0 ? 1 : 0; out4[3] = e->nhist;
    if (ids) for (int i = 0; i < e->nhist && i < max_ids; i++) ids[i] = e->hist[i];
    return 0;
}

#define PARK_MAGIC 0x59534b4e5041524bULL   /* "YSKNPARK" */
struct yskn_park {
    uint64_t magic, fp;
    size_t bytes;
    int pos, tok, nids, nhist;
    int32_t *ids, *hist;
};

EXPORT int yskn_park_size(yskn_engine *e, int pos, size_t *bytes) {
    (void)e;
    if (pos < 0) { set_err("park: position out of range"); return 1; }
    *bytes = sizeof(struct yskn_park) + (size_t)pos * 8 + 4;
    return 0;
}
EXPORT int yskn_park(yskn_engine *e, const int32_t *ids, int n, struct yskn_park **out) {
    g_err[0] = 0;
    if (!e || e->pos < 1) { set_err("park: nothing to park"); return 1; }
    const int32_t *src; int cnt;
    if (ids) {
        if (n != e->pos) { set_err("park: ids do not cover the current position"); return 1; }
        for (int i = 0; i < n && i < e->nhist; i++)
            if (ids[i] != e->hist[i]) { set_err("park: ids do not cover the current position"); return 1; }
        src = ids; cnt = n;
    } else {
        if (n != 0 || e->nhist < 1) { set_err("park: nothing to park"); return 1; }
        src = e->hist; cnt = e->nhist;
    }
    struct yskn_park *h = calloc(1, sizeof *h);
    h->ids = malloc(sizeof(int32_t) * (cnt > 0 ? cnt : 1));
    h->hist = malloc(sizeof(int32_t) * (cnt > 0 ? cnt : 1));
    memcpy(h->ids, src, sizeof(int32_t) * cnt);
    memcpy(h->hist, src, sizeof(int32_t) * cnt);
    h->pos = cnt; h->nids = cnt; h->nhist = cnt;
    h->tok = e->tok;
    h->bytes = sizeof(struct yskn_park) + (size_t)cnt * 8 + 4;
    h->fp = (uint64_t)(uint32_t)e->max_ctx;
    h->magic = PARK_MAGIC;
    *out = h;
    return 0;
}
EXPORT int yskn_resume(yskn_engine *e, const struct yskn_park *h) {
    g_err[0] = 0;
    if (!h || h->magic != PARK_MAGIC) { set_err("resume: handle is not live"); return 1; }
    if (h->fp != (uint64_t)(uint32_t)e->max_ctx || h->pos < 1 || h->pos > e->max_ctx) {
        set_err("resume: handle does not match this engine"); return 1;
    }
    free(e->hist);
    e->hist = malloc(sizeof(int32_t) * (h->nhist > 0 ? h->nhist : 1));
    memcpy(e->hist, h->hist, sizeof(int32_t) * h->nhist);
    e->nhist = h->nhist; e->pos = h->pos; e->tok = h->tok;
    return 0;
}
EXPORT int yskn_park_info(const struct yskn_park *h, int32_t *out4, size_t *bytes) {
    g_err[0] = 0;
    if (!h || h->magic != PARK_MAGIC) { set_err("park: handle is not live"); return 1; }
    if (out4) { out4[0] = h->pos; out4[1] = h->pos; out4[2] = 0; out4[3] = h->tok; }
    if (bytes) *bytes = h->bytes;
    return 0;
}
EXPORT void yskn_park_free(struct yskn_park *h) {
    if (!h || h->magic != PARK_MAGIC) return;
    h->magic = 0;
    free(h->ids); free(h->hist); free(h);
}


static int lcp(const int32_t *a, int na, const int32_t *b, int nb) { int i = 0; while (i < na && i < nb && a[i] == b[i]) i++; return i; }

EXPORT int yskn_spec_generate(yskn_engine *e, const int32_t *prompt, int n_prompt, int n_gen, const int32_t *cfg, int n_cfg,
                              int32_t *out, double *stats, double *times_ms) {
    g_err[0] = 0;
    atomic_store(&e->cancel, 0);
    g_calls++; g_last_n_prompt = n_prompt;
    if (n_cfg != 41 && n_cfg != 44) { set_err("spec_generate: bad config length"); return 1; }
    memset(g_last_cfg, 0, sizeof g_last_cfg); memcpy(g_last_cfg, cfg, sizeof(int32_t) * n_cfg); g_last_n_cfg = n_cfg;
    if (n_prompt < 1 || n_prompt + n_gen + 1 > e->max_ctx) { set_err("spec_generate: context overflow"); return 1; }
    if (prompt[0] == 7) { set_err("stub: injected failure"); return 1; }
    int eos1 = cfg[20], eos2 = cfg[21];
    int snap_on = n_cfg >= 44 ? cfg[42] : 0, snap_tok = n_cfg >= 44 ? cfg[43] : -1;

    /* prompt reuse: live state, snapshot, or fresh */
    int mode = 0, cached = 0;
    int l = lcp(e->hist, e->nhist, prompt, n_prompt);
    if (snap_on && e->nhist > 0 && l == e->nhist) { mode = 1; cached = l; }
    else if (snap_on && e->nsnap > 0 && lcp(e->snap, e->nsnap, prompt, n_prompt) == e->nsnap) { mode = 2; cached = e->nsnap; }

    /* the scripted output of this call */
    script_t sc;
    pthread_mutex_lock(&g_mu);
    if (g_qhead != g_qtail) sc = g_queue[g_qhead++ % 64];
    else { sc.n = 40; sc.ids = malloc(sizeof(int32_t) * 40); for (int i = 0; i < 40; i++) sc.ids[i] = 1000 + i; sc.loop_from = -1; }
    pthread_mutex_unlock(&g_mu);
    int k = 0;
    #define NEXT() (k < sc.n ? sc.ids[k++] : (sc.loop_from >= 0 && sc.n > 0 ? (k = sc.loop_from, sc.ids[k++]) : EOS_ID))

    int idx = 0, passes = 0, hit = 0;
    static const int pat[4] = { 1, 3, 2, 4 };
    out[idx++] = NEXT();                                  /* the prompt pass predicts the first id */
    hit = (out[0] == eos1 || out[0] == eos2);
    sleep_us(g_delay_us);
    while (!hit && idx < n_gen + 1 && !atomic_load(&e->cancel)) {
        int cnt = pat[passes % 4];
        for (int j = 0; j < cnt && idx < n_gen + 1; j++) {
            int32_t t = NEXT();
            out[idx++] = t;
            if (t == eos1 || t == eos2) { hit = 1; break; }
        }
        passes++;
        sleep_us(g_delay_us);
    }
    free(sc.ids);

    memset(stats, 0, sizeof(double) * 48);
    stats[0] = passes + 1; stats[1] = idx; stats[30] = idx - 1; stats[31] = hit; stats[41] = mode; stats[42] = cached;
    times_ms[0] = 0.01 * (n_prompt - cached); times_ms[1] = 0.1 * idx;

    /* the engine's state: prompt + all generated ids except the last one */
    int nh = n_prompt + idx - 1;
    int32_t *h = malloc(sizeof(int32_t) * (nh > 0 ? nh : 1));
    memcpy(h, prompt, sizeof(int32_t) * n_prompt); memcpy(h + n_prompt, out, sizeof(int32_t) * (idx - 1));
    free(e->hist); e->hist = h; e->nhist = nh; e->pos = nh; e->tok = idx > 0 ? out[idx - 1] : 0;
    if (snap_on && snap_tok >= 0) {                       /* snapshot right after the last chat-start marker of the prompt */
        int last = -1; for (int i = 0; i < n_prompt; i++) if (prompt[i] == snap_tok) last = i;
        if (last >= 0) { free(e->snap); e->nsnap = last + 1; e->snap = malloc(sizeof(int32_t) * e->nsnap); memcpy(e->snap, prompt, sizeof(int32_t) * e->nsnap); }
    }
    return 0;
}
