# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""ctypes bindings to the multimodal library of llama.cpp (``mtmd.dll``), checked against the b10488 release for Windows.

The projector (mmproj) turns an image into features of 5120 values per image token.  The library also needs the text model of
llama.cpp, for its settings and vocabulary only: the model is opened with ``no_alloc``, so its weights are never read and any
Qwen3.8-27B GGUF works.  The images are encoded on the CPU: on the GPU the features differ slightly, and in the image test of the
contributor of this code (pull request #4) the score fell (87.8 on the CPU against 79.6 and 81.6 in two GPU runs).

Only the CPU build ``llama-b10488-bin-win-cpu-x64.zip`` is accepted: the structures below are declared for exactly this build, so
its libraries are compared by SHA-256 before any of them is loaded.
"""
from __future__ import annotations

import atexit
import ctypes as C
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

P, I, S, B = C.c_void_p, C.c_int32, C.c_size_t, C.c_bool

LLAMA_ZIP = "llama-b10488-bin-win-cpu-x64.zip"
# sha256 of the libraries of LLAMA_ZIP (release b10488 of ggml-org/llama.cpp) that are bound here or that they import
B10488_CPU_X64 = {
    "llama.dll": "1050f5ddc24e12ce136cb02fda3eee3bb2aa589275b2559b397dea340a3469f6",
    "mtmd.dll": "349c3b83be8eadb85928f4cc0e87f99fff50cb411115910588d20fb1593de73c",
    "ggml.dll": "aaca2878d4f10da490f12be47cf11d6902a8a1502c3ca68045114a2c71d794e9",
    "ggml-base.dll": "82d02fcaf8c9badda59f3b1ced19dc6c708fe9865662b887d1daede81601545c",
}


class _ModelParams(C.Structure):
    _fields_ = [("devices", P), ("tensor_buft_overrides", P), ("n_gpu_layers", I),
                ("split_mode", I), ("load_mode", I), ("main_gpu", I), ("tensor_split", P),
                ("progress_callback", P), ("progress_callback_user_data", P), ("kv_overrides", P),
                ("vocab_only", B), ("check_tensors", B), ("use_extra_bufts", B),
                ("no_host", B), ("no_alloc", B), ("load_mtp", B)]


class _ContextParams(C.Structure):
    _fields_ = [("use_gpu", B), ("print_timings", B), ("n_threads", I), ("image_marker", C.c_char_p),
                ("media_marker", C.c_char_p), ("flash_attn_type", I), ("warmup", B),
                ("image_min_tokens", I), ("image_max_tokens", I), ("cb_eval", P),
                ("cb_eval_user_data", P), ("batch_max_tokens", I), ("progress_callback", P),
                ("progress_callback_user_data", P)]


class _InputText(C.Structure):
    _fields_ = [("text", C.c_char_p), ("text_len", S), ("add_special", B), ("parse_special", B)]


class _DecoderPos(C.Structure):
    _fields_ = [("t", C.c_uint32), ("x", C.c_uint32), ("y", C.c_uint32), ("z", C.c_uint32)]


def _bind(lib, name, args, result):
    try:
        fn = getattr(lib, name)
    except AttributeError:
        raise RuntimeError("vision: the llama.cpp library has no function %s; image input needs %s" % (name, LLAMA_ZIP)) from None
    fn.argtypes, fn.restype = args, result
    return fn


def check_llama_dir(llama_dir) -> None:
    """The folder must hold the libraries of LLAMA_ZIP (compared by SHA-256) and no GPU backend of ggml."""
    d = Path(llama_dir)
    if not d.is_dir():
        raise FileNotFoundError("vision: the llama.cpp folder %s does not exist" % d)
    for name, want in B10488_CPU_X64.items():
        p = d / name
        if not p.is_file():
            raise FileNotFoundError("vision: missing %s (unpack %s into this folder)" % (p, LLAMA_ZIP))
        if hashlib.sha256(p.read_bytes()).hexdigest() != want:
            raise RuntimeError("vision: %s is not the file of %s; image input needs exactly this build of llama.cpp" % (p, LLAMA_ZIP))
    gpu = sorted(p.name for p in d.glob("ggml-*.dll") if not p.name.lower().startswith(("ggml-base.", "ggml-cpu", "ggml-rpc.")))
    if gpu:
        raise RuntimeError("vision: %s holds GPU backends of llama.cpp (%s); use a folder with only %s (the projector runs on the CPU)"
                           % (d, ", ".join(gpu), LLAMA_ZIP))


def _load_libraries(llama_dir: Path):
    """llama.dll and mtmd.dll of the folder, with the CPU backend of ggml loaded.  Returns (llama, mtmd, add_dll_directory handle).
    The folder must be an absolute path: os.add_dll_directory refuses a relative one (WinError 87)."""
    dll_dir = os.add_dll_directory(str(llama_dir))
    llama = C.CDLL(str(llama_dir / "llama.dll"))
    mtmd = C.CDLL(str(llama_dir / "mtmd.dll"))
    ggml = C.CDLL(str(llama_dir / "ggml.dll"))
    # ggml_backend_load_all_from_path loads the backends with a plain LoadLibrary: without the folder on PATH their
    # dependencies are not found and the backends are skipped without an error.  The backends and everything they import are
    # loaded during this call (none of these libraries delay-loads another), so PATH is restored right after it.
    old = os.environ.get("PATH")
    os.environ["PATH"] = str(llama_dir) + os.pathsep + (old or "")
    try:
        _bind(ggml, "ggml_backend_load_all_from_path", [C.c_char_p], None)(str(llama_dir).encode())
    finally:
        if old is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = old
    return llama, mtmd, dll_dir


# llama.cpp prints every detail of the model and of each encode.  Only its warnings and errors are passed on to stderr.
_LOG_FN = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)
_log_level = [0]
_log_libs = []


def _log(level, text, _user):
    try:
        if level != 5:                                       # GGML_LOG_LEVEL_CONT continues the line before
            _log_level[0] = level
        if _log_level[0] >= 3 and text:                      # GGML_LOG_LEVEL_WARN, GGML_LOG_LEVEL_ERROR
            sys.stderr.write(text.decode("utf-8", "replace"))
    except Exception:                                        # noqa: BLE001 - never raise into the library
        pass


_LOG_CALLBACK = _LOG_FN(_log)


def _set_logs(llama, mtmd, quiet: bool) -> None:
    cb = C.cast(_LOG_CALLBACK, P) if quiet else None         # passed as a plain pointer: NULL = the library's own logger again
    _bind(llama, "llama_log_set", [P, P], None)(cb, None)
    _bind(mtmd, "mtmd_log_set", [P, P], None)(cb, None)
    if quiet:
        if not _log_libs:
            atexit.register(_restore_logs)                   # before Python shuts down: no callback into a finished interpreter
        _log_libs[:] = [(llama, mtmd)]


def _restore_logs() -> None:
    for llama, mtmd in _log_libs:
        try:
            _set_logs(llama, mtmd, False)
        except Exception:                                    # noqa: BLE001
            pass


def to_bfloat16(values) -> np.ndarray:
    """float32 -> bfloat16 bits, rounded to nearest even."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


class Tokenized:
    """One preprocessed image (the input chunks of mtmd) waiting to be encoded.  ``n`` is its number of image tokens; close() frees it."""

    def __init__(self, encoder, chunks, chunk, n, t0):
        self._encoder, self.chunks, self.chunk, self.n, self.t0 = encoder, chunks, chunk, int(n), t0

    def close(self) -> None:
        if self.chunks:
            self._encoder.f["mtmd_input_chunks_free"](self.chunks)
            self.chunks = self.chunk = None


class MtmdEncoder:
    HIDDEN = 5120

    def __init__(self, llama_dir, mmproj, model, *, threads: int = 8, max_image_tokens=None):
        """``max_image_tokens``: larger images are downsized by llama.cpp to at most this many image tokens (None: the
        projector's own limit, 4,096 for this one)."""
        llama_dir, mmproj, model = (Path(os.path.abspath(os.fspath(p))) for p in (llama_dir, mmproj, model))
        check_llama_dir(llama_dir)
        for p in (mmproj, model):
            if not p.is_file():
                raise FileNotFoundError("vision: missing %s" % p)
        self.ctx = self.model = self.lib = self.llama = self._dll_dir = None
        self._logs_set = False
        try:
            self._open(llama_dir, mmproj, model, int(threads), max_image_tokens)
        except BaseException:
            self.close()
            raise

    def _open(self, llama_dir, mmproj, model, threads, max_image_tokens):
        self.llama, self.lib, self._dll_dir = _load_libraries(llama_dir)
        sig = {
            "mtmd_context_params_default": ([], _ContextParams),
            "mtmd_init_from_file": ([C.c_char_p, P, _ContextParams], P),
            "mtmd_free": ([P], None), "mtmd_default_marker": ([], C.c_char_p),
            "mtmd_bitmap_init": ([C.c_uint32, C.c_uint32, P], P), "mtmd_bitmap_free": ([P], None),
            "mtmd_input_chunks_init": ([], P), "mtmd_input_chunks_free": ([P], None),
            "mtmd_input_chunks_size": ([P], S), "mtmd_input_chunks_get": ([P, S], P),
            "mtmd_input_chunk_get_type": ([P], I), "mtmd_input_chunk_get_n_tokens": ([P], S),
            "mtmd_input_chunk_get_n_pos": ([P], I), "mtmd_input_chunk_get_tokens_image": ([P], P),
            "mtmd_image_tokens_get_decoder_pos": ([P, I, S], _DecoderPos),
            "mtmd_tokenize": ([P, P, C.POINTER(_InputText), C.POINTER(P), S], I),
            "mtmd_encode_chunk": ([P, P], I), "mtmd_get_output_embd": ([P], C.POINTER(C.c_float)),
        }
        self.f = {k: _bind(self.lib, k, *v) for k, v in sig.items()}      # all found before any of them is called
        self._model_free = _bind(self.llama, "llama_model_free", [P], None)
        _set_logs(self.llama, self.lib, True)
        self._logs_set = True
        _bind(self.llama, "llama_backend_init", [], None)()
        mp = _bind(self.llama, "llama_model_default_params", [], _ModelParams)()
        mp.no_alloc, mp.n_gpu_layers, mp.load_mode = True, 0, 0     # settings and vocabulary only; no_alloc cannot memory-map
        self.model = _bind(self.llama, "llama_model_load_from_file", [C.c_char_p, _ModelParams], P)(str(model).encode(), mp)
        if not self.model:
            raise RuntimeError("vision: cannot open the text model %s" % model)
        hidden = _bind(self.llama, "llama_model_n_embd_inp", [P], I)(self.model)
        if hidden != self.HIDDEN:
            raise RuntimeError("vision: the text model has %d input dimensions, expected %d (Qwen3.8-27B)" % (hidden, self.HIDDEN))
        cp = self.f["mtmd_context_params_default"]()
        if (cp.n_threads, cp.image_min_tokens, cp.image_max_tokens, cp.batch_max_tokens) != (4, -1, -1, 1024):
            raise RuntimeError("vision: mtmd.dll is not the expected llama.cpp release (b10488): structure layout differs")
        cp.use_gpu, cp.warmup, cp.print_timings, cp.n_threads = False, False, False, threads
        if max_image_tokens is not None:                     # the projector then shrinks a larger image (keeping its aspect ratio)
            cp.image_max_tokens = int(max_image_tokens)
        self.ctx = self.f["mtmd_init_from_file"](str(mmproj).encode(), self.model, cp)
        if not self.ctx:
            raise RuntimeError("vision: cannot load the projector %s" % mmproj)

    def tokenize(self, image) -> Tokenized:
        """Preprocess an RGB PIL image.  This fixes its number of image tokens before the expensive encode."""
        rgb = np.ascontiguousarray(image.convert("RGB"), dtype=np.uint8)
        f = self.f
        t0 = time.perf_counter()
        bitmap = f["mtmd_bitmap_init"](rgb.shape[1], rgb.shape[0], rgb.ctypes.data)
        if not bitmap:
            raise RuntimeError("vision: mtmd_bitmap_init failed")
        chunks = f["mtmd_input_chunks_init"]()
        try:
            marker = f["mtmd_default_marker"]()
            text = _InputText(marker, len(marker), False, True)
            if f["mtmd_tokenize"](self.ctx, chunks, C.byref(text), (P * 1)(bitmap), 1):
                raise ValueError("vision: image preprocessing failed")
            chunk = [f["mtmd_input_chunks_get"](chunks, i) for i in range(f["mtmd_input_chunks_size"](chunks))]
            chunk = [c for c in chunk if f["mtmd_input_chunk_get_type"](c) == 1]
            if len(chunk) != 1:
                raise RuntimeError("vision: expected one image chunk, got %d" % len(chunk))
            job = Tokenized(self, chunks, chunk[0], f["mtmd_input_chunk_get_n_tokens"](chunk[0]), t0)
            chunks = None                                    # now owned by job
            return job
        finally:
            f["mtmd_bitmap_free"](bitmap)                     # mtmd_tokenize copied the pixels
            if chunks:
                f["mtmd_input_chunks_free"](chunks)

    def encode_tokenized(self, job: Tokenized) -> dict:
        """A preprocessed image -> features (bfloat16 bits, (n, 5120)), positions ([time, row, column], (3, n)), grid (rows, columns)."""
        f = self.f
        chunk, n = job.chunk, job.n
        tokens = f["mtmd_input_chunk_get_tokens_image"](chunk)
        pos = [f["mtmd_image_tokens_get_decoder_pos"](tokens, 0, i) for i in range(n)]
        positions = np.array([(p.t, p.y, p.x) for p in pos], dtype=np.int32).T.copy()   # .y is the row, .x the column
        h, w = int(positions[1].max()) + 1, int(positions[2].max()) + 1
        expected = np.stack([np.zeros(n, np.int32), np.arange(n, dtype=np.int32) // w, np.arange(n, dtype=np.int32) % w])
        if n != h * w or not np.array_equal(positions, expected) or f["mtmd_input_chunk_get_n_pos"](chunk) != max(h, w):
            raise RuntimeError("vision: unexpected image positions from mtmd")
        if f["mtmd_encode_chunk"](self.ctx, chunk):
            raise RuntimeError("vision: mtmd encode failed")
        feats = np.ctypeslib.as_array(f["mtmd_get_output_embd"](self.ctx), shape=(n * self.HIDDEN,)).reshape(n, self.HIDDEN).copy()
        if not np.isfinite(feats).all():
            raise RuntimeError("vision: non-finite image features")
        return dict(features=to_bfloat16(feats), positions=positions, grid=np.array([h, w], dtype=np.int32),
                    seconds=np.array(time.perf_counter() - job.t0))

    def encode(self, image, max_tokens=None) -> dict:
        """RGB PIL image -> features, positions and grid (see encode_tokenized).  ValueError above ``max_tokens`` image tokens."""
        job = self.tokenize(image)
        try:
            if max_tokens is not None and job.n > max_tokens:
                raise ValueError("vision: the image needs %d image tokens, more than %d" % (job.n, max_tokens))
            return self.encode_tokenized(job)
        finally:
            job.close()

    def close(self) -> None:
        if self.ctx:
            self.f["mtmd_free"](self.ctx)
            self.ctx = None
        if self.model:
            self._model_free(self.model)
            self.model = None
        if self._logs_set:
            self._logs_set = False
            try:
                _set_logs(self.llama, self.lib, False)
            except Exception:                                # noqa: BLE001 - closing must not fail on the logger
                pass
        if self._dll_dir is not None:                        # the folder leaves the DLL search path; the loaded libraries stay
            try:
                self._dll_dir.close()
            except OSError:
                pass
            self._dll_dir = None
