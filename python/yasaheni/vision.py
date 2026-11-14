# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Optional still-image input. No Pillow or native vision library is imported until an image is requested."""
from __future__ import annotations

import base64
from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
from urllib.parse import unquote, urlsplit
import warnings

import numpy as np

MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
MAX_IMAGE_BYTES = 24 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_IMAGES = 16                     # per request
MAX_ASPECT = 200                    # longer side / shorter side, as in Qwen's own image processor
# Image tokens.  The encode runs on the CPU and its time grows with the square of the image tokens (on our 16-core CPU: about 15 s
# for 1,024 tokens with 16 threads; 86 s for 2,040 and 5 1/2 minutes for 4,096 with 8 threads).  So llama.cpp is asked to downsize
# every image to at most DEFAULT_IMAGE_TOKENS (--vision-max-image-tokens); a 1920 x 1080 screenshot then takes 1,008 tokens, not 2,040.
DEFAULT_IMAGE_TOKENS = 1024         # per image, default of --vision-max-image-tokens
IMAGE_TOKENS_RANGE = (64, 4096)     # the values --vision-max-image-tokens accepts
MAX_IMAGE_TOKENS = 4096             # per image, hard ceiling (the projector's own limit): an image above it is refused (400)
REQUEST_IMAGE_TOKENS_FACTOR = 8     # per request, default of --vision-max-request-image-tokens: 8 x the per-image limit (8,192)
# Features take 10 KiB per image token (5,120 bfloat16 values): 10 MiB for an image of 1,024 tokens, 40 MiB for 4,096.  A client
# sends the whole conversation again with every turn, so every image of a chat comes back each turn; the memory cache must hold
# them all or each turn encodes them again (64 MiB held only one or two large photos).  512 MiB holds about 50 images at the default
# limit, and all the images of one request at the default per-request limit (80 MiB).  It is filled only as images are seen.
MEMORY_CACHE_BYTES = 512 * 1024 * 1024
LOCAL_FILES_HINT = "--vision-allow-local-files"
PROJECTOR_DIM = 5120                # values per image token that Qwen3.8-27B takes (its hidden size)


def physical_cores() -> int:
    """The number of physical CPU cores: Windows GetLogicalProcessorInformationEx (one record per core), else os.cpu_count() // 2,
    at least 1.  It is the default thread count of the image encoder."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            fn = ctypes.WinDLL("kernel32", use_last_error=True).GetLogicalProcessorInformationEx
            fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
            fn.restype = wintypes.BOOL
            size = wintypes.DWORD(0)
            fn(0, None, ctypes.byref(size))                  # RelationProcessorCore = 0; the first call only gives the size
            if size.value:
                buf = ctypes.create_string_buffer(size.value)
                if fn(0, buf, ctypes.byref(size)):
                    raw, n, off = buf.raw[:size.value], 0, 0
                    while off + 8 <= len(raw):               # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship, Size, ...
                        relation, step = struct.unpack_from("<II", raw, off)
                        if step < 8:
                            break
                        n += relation == 0
                        off += step
                    if n:
                        return n
        except (OSError, AttributeError, ValueError, struct.error):
            pass
    return max(1, (os.cpu_count() or 2) // 2)


def error_text(ex) -> str:
    """The message of an image input error without the module prefix "vision: " (the callers say "image input: ...")."""
    text = str(ex) or type(ex).__name__
    return text[len("vision: "):] if text.startswith("vision: ") else text


def _absolute(path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_keys(path, max_kv: int = 64) -> dict:
    """The scalar and string values at the start of a GGUF header (``general.architecture`` and the like).  Arrays of numbers and
    short lists of strings (``general.tags``) are skipped; the scan stops at a long list of strings (the vocabulary), after
    ``max_kv`` keys, or where the file ends, so it reads a few KB at most.  ValueError if the file does not start with the GGUF
    magic."""
    out = {}
    with open(path, "rb") as f:
        head = f.read(24)
        if head[:4] != b"GGUF":
            raise ValueError("not a GGUF file")
        if len(head) < 24 or struct.unpack_from("<I", head, 4)[0] < 2:       # cut short, or version 1 (32-bit lengths)
            return out

        def read(n):
            b = f.read(n)
            if len(b) != n:
                raise EOFError
            return b

        def string():
            (n,) = struct.unpack("<Q", read(8))
            if n > 1 << 20:
                raise EOFError
            return read(n).decode("utf-8", "replace")
        try:
            for _ in range(min(struct.unpack_from("<Q", head, 16)[0], max_kv)):
                key = string()
                (kind,) = struct.unpack("<I", read(4))
                if kind in _GGUF_SCALAR:
                    fmt = _GGUF_SCALAR[kind]
                    (out[key],) = struct.unpack(fmt, read(struct.calcsize(fmt)))
                elif kind == 8:
                    out[key] = string()
                elif kind == 9:
                    item, n = struct.unpack("<IQ", read(12))
                    if item in _GGUF_SCALAR:
                        f.seek(struct.calcsize(_GGUF_SCALAR[item]) * n, 1)
                    elif item == 8 and n <= 256:             # e.g. general.tags, before the keys that matter here
                        for _ in range(n):
                            string()
                    else:
                        break
                else:
                    break
        except EOFError:
            pass
    return out


class PreparedVisual(tuple):
    """Preserve the (ids, in_reasoning) interface; attach visual data to this preparation, not shared runner state."""
    def __new__(cls, ids, in_reasoning, visual):
        value = super().__new__(cls, (ids, in_reasoning))
        value.visual = visual
        return value


def split_images(messages):
    sources, out = [], []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list) or not any(isinstance(p, dict) and p.get("type") == "image_url" for p in content):
            out.append(message)
            continue
        if message.get("role") != "user":
            from .chat import RequestError
            raise RequestError("images are supported only in user messages")
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                value = part.get("image_url")
                if not isinstance(value, dict) or not isinstance(value.get("url"), str) or not value["url"]:
                    from .chat import RequestError
                    raise RequestError("image_url needs a nonempty url string")
                sources.append(value["url"])
                parts.append({"type": "text", "text": MARKER})
            else:
                parts.append(part)
        out.append(dict(message, content=parts))
    if len(sources) > MAX_IMAGES:
        from .chat import RequestError
        raise RequestError("at most %d images are supported per request" % MAX_IMAGES)
    return (out if sources else messages), sources


def read_image(source, *, allow_local_files=False):
    """Read a base64 data URI. Local files require an explicit server-side opt-in: True, or the folder they must be in.
    Never fetch a remote URL.  A source that is not a usable image raises ValueError."""
    if source.startswith("data:"):
        header, sep, data = source.partition(",")
        if not sep or not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("images must use a base64 data:image URI")
        if len(data) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise ValueError("image exceeds 24 MiB")
        raw = base64.b64decode(data, validate=True)
    else:
        url = urlsplit(source)
        drive = len(url.scheme) == 1 and source[1:2] == ":"           # C:\... is a local path, not a URL
        if url.scheme == "data":
            raise ValueError("images must use a base64 data:image URI")
        if url.scheme and url.scheme != "file" and not drive:
            raise ValueError("remote image URLs are not supported; send the image as a data:image/...;base64 URI")
        if not allow_local_files:
            raise ValueError("only data:image URIs are enabled; local files require %s" % LOCAL_FILES_HINT)
        if url.scheme == "file":
            if url.netloc not in ("", "localhost") or url.query or url.fragment:
                raise ValueError("remote file hosts, queries and fragments are not supported")
            path = unquote(url.path)
            if len(path) > 2 and path[0] == "/" and path[2] == ":":
                path = path[1:]
        else:
            path = source
        if path.replace("\\", "/").startswith("//"):
            raise ValueError("network shares and device paths are not supported")
        try:
            path = Path(path).resolve()
            root = None if allow_local_files is True else Path(allow_local_files).resolve()
        except OSError:
            raise ValueError("invalid local file path") from None
        if str(path).replace("\\", "/").startswith("//"):
            raise ValueError("network shares are not supported")
        if root is not None and not path.is_relative_to(root):
            raise ValueError("local image files must be inside the folder given with %s" % LOCAL_FILES_HINT)
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_IMAGE_BYTES + 1)
        except OSError:                                      # one message for a missing, locked or unreadable file
            raise ValueError("cannot read the local image file") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 24 MiB")
    try:
        from PIL import Image, ImageOps
    except ImportError as ex:
        raise ImportError("image input needs Pillow: pip install Pillow (into the Python that runs yasaheni)") from ex
    try:
        with warnings.catch_warnings():                      # Pillow warns above 89 million pixels: refused below anyway
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(raw))
        with image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("image exceeds 40 million pixels")
            if max(image.size) > MAX_ASPECT * min(image.size):
                raise ValueError("image aspect ratio exceeds %d:1" % MAX_ASPECT)
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("only still images are supported")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except ValueError:
        raise
    except Image.DecompressionBombError:                     # Pillow's own size limit, checked inside Image.open
        raise ValueError("image exceeds 40 million pixels") from None
    except Image.UnidentifiedImageError:
        raise ValueError("the data is not an image format Pillow can read") from None
    except Exception as ex:                                  # noqa: BLE001 - truncated or damaged image data
        raise ValueError("the image cannot be decoded (%s)" % (str(ex)[:120] or type(ex).__name__)) from None


def validate_features(item):
    grid = np.asarray(item["grid"])
    if grid.shape != (2,) or grid.dtype.kind not in "iu" or np.any(grid <= 0):
        raise ValueError("invalid image grid")
    h, w = map(int, grid)
    n = h * w
    features = item["features"]
    if features.dtype != np.uint16 or features.shape != (n, 5120):
        raise ValueError("invalid bfloat16 image feature shape")
    # Reject infinities/NaNs in cached bfloat16 data too.
    if np.any((features & 0x7f80) == 0x7f80):
        raise ValueError("non-finite image features")
    positions = np.stack([np.zeros(n, dtype=np.int32), np.arange(n, dtype=np.int32) // w,
                          np.arange(n, dtype=np.int32) % w])
    if item["positions"].dtype != np.int32 or not np.array_equal(positions, item["positions"]):
        raise ValueError("image positions differ from mtmd")
    return h, w, positions


def _check_length(size, max_tokens, max_ctx):
    if max_tokens is not None and size >= max_tokens:
        from .chat import context_length_error
        raise context_length_error(size, max_ctx if max_ctx is not None else max_tokens)


def build_visual_prompt(prompt, images, image_id, *, max_tokens=None, max_ctx=None):
    """Expand image placeholders and calculate Qwen's [time, height, width] rotary positions."""
    if not images or sum(int(t) == image_id for t in prompt) != len(images):
        raise ValueError("image placeholder count does not match supplied images")
    grids = [validate_features(item) for item in images]
    size = len(prompt) - len(images) + sum(h * w for h, w, _ in grids)
    _check_length(size, max_tokens, max_ctx)
    ids, positions, slots, features = [], [], [], []
    offset, index = 0, 0
    for token in prompt:
        if int(token) != image_id:
            ids.append(int(token))
            positions.append([offset] * 3)
            offset += 1
        else:
            h, w, relative = grids[index]
            n = h * w
            slots.extend(range(len(ids), len(ids) + n))
            ids.extend([image_id] * n)
            positions.extend((relative + offset).T.tolist())
            features.append(images[index]["features"])
            offset += max(h, w)
            index += 1
    return np.asarray(ids, np.int32), dict(features=np.ascontiguousarray(np.concatenate(features)),
        slots=np.asarray(slots, np.int32), positions=np.ascontiguousarray(np.asarray(positions, np.int32).T),
        delta=offset - len(ids))


class VisionProcessor:
    """Lazy CPU encoder with a bounded memory cache and optional disk cache. Calls are serialized.

    ``threads``: CPU threads of the encoder (None: one per physical core).  ``max_image_tokens``: llama.cpp downsizes every image to
    at most this many image tokens.  ``max_request_image_tokens``: the image tokens of one request together (None: 8 x the per-image
    limit)."""
    def __init__(self, *, mmproj, llama_dir, model, cache=None, threads=None, allow_local_files=False,
                 max_image_tokens=DEFAULT_IMAGE_TOKENS, max_request_image_tokens=None):
        # Absolute paths, fixed here once: a relative --llama-dir passed the start-up check and then failed every image request
        # (os.add_dll_directory refuses a relative path with WinError 87).
        self.mmproj, self.llama_dir, self.model = (_absolute(p) for p in (mmproj, llama_dir, model))
        self.cache = _absolute(cache) if cache else None
        self.threads = physical_cores() if threads is None else int(threads)
        if self.threads < 1:
            raise ValueError("--vision-threads must be at least 1")
        self.max_image_tokens = int(max_image_tokens)
        if not IMAGE_TOKENS_RANGE[0] <= self.max_image_tokens <= IMAGE_TOKENS_RANGE[1]:
            raise ValueError("--vision-max-image-tokens must be between %d and %d" % IMAGE_TOKENS_RANGE)
        self.max_request_image_tokens = (REQUEST_IMAGE_TOKENS_FACTOR * self.max_image_tokens if max_request_image_tokens is None
                                         else int(max_request_image_tokens))
        if not self.max_image_tokens <= self.max_request_image_tokens <= MAX_IMAGES * MAX_IMAGE_TOKENS:
            raise ValueError("--vision-max-request-image-tokens must be between the per-image limit (%d) and %d"
                             % (self.max_image_tokens, MAX_IMAGES * MAX_IMAGE_TOKENS))
        if allow_local_files not in (False, None, True):
            allow_local_files = str(_absolute(allow_local_files))
        self.allow_local_files = allow_local_files or False  # False, True (any path) or the folder local files must be in
        self._encoder = None
        self._memory = OrderedDict()
        self._memory_bytes = 0
        self._identity = None
        self._lock = threading.RLock()

    def check(self):
        """Check the options at start, before the model loads.  The encoder itself still opens at the first image."""
        if sys.platform != "win32":
            raise RuntimeError("image input works only on Windows (it uses the Windows build of llama.cpp b10488)")
        keys = {}
        for option, path in (("--mmproj", self.mmproj), ("--vision-model", self.model)):
            if not path.is_file():
                raise FileNotFoundError("%s: no such file: %s" % (option, path))
            try:
                keys[option] = gguf_keys(path)               # the first keys of the header only
            except ValueError:
                raise ValueError("%s: %s is not a GGUF file" % (option, path)) from None
        arch = keys["--mmproj"].get("general.architecture")
        if arch is not None and arch != "clip":
            raise ValueError("--mmproj: %s is not a projector (mmproj) file: its architecture is %r, not 'clip'" % (self.mmproj, arch))
        dim = keys["--mmproj"].get("clip.vision.projection_dim")
        if dim is not None and dim != PROJECTOR_DIM:
            raise ValueError("--mmproj: %s is the projector of another model (%d values per image token; Qwen3.8-27B takes %d)"
                             % (self.mmproj, dim, PROJECTOR_DIM))
        arch = keys["--vision-model"].get("general.architecture")
        if arch == "clip":
            raise ValueError("--vision-model: %s is a projector (mmproj) file; give a Qwen3.8-27B GGUF here" % self.model)
        dim = keys["--vision-model"].get("%s.embedding_length" % arch)
        if dim is not None and dim != PROJECTOR_DIM:
            raise ValueError("--vision-model: %s is not a Qwen3.8-27B GGUF (its embedding length is %d, not %d)"
                             % (self.model, dim, PROJECTOR_DIM))
        from ._mtmd import check_llama_dir
        check_llama_dir(self.llama_dir)
        try:
            import PIL.Image  # noqa: F401
        except ImportError as ex:
            raise ImportError("image input needs Pillow: pip install Pillow (into the Python that runs yasaheni)") from ex
        if self.allow_local_files not in (False, None, True) and not Path(self.allow_local_files).is_dir():
            raise FileNotFoundError("%s: no such folder: %s" % (LOCAL_FILES_HINT, self.allow_local_files))
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        """One line for the server's start-up message."""
        return ("image input: at most %d image tokens per image (larger images are downsized), %d per request, %d images per request; "
                "%d CPU threads.  While an image is encoded, other requests wait" % (self.max_image_tokens, self.max_request_image_tokens,
                                                                                     MAX_IMAGES, self.threads))

    def _fingerprint(self):
        # Replacing a projector/model/backend invalidates cached features, including after a server restart.  The per-image limit
        # changes the features (the image is downsized differently).  The thread count did not change them in our tests (1 to 32
        # threads, bit for bit), but it stays in the key: a cache entry is only reused under exactly the settings that made it.
        paths = [self.mmproj, self.model, self.llama_dir / "llama.dll", self.llama_dir / "mtmd.dll"]
        paths += sorted(self.llama_dir.glob("ggml*.dll"))
        files = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        return json.dumps(["b10488-cpu-bf16-rne-v2", self.threads, self.max_image_tokens, files]).encode()

    def _cached(self, key):
        """Features from the memory or the disk cache, or None.  A damaged or unreadable disk entry counts as missing."""
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        path = self.cache / (key + ".npz") if self.cache else None
        if not path or not path.is_file():
            return None
        try:                                                 # our own handle: closed even when numpy cannot read the file
            with path.open("rb") as stream, np.load(stream, allow_pickle=False) as stored:
                item = {k: stored[k].copy() for k in ("features", "positions", "grid")}
            validate_features(item)
        except Exception:                                    # noqa: BLE001 - it is encoded and written again
            return None
        self._remember(key, item)
        return item

    def _remember(self, key, item):
        size = sum(v.nbytes for v in item.values())
        while self._memory and self._memory_bytes + size > MEMORY_CACHE_BYTES:      # least recently used first
            _, old = self._memory.popitem(last=False)
            self._memory_bytes -= sum(v.nbytes for v in old.values())
        if size <= MEMORY_CACHE_BYTES:
            self._memory[key] = item
            self._memory_bytes += size

    def _store(self, key, item):
        """Write a disk cache entry under a temporary name, then rename it: a reader never sees a half-written file."""
        if not self.cache:
            return
        path, tmp = self.cache / (key + ".npz"), None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=key[:16] + ".", suffix=".tmp")
            with os.fdopen(fd, "wb") as stream:
                np.savez(stream, **item)
            os.replace(tmp, path)
            tmp = None
        except Exception as ex:                              # noqa: BLE001 - the cache only saves time
            print("note: the image cache was not written (%s)" % ex, file=sys.stderr, flush=True)
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def prepare(self, prompt, sources, fmt, max_tokens, max_ctx=None):
        """Read, check and encode the images of one request; returns the expanded prompt ids and the visual input.

        ``max_tokens`` is the prompt length that must not be reached (``max_ctx`` is only named in the error).  Every limit is
        checked before the expensive encode: each image is downsized by llama.cpp to at most ``max_image_tokens`` and refused above
        MAX_IMAGE_TOKENS, the images together at most ``max_request_image_tokens``, and the whole input within the context (counted
        with every image).  Raises RequestError / ContextLengthError or ValueError for the request, other errors for the server.

        The server calls this inside its generation lock, so while an image is encoded (seconds to minutes on the CPU) no other
        request is served.  Moving the encode out of that lock is a later change."""
        from .chat import RequestError
        image_id = fmt.token_id("<|image_pad|>")
        expected = [fmt.token_id(t) for t in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")]
        if fmt.encode(MARKER).tolist() != expected:
            raise RuntimeError("tokenizer does not encode image markers as single tokens")
        markers = sum(int(t) == image_id for t in prompt)
        if markers != len(sources):
            raise RequestError("the messages hold %d image markers for %d image(s); do not write <|vision_start|>, <|image_pad|> "
                               "or <|vision_end|> as text" % (markers, len(sources)))
        with self._lock:
            identity = self._fingerprint()
            if identity != self._identity:
                self.close()
                self._identity = identity
            items, pending, total = [], [], 0
            try:
                for i, source in enumerate(sources):
                    image = read_image(source, allow_local_files=self.allow_local_files)
                    key = hashlib.sha256(identity + str(image.size).encode() + image.tobytes()).hexdigest()
                    item = self._cached(key)
                    if item is None:
                        if self._encoder is None:
                            from ._mtmd import MtmdEncoder
                            self._encoder = MtmdEncoder(self.llama_dir, self.mmproj, self.model, threads=self.threads,
                                                        max_image_tokens=self.max_image_tokens)
                        job = self._encoder.tokenize(image)
                        pending.append((i, key, job))
                        n = job.n
                    else:
                        n = len(item["features"])
                    del image
                    if n > MAX_IMAGE_TOKENS:
                        raise RequestError("image %d needs %d image tokens; at most %d are allowed per image"
                                           % (i + 1, n, MAX_IMAGE_TOKENS))
                    total += n
                    if total > self.max_request_image_tokens:
                        raise RequestError("the images need %d image tokens or more; at most %d are allowed per request (server option "
                                           "--vision-max-request-image-tokens; the images of earlier turns count too)"
                                           % (total, self.max_request_image_tokens))
                    items.append(item)
                _check_length(len(prompt) - len(sources) + total, max_tokens, max_ctx)     # every image counted
                for i, key, job in pending:                  # all limits are met: now encode
                    item = self._encoder.encode_tokenized(job)
                    validate_features(item)
                    self._remember(key, item)
                    self._store(key, item)
                    items[i] = item
            finally:
                for _, _, job in pending:
                    job.close()
            return build_visual_prompt(prompt, items, image_id, max_tokens=max_tokens, max_ctx=max_ctx)

    def close(self):
        with self._lock:
            if self._encoder is not None:
                self._encoder.close()
                self._encoder = None
            self._memory.clear()
            self._memory_bytes = 0
