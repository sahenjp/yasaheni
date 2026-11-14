"""Image input: message splitting, placeholder expansion and positions, feature checks, image reading, the limits checked before
encoding, the feature cache and the start-up checks.  No GPU, no stub library, no llama.cpp (a fake encoder stands in for it)."""
import argparse
import base64
import contextlib
import hashlib
import io
import os
import struct
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from yasaheni import _mtmd, cli, vision
from yasaheni._native import YasakaniError
from yasaheni.chat import ContextLengthError, RequestError
from yasaheni.server import ServerConfig, parse_chat_request
from yasaheni.vision import (DEFAULT_IMAGE_TOKENS, IMAGE_TOKENS_RANGE, MARKER, MAX_IMAGE_TOKENS, MAX_IMAGES, MEMORY_CACHE_BYTES,
                             REQUEST_IMAGE_TOKENS_FACTOR, VisionProcessor, build_visual_prompt, read_image, split_images,
                             validate_features)

IMG = 99   # stands in for the <|image_pad|> token id


def fake_image(h, w, fill=1):
    n = h * w
    return dict(features=np.full((n, 5120), fill, np.uint16),
                positions=np.stack([np.zeros(n, np.int32), np.arange(n, dtype=np.int32) // w, np.arange(n, dtype=np.int32) % w]),
                grid=np.array([h, w], np.int32))


def image_part(url="data:image/png;base64,AAAA"):
    return {"type": "image_url", "image_url": {"url": url}}


class SplitImagesTests(unittest.TestCase):
    def test_text_only_is_returned_unchanged(self):
        msgs = [{"role": "user", "content": "hello"}]
        out, sources = split_images(msgs)
        self.assertIs(out, msgs)
        self.assertEqual(sources, [])

    def test_images_become_markers_in_order(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "a"}, image_part("data:1"), image_part("data:2")]}]
        out, sources = split_images(msgs)
        self.assertEqual(sources, ["data:1", "data:2"])
        self.assertEqual([p["text"] for p in out[0]["content"]], ["a", MARKER, MARKER])
        self.assertEqual(msgs[0]["content"][1]["type"], "image_url")      # the caller's messages are not changed

    def test_rejections(self):
        with self.assertRaises(RequestError):
            split_images([{"role": "assistant", "content": [image_part()]}])
        with self.assertRaises(RequestError):
            split_images([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": ""}}]}])
        with self.assertRaises(RequestError):
            split_images([{"role": "user", "content": [image_part()] * (MAX_IMAGES + 1)}])


class VisualPromptTests(unittest.TestCase):
    def test_expansion_slots_and_positions(self):
        ids, v = build_visual_prompt([1, IMG, 2], [fake_image(2, 3)], IMG)
        self.assertEqual(ids.tolist(), [1] + [IMG] * 6 + [2])
        self.assertEqual(v["slots"].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(v["positions"].shape, (3, 8))
        # text before: position 0 on all three axes; the image: [time, row, column] from 1; text after: 1 + max(2, 3)
        self.assertEqual(v["positions"][:, 0].tolist(), [0, 0, 0])
        self.assertEqual(v["positions"][:, 1:7].T.tolist(), [[1, 1 + r, 1 + c] for r in range(2) for c in range(3)])
        self.assertEqual(v["positions"][:, 7].tolist(), [4, 4, 4])
        self.assertEqual(v["delta"], 5 - 8)                                # next position minus the number of tokens
        self.assertEqual(v["features"].shape, (6, 5120))

    def test_two_images(self):
        ids, v = build_visual_prompt([IMG, 7, IMG], [fake_image(1, 2, 1), fake_image(2, 2, 2)], IMG)
        self.assertEqual(len(ids), 2 + 1 + 4)
        self.assertEqual(v["slots"].tolist(), [0, 1, 3, 4, 5, 6])
        self.assertEqual(v["features"][:2, 0].tolist(), [1, 1])
        self.assertEqual(v["features"][2:, 0].tolist(), [2, 2, 2, 2])

    def test_count_mismatch_and_context(self):
        with self.assertRaises(ValueError):
            build_visual_prompt([1, 2], [fake_image(1, 1)], IMG)
        with self.assertRaises(ContextLengthError):
            build_visual_prompt([1, IMG], [fake_image(4, 4)], IMG, max_tokens=10)


class FeatureCheckTests(unittest.TestCase):
    def test_bad_features(self):
        item = fake_image(2, 2)
        item["features"] = item["features"][:3]
        with self.assertRaises(ValueError):
            validate_features(item)
        item = fake_image(2, 2)
        item["features"][0, 0] = 0x7F80                                    # bfloat16 infinity
        with self.assertRaises(ValueError):
            validate_features(item)
        item = fake_image(2, 2)
        item["positions"] = item["positions"][:, ::-1].copy()
        with self.assertRaises(ValueError):
            validate_features(item)


class ReadImageTests(unittest.TestCase):
    def test_sources(self):
        with self.assertRaises(ValueError):
            read_image("https://example.com/a.png")
        with self.assertRaises(ValueError):
            read_image("C:/a.png")                                         # local files need the server option
        with self.assertRaises(ValueError):
            read_image("data:text/plain;base64,AAAA")

    def test_data_uri(self):
        try:
            from PIL import Image
        except ImportError:
            raise unittest.SkipTest("Pillow is not installed")
        buf = io.BytesIO()
        Image.new("RGBA", (5, 3), (255, 0, 0, 128)).save(buf, format="PNG")
        im = read_image("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
        self.assertEqual((im.mode, im.size), ("RGB", (5, 3)))


# ---- the fixes after review: every failure is an HTTP error, limits before encoding, start-up checks --------------------------------
def need_pillow():
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("Pillow is not installed")


def png_bytes(size, mode="RGB"):
    from PIL import Image
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format="PNG")
    return buf.getvalue()


def png_uri(size, mode="RGB"):
    return "data:image/png;base64," + base64.b64encode(png_bytes(size, mode)).decode()


class FakeJob:
    def __init__(self, n):
        self.n, self.closed = n, False

    def close(self):
        self.closed = True


class FakeEncoder:
    """Stands in for _mtmd.MtmdEncoder: every image becomes GRID[0] x GRID[1] image tokens."""
    GRID = [2, 3]
    jobs, encoded, fail, opened = [], [], None, []

    def __init__(self, *a, **k):
        FakeEncoder.opened.append((a, k))

    def tokenize(self, image):
        job = FakeJob(self.GRID[0] * self.GRID[1])
        FakeEncoder.jobs.append(job)
        return job

    def encode_tokenized(self, job):
        if FakeEncoder.fail is not None:
            raise FakeEncoder.fail
        FakeEncoder.encoded.append(job.n)
        return fake_image(*self.GRID)

    def close(self):
        pass


class FakeFormatter:
    IDS = {"<|vision_start|>": 97, "<|image_pad|>": IMG, "<|vision_end|>": 98}

    def token_id(self, name):
        return self.IDS[name]

    def encode(self, text):
        return np.array([97, IMG, 98] if text == MARKER else [1], np.int32)


class ProcessorCase(unittest.TestCase):
    """A VisionProcessor over placeholder files, with the fake encoder."""

    def setUp(self):
        need_pillow()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = t = Path(tmp.name)
        (t / "llama").mkdir()
        for f in ("llama.dll", "mtmd.dll", "ggml.dll", "ggml-base.dll"):
            (t / "llama" / f).write_bytes(b"not a library: " + f.encode())
        (t / "mmproj.gguf").write_bytes(b"GGUF placeholder")
        (t / "text.gguf").write_bytes(b"GGUF placeholder")
        FakeEncoder.GRID[:] = [2, 3]
        FakeEncoder.fail = None
        del FakeEncoder.jobs[:], FakeEncoder.encoded[:], FakeEncoder.opened[:]
        patcher = mock.patch.object(_mtmd, "MtmdEncoder", FakeEncoder)
        patcher.start()
        self.addCleanup(patcher.stop)

    def processor(self, **kw):
        return VisionProcessor(mmproj=self.t / "mmproj.gguf", llama_dir=self.t / "llama", model=self.t / "text.gguf", **kw)

    @staticmethod
    def prepare(vp, uris, prompt=None, max_tokens=10000):
        prompt = [1] + [IMG] * len(uris) + [2] if prompt is None else prompt
        return vp.prepare(prompt, uris, FakeFormatter(), max_tokens, max_tokens + 64)


class CacheTests(ProcessorCase):
    def test_damaged_entry_is_rebuilt(self):
        uri = png_uri((9, 4))
        self.prepare(self.processor(cache=self.t / "cache"), [uri])
        files = list((self.t / "cache").glob("*.npz"))
        self.assertEqual((len(files), FakeEncoder.encoded), (1, [6]))
        files[0].write_bytes(b"PK\x03\x04 cut short")                  # a damaged entry (zipfile.BadZipFile when read)
        ids, v = self.prepare(self.processor(cache=self.t / "cache"), [uri])   # a new processor: nothing in memory
        self.assertEqual(len(FakeEncoder.encoded), 2)                    # treated as missing: encoded again ...
        with np.load(files[0], allow_pickle=False) as stored:            # ... and written again, readable
            self.assertEqual(stored["features"].shape, (6, 5120))
        self.prepare(self.processor(cache=self.t / "cache"), [uri])
        self.assertEqual(len(FakeEncoder.encoded), 2)                    # now read from the disk cache
        self.assertEqual(list((self.t / "cache").glob("*.tmp")), [])

    def test_failed_write_is_not_fatal(self):
        with mock.patch.object(vision.np, "savez", side_effect=OSError("disk full")), contextlib.redirect_stderr(io.StringIO()) as err:
            ids, v = self.prepare(self.processor(cache=self.t / "cache"), [png_uri((9, 4))])
        self.assertEqual(v["features"].shape, (6, 5120))
        self.assertIn("not written", err.getvalue())
        self.assertEqual(list((self.t / "cache").iterdir()), [])         # no final file and no temporary file left


class LimitTests(ProcessorCase):
    def test_tokens_per_image(self):
        FakeEncoder.GRID[:] = [64, 65]                                   # 4160 image tokens
        with self.assertRaises(RequestError) as cm:
            self.prepare(self.processor(), [png_uri((8, 8))])
        self.assertIn("at most %d" % MAX_IMAGE_TOKENS, str(cm.exception))
        self.assertEqual(FakeEncoder.encoded, [])                        # refused before the encode
        self.assertTrue(all(j.closed for j in FakeEncoder.jobs))

    def test_tokens_per_request(self):
        FakeEncoder.GRID[:] = [32, 32]                                   # 1024 per image (the default per-image limit), 10 images
        with self.assertRaises(RequestError) as cm:
            self.prepare(self.processor(), [png_uri((8, 8))] * 10, max_tokens=100000)
        self.assertNotIsInstance(cm.exception, ContextLengthError)
        self.assertEqual(REQUEST_IMAGE_TOKENS_FACTOR * DEFAULT_IMAGE_TOKENS, 8192)
        self.assertIn("at most 8192 are allowed per request", str(cm.exception))
        self.assertEqual(FakeEncoder.encoded, [])
        self.assertEqual(len(FakeEncoder.jobs), 9)                       # stopped at the image that crossed the limit
        self.assertTrue(all(j.closed for j in FakeEncoder.jobs))
        del FakeEncoder.jobs[:]
        ids, v = self.prepare(self.processor(), [png_uri((8, 8))] * 8, max_tokens=100000)    # 8 x 1024 = the limit: accepted
        self.assertEqual(v["features"].shape, (8192, 5120))
        with self.assertRaises(RequestError):                            # the limit can be set
            self.prepare(self.processor(max_request_image_tokens=2047), [png_uri((8, 8))] * 2, max_tokens=100000)

    def test_context_error_counts_every_image(self):
        with self.assertRaises(ContextLengthError) as cm:                # 5 - 3 + 3 x 6 = 20 tokens; the limit is crossed at image 2
            self.prepare(self.processor(), [png_uri((8, 8))] * 3, max_tokens=10)
        self.assertIn("the input is too long: 20 tokens exceeds the model's maximum context length of 74 tokens", str(cm.exception))
        self.assertEqual(FakeEncoder.encoded, [])
        self.assertEqual(len(FakeEncoder.jobs), 3)
        self.assertTrue(all(j.closed for j in FakeEncoder.jobs))

    def test_context_length_before_encode(self):
        with self.assertRaises(ContextLengthError) as cm:
            self.prepare(self.processor(), [png_uri((8, 8))], max_tokens=5)     # 3 - 1 + 6 = 8 tokens
        self.assertEqual(cm.exception.code, "context_length_exceeded")
        self.assertIn("8 tokens exceeds the model's maximum context length of 69 tokens", str(cm.exception))
        self.assertEqual(FakeEncoder.encoded, [])

    def test_marker_count_is_checked_first(self):
        with self.assertRaises(RequestError) as cm:                     # the source is not even read
            self.prepare(self.processor(), ["data:image/png;base64,!!"], prompt=[1, IMG, IMG, 2])
        self.assertIn("2 image markers for 1 image(s)", str(cm.exception))

    def test_encoder_errors_pass_through(self):
        FakeEncoder.fail = KeyError("mtmd")                              # chat.ChatRunner turns it into a server error (500)
        with self.assertRaises(KeyError):
            self.prepare(self.processor(), [png_uri((8, 8))])
        self.assertTrue(all(j.closed for j in FakeEncoder.jobs))


class ImageFileTests(unittest.TestCase):
    def setUp(self):
        need_pillow()

    def test_decompression_bomb(self):
        from PIL import Image
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 1000):         # Pillow refuses above twice its limit, inside Image.open
            with self.assertRaises(ValueError) as cm:
                read_image(png_uri((100, 100)))
        self.assertIn("40 million pixels", str(cm.exception))

    def test_aspect_ratio(self):
        self.assertEqual(read_image(png_uri((2000, 10))).size, (2000, 10))          # 200:1 is allowed
        for size, mode in (((2001, 10), "RGB"), ((10, 2001), "RGB"), ((40_000_000, 1), "1")):   # the last one: 5 KB of PNG
            with self.assertRaises(ValueError) as cm:
                read_image(png_uri(size, mode))
            self.assertIn("aspect ratio", str(cm.exception))

    def test_damaged_data(self):
        from PIL import Image
        buf = io.BytesIO()
        Image.effect_noise((64, 64), 60).save(buf, format="PNG")         # noise: the pixel data is most of the file
        good = buf.getvalue()
        for raw in (good[: len(good) // 2], b"\x89PNG\r\n\x1a\n" + b"\0" * 64, b"plain text, not an image"):
            with self.assertRaises(ValueError) as cm:                    # never an OSError or a Pillow exception
                read_image("data:image/png;base64," + base64.b64encode(raw).decode())
            self.assertNotIn("BytesIO", str(cm.exception))

    def test_local_files_only_inside_the_folder(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root, other = Path(d) / "images", Path(d) / "other"
            root.mkdir()
            other.mkdir()
            Image.new("RGB", (4, 2)).save(root / "a.png")
            Image.new("RGB", (4, 2)).save(other / "b.png")
            (root / "note.txt").write_text("not an image")
            self.assertEqual(read_image(str(root / "a.png"), allow_local_files=str(root)).size, (4, 2))
            self.assertEqual(read_image((root / "a.png").as_uri(), allow_local_files=str(root)).size, (4, 2))
            for bad in (str(other / "b.png"), str(root / ".." / "other" / "b.png")):
                with self.assertRaises(ValueError) as cm:
                    read_image(bad, allow_local_files=str(root))
                self.assertIn("inside the folder", str(cm.exception))
            with self.assertRaises(ValueError) as cm:
                read_image(str(root / "missing.png"), allow_local_files=str(root))
            self.assertEqual(str(cm.exception), "cannot read the local image file")      # no path in the message
            with self.assertRaises(ValueError):
                read_image(str(root / "note.txt"), allow_local_files=str(root))
            self.assertEqual(read_image(str(other / "b.png"), allow_local_files=True).size, (4, 2))   # library use: any path
            with self.assertRaises(ValueError):
                read_image(str(root / "a.png"))                          # local files are off by default


class LibraryTests(unittest.TestCase):
    def test_missing_function(self):
        with self.assertRaises(RuntimeError) as cm:
            _mtmd._bind(object(), "mtmd_no_such_function", [], None)
        self.assertIn("mtmd_no_such_function", str(cm.exception))
        self.assertIn(_mtmd.LLAMA_ZIP, str(cm.exception))

    def test_encoder_refuses_a_library_without_a_function(self):
        class Fn:
            def __call__(self, *a):
                calls.append(a)

        class Lib:
            def __init__(self, missing=()):
                self.missing = missing

            def __getattr__(self, name):
                if name in self.missing:
                    raise AttributeError(name)
                return Fn()

        calls = []
        with tempfile.TemporaryDirectory() as d:
            for f in ("mmproj.gguf", "text.gguf"):
                (Path(d) / f).write_bytes(b"GGUF")
            with mock.patch.object(_mtmd, "check_llama_dir"), \
                    mock.patch.object(_mtmd, "_load_libraries", return_value=(Lib(), Lib({"mtmd_encode_chunk"}), None)):
                with self.assertRaises(RuntimeError) as cm:
                    _mtmd.MtmdEncoder(d, Path(d) / "mmproj.gguf", Path(d) / "text.gguf")
        self.assertIn("mtmd_encode_chunk", str(cm.exception))
        self.assertEqual(calls, [])                                      # refused before any function of it was called

    def test_logs_are_set_and_restored(self):
        seen = []

        class Fn:
            def __init__(self, name):
                self.name = name

            def __call__(self, *a):                                     # converts the arguments as ctypes does
                seen.append((self.name, [t.from_param(v) for t, v in zip(self.argtypes, a)]))

        class Lib:
            def __getattr__(self, name):
                return Fn(name)

        with mock.patch.object(_mtmd.atexit, "register") as register, mock.patch.object(_mtmd, "_log_libs", []):
            _mtmd._set_logs(Lib(), Lib(), True)
            _mtmd._set_logs(Lib(), Lib(), False)
            _mtmd._set_logs(Lib(), Lib(), True)
        self.assertEqual([s[0] for s in seen], ["llama_log_set", "mtmd_log_set"] * 3)
        self.assertIsNone(seen[2][1][0])                                 # NULL: the library's own logger again
        register.assert_called_once()                                    # restored before Python shuts down
        with contextlib.redirect_stderr(io.StringIO()) as err:
            for level, text in ((2, b"info\n"), (5, b"info, continued\n"), (3, b"warning\n"), (5, b"warning, continued\n"),
                                (4, b"error\n"), (1, b"debug\n")):
                _mtmd._LOG_CALLBACK(level, text, None)                    # through the C callback, as llama.cpp calls it
        self.assertEqual(err.getvalue(), "warning\nwarning, continued\nerror\n")

    def test_llama_dir_check(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with self.assertRaises(FileNotFoundError):
                _mtmd.check_llama_dir(d)                                 # empty folder
            for f in _mtmd.B10488_CPU_X64:
                (d / f).write_bytes(b"another build of " + f.encode())
            with self.assertRaises(RuntimeError) as cm:
                _mtmd.check_llama_dir(d)
            self.assertIn(_mtmd.LLAMA_ZIP, str(cm.exception))
            sums = {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in _mtmd.B10488_CPU_X64}
            with mock.patch.dict(_mtmd.B10488_CPU_X64, sums):
                for f in ("ggml-cpu-haswell.dll", "ggml-cpu-zen4.dll", "ggml-rpc.dll", "llama-server.exe"):
                    (d / f).write_bytes(b"x")
                _mtmd.check_llama_dir(d)                                 # the CPU build: accepted
                (d / "ggml-cuda.dll").write_bytes(b"x")
                with self.assertRaises(RuntimeError) as cm:
                    _mtmd.check_llama_dir(d)
                self.assertIn("ggml-cuda.dll", str(cm.exception))


class StartupCheckTests(ProcessorCase):
    def win(self):
        return mock.patch.object(vision.sys, "platform", "win32")

    def test_windows_only(self):
        with mock.patch.object(vision.sys, "platform", "linux"):
            with self.assertRaises(RuntimeError) as cm:
                self.processor().check()
        self.assertIn("Windows", str(cm.exception))

    def test_files(self):
        with self.win(), mock.patch.object(_mtmd, "check_llama_dir") as lib_check:
            self.processor(cache=self.t / "cache").check()               # all present
            self.assertTrue((self.t / "cache").is_dir())
            lib_check.assert_called_once()
            (self.t / "mmproj.gguf").write_bytes(b"<html>")
            with self.assertRaises(ValueError):
                self.processor().check()                                 # not a GGUF file
            (self.t / "mmproj.gguf").unlink()
            with self.assertRaises(FileNotFoundError):
                self.processor().check()
        with self.win():
            (self.t / "mmproj.gguf").write_bytes(b"GGUF placeholder")
            with self.assertRaises(RuntimeError):
                self.processor().check()                                 # placeholder libraries: not the b10488 CPU build

    def test_pillow_and_options(self):
        with self.win(), mock.patch.object(_mtmd, "check_llama_dir"):
            with mock.patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
                with self.assertRaises(ImportError) as cm:
                    self.processor().check()
                self.assertIn("pip install Pillow", str(cm.exception))
            with self.assertRaises(FileNotFoundError):
                self.processor(allow_local_files=str(self.t / "no such folder")).check()
        with self.assertRaises(ValueError):
            self.processor(threads=0)

    def test_cli_stops_before_the_model_loads(self):
        args = argparse.Namespace(mmproj=str(self.t / "missing.gguf"), llama_dir=str(self.t / "llama"), vision_model=str(self.t / "text.gguf"),
                                  vision_cache=None, vision_threads=8, vision_allow_local_files=None, model_dir=str(self.t))
        with mock.patch.object(cli, "ModelPack") as pack, mock.patch("yasaheni.textio.ChatFormatter"), \
                mock.patch("yasaheni.engine.Engine") as engine, self.win():
            with self.assertRaises(YasakaniError) as cm:
                cli._open_stack(args)
            self.assertIn("--mmproj", str(cm.exception))
            pack.load.assert_called_once()
            engine.assert_not_called()                                   # the long model load never started
        for kw in (dict(mmproj=None), dict(vision_threads=0)):
            ns = dict(vars(args), mmproj=str(self.t / "mmproj.gguf"))
            ns.update(kw)
            with self.assertRaises(YasakaniError):
                cli._vision(argparse.Namespace(**ns))

    def test_cli_options(self):
        a = cli.build_parser().parse_args(["serve", "--vision-allow-local-files", "pictures"])
        self.assertEqual(a.vision_allow_local_files, "pictures")
        self.assertIsNone(cli.build_parser().parse_args(["serve"]).vision_allow_local_files)


class ParseTests(unittest.TestCase):
    BODY = {"messages": [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                                       {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}

    def test_text_only_server_refuses_images_while_parsing(self):
        with self.assertRaises(RequestError) as cm:
            parse_chat_request(self.BODY, ServerConfig())
        self.assertEqual(str(cm.exception), "messages[0]: only text content is supported")    # the message of 0.1.1

    def test_image_server_keeps_them(self):
        p = parse_chat_request(self.BODY, ServerConfig(), images=True)
        self.assertEqual([part["type"] for part in p.messages[0]["content"]], ["text", "image_url"])


# ---- fix v2 (2026-11-14): per-image limit passed to llama.cpp, request limit, physical cores, cache size, absolute paths ---------------
def gguf_bytes(keys):
    """A GGUF header with these keys (str, int as uint32, bool, list of str) and no tensors."""
    out = [b"GGUF", struct.pack("<IQQ", 3, 0, len(keys))]
    for k, v in keys.items():
        out.append(struct.pack("<Q", len(k)) + k.encode())
        if isinstance(v, list):
            out.append(struct.pack("<IIQ", 9, 8, len(v)) + b"".join(struct.pack("<Q", len(s)) + s.encode() for s in v))
        elif isinstance(v, bool):
            out.append(struct.pack("<I?", 7, v))
        elif isinstance(v, int):
            out.append(struct.pack("<II", 4, v))
        else:
            out.append(struct.pack("<IQ", 8, len(v)) + v.encode())
    return b"".join(out)


MMPROJ_KEYS = {"general.architecture": "clip", "general.type": "mmproj", "general.tags": ["image-text-to-text"],
               "clip.has_vision_encoder": True, "clip.vision.projection_dim": 5120}       # the order of the real file
TEXT_KEYS = {"general.architecture": "qwen35", "general.type": "model", "general.tags": ["text-generation"], "qwen35.block_count": 65,
             "qwen35.embedding_length": 5120}


class OptionTests(ProcessorCase):
    def test_defaults(self):
        vp = self.processor()
        self.assertEqual((vp.max_image_tokens, vp.max_request_image_tokens), (1024, 8192))
        self.assertEqual(vp.threads, vision.physical_cores())
        self.assertEqual(self.processor(max_image_tokens=2048).max_request_image_tokens, 16384)
        self.assertEqual((DEFAULT_IMAGE_TOKENS, IMAGE_TOKENS_RANGE, MAX_IMAGE_TOKENS, MEMORY_CACHE_BYTES),
                         (1024, (64, 4096), 4096, 512 * 1024 * 1024))
        self.assertIn("at most 1024 image tokens per image", vp.summary())
        self.assertIn("8192 per request", vp.summary())

    def test_ranges(self):
        for kw in (dict(max_image_tokens=63), dict(max_image_tokens=4097), dict(max_request_image_tokens=1023),
                   dict(max_request_image_tokens=16 * 4096 + 1), dict(threads=0)):
            with self.subTest(**kw), self.assertRaises(ValueError):
                self.processor(**kw)
        self.processor(max_image_tokens=64, max_request_image_tokens=64)
        self.processor(max_image_tokens=4096, max_request_image_tokens=16 * 4096)

    def test_limit_and_threads_reach_the_encoder(self):
        self.prepare(self.processor(max_image_tokens=512, threads=3), [png_uri((8, 8))])
        (args, kw), = FakeEncoder.opened
        self.assertEqual(kw, dict(threads=3, max_image_tokens=512))

    def test_cache_key_holds_the_limit(self):
        uri = png_uri((9, 4))
        self.prepare(self.processor(cache=self.t / "cache"), [uri])
        self.prepare(self.processor(cache=self.t / "cache"), [uri])                         # same settings: from the disk cache
        self.assertEqual(len(FakeEncoder.encoded), 1)
        self.prepare(self.processor(cache=self.t / "cache", max_image_tokens=512), [uri])    # another limit: encoded again
        self.assertEqual(len(FakeEncoder.encoded), 2)
        self.assertEqual(len(list((self.t / "cache").glob("*.npz"))), 2)

    def test_memory_cache_keeps_a_chat_with_large_images(self):
        FakeEncoder.GRID[:] = [48, 64]                                   # 3072 tokens = 30 MiB of features per image
        vp = self.processor(max_image_tokens=4096)
        chat = [png_uri((8, 8)), png_uri((9, 8)), png_uri((10, 8))]    # three different images, 90 MiB together
        for turn in range(3):                                            # the client sends every image again with each turn
            self.prepare(vp, chat, max_tokens=100000)
        self.assertEqual(len(FakeEncoder.encoded), 3)                    # each image encoded once (64 MiB: every turn again)
        self.assertGreater(vp._memory_bytes, 64 * 1024 * 1024)
        with mock.patch.object(vision, "MEMORY_CACHE_BYTES", 64 * 1024 * 1024):
            vp2 = self.processor(max_image_tokens=4096)
            for turn in range(3):
                self.prepare(vp2, chat, max_tokens=100000)
        self.assertEqual(len(FakeEncoder.encoded), 3 + 3 + 1 + 1)        # the old 64 MiB: one image encoded again on every turn
        self.assertLessEqual(vp2._memory_bytes, 64 * 1024 * 1024)


class PathTests(ProcessorCase):
    def test_relative_paths_become_absolute(self):
        (self.t / "pictures").mkdir()
        old = os.getcwd()
        os.chdir(self.t)
        try:
            vp = VisionProcessor(mmproj="mmproj.gguf", llama_dir="llama", model="text.gguf", cache="cache",
                                 allow_local_files="pictures")
        finally:
            os.chdir(old)                                                # the server's working folder does not matter afterwards
        for p in (vp.mmproj, vp.llama_dir, vp.model, vp.cache, Path(vp.allow_local_files)):
            self.assertTrue(p.is_absolute(), p)
            self.assertTrue(str(p).startswith(str(self.t)), p)
        with mock.patch.object(vision.sys, "platform", "win32"), mock.patch.object(_mtmd, "check_llama_dir") as lib_check:
            vp.check()
        self.assertTrue(lib_check.call_args[0][0].is_absolute())
        self.prepare(vp, [png_uri((8, 8))])
        (args, kw), = FakeEncoder.opened
        self.assertTrue(all(Path(a).is_absolute() for a in args), args)

    def test_cli_resolves_the_paths(self):
        args = argparse.Namespace(mmproj="mmproj.gguf", llama_dir="llama", vision_model="text.gguf", vision_cache="cache",
                                  vision_threads=None, vision_allow_local_files=None, vision_max_image_tokens=1024,
                                  vision_max_request_image_tokens=None)
        old = os.getcwd()
        os.chdir(self.t)
        try:
            with mock.patch.object(vision.sys, "platform", "win32"), mock.patch.object(_mtmd, "check_llama_dir"):
                vp = cli._vision(args)
        finally:
            os.chdir(old)
        self.assertEqual(vp.llama_dir, self.t / "llama")
        self.assertEqual(vp.cache, self.t / "cache")

    def test_encoder_uses_an_absolute_folder(self):
        seen = []

        def load(folder):
            seen.append(folder)
            raise OSError("stop here")
        old = os.getcwd()
        os.chdir(self.t)
        try:
            with mock.patch.object(_mtmd, "check_llama_dir"), mock.patch.object(_mtmd, "_load_libraries", side_effect=load):
                with self.assertRaises(OSError):
                    _REAL_ENCODER("llama", "mmproj.gguf", "text.gguf")
        finally:
            os.chdir(old)
        self.assertEqual(seen, [self.t / "llama"])


_REAL_ENCODER = _mtmd.MtmdEncoder        # the real class (ProcessorCase replaces _mtmd.MtmdEncoder with the fake one)


class EncoderCloseTests(unittest.TestCase):
    def test_dll_directory_is_given_back(self):
        handle = mock.Mock()

        class Lib:
            def __init__(self, missing=()):
                self.missing = missing

            def __getattr__(self, name):
                if name in self.missing:
                    raise AttributeError(name)
                return lambda *a: None
        with tempfile.TemporaryDirectory() as d:
            for f in ("mmproj.gguf", "text.gguf"):
                (Path(d) / f).write_bytes(b"GGUF")
            with mock.patch.object(_mtmd, "check_llama_dir"), \
                    mock.patch.object(_mtmd, "_load_libraries", return_value=(Lib(), Lib({"mtmd_encode_chunk"}), handle)):
                with self.assertRaises(RuntimeError):                    # fails half way: everything is closed again
                    _REAL_ENCODER(d, Path(d) / "mmproj.gguf", Path(d) / "text.gguf")
        handle.close.assert_called_once()
        enc = _REAL_ENCODER.__new__(_REAL_ENCODER)                       # an open encoder: close() gives the folder back once
        enc.ctx = enc.model = None
        enc.lib = enc.llama = None
        enc._logs_set = False
        enc._dll_dir = handle2 = mock.Mock()
        enc.close()
        enc.close()
        handle2.close.assert_called_once()


class ProjectorCheckTests(ProcessorCase):
    def check(self, mmproj=MMPROJ_KEYS, text=TEXT_KEYS):
        (self.t / "mmproj.gguf").write_bytes(gguf_bytes(mmproj))
        (self.t / "text.gguf").write_bytes(gguf_bytes(text))
        with mock.patch.object(vision.sys, "platform", "win32"), mock.patch.object(_mtmd, "check_llama_dir"):
            self.processor().check()

    def test_header_keys(self):
        (self.t / "x.gguf").write_bytes(gguf_bytes(MMPROJ_KEYS) + b"tensor data")
        self.assertEqual(vision.gguf_keys(self.t / "x.gguf"), {k: v for k, v in MMPROJ_KEYS.items() if k != "general.tags"})
        vocab = dict(TEXT_KEYS, **{"tokenizer.ggml.tokens": ["t%d" % i for i in range(300)], "after.the.vocabulary": 1})
        (self.t / "x.gguf").write_bytes(gguf_bytes(vocab))
        self.assertNotIn("after.the.vocabulary", vision.gguf_keys(self.t / "x.gguf"))      # stops at a long list of strings
        self.assertEqual(vision.gguf_keys(self.t / "x.gguf")["qwen35.embedding_length"], 5120)
        (self.t / "x.gguf").write_bytes(gguf_bytes(MMPROJ_KEYS)[:40])                     # cut short: what was read so far
        self.assertEqual(vision.gguf_keys(self.t / "x.gguf"), {})
        (self.t / "x.gguf").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 40)
        with self.assertRaises(ValueError):
            vision.gguf_keys(self.t / "x.gguf")

    def test_the_right_files_pass(self):
        self.check()
        self.check(mmproj={}, text={})                                   # no keys to read: only the GGUF magic is checked

    def test_swapped_or_wrong_files_are_refused(self):
        cases = [(dict(MMPROJ_KEYS, **{"general.architecture": "qwen35"}), TEXT_KEYS, "--mmproj", "not a projector"),
                 (dict(MMPROJ_KEYS, **{"clip.vision.projection_dim": 4096}), TEXT_KEYS, "--mmproj", "another model"),
                 (MMPROJ_KEYS, MMPROJ_KEYS, "--vision-model", "is a projector"),
                 (MMPROJ_KEYS, dict(TEXT_KEYS, **{"qwen35.embedding_length": 4096}), "--vision-model", "not a Qwen3.8-27B")]
        for mmproj, text, option, words in cases:
            with self.subTest(option=option, words=words):
                with self.assertRaises(ValueError) as cm:
                    self.check(mmproj, text)
                self.assertTrue(str(cm.exception).startswith(option), cm.exception)
                self.assertIn(words, str(cm.exception))


class MessageTests(ProcessorCase):
    def test_remote_urls(self):
        for allow in (False, True):
            for url in ("http://example.com/a.png", "HTTPS://example.com/a.png", "ftp://example.com/a.png"):
                with self.subTest(allow=allow, url=url), self.assertRaises(ValueError) as cm:
                    read_image(url, allow_local_files=allow)
                self.assertIn("remote image URLs are not supported", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            read_image("DATA:text/plain;base64,AAAA")
        self.assertIn("base64 data:image URI", str(cm.exception))

    def test_no_pillow_warning_for_large_images(self):
        from PIL import Image
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 1000), warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            self.assertEqual(read_image(png_uri((40, 40))).size, (40, 40))      # 1600 pixels: Pillow would warn
        self.assertEqual([w for w in seen if issubclass(w.category, Image.DecompressionBombWarning)], [])

    def test_one_prefix(self):
        self.assertEqual(vision.error_text(RuntimeError("vision: missing x")), "missing x")
        self.assertEqual(vision.error_text(KeyError()), "KeyError")
        args = argparse.Namespace(mmproj=str(self.t / "mmproj.gguf"), llama_dir=str(self.t / "no such folder"),
                                  vision_model=str(self.t / "text.gguf"), vision_cache=None, vision_threads=None,
                                  vision_allow_local_files=None)
        with mock.patch.object(vision.sys, "platform", "win32"), self.assertRaises(YasakaniError) as cm:
            cli._vision(args)
        self.assertTrue(str(cm.exception).startswith("image input: the llama.cpp folder"), cm.exception)


class CoresTests(unittest.TestCase):
    def test_physical_cores(self):
        n = vision.physical_cores()
        self.assertGreaterEqual(n, 1)
        self.assertLessEqual(n, os.cpu_count() or n)
        with mock.patch.object(vision.sys, "platform", "linux"), mock.patch.object(vision.os, "cpu_count", return_value=32):
            self.assertEqual(vision.physical_cores(), 16)
        with mock.patch.object(vision.sys, "platform", "linux"), mock.patch.object(vision.os, "cpu_count", return_value=None):
            self.assertEqual(vision.physical_cores(), 1)


class CliOptionTests(unittest.TestCase):
    def test_options_and_help(self):
        a = cli.build_parser().parse_args(["serve"])
        self.assertEqual((a.vision_threads, a.vision_max_image_tokens, a.vision_max_request_image_tokens), (None, 1024, None))
        self.assertEqual(a.vision_max_image_tokens, DEFAULT_IMAGE_TOKENS)
        a = cli.build_parser().parse_args(["serve", "--vision-threads", "4", "--vision-max-image-tokens", "2048",
                                           "--vision-max-request-image-tokens", "4096"])
        self.assertEqual((a.vision_threads, a.vision_max_image_tokens, a.vision_max_request_image_tokens), (4, 2048, 4096))
        sp = cli.build_parser()._subparsers._group_actions[0].choices["serve"]
        text = sp.format_help()
        for words in ("default 1024", "64 to 4096", "8 x the per-image limit", "one per physical CPU core"):
            self.assertIn(words, " ".join(text.split()))
        self.assertEqual((IMAGE_TOKENS_RANGE, REQUEST_IMAGE_TOKENS_FACTOR), ((64, 4096), 8))     # the numbers of the help text


if __name__ == "__main__":
    unittest.main()
