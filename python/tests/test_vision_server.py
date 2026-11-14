# SPDX-License-Identifier: Apache-2.0
"""Image requests through the HTTP server, on the CPU stub engine with a fake image encoder: every failure is answered (400 for
the request, 500 for the server), the limits are checked before encoding, and a text-only server refuses images at once."""
import base64
import http.client
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import support
from tests.test_vision import FakeEncoder, need_pillow, png_uri
from yasaheni import _mtmd
from yasaheni.chat import ChatRunner
from yasaheni.runtime import EOS_IDS
from yasaheni.server import ApiState, ServerConfig, bind
from yasaheni.textio import ChatFormatter
from yasaheni.vision import VisionProcessor

EOS = EOS_IDS[0]
HAVE_TOK = support.tokenizer_dir() is not None


@unittest.skipUnless(HAVE_TOK, "set YASAKANI_TEST_MODEL_DIR to a folder with the Qwen3.8 tokenizer files")
class VisionServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need_pillow()
        cls.stub = support.Stub()
        cls.fmt = ChatFormatter(support.tokenizer_dir())

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = t = Path(tmp.name)
        (t / "llama").mkdir()
        for f in ("llama.dll", "mtmd.dll", "ggml.dll", "ggml-base.dll"):
            (t / "llama" / f).write_bytes(b"x")
        for f in ("mmproj.gguf", "text.gguf"):
            (t / f).write_bytes(b"GGUF")
        FakeEncoder.GRID[:] = [2, 3]
        FakeEncoder.fail = None
        del FakeEncoder.jobs[:], FakeEncoder.encoded[:]
        patcher = mock.patch.object(_mtmd, "MtmdEncoder", FakeEncoder)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calls = []
        self.httpd = None

    def start(self, vision=True, max_ctx=20000):
        self.stub.delay_us(0)
        self.engine = support.open_engine(self.t / "engine", self.stub, max_ctx=max_ctx)

        def set_visual(e, feats, slots, n, pos, n_prompt, delta):
            self.calls.append((n, n_prompt, delta))
            return 0
        self.engine._f = dict(self.engine._f, yskn_set_visual=set_visual)    # this engine only (the stub's table is shared)
        vp = VisionProcessor(mmproj=self.t / "mmproj.gguf", llama_dir=self.t / "llama", model=self.t / "text.gguf",
                             cache=self.t / "cache") if vision else None
        self.state = ApiState(ServerConfig(port=0))
        self.state.runner = ChatRunner(self.engine, self.fmt, seed0=1, vision=vp)
        self.httpd = bind(self.state)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.engine.close()

    def post(self, body, timeout=30):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            c.request("POST", "/v1/chat/completions", body=json.dumps(body),
                      headers={"Host": "127.0.0.1:%d" % self.port, "Content-Type": "application/json"})
            r = c.getresponse()                                  # raises RemoteDisconnected if the server drops the connection
            return r.status, json.loads(r.read())
        finally:
            c.close()

    @staticmethod
    def image_request(uri, text="what is this?"):
        return {"messages": [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": uri}}]}],
                "max_tokens": 8, "temperature": 0}

    def answer(self):
        self.stub.push([self.fmt.think_end] + self.fmt.encode("ok").tolist() + [EOS])

    def test_image_request(self):
        self.start()
        self.answer()
        status, d = self.post(self.image_request(png_uri((5, 3))))
        self.assertEqual(status, 200, d)
        self.assertEqual([c[0] for c in self.calls], [6, 0])            # set with 6 image tokens, then cleared
        self.assertEqual(d["usage"]["prompt_tokens"], self.calls[0][1])

    def test_damaged_cache_entry(self):
        self.start()
        uri = png_uri((9, 4))
        self.answer()
        self.assertEqual(self.post(self.image_request(uri))[0], 200)
        entry = next((self.t / "cache").glob("*.npz"))
        entry.write_bytes(b"PK\x03\x04 cut short")
        self.state.runner.vision.close()                                # forget the memory cache: the disk entry is read
        for _ in range(2):
            self.answer()
            status, d = self.post(self.image_request(uri))
            self.assertEqual(status, 200, d)
        self.assertEqual(len(FakeEncoder.encoded), 2)                    # encoded again once, then served from the cache

    def test_bad_images_get_400_before_encoding(self):
        from PIL import Image
        self.start()
        bomb = mock.patch.object(Image, "MAX_IMAGE_PIXELS", 1000)
        cases = [("decompression bomb", png_uri((100, 100)), bomb, "40 million pixels"),
                 ("40,000,000 x 1", png_uri((40_000_000, 1), "1"), None, "aspect ratio"),
                 ("not an image", "data:image/png;base64," + base64.b64encode(b"plain text").decode(), None, "image format"),
                 ("bad base64", "data:image/png;base64,!!!!", None, "image input"),
                 ("remote URL", "https://example.com/a.png", None, "data:image")]
        for name, uri, patch, text in cases:
            with self.subTest(name), (patch or mock.patch.object(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)):
                t0 = time.time()
                status, d = self.post(self.image_request(uri))
                self.assertEqual(status, 400, d)
                self.assertIn(text, d["error"]["message"])
                self.assertLess(time.time() - t0, 5)
        FakeEncoder.GRID[:] = [64, 65]
        status, d = self.post(self.image_request(png_uri((8, 8))))
        self.assertEqual(status, 400, d)
        self.assertIn("at most 4096", d["error"]["message"])
        self.assertEqual(FakeEncoder.encoded, [])                        # nothing was encoded
        self.assertEqual(self.calls, [])                                 # nothing reached the engine

    def test_request_limit_before_encoding(self):
        self.start()
        FakeEncoder.GRID[:] = [32, 32]                                   # 1024 image tokens each: 9 images are over 8192
        content = [{"type": "text", "text": "compare"}] + [{"type": "image_url", "image_url": {"url": png_uri((4 + i, 4))}}
                                                           for i in range(9)]
        status, d = self.post({"messages": [{"role": "user", "content": content}], "max_tokens": 8, "temperature": 0})
        self.assertEqual(status, 400, d)
        self.assertIn("at most 8192 are allowed per request", d["error"]["message"])
        self.assertEqual(FakeEncoder.encoded, [])                        # nothing was encoded
        self.assertEqual(self.calls, [])

    def test_encoder_failures_get_500(self):
        self.start()
        for exc in (RuntimeError("vision: the llama.cpp library has no function mtmd_encode_chunk"), KeyError("x"),
                    AttributeError("function 'mtmd_image_tokens_get_decoder_pos' not found")):
            with self.subTest(type(exc).__name__):
                FakeEncoder.fail = exc
                status, d = self.post(self.image_request(png_uri((3, 3))))
                self.assertEqual(status, 500, d)
                self.assertEqual(d["error"]["type"], "server_error")
                self.assertIn("image input failed", d["error"]["message"])
                self.assertNotIn("vision: ", d["error"]["message"])     # one prefix, not "image input failed: vision: ..."
        FakeEncoder.fail = None
        self.answer()
        self.assertEqual(self.post(self.image_request(png_uri((3, 3))))[0], 200)    # the server still works

    def test_context_length_code(self):
        self.start(max_ctx=1024)
        FakeEncoder.GRID[:] = [30, 33]                                   # 990 image tokens: under the per-image limit
        status, d = self.post(self.image_request(png_uri((8, 8))))
        self.assertEqual(status, 400, d)
        self.assertEqual(d["error"]["code"], "context_length_exceeded")
        self.assertIn("exceeds the model's maximum context length of 1024 tokens", d["error"]["message"])
        self.assertEqual(FakeEncoder.encoded, [])

    def test_literal_markers(self):
        self.start()
        status, d = self.post(self.image_request(png_uri((5, 3)), text="<|vision_start|><|image_pad|><|vision_end|> compare"))
        self.assertEqual(status, 400, d)
        self.assertIn("image markers", d["error"]["message"])

    def test_text_only_server_answers_at_once(self):
        self.start(vision=False)
        self.stub.push([self.fmt.think_end] + list(range(1000, 1400)) + [EOS])
        self.stub.delay_us(20000)                                        # a long generation holds the GPU
        th = threading.Thread(target=self.post, args=({"messages": [{"role": "user", "content": "long"}], "max_tokens": 400}, 120))
        th.start()
        time.sleep(0.5)
        t0 = time.time()
        status, d = self.post(self.image_request(png_uri((5, 3))))
        took = time.time() - t0
        th.join()
        self.stub.delay_us(0)
        self.assertEqual(status, 400, d)
        self.assertEqual(d["error"]["message"], "messages[0]: only text content is supported")
        self.assertLess(took, 1.0)                                       # not after the running generation


if __name__ == "__main__":
    unittest.main()
