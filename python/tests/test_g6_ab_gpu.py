# SPDX-License-Identifier: Apache-2.0
"""GPU A/B tests for the q35_g6 prefill tile grid: the switch may only change how the int8 products are scheduled, never the
greedy continuation of an 8K-token prompt.

They need the real engine library and the model, so they run only when YASAKANI_TEST_GPU is set and YASAKANI_TEST_MODEL_DIR points
at the model folder:

    YASAKANI_TEST_GPU=1 YASAKANI_TEST_MODEL_DIR=/models/qwen3.8-27b \\
        python3 -m unittest discover -s python/tests -t python -p 'test_g6_ab_gpu.py' -v

The test compares the default engine with YSKN_G6_TILEGRID=1 on the same engine and with YSKN_G6_TILEGRID=0 on an engine opened
with that profile override. A split-K launch that falls back to PFA still produces tokens, so its stderr note is checked too.
"""
import contextlib
import os
import tempfile
import unittest

import numpy as np

from yasaheni._native import YasakaniError
from yasaheni.engine import Engine, Sampling
from yasaheni.runtime import VOCAB_SIZE, ModelPack, RuntimeProfile

MODEL_DIR = os.environ.get("YASAKANI_TEST_MODEL_DIR")
GPU = os.environ.get("YASAKANI_TEST_GPU", "") not in ("", "0")

BASE_ENV = {"YSKN_Q35_G6": "1", "YSKN_Q35_PMAX": "1024"}
SWITCHES = ("YSKN_G6_TILEGRID",)
MAX_CTX = 12288
PROMPT_TOKENS = 8192
NEW_TOKENS = 4


def _have_model() -> bool:
    """The quality folder names its files in yasaheni.json, so ask ModelPack instead of guessing a file name."""
    if not (GPU and MODEL_DIR):
        return False
    try:
        ModelPack.load(MODEL_DIR)
    except (YasakaniError, OSError, ValueError):
        return False
    return True


class _switch_env:
    """Set exactly the named switch values for the with block and remove the others, so no run inherits a stale switch."""

    def __init__(self, values=()):
        self.values = dict(values)

    def __enter__(self):
        for name in SWITCHES:
            os.environ.pop(name, None)
        os.environ.update(self.values)

    def __exit__(self, *exc):
        for name in self.values:
            os.environ.pop(name, None)


@contextlib.contextmanager
def _capture_stderr():
    """Capture file descriptor 2: the engine writes its notes from C, so redirect_stderr would not see them."""
    saved = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 2)
    captured = []

    def read():
        """Return the log so far; after the with block ends, the content read just before the file was closed."""
        if not tmp.closed:
            tmp.flush()
            tmp.seek(0)
            captured[:] = [tmp.read().decode("utf-8", "replace")]
        return captured[0]

    try:
        yield read
    finally:
        read()
        os.dup2(saved, 2)
        os.close(saved)
        tmp.close()


HAVE = _have_model()


@unittest.skipUnless(HAVE, "set YASAKANI_TEST_GPU=1 and YASAKANI_TEST_MODEL_DIR to the model folder to run the q35_g6 GPU tests")
class G6AbGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = ModelPack.load(MODEL_DIR)
        cls._engines = {}
        cls.addClassCleanup(cls._close)

    @classmethod
    def _close(cls):
        for e in cls._engines.values():
            try:
                e.close()
            except Exception:                                  # noqa: BLE001
                pass

    @classmethod
    def engine(cls, overrides=()):
        """One engine per open-time profile, so the forced tile-grid setting is in place before the engine opens."""
        key = tuple(sorted(overrides))
        if key not in cls._engines:
            cls._engines[key] = Engine(cls.pack, max_ctx=MAX_CTX,
                                       profile=RuntimeProfile.default(cls.pack, {**BASE_ENV, **dict(overrides)}))
        return cls._engines[key]

    @staticmethod
    def ids(n, seed):
        return np.random.default_rng(seed).integers(0, VOCAB_SIZE, size=n, dtype=np.int32).tolist()

    def prefill(self, prompt, engine, switches=()):
        """Greedy continuation after the prompt with exactly the named switches set (none of the others)."""
        with _switch_env(switches):
            engine.set_sampling(Sampling(0.0))
            engine.reset()
            return engine.generate(prompt, NEW_TOKENS, stop_ids=None).ids

    def test_tilegrid_matches_default(self):
        prompt = self.ids(PROMPT_TOKENS, 4242)
        base = self.prefill(prompt, self.engine())
        self.assertEqual(len(base), NEW_TOKENS)
        self.assertEqual(self.prefill(prompt, self.engine(), (("YSKN_G6_TILEGRID", "1"),)), base,
                         "YSKN_G6_TILEGRID=1 changed the prefill result")
        splitk = self.engine((("YSKN_G6_TILEGRID", "0"),))
        with _capture_stderr() as log:
            got = self.prefill(prompt, splitk, (("YSKN_G6_TILEGRID", "0"),))
        self.assertEqual(got, base, "YSKN_G6_TILEGRID=0 changed the prefill result")
        self.assertNotIn("split-K grid needs a workspace", log(),
                         "the split-K launch fell back to PFA")


if __name__ == "__main__":
    unittest.main()
