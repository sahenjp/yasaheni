# SPDX-License-Identifier: Apache-2.0
"""GPU tests for parking and resuming a qwen3_5 context (design section 4.3).

They need the real engine library and the model, so they run only when YASAKANI_TEST_GPU is set and YASAKANI_TEST_MODEL_DIR points
at the model folder:

    YASAKANI_TEST_GPU=1 YASAKANI_TEST_MODEL_DIR=/models/qwen3.8-27b \\
        python3 -m unittest discover -s python/tests -t python -p 'test_park_gpu.py' -v
"""
import os
import unittest

import numpy as np

from yasaheni._native import YasakaniError
from yasaheni.engine import Engine, Sampling
from yasaheni.runtime import VOCAB_SIZE, ModelPack, RuntimeProfile

MODEL_DIR = os.environ.get("YASAKANI_TEST_MODEL_DIR")
GPU = os.environ.get("YASAKANI_TEST_GPU", "") not in ("", "0")


def _have_model() -> bool:
    """The quality folder names its files in yasaheni.json, so ask ModelPack instead of guessing a file name."""
    if not (GPU and MODEL_DIR):
        return False
    try:
        ModelPack.load(MODEL_DIR)
    except (YasakaniError, OSError, ValueError):
        return False
    return True


HAVE = _have_model()
MAX_CTX = 8192

# KV storage configurations: the shipped one, INT8 without the packable rows, and FP16 (only a non-KVX build can open it).
FORMATS = [("shipped", {}), ("int8", {"YSKN_TB_CAP": "0"}), ("fp16", {"YSKN_Q35_KV8": "0", "YSKN_Q35_KV_FP16": "1"})]


@unittest.skipUnless(HAVE, "set YASAKANI_TEST_GPU=1 and YASAKANI_TEST_MODEL_DIR to the model folder to run the park GPU tests")
class ParkGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = ModelPack.load(MODEL_DIR)
        cls._engines = {}
        cls.addClassCleanup(cls._close_all)

    @classmethod
    def _close_all(cls):
        for e in cls._engines.values():
            if e is not None:
                try:
                    e.close()
                except Exception:                                  # noqa: BLE001
                    pass

    def engine(self, fmt="shipped", max_ctx=MAX_CTX):
        e = self._engines.get(fmt)
        if e is None:
            overrides = dict(FORMATS)[fmt]
            try:
                e = Engine(self.pack, max_ctx=max_ctx, profile=RuntimeProfile.default(self.pack, overrides))
            except YasakaniError as ex:
                self.skipTest("%s KV storage is not available on this build: %s" % (fmt, ex))
            self._engines[fmt] = e
        return e

    @staticmethod
    def ids(n, seed):
        return np.random.default_rng(seed).integers(0, VOCAB_SIZE, size=n, dtype=np.int32).tolist()

    def test_park_hash_roundtrip(self):
        for fmt, _ in FORMATS:
            with self.subTest(format=fmt):
                e = self.engine(fmt)
                e.set_sampling(Sampling(0.0))
                e.reset()
                prompt, other = self.ids(4096, 1), self.ids(512, 2)
                r = e.generate(prompt, 4, stop_ids=None)
                pos = e.position()
                self.assertEqual(pos, 4096 + len(r.ids) - 1)
                h0 = e.state_hash()
                h = e.park()
                try:
                    self.assertEqual(h.position, pos)
                    self.assertEqual(h.last_token, r.ids[-1])
                    self.assertEqual(h.token_ids, e.live_ids())
                    self.assertGreater(h.nbytes, 0)
                    np.testing.assert_array_equal(e.state_hash(), h0)       # park does not touch the engine
                    e.reset()
                    e.generate(other, 8, stop_ids=None)                     # overwrite the KV rows
                    e.resume(h)
                    self.assertEqual(e.position(), pos)
                    np.testing.assert_array_equal(e.state_hash(), h0)
                finally:
                    h.close()

    def test_park_interleaved_greedy(self):
        e = self.engine()
        e.set_sampling(Sampling(0.0))
        e.reset()
        p, q = self.ids(512, 3), self.ids(700, 4)
        t = e.generate(p, 12, stop_ids=None).ids
        h0 = e.state_hash()
        h = e.park()
        try:
            u = e.generate(p + t, 8, stop_ids=None).ids
            e.reset()
            e.generate(q, 6, stop_ids=None)
            e.resume(h)
            np.testing.assert_array_equal(e.state_hash(), h0)
            self.assertEqual(e.generate(p + t, 8, stop_ids=None).ids, u)
        finally:
            h.close()

    def test_park_interleaved_sampled(self):
        """A resumed context must reproduce the uninterrupted continuation for every seed, not only the seed that was used first.

        A different seed changing the first token is not guaranteed (a high-probability token stays the same), so the test compares
        full continuations per seed instead of asserting that a new seed must pick a different first token.
        """
        e = self.engine()
        e.reset()
        p, q = self.ids(512, 5), self.ids(700, 6)
        seeds = (1234, 4321, 2026, 77)
        sampling = lambda seed: Sampling(0.8, 0.95, 20, 0.0, seed)
        e.set_sampling(sampling(seeds[0]))
        t = e.generate(p, 12, stop_ids=None).ids
        h0 = e.state_hash()
        h = e.park()
        try:
            base = {}
            for seed in seeds:
                e.set_sampling(sampling(seed))
                base[seed] = e.generate(p + t, 8, stop_ids=None).ids
                e.resume(h)                                          # back to the parked state for the next baseline
            e.reset()
            e.generate(q, 6, stop_ids=None)
            for seed in seeds:
                e.resume(h)
                np.testing.assert_array_equal(e.state_hash(), h0)
                e.set_sampling(sampling(seed))
                self.assertEqual(e.generate(p + t, 8, stop_ids=None).ids, base[seed])
        finally:
            h.close()

    def test_park_non_destructive(self):
        e = self.engine()
        e.set_sampling(Sampling(0.0))
        e.reset()
        e.generate(self.ids(256, 7), 6, stop_ids=None)
        h0 = e.state_hash()
        h1 = e.park()
        h2 = e.park()
        try:
            self.assertEqual(h1.nbytes, h2.nbytes)
            self.assertIsNot(h1._h, h2._h)
            e.reset()
            e.resume(h1)
            np.testing.assert_array_equal(e.state_hash(), h0)
            e.reset()
            e.resume(h2)
            np.testing.assert_array_equal(e.state_hash(), h0)
            h1.close()
            e.reset()
            e.resume(h2)                                            # freeing one handle leaves the other usable
            np.testing.assert_array_equal(e.state_hash(), h0)
        finally:
            h1.close()
            h2.close()

    def test_resume_fingerprint(self):
        e = self.engine()
        e.set_sampling(Sampling(0.0))
        e.reset()
        e.generate(self.ids(256, 8), 4, stop_ids=None)
        h0 = e.state_hash()
        h = e.park()
        other = Engine(self.pack, max_ctx=4096, profile=RuntimeProfile.default(self.pack))
        try:
            self.assertNotEqual(other.max_ctx, e.max_ctx)
            with self.assertRaisesRegex(YasakaniError, "does not match this engine"):
                other.resume(h)
        finally:
            other.close()
            h.close()
        np.testing.assert_array_equal(e.state_hash(), h0)           # a refused resume changes nothing

    def test_park_after_cancel(self):
        e = self.engine()
        e.set_sampling(Sampling(1.0, 0.95, 20, 0.0, 7))
        e.reset()
        started = []

        def on_tokens(ids):
            if not started:
                started.append(len(ids))
                e.cancel()

        r = e.generate(self.ids(1024, 9), 4096, stop_ids=None, on_tokens=on_tokens)
        self.assertEqual(r.finish, "cancelled")
        self.assertGreater(len(r.ids), 0)
        pos = e.position()
        h0 = e.state_hash()
        h = e.park()
        try:
            e.reset()
            e.resume(h)
            self.assertEqual(e.position(), pos)
            np.testing.assert_array_equal(e.state_hash(), h0)
        finally:
            h.close()

    def test_park_position_and_token(self):
        e = self.engine()
        e.set_sampling(Sampling(0.0))
        e.reset()
        p = self.ids(256, 10)
        t = e.generate(p, 5, stop_ids=None).ids
        h = e.park()
        try:
            self.assertEqual(h.position, e.position())
            self.assertEqual(h.last_token, t[-1])                   # d_toks_[0], emitted but not consumed
            e.reset()
            e.resume(h)
            self.assertEqual(e.position(), h.position)
            self.assertEqual(e.live_ids(), h.token_ids)
            nxt = self.ids(1, 11)[0]
            r = e.generate(h.token_ids + [nxt], 3, stop_ids=None)   # continues at pos, no re-prefill
            self.assertEqual(r.snap_mode, 1)
            self.assertEqual(r.cached_tokens, h.position)
        finally:
            h.close()

    def test_park_resume_matches_tokens(self):
        """Compare generated tokens over several steps, not only the state hash."""
        e = self.engine()
        e.set_sampling(Sampling(0.0))
        e.reset()
        p, q, tail = self.ids(512, 20), self.ids(700, 21), self.ids(2, 22)
        t = e.generate(p, 8, stop_ids=None).ids
        h = e.park()
        try:
            base = e.generate(p + t + tail, 6, stop_ids=None).ids
            e.reset()
            e.generate(q, 8, stop_ids=None)
            e.resume(h)
            self.assertEqual(e.generate(p + t + tail, 6, stop_ids=None).ids, base)
        finally:
            h.close()

    def test_resume_staging_paths_agree(self):
        """The staged resume (contiguous HtoD, then a device scatter) and the 3D copies must restore the same state."""
        e = self.engine()
        e.set_sampling(Sampling(0.0))
        e.reset()
        e.generate(self.ids(1024, 26), 6, stop_ids=None)
        h0 = e.state_hash()
        h_on = e.park()
        off = Engine(self.pack, max_ctx=MAX_CTX, profile=RuntimeProfile.default(self.pack, {"YSKN_PARK_STAGE_MB": "0"}))
        h_off = None
        try:
            e.reset()
            e.generate(self.ids(700, 27), 8, stop_ids=None)          # overwrite the KV rows first
            e.resume(h_on)                                           # staging on: chunks through the staging halves
            np.testing.assert_array_equal(e.state_hash(), h0)

            off.resume(h_on)                                         # staging off: the 3D mirror of the same handle
            np.testing.assert_array_equal(off.state_hash(), h0)
            self.assertEqual(off.live_ids(), h_on.token_ids)

            h_off = off.park()                                       # and the park side of staging off
            self.assertEqual(h_off.nbytes, h_on.nbytes)
            off.reset()
            off.resume(h_off)
            np.testing.assert_array_equal(off.state_hash(), h0)
            e.reset()
            e.resume(h_off)                                          # a 3D handle through the staging-on path
            np.testing.assert_array_equal(e.state_hash(), h0)
        finally:
            if h_off is not None:
                h_off.close()
            h_on.close()
            off.close()

    def test_park_resume_after_ring_overwrite(self):
        """A run that overwrote the bdraft rings and the capture buffer must not change the resumed continuation."""
        e = self.engine()
        e.set_sampling(Sampling(0.0))
        e.reset()
        p, q, tail = self.ids(1024, 23), self.ids(1024, 24), self.ids(1, 25)
        t = e.generate(p, 12, stop_ids=None).ids
        h = e.park()
        try:
            base = e.generate(p + t + tail, 5, stop_ids=None).ids
            e.reset()
            e.generate(q, 64, stop_ids=None)                    # overwrites the parked context's slots in the drafter rings
            e.resume(h)
            self.assertEqual(e.generate(p + t + tail, 5, stop_ids=None).ids, base)
        finally:
            h.close()


if __name__ == "__main__":
    unittest.main()
