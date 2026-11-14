# SPDX-License-Identifier: Apache-2.0
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from tests import support
from yasaheni._native import YasakaniError
from yasaheni.runtime import EOS_IDS

EOS = EOS_IDS[0]


class ParkStubTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(0)
        self.e = support.open_engine(Path(self.tmp.name), self.stub)

    def tearDown(self):
        self.e.close()
        self.tmp.cleanup()

    def _generate(self, prompt, script, max_new=50):
        self.stub.push(script)
        return self.e.generate(prompt, max_new)

    def test_park_roundtrip_stub(self):
        self.assertEqual(self.e.live_ids(), [])
        self._generate([5, 6, 7], [11, 12, 13, EOS])
        pos = self.e.position()
        live = self.e.live_ids()
        h0 = self.e.state_hash()
        h = self.e.park()
        self.assertFalse(h.closed)
        self.assertEqual(h.position, pos)
        self.assertEqual(h.token_ids, live)
        self.assertEqual(h.token_ids, [5, 6, 7, 11, 12, 13])
        np.testing.assert_array_equal(self.e.state_hash(), h0)      # parked without changing the engine

        self.e.reset()
        self._generate([9, 9, 9, 9], [21, 22, EOS], max_new=10)
        self.e.resume(h)
        self.assertEqual(self.e.position(), pos)
        np.testing.assert_array_equal(self.e.state_hash(), h0)
        self.assertEqual(self.e.live_ids(), live)

        r = self._generate(h.token_ids + [40, 41], [31, 32, EOS], max_new=10)
        self.assertEqual(r.snap_mode, 1)                            # continued the live state
        self.assertEqual(r.cached_tokens, h.position)

    def test_park_size_stub(self):
        self._generate([5, 6, 7], [11, 12, 13, EOS])
        pos = self.e.position()
        h = self.e.park()
        self.assertEqual(self.e.park_size(pos), h.nbytes)
        self.assertGreater(self.e.park_size(pos + 1000), h.nbytes)

    def test_park_busy(self):
        started = threading.Event()

        def on_tokens(ids):
            started.set()
        self.stub.push([100, 101, 102, 103], loop_from=0)
        self.stub.delay_us(5000)
        box = {}

        def run():
            box["r"] = self.e.generate([5, 6], 1000, on_tokens=on_tokens)

        th = threading.Thread(target=run)
        th.start()
        try:
            self.assertTrue(started.wait(5))
            with self.assertRaisesRegex(YasakaniError, "busy"):
                self.e.park()
        finally:
            th.join()
        self.assertEqual(box["r"].finish, "length")

    def test_park_wrong_engine(self):
        self._generate([5, 6, 7], [11, 12, 13, EOS])
        h = self.e.park()
        other = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=2048, name="other")
        try:
            self.assertNotEqual(other.max_ctx, self.e.max_ctx)
            with self.assertRaisesRegex(YasakaniError, "does not match this engine"):
                other.resume(h)
        finally:
            other.close()
        self.assertFalse(h.closed)                                  # a refused resume leaves the handle live

    def test_park_ids_must_cover_position(self):
        self._generate([5, 6, 7], [11, 12, 13, EOS])
        with self.assertRaisesRegex(YasakaniError, "do not cover"):
            self.e.park([5, 6])
        with self.assertRaisesRegex(YasakaniError, "do not cover"):
            self.e.park([5, 6, 8, 11, 12, 13])

    def test_park_free_twice_and_closed(self):
        self._generate([5, 6, 7], [11, 12, 13, EOS])
        h = self.e.park()
        self.e.park_free(h)
        self.assertTrue(h.closed)
        self.e.park_free(h)                                         # freeing again does nothing
        h2 = self.e.park()
        h2.close()
        self.assertTrue(h2.closed)
        with self.assertRaisesRegex(YasakaniError, "closed"):
            self.e.resume(h2)
        h3 = self.e.park()
        self.e.close()
        self.assertTrue(h3.closed)                                  # closing the engine releases its handles

    def test_park_free_foreign_handle(self):
        self._generate([5, 6, 7], [11, 12, 13, EOS])
        h = self.e.park()
        other = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=2048, name="other")
        try:
            with self.assertRaisesRegex(YasakaniError, "another engine"):
                other.park_free(h)
        finally:
            other.close()
        self.assertFalse(h.closed)


if __name__ == "__main__":
    unittest.main()
