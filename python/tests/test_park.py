# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

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
        self.assertEqual(h.last_token, EOS)                     # emitted, not yet consumed
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

    def test_park_resume_busy(self):
        self._generate([5, 6, 7], [11, 12, 13, EOS])
        h = self.e.park()
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
                self.e.resume(h)
        finally:
            th.join()
            h.close()
        self.assertEqual(box["r"].finish, "length")

    def test_park_size_negative(self):
        with self.assertRaisesRegex(YasakaniError, "out of range"):
            self.e.park_size(-1)

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


class ParkStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(0)
        self.e = support.open_engine(Path(self.tmp.name), self.stub)
        from yasaheni.parking import ParkStore
        self.store = ParkStore(self.e, 1 << 30)

    def tearDown(self):
        self.store.clear()
        self.e.close()
        self.tmp.cleanup()

    def _state(self, prompt, script, conversation_id=None):
        self.stub.push(script)
        self.e.generate(prompt, 20)
        self.store.note_live(conversation_id)

    def test_store_prefers_longest_prefix(self):
        self._state([5, 6, 7], [31])
        short = self.store.keep_live()
        self.assertIsNotNone(short)
        self._state([5, 6, 7, 31, 8], [41])
        long = self.store.keep_live()
        self.assertIsNotNone(long)
        self.assertEqual(short.token_ids, [5, 6, 7, 31])
        self.assertEqual(long.token_ids, [5, 6, 7, 31, 8, 41])
        self.assertIs(self.store.find([5, 6, 7, 31, 8, 41, 9]), long)   # both are prefixes; the longer one wins
        self.assertEqual(self.store.find([9, 9]), None)
        self.assertEqual(self.store.find([5, 6, 7, 31]), None)      # equal length is not an extension

    def test_store_conversation_id_preferred(self):
        self._state([5, 6, 7], [31], conversation_id="a")
        ha = self.store.keep_live()
        self.assertIsNotNone(ha)
        self._state([5, 6, 7, 31, 8], [41], conversation_id="b")
        hb = self.store.keep_live()
        prompt = [5, 6, 7, 31, 8, 41, 9]
        self.assertIs(self.store.find(prompt), hb)              # ids alone pick the longer prefix
        self.assertIs(self.store.find(prompt, "a"), ha)         # the id wins over the prefix when it matches
        self.assertIs(self.store.find(prompt, "nope"), hb)      # an unknown id falls back to the prefix

    def test_switch_context_modes(self):
        from yasaheni.server import switch_context
        self._state([5, 6, 7], [31, 32])
        live = self.e.live_ids()
        switch_context(self.e, self.store, live + [99])         # the live state already continues the prompt
        self.assertEqual(len(self.store), 0)
        self.assertEqual(self.e.position(), len(live))
        switch_context(self.e, self.store, [1, 2, 3])           # a different prompt parks the live state
        self.assertEqual(len(self.store), 1)
        self.e.reset()
        switch_context(self.e, self.store, live + [77])         # the parked state is resumed
        self.assertEqual(self.e.position(), len(live))
        self.e.reset()
        with mock.patch.object(self.e, "resume", side_effect=YasakaniError("resume: handle does not match this engine")):
            with contextlib.redirect_stderr(io.StringIO()):
                switch_context(self.e, self.store, [1, 2, 3])
        self.assertEqual(self.e.position(), 0)                  # a failed resume falls back to a fresh prefill

    def test_store_lru_eviction(self):
        self._state([5, 6, 7], [31], conversation_id="a")
        ha = self.store.keep_live()
        self._state([9, 9, 9], [41], conversation_id="b")
        hb = self.store.keep_live()
        self.assertEqual(len(self.store), 2)
        size = ha.nbytes
        self.assertEqual(hb.nbytes, size)
        self.store.budget = 2 * size + size // 2               # room for two handles, not three
        self._state([8, 8, 8], [51], conversation_id="c")
        hc = self.store.keep_live()
        self.assertIsNotNone(hc)
        self.assertEqual(len(self.store), 2)
        self.assertTrue(ha.closed)                              # the least recently used handle was freed
        self.assertFalse(hb.closed)
        self.assertLessEqual(self.store.used, self.store.budget)

    def test_store_oversized_handle_is_not_parked(self):
        self._state([5, 6, 7], [31])
        self.store.budget = 1
        with contextlib.redirect_stderr(io.StringIO()) as log:
            self.assertIsNone(self.store.keep())
        self.assertIn("does not fit", log.getvalue())
        self.assertEqual(len(self.store), 0)
        self.assertEqual(self.store.used, 0)
        self.assertEqual(self.e.position(), 4)                  # the engine was not touched


class ParkBudgetTests(unittest.TestCase):
    def test_park_budget_parsing(self):
        from yasaheni.parking import DEFAULT_PARK_MB, park_budget_bytes
        mib = 1024 * 1024
        cases = [("3", 3 * mib), ("0", 0), ("-5", 0), ("not a number", DEFAULT_PARK_MB * mib), ("", DEFAULT_PARK_MB * mib)]
        for value, want in cases:
            with self.subTest(value=value), mock.patch.dict("os.environ", {"YSKN_PARK_MB": value}):
                self.assertEqual(park_budget_bytes(), want)


if __name__ == "__main__":
    unittest.main()
