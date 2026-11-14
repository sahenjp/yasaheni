# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
import unittest
from pathlib import Path

from tests import support                                   # noqa: F401
from yasaheni import runtime as R

# the generation call of the published configuration, as the integers the engine reads (44 values)
GOLDEN_SPEC = [7, 1, 1, 2, 8, 3, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 15, 6, 1, 248046, 248044, 8, 0, 0, 0, 0, 0, 0, 0, 0, 300, -1, 0, 0, 0,
               1, 4, 1, 4, 15, 1, 1, 248045]


class RuntimeTests(unittest.TestCase):
    def test_spec_vector(self):
        self.assertEqual(R.spec_array(R.DEFAULT_SPEC).tolist(), GOLDEN_SPEC)
        self.assertEqual(len(R.SPEC_FIELDS), 44)

    def test_unknown_spec_key(self):
        with self.assertRaises(ValueError):
            R.spec_array(dict(R.DEFAULT_SPEC, bogus=1))

    def test_env_has_seventeen_settings(self):
        with tempfile.TemporaryDirectory() as t:
            pack = support.make_pack(Path(t))
            prof = R.RuntimeProfile.default(pack)
        self.assertEqual(len(prof.env), 17)
        self.assertEqual(set(R.PATH_ENV) - set(prof.env), set())
        self.assertTrue(all(k.startswith("YSKN_") for k in prof.env))

    def test_apply_removes_stale_settings(self):
        env = {"YSKN_Q35_PMAX": "7", "YSKN_SOMETHING_ELSE": "1", "PATH": "/bin"}
        with tempfile.TemporaryDirectory() as t:
            prof = R.RuntimeProfile.default(support.make_pack(Path(t)))
        removed = prof.apply(env)
        self.assertEqual(removed, ["YSKN_SOMETHING_ELSE"])
        self.assertEqual(env["YSKN_Q35_PMAX"], "1024")
        self.assertEqual(env["PATH"], "/bin")

    def test_apply_keeps_tuning_settings(self):
        env = {"YSKN_LNO_CFG": "1", "YSKN_LNO_CFG_TAB": "0,1,1,1,1,1,1,1,1", "YSKN_BDR_CFG": "2",
               "YSKN_SOMETHING_ELSE": "1", "PATH": "/bin"}
        with tempfile.TemporaryDirectory() as t:
            prof = R.RuntimeProfile.default(support.make_pack(Path(t)))
        removed = prof.apply(env)
        self.assertEqual(removed, ["YSKN_SOMETHING_ELSE"])
        self.assertEqual(prof.kept_env, ["YSKN_LNO_CFG", "YSKN_LNO_CFG_TAB", "YSKN_BDR_CFG"])
        self.assertEqual(env["YSKN_LNO_CFG"], "1")
        self.assertEqual(env["YSKN_LNO_CFG_TAB"], "0,1,1,1,1,1,1,1,1")
        self.assertEqual(env["YSKN_BDR_CFG"], "2")
        self.assertEqual(env["PATH"], "/bin")

    def test_profile_tuning_value_wins_over_shell(self):
        env = {"YSKN_LNO_CFG": "2", "YSKN_LNO_CFG_TAB": "0,2,2,2,2,2,2,2,2", "PATH": "/bin"}
        with tempfile.TemporaryDirectory() as t:
            prof = R.RuntimeProfile.default(support.make_pack(Path(t)),
                                            {"YSKN_LNO_CFG": "1", "YSKN_LNO_CFG_TAB": "0,1,1,1,1,1,1,1,1"})
        removed = prof.apply(env)
        self.assertEqual(removed, [])
        self.assertEqual(prof.kept_env, [])
        self.assertEqual(env["YSKN_LNO_CFG"], "1")
        self.assertEqual(env["YSKN_LNO_CFG_TAB"], "0,1,1,1,1,1,1,1,1")
        self.assertNotIn("YSKN_BDR_CFG", env)

    def test_override_must_be_engine_setting(self):
        with tempfile.TemporaryDirectory() as t:
            pack = support.make_pack(Path(t))
            with self.assertRaises(ValueError):
                R.RuntimeProfile.default(pack, {"PATH": "x"})
            prof = R.RuntimeProfile.default(pack, {"YSKN_Q35_PMAX": 512})
        self.assertEqual(prof.env["YSKN_Q35_PMAX"], "512")

    def test_model_pack_missing_files(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "m"
            root.mkdir()
            (root / "qwen38_27b_e3x.yskn").write_bytes(b"x")
            with self.assertRaises(Exception) as cm:
                R.ModelPack.load(root)
            self.assertIn("missing", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
