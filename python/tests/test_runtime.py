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
        env = {"YSKN_Q35_PMAX": "7", "YSKN_SOMETHING_ELSE": "1", "YSKN_PARK_REUSE": "0", "PATH": "/bin"}
        with tempfile.TemporaryDirectory() as t:
            prof = R.RuntimeProfile.default(support.make_pack(Path(t)))
        removed = prof.apply(env)
        self.assertEqual(removed, ["YSKN_SOMETHING_ELSE"])
        self.assertEqual(env["YSKN_Q35_PMAX"], "1024")
        self.assertEqual(env["YSKN_PARK_REUSE"], "0")
        self.assertEqual(env["PATH"], "/bin")

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

    @staticmethod
    def _manifest_folder(tmp, names, manifest):
        root = Path(tmp) / "m"
        root.mkdir()
        for f in list(names) + ["bdraft2_k8.yskn", "bdraft2_k8.shead", "kvq_u5.bin", "tokenizer.json", "tokenizer_config.json"]:
            (root / f).write_bytes(b"x")
        (root / manifest[0]).write_text(manifest[1], encoding="utf-8")
        return root

    def test_yasakani_manifest_is_read(self):
        with tempfile.TemporaryDirectory() as t:
            root = self._manifest_folder(t, ["qwen38_27b_quality.yskn"],
                                         ("yasakani.json", '{"files": {"weights": "qwen38_27b_quality.yskn"}, "max_ctx": 131072}'))
            pack = R.ModelPack.load(root)
        self.assertEqual(pack.weights.name, "qwen38_27b_quality.yskn")
        self.assertEqual(pack.max_ctx, 131072)

    def test_yasaheni_manifest_wins_when_both_exist(self):
        with tempfile.TemporaryDirectory() as t:
            root = self._manifest_folder(t, ["qwen38_27b_quality.yskn", "qwen38_27b_speed.yskn"],
                                         ("yasakani.json", '{"files": {"weights": "qwen38_27b_quality.yskn"}}'))
            (root / "yasaheni.json").write_text('{"files": {"weights": "qwen38_27b_speed.yskn"}}', encoding="utf-8")
            pack = R.ModelPack.load(root)
        self.assertEqual(pack.weights.name, "qwen38_27b_speed.yskn")


if __name__ == "__main__":
    unittest.main()
