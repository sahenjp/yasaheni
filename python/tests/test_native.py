# SPDX-License-Identifier: Apache-2.0
"""Library discovery: a build's ``bin`` folder comes before the pre-83c19d5 folder that may hold a stale library."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from yasaheni import _native


@unittest.skipIf(sys.platform == "win32", "the build-folder candidates are Linux only")
class CandidateOrderTests(unittest.TestCase):
    def test_the_bin_folder_comes_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "pkg" / "yasaheni"
            pkg.mkdir(parents=True)
            for build in ("build-linux", "build"):
                (root / build / "bin").mkdir(parents=True)
                (root / build / "bin" / _native._LIB_NAME).write_bytes(b"")
                (root / build / _native._LIB_NAME).write_bytes(b"")
            with mock.patch.object(_native, "__file__", str(pkg / "_native.py")), \
                    mock.patch.dict(os.environ, {"YASAKANI_DLL": ""}):
                paths = list(_native._candidates(None))
                for build in ("build-linux", "build"):
                    new = root / build / "bin" / _native._LIB_NAME
                    old = root / build / _native._LIB_NAME
                    self.assertLess(paths.index(new), paths.index(old), "%s/bin must come before %s" % (build, build))
                self.assertEqual(_native.find_library(), root / "build-linux" / "bin" / _native._LIB_NAME)


if __name__ == "__main__":
    unittest.main()
