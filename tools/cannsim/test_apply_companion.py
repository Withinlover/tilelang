"""The companion installer must not overwrite a user's helper checkout."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from apply_companion import apply_companion


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


class CompanionInstallerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "helper"
        self.bundle = Path(temporary.name) / "bundle"
        self.root.mkdir()
        self.bundle.mkdir()
        git(self.root, "init", "-q")
        (self.root / "original").write_text("preserve this\n")
        git(self.root, "add", "original")
        git(self.root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "base")
        self.base = git(self.root, "rev-parse", "HEAD")
        content = b"companion addition\n"
        patch = (
            b"diff --git a/added.txt b/added.txt\nnew file mode 100644\n"
            b"--- /dev/null\n+++ b/added.txt\n@@ -0,0 +1 @@\n+companion addition\n"
        )
        (self.bundle / "p0_companion.patch").write_bytes(patch)
        self.manifest = {
            "base_revision": self.base,
            "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "files": {"added.txt": {"sha256": hashlib.sha256(content).hexdigest()}},
        }
        self.write_manifest()

    def write_manifest(self):
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest))

    def test_check_apply_verify_preserves_base(self):
        apply_companion(self.root, self.bundle, check=True)
        self.assertFalse((self.root / "added.txt").exists())
        apply_companion(self.root, self.bundle)
        apply_companion(self.root, self.bundle, verify=True)
        self.assertEqual(git(self.root, "diff", "HEAD"), "")
        self.assertEqual(git(self.root, "rev-parse", "HEAD"), self.base)
        (self.root / "added.txt").write_text("modified by user\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            apply_companion(self.root, self.bundle, verify=True)

    def test_dirty_checkout_refused_without_modification(self):
        for name in ("original", "untracked"):
            with self.subTest(name=name):
                path = self.root / name
                before = path.read_bytes() if path.exists() else None
                path.write_text("user work\n")
                with self.assertRaisesRegex(ValueError, "dirty"):
                    apply_companion(self.root, self.bundle)
                self.assertEqual(path.read_text(), "user work\n")
                self.assertFalse((self.root / "added.txt").exists())
                if before is None:
                    path.unlink()
                else:
                    path.write_bytes(before)

    def test_verify_rejects_modified_upstream_but_allows_untracked_logs(self):
        apply_companion(self.root, self.bundle)
        (self.root / "record.log").write_text("local diagnostics\n")
        apply_companion(self.root, self.bundle, verify=True)
        (self.root / "original").write_text("modified upstream\n")
        with self.assertRaisesRegex(ValueError, "upstream tracked files changed"):
            apply_companion(self.root, self.bundle, verify=True)

    def test_wrong_revision_refused(self):
        self.manifest["base_revision"] = "0" * 40
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "HEAD must equal"):
            apply_companion(self.root, self.bundle)
        self.assertFalse((self.root / "added.txt").exists())

    def test_patch_tampering_refused(self):
        with (self.bundle / "p0_companion.patch").open("a") as stream:
            stream.write("unexpected change\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            apply_companion(self.root, self.bundle)

    def test_repeat_application_refused(self):
        apply_companion(self.root, self.bundle)
        with self.assertRaisesRegex(ValueError, "dirty"):
            apply_companion(self.root, self.bundle)
        self.assertEqual((self.root / "added.txt").read_text(), "companion addition\n")

    def test_manifest_file_mismatch_refused(self):
        self.manifest["files"] = {"different.txt": {"sha256": "0" * 64}}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "file list"):
            apply_companion(self.root, self.bundle)
        self.assertFalse((self.root / "added.txt").exists())

    def test_ignored_existing_path_refused(self):
        (self.root / ".git" / "info" / "exclude").write_text("added.txt\n")
        (self.root / "added.txt").write_text("ignored user work\n")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            apply_companion(self.root, self.bundle)
        self.assertEqual((self.root / "added.txt").read_text(), "ignored user work\n")


if __name__ == "__main__":
    unittest.main()
