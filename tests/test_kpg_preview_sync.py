"""Preview-sync helpers for the K–Pg cinema."""

from __future__ import annotations

import inspect
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from animate.scenes.blender.render_kpg import (
    _ejectaTailShape,
    _replaceKpgFrameHandler,
    applyKpgJobInBlender,
)
from animate.scenes.blender.sync_kpg_preview import (
    PREVIEW_DIR_ENV,
    _previewFrames,
    _previewRoot,
)


class PreviewRootTests(unittest.TestCase):
    def test_preview_root_reuses_the_env_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix='solsys_kpg_test_') as raw:
            root = Path(raw)
            with patch.dict(os.environ, {PREVIEW_DIR_ENV: str(root)}, clear=False):
                self.assertEqual(_previewRoot(), root)
                self.assertEqual(_previewFrames(), root / 'frames')

    def test_preview_root_creates_a_private_directory(self) -> None:
        previous = os.environ.pop(PREVIEW_DIR_ENV, None)
        try:
            root = _previewRoot()
            self.assertTrue(root.is_dir())
            self.assertEqual(os.environ[PREVIEW_DIR_ENV], str(root))
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
        finally:
            created = os.environ.get(PREVIEW_DIR_ENV)
            if previous is None:
                os.environ.pop(PREVIEW_DIR_ENV, None)
            else:
                os.environ[PREVIEW_DIR_ENV] = previous
            if created:
                leftover = Path(created)
                if leftover.is_dir() and leftover.name.startswith('solsys_kpg_'):
                    leftover.rmdir()


class PreviewApplyTests(unittest.TestCase):
    def test_ejecta_tails_are_not_one_length(self) -> None:
        depths = [_ejectaTailShape(index, 1.0)[2] for index in range(480)]
        self.assertGreater(max(depths) / max(min(depths), 1e-9), 4.0)
        self.assertGreater(len({round(depth, 4) for depth in depths}), 40)

    def test_apply_job_skips_render_when_asked(self) -> None:
        params = inspect.signature(applyKpgJobInBlender).parameters
        self.assertIn('render', params)
        self.assertTrue(params['render'].default)

    def test_frame_handler_replaces_only_marked_kpg_callbacks(self) -> None:
        def other(_scene: object) -> None:
            return None

        def old(_scene: object) -> None:
            return None

        def new(_scene: object) -> None:
            return None

        bag: list[object] = [other]
        bpy = SimpleNamespace(app=SimpleNamespace(handlers=SimpleNamespace(frame_change_pre=bag)))
        _replaceKpgFrameHandler(bpy, old)
        _replaceKpgFrameHandler(bpy, new)
        self.assertEqual(bag, [other, new])
        self.assertTrue(new._solsys_kpg_frame)


if __name__ == '__main__':
    unittest.main()
