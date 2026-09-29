#!/usr/bin/env python3
"""Load the current K–Pg cinema into a Blender window.

    .venv/bin/python animate/scenes/blender/sync_kpg_preview.py

Writes the job, then opens Blender (not --background) and rebuilds the scene.
Does not render the GIF. Scrub the timeline after it loads.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PREVIEW_DIR_ENV = 'SOLSYS_KPG_PREVIEW_DIR'
DUMMY_PNG = bytes.fromhex(
    '89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de'
    '0000000c4944415408d763f8cf0000010101005f9d4e0e0000000049454e44ae426082'
)
PREVIEW_FRAME = 500


def _previewRoot() -> Path:
    existing = os.environ.get(PREVIEW_DIR_ENV)
    if existing:
        root = Path(existing)
        root.mkdir(parents=True, exist_ok=True)
        return root
    root = Path(tempfile.mkdtemp(prefix='solsys_kpg_'))
    os.chmod(root, 0o700)
    os.environ[PREVIEW_DIR_ENV] = str(root)
    return root


def _jobPath() -> Path:
    return _previewRoot() / 'earth_kpg_dark_job.json'


def _previewFrames() -> Path:
    return _previewRoot() / 'frames'


def _writeJob() -> None:
    os.environ.setdefault('MPLCONFIGDIR', str(_previewRoot() / 'mpl'))
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from animate.scenes.blender.flyby_scene import writeFlybyJob
    from animate.scenes.kpg_cinematic import (
        DEFAULT_EVENT_CSV,
        buildImpactSamples,
        buildKpgJob,
        loadImpactEvent,
    )

    event = loadImpactEvent(REPO / DEFAULT_EVENT_CSV)
    frames = _previewFrames()
    frames.mkdir(parents=True, exist_ok=True)
    job = buildKpgJob(event, buildImpactSamples(event), theme='dark', framesDirectory=frames)
    jobPath = _jobPath()
    writeFlybyJob(job, jobPath)
    print(f'Wrote {jobPath} ({len(job["frames"])} frames)', flush=True)


def _applyInBlender() -> None:
    import importlib.util

    import bpy  # type: ignore[import-not-found]

    script = REPO / 'animate' / 'scenes' / 'blender' / 'render_kpg.py'
    spec = importlib.util.spec_from_file_location('solsys_render_kpg', script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load {script}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frames = _previewFrames()
    frames.mkdir(parents=True, exist_ok=True)
    job = module.loadKpgJob(_jobPath())
    job['outputDirectory'] = str(frames)
    (frames / f'frame_{PREVIEW_FRAME:04d}.png').write_bytes(DUMMY_PNG)
    module.applyKpgJobInBlender(job, render=False)
    bpy.context.scene.frame_set(PREVIEW_FRAME)
    print(
        f'Synced Blender  frame={PREVIEW_FRAME}  end={int(bpy.context.scene.frame_end)}',
        flush=True,
    )


def _blenderExecutable() -> str:
    found = shutil.which('blender')
    if found:
        return found
    mac = Path('/Applications/Blender.app/Contents/MacOS/Blender')
    if mac.is_file():
        return str(mac)
    raise RuntimeError('blender not found on PATH')


def main() -> int:
    try:
        import bpy  # type: ignore[import-not-found]
    except ImportError:
        bpy = None
    if bpy is not None:
        _applyInBlender()
        return 0
    _writeJob()
    blender = _blenderExecutable()
    os.execv(blender, [blender, '--python', str(Path(__file__).resolve())])
    raise RuntimeError(f'Failed to exec {blender}')


if __name__ == '__main__':
    raise SystemExit(main())
