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
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
JOB_PATH = Path('/tmp/earth_kpg_dark_job.json')
PREVIEW = Path('/tmp/solsys_kpg_preview')
DUMMY_PNG = bytes.fromhex(
    '89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de'
    '0000000c4944415408d763f8cf0000010101005f9d4e0e0000000049454e44ae426082'
)
PREVIEW_FRAME = 500


def _writeJob() -> None:
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-solsys')
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
    PREVIEW.mkdir(parents=True, exist_ok=True)
    job = buildKpgJob(event, buildImpactSamples(event), theme='dark', framesDirectory=PREVIEW)
    writeFlybyJob(job, JOB_PATH)
    print(f'Wrote {JOB_PATH} ({len(job["frames"])} frames)', flush=True)


def _applyInBlender() -> None:
    import importlib.util

    import bpy  # type: ignore[import-not-found]

    bpy.app.handlers.frame_change_pre.clear()
    script = REPO / 'animate' / 'scenes' / 'blender' / 'render_kpg.py'
    spec = importlib.util.spec_from_file_location('solsys_render_kpg', script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load {script}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    job = module.loadKpgJob(JOB_PATH)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    job['outputDirectory'] = str(PREVIEW)
    job['stillFrames'] = [PREVIEW_FRAME]
    (PREVIEW / f'frame_{PREVIEW_FRAME:04d}.png').write_bytes(DUMMY_PNG)
    original = bpy.ops.render.render

    def _skip(*_args: object, **_kwargs: object) -> set[str]:
        return {'FINISHED'}

    bpy.ops.render.render = _skip
    try:
        module.applyKpgJobInBlender(job)
    finally:
        bpy.ops.render.render = original
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


if __name__ == '__main__':
    raise SystemExit(main())
