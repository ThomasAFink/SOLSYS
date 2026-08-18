"""Blender renderer for the K–Pg Earth-impact job.

Host writes a kpg-job JSON; this script (stdlib + bpy) places Earth, a true-scale
impactor, a diving camera and a schematic flash/veil::

    blender --background --python animate/scenes/blender/render_kpg.py -- \\
        output/animate/blender/planets/earth/earth_kpg_dark_job.json
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

JOB_SCHEMA_ID = 'solsys.blender_kpg_job/v1'


def _argvAfterDoubleDash(argv: list[str]) -> list[str]:
    if '--' in argv:
        return argv[argv.index('--') + 1 :]
    return argv[1:]


def _flybyModule() -> ModuleType:
    """Load sibling render_flyby.py by path so Blender does not need the venv."""
    path = Path(__file__).resolve().with_name('render_flyby.py')
    spec = importlib.util.spec_from_file_location('solsys_render_flyby', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loadKpgJob(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema') != JOB_SCHEMA_ID:
        raise ValueError(
            f'Unsupported K–Pg job schema: {payload.get("schema")!r} (expected {JOB_SCHEMA_ID!r})'
        )
    for key in ('theme', 'body', 'impactor', 'frames', 'outputDirectory', 'resolution'):
        if key not in payload:
            raise ValueError(f'K–Pg job missing required key: {key}')
    if not payload['frames']:
        raise ValueError('K–Pg job must include frames')
    return payload


def _keyLocation(obj: Any, location: tuple[float, float, float], frame: int) -> None:
    obj.location = location
    obj.keyframe_insert(data_path='location', frame=frame)


def _softenCloseupTextures(material: Any) -> None:
    """Flyby uses Closest sampling to hide the date line; the dive needs filtering."""
    nodeTree = getattr(material, 'node_tree', None)
    if nodeTree is None:
        return
    for node in nodeTree.nodes:
        if getattr(node, 'type', '') == 'TEX_IMAGE' and hasattr(node, 'interpolation'):
            node.interpolation = 'Cubic'


def _buildEarth(bpy: Any, flyby: ModuleType, job: dict[str, Any]) -> tuple[Any, float]:
    body = job['body']
    radius = float(body['displayRadiusAu'])
    appearance = job.get('appearance') if isinstance(job.get('appearance'), dict) else None
    earth = flyby._createBodySphere(bpy, 'Earth', radius)
    material = bpy.data.materials.new(name='EarthKpgMaterial')
    flyby._applyBodyMaterial(
        bpy,
        material,
        color=[float(channel) for channel in body['colorRgba']],
        appearance=appearance,
        theme=str(job['theme']),
        bodyKind='planet',
    )
    _softenCloseupTextures(material)
    if earth.data.materials:
        earth.data.materials[0] = material
    else:
        earth.data.materials.append(material)
    flyby._attachAppearanceExtras(
        bpy, earth, radius=radius, appearance=appearance, theme=str(job['theme'])
    )
    return earth, radius


def _buildImpactor(bpy: Any, flyby: ModuleType, job: dict[str, Any], earthRadius: float) -> Any:
    scale = float(job['impactor']['radiusScale'])
    impactor = flyby._createBodySphere(
        bpy, 'Impactor', max(earthRadius * scale, earthRadius * 1e-4)
    )
    rock = bpy.data.materials.new(name='ImpactorRock')
    nodeTree = getattr(rock, 'node_tree', None)
    if nodeTree is None:
        rock.use_nodes = True
        nodeTree = rock.node_tree
    principled = next(node for node in nodeTree.nodes if node.type == 'BSDF_PRINCIPLED')
    color = [float(channel) for channel in job['impactor']['colorRgba']]
    principled.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1.0)
    if 'Roughness' in principled.inputs:
        principled.inputs['Roughness'].default_value = 0.88
    impactor.data.materials.append(rock)
    return impactor


def _keyframeShot(
    camera: Any,
    lookAt: Any,
    impactor: Any,
    flash: Any,
    lightData: Any,
    flashData: Any,
    sunEnergy: float,
    frames: list[dict[str, Any]],
) -> None:
    for sample in frames:
        frame = int(sample['frame'])
        _keyLocation(camera, tuple(float(value) for value in sample['cameraAu']), frame)
        _keyLocation(lookAt, tuple(float(value) for value in sample['lookAtAu']), frame)
        _keyLocation(impactor, tuple(float(value) for value in sample['impactorAu']), frame)
        _keyLocation(flash, tuple(float(value) for value in sample['impactorAu']), frame)
        lightData.energy = sunEnergy * float(sample['sunScale'])
        lightData.keyframe_insert(data_path='energy', frame=frame)
        flashData.energy = 8.0 * float(sample['flashScale'])
        flashData.keyframe_insert(data_path='energy', frame=frame)


def applyKpgJobInBlender(job: dict[str, Any]) -> Path:
    import bpy  # type: ignore[import-not-found]

    flyby = _flybyModule()
    theme = str(job['theme'])
    frames = job['frames']
    outputDirectory = Path(job['outputDirectory'])
    outputDirectory.mkdir(parents=True, exist_ok=True)

    flyby._clearSceneObjects(bpy)
    earth, radius = _buildEarth(bpy, flyby, job)
    impactor = _buildImpactor(bpy, flyby, job, radius)

    lookAt = bpy.data.objects.new('KpgLookAt', None)
    bpy.context.scene.collection.objects.link(lookAt)

    scene = bpy.context.scene
    scene.frame_start = int(frames[0]['frame'])
    scene.frame_end = int(frames[-1]['frame'])
    scene.frame_current = scene.frame_start

    cameraData = bpy.data.cameras.new('KpgCameraData')
    cameraData.lens = 50
    cameraData.clip_start = max(radius * 0.004, 1e-6)
    cameraData.clip_end = max(radius * 120.0, 10.0)
    camera = bpy.data.objects.new('KpgCamera', cameraData)
    scene.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new(type='TRACK_TO')
    track.target = lookAt
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    sunEnergy = 2.8 if theme == 'dark' else 4.2
    lightData = bpy.data.lights.new('KpgSun', type='SUN')
    lightData.energy = sunEnergy
    lightData.angle = math.radians(4.0)
    sun = bpy.data.objects.new('KpgSun', lightData)
    strike = max(frames, key=lambda sample: float(sample['flashScale']))
    site = [float(value) for value in strike['impactorAu']]
    siteLength = math.sqrt(sum(component * component for component in site)) or 1.0
    sun.location = tuple(component / siteLength * radius * 16.0 for component in site)
    scene.collection.objects.link(sun)
    sunTrack = sun.constraints.new(type='TRACK_TO')
    sunTrack.target = earth
    sunTrack.track_axis = 'TRACK_NEGATIVE_Z'
    sunTrack.up_axis = 'UP_Y'

    flashData = bpy.data.lights.new('KpgFlash', type='POINT')
    flashData.energy = 0.0
    flash = bpy.data.objects.new('KpgFlash', flashData)
    scene.collection.objects.link(flash)
    _keyframeShot(camera, lookAt, impactor, flash, lightData, flashData, sunEnergy, frames)

    flyby._configureWorld(bpy, theme)
    flyby._configureFlybyRenderer(
        scene,
        ringsEnabled=False,
        resolution=int(job['resolution']),
        fps=int(job.get('fps', 20)),
        filmTransparent=bool(job.get('filmTransparent', False)),
        theme=theme,
        outputDirectory=outputDirectory,
        isStar=False,
    )
    bpy.ops.render.render(animation=True)
    written = sorted(outputDirectory.glob('frame_*.png'))
    if not written:
        raise RuntimeError(f'Blender produced no PNG frames in {outputDirectory}')
    return outputDirectory


def main(argv: list[str] | None = None) -> int:
    args = _argvAfterDoubleDash(list(argv if argv is not None else sys.argv))
    if not args:
        print(
            'Usage: render_kpg.py <kpg_job.json>\n'
            '   or: blender --background --python render_kpg.py -- <kpg_job.json>',
            file=sys.stderr,
        )
        return 2
    flyby = _flybyModule()
    job = loadKpgJob(Path(args[0]))
    print(
        f'K–Pg job: theme={job["theme"]} frames={len(job["frames"])} resolution={job["resolution"]}'
    )
    if not flyby._bpyAvailable():
        print('bpy not available — dry-run validation only.')
        return 0
    outputDirectory = applyKpgJobInBlender(job)
    print(f'Rendered PNG frames → {outputDirectory}')
    return 0


if __name__ == '__main__':
    flyby = _flybyModule()
    exitCode = main()
    if not flyby._bpyAvailable():
        raise SystemExit(exitCode)
    if '--background' in sys.argv or '-b' in sys.argv:
        raise SystemExit(exitCode)
