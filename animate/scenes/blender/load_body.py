"""Blender ingest for SOLSYS body-scene JSON.

Runs two ways:

1. Host dry-run (no bpy)::

       python animate/scenes/blender/load_body.py \\
           output/animate/blender/planets/earth/earth_body_scene.json

2. Inside Blender::

       blender --background --python animate/scenes/blender/load_body.py -- \\
           output/animate/blender/planets/earth/earth_body_scene.json

Keep this module stdlib-only so Blender's bundled Python can import it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_ID = 'solsys.blender_body_scene/v1'


def _bpyAvailable() -> bool:
    """True when running inside Blender (avoid unused ``import bpy`` for CodeQL)."""
    return importlib.util.find_spec('bpy') is not None


def _argvAfterDoubleDash(argv: list[str]) -> list[str]:
    if '--' in argv:
        return argv[argv.index('--') + 1 :]
    # Host dry-run: script path is argv[0].
    return argv[1:]


def loadPayload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema') != SCHEMA_ID:
        raise ValueError(
            f'Unsupported body scene schema: {payload.get("schema")!r} (expected {SCHEMA_ID!r})'
        )
    if 'body' not in payload or 'keyframes' not in payload:
        raise ValueError('Body scene JSON must include body and keyframes')
    if not payload['keyframes']:
        raise ValueError('Body scene JSON must include at least one keyframe')
    return payload


def summarizePayload(payload: dict[str, Any]) -> str:
    body = payload['body']
    keyframes = payload['keyframes']
    return (
        f'Loaded {body["name"]} ({body["kind"]}, system={body["systemId"]}) '
        f'with {len(keyframes)} keyframes; '
        f'displayRadiusAu={body["displayRadiusAu"]:.4f}; '
        f'cameraHintDistanceAu={payload["cameraHintDistanceAu"]:.4f}'
    )


def _clearSceneObjects(bpy: Any) -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _createUvSphere(bpy: Any, name: str, radius: float) -> Any:
    """Build a UV sphere via bmesh (works in GUI and --background; no active_object)."""
    import bmesh  # type: ignore[import-not-found]

    mesh = bpy.data.meshes.new(f'{name}Mesh')
    builder = bmesh.new()
    bmesh.ops.create_uvsphere(builder, u_segments=32, v_segments=16, radius=radius)
    builder.to_mesh(mesh)
    builder.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def applyPayloadInBlender(payload: dict[str, Any]) -> str:
    """Create a UV sphere, material, and location keyframes from the payload."""
    import bpy  # type: ignore[import-not-found]

    body = payload['body']
    keyframes = payload['keyframes']
    name = str(body['name'])
    radius = float(body['displayRadiusAu'])
    color = [float(channel) for channel in body['colorRgba']]

    # Avoid wm.read_factory_settings / ops that need a VIEW_3D active object —
    # those fail when the script runs at GUI startup in Blender 5.x.
    _clearSceneObjects(bpy)
    obj = _createUvSphere(bpy, name, radius)

    material = bpy.data.materials.new(name=f'{name}Material')
    nodeTree = getattr(material, 'node_tree', None)
    if nodeTree is not None:
        principled = nodeTree.nodes.get('Principled BSDF')
        if principled is not None:
            principled.inputs['Base Color'].default_value = color
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

    scene = bpy.context.scene
    scene.frame_start = int(keyframes[0]['frame'])
    scene.frame_end = int(keyframes[-1]['frame'])
    scene.frame_current = int(keyframes[0]['frame'])

    for keyframe in keyframes:
        frame = int(keyframe['frame'])
        position = keyframe['positionAu']
        obj.location = (float(position[0]), float(position[1]), float(position[2]))
        obj.keyframe_insert(data_path='location', frame=frame)

    first = keyframes[0]['positionAu']
    distance = float(payload['cameraHintDistanceAu'])
    cameraData = bpy.data.cameras.new(f'{name}CameraData')
    camera = bpy.data.objects.new(f'{name}Camera', cameraData)
    camera.location = (
        float(first[0]) + distance,
        float(first[1]) - distance,
        float(first[2]) + distance * 0.6,
    )
    scene.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new(type='TRACK_TO')
    track.target = obj
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    lightData = bpy.data.lights.new(f'{name}Sun', type='SUN')
    light = bpy.data.objects.new(f'{name}Sun', lightData)
    light.location = (float(first[0]), float(first[1]) - distance, float(first[2]) + distance)
    scene.collection.objects.link(light)

    return f'Blender scene built for {name} ({len(keyframes)} location keyframes)'


def _saveBlendBesideJson(bpy: Any, jsonPath: Path) -> Path:
    blendPath = jsonPath.with_suffix('.blend').resolve()
    blendPath.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blendPath))
    return blendPath


def main(argv: list[str] | None = None) -> int:
    args = _argvAfterDoubleDash(list(argv if argv is not None else sys.argv))
    if not args:
        print(
            'Usage: load_body.py <body_scene.json>\n'
            '   or: blender --python load_body.py -- <body_scene.json>',
            file=sys.stderr,
        )
        return 2

    path = Path(args[0])
    payload = loadPayload(path)
    summary = summarizePayload(payload)
    print(summary)

    try:
        import bpy  # type: ignore[import-not-found]
    except ImportError:
        print('bpy not available — dry-run validation only (export JSON is ready for Blender).')
        return 0

    message = applyPayloadInBlender(payload)
    print(message)
    blendPath = _saveBlendBesideJson(bpy, path)
    print(f'Saved {blendPath}')
    print('Leave this Blender window open — select Earth, View → Frame Selected, scrub timeline.')
    return 0


if __name__ == '__main__':
    exitCode = main()
    # SystemExit terminates the Blender GUI process. Only exit for host dry-runs.
    if not _bpyAvailable():
        raise SystemExit(exitCode)
