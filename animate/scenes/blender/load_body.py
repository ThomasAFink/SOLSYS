"""Blender ingest for SOLSYS body-scene JSON.

Runs two ways:

1. Host dry-run (no bpy)::

       python animate/scenes/blender/load_body.py output/animate/blender/earth_body_scene.json

2. Inside Blender::

       blender --background --python animate/scenes/blender/load_body.py -- \\
           output/animate/blender/earth_body_scene.json

Keep this module stdlib-only so Blender's bundled Python can import it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_ID = 'solsys.blender_body_scene/v1'


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


def applyPayloadInBlender(payload: dict[str, Any]) -> str:
    """Create a UV sphere, material, and location keyframes from the payload."""
    import bpy  # type: ignore[import-not-found]

    body = payload['body']
    keyframes = payload['keyframes']
    name = str(body['name'])
    radius = float(body['displayRadiusAu'])
    color = [float(channel) for channel in body['colorRgba']]

    # Fresh scene for the stub ingest.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    assert obj is not None
    obj.name = name

    material = bpy.data.materials.new(name=f'{name}Material')
    # Blender 5+ materials are node-based by default; avoid deprecated use_nodes.
    nodeTree = getattr(material, 'node_tree', None)
    if nodeTree is not None:
        principled = nodeTree.nodes.get('Principled BSDF')
        if principled is not None:
            principled.inputs['Base Color'].default_value = color
    obj.data.materials.append(material)

    scene = bpy.context.scene
    assert scene is not None
    scene.frame_start = int(keyframes[0]['frame'])
    scene.frame_end = int(keyframes[-1]['frame'])

    for keyframe in keyframes:
        frame = int(keyframe['frame'])
        position = keyframe['positionAu']
        obj.location = (float(position[0]), float(position[1]), float(position[2]))
        obj.keyframe_insert(data_path='location', frame=frame)

    distance = float(payload['cameraHintDistanceAu'])
    bpy.ops.object.camera_add(location=(distance, -distance, distance * 0.6))
    camera = bpy.context.active_object
    assert camera is not None
    camera.name = f'{name}Camera'
    scene.camera = camera

    # Point camera at the body origin of the first keyframe.
    first = keyframes[0]['positionAu']
    track = camera.constraints.new(type='TRACK_TO')
    track.target = obj
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    camera.location = (
        float(first[0]) + distance,
        float(first[1]) - distance,
        float(first[2]) + distance * 0.6,
    )

    return f'Blender scene built for {name} ({len(keyframes)} location keyframes)'


def main(argv: list[str] | None = None) -> int:
    args = _argvAfterDoubleDash(list(argv if argv is not None else sys.argv))
    if not args:
        print(
            'Usage: load_body.py <body_scene.json>\n'
            '   or: blender --background --python load_body.py -- <body_scene.json>',
            file=sys.stderr,
        )
        return 2

    path = Path(args[0])
    payload = loadPayload(path)
    summary = summarizePayload(payload)
    print(summary)

    try:
        import bpy  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        print('bpy not available — dry-run validation only (export JSON is ready for Blender).')
        return 0

    message = applyPayloadInBlender(payload)
    print(message)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
