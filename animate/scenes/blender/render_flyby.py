"""Blender renderer for SOLSYS planet flyby jobs.

Host writes a flyby-job JSON; this script (stdlib + bpy) builds a body-centered
close-up and writes PNG frames::

    blender --background --python animate/scenes/blender/render_flyby.py -- \\
        output/animate/blender/earth_flyby_dark_job.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

JOB_SCHEMA_ID = 'solsys.blender_flyby_job/v1'


def _argvAfterDoubleDash(argv: list[str]) -> list[str]:
    if '--' in argv:
        return argv[argv.index('--') + 1 :]
    return argv[1:]


def loadJob(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema') != JOB_SCHEMA_ID:
        raise ValueError(
            f'Unsupported flyby job schema: {payload.get("schema")!r} (expected {JOB_SCHEMA_ID!r})'
        )
    for key in ('theme', 'body', 'frames', 'outputDirectory', 'resolution'):
        if key not in payload:
            raise ValueError(f'Flyby job missing required key: {key}')
    if not payload['frames']:
        raise ValueError('Flyby job must include frames')
    return payload


def _clearSceneObjects(bpy: Any) -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for camera in list(bpy.data.cameras):
        bpy.data.cameras.remove(camera)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)
    for world in list(bpy.data.worlds):
        bpy.data.worlds.remove(world)


def _createUvSphere(bpy: Any, name: str, radius: float) -> Any:
    import bmesh  # type: ignore[import-not-found]

    mesh = bpy.data.meshes.new(f'{name}Mesh')
    builder = bmesh.new()
    bmesh.ops.create_uvsphere(builder, u_segments=48, v_segments=24, radius=radius)
    bmesh.ops.recalc_face_normals(builder, faces=builder.faces)
    builder.to_mesh(mesh)
    builder.free()
    # Smooth shading for a softer planet look.
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _setPrincipled(material: Any, *, color: list[float], roughness: float, specular: float) -> None:
    nodeTree = getattr(material, 'node_tree', None)
    if nodeTree is None:
        return
    principled = nodeTree.nodes.get('Principled BSDF')
    if principled is None:
        return
    principled.inputs['Base Color'].default_value = color
    if 'Roughness' in principled.inputs:
        principled.inputs['Roughness'].default_value = roughness
    # Blender 4+/5 name variants for specular-ish controls.
    for inputName in ('Specular IOR Level', 'Specular', 'Specular Tint'):
        if inputName in principled.inputs and inputName != 'Specular Tint':
            principled.inputs[inputName].default_value = specular
            break


def _configureWorld(bpy: Any, theme: str) -> None:
    world = bpy.data.worlds.new('FlybyWorld')
    bpy.context.scene.world = world
    if theme == 'light':
        color = (0.92, 0.93, 0.95)
        # Soft ambient fill (no second lamp) so the night side isn't a hard wedge.
        strength = 0.55
    else:
        color = (0.02, 0.025, 0.035)
        strength = 0.18
    # Prefer nodes when available; fall back to solid world color.
    nodeTree = getattr(world, 'node_tree', None)
    if nodeTree is None:
        world.color = color
        return
    background = nodeTree.nodes.get('Background')
    if background is None:
        background = nodeTree.nodes.new('ShaderNodeBackground')
        output = nodeTree.nodes.get('World Output')
        if output is not None:
            nodeTree.links.new(background.outputs['Background'], output.inputs['Surface'])
    background.inputs['Color'].default_value = (*color, 1.0)
    background.inputs['Strength'].default_value = strength


def applyFlybyJobInBlender(job: dict[str, Any]) -> Path:
    import bpy  # type: ignore[import-not-found]

    body = job['body']
    theme = str(job['theme'])
    frames = job['frames']
    name = str(body['name'])
    radius = float(body['displayRadiusAu'])
    color = [float(channel) for channel in body['colorRgba']]
    resolution = int(job['resolution'])
    outputDirectory = Path(job['outputDirectory'])
    outputDirectory.mkdir(parents=True, exist_ok=True)

    _clearSceneObjects(bpy)
    planet = _createUvSphere(bpy, name, radius)
    material = bpy.data.materials.new(name=f'{name}FlybyMaterial')
    roughness = 0.42 if theme == 'light' else 0.55
    specular = 0.35 if theme == 'light' else 0.22
    _setPrincipled(material, color=color, roughness=roughness, specular=specular)
    if planet.data.materials:
        planet.data.materials[0] = material
    else:
        planet.data.materials.append(material)

    scene = bpy.context.scene
    scene.frame_start = int(frames[0]['frame'])
    scene.frame_end = int(frames[-1]['frame'])
    scene.frame_current = scene.frame_start

    for sample in frames:
        frame = int(sample['frame'])
        rotationRad = math.radians(float(sample['bodyRotationDeg']))
        planet.rotation_euler = (0.0, 0.0, rotationRad)
        planet.keyframe_insert(data_path='rotation_euler', frame=frame)

    cameraData = bpy.data.cameras.new(f'{name}FlybyCameraData')
    cameraData.lens = 50
    cameraData.clip_start = max(radius * 0.05, 0.0001)
    cameraData.clip_end = max(radius * 80.0, 10.0)
    camera = bpy.data.objects.new(f'{name}FlybyCamera', cameraData)
    scene.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new(type='TRACK_TO')
    track.target = planet
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    for sample in frames:
        frame = int(sample['frame'])
        position = sample['cameraAu']
        camera.location = (float(position[0]), float(position[1]), float(position[2]))
        camera.keyframe_insert(data_path='location', frame=frame)

    # Single distant sun aimed at the body. No area fill — that was carving the
    # pointy false "terminator". Ambient lift comes from the world background.
    lightData = bpy.data.lights.new(f'{name}KeySun', type='SUN')
    lightData.energy = 2.8 if theme == 'dark' else 2.1
    lightData.angle = math.radians(5.0)  # slightly soft limb, still one light
    light = bpy.data.objects.new(f'{name}KeySun', lightData)
    light.location = (radius * 12.0, -radius * 5.0, radius * 8.0)
    scene.collection.objects.link(light)
    sunTrack = light.constraints.new(type='TRACK_TO')
    sunTrack.target = planet
    sunTrack.track_axis = 'TRACK_NEGATIVE_Z'
    sunTrack.up_axis = 'UP_Y'

    _configureWorld(bpy, theme)

    # EEVEE for fast local GIF frames (Blender 4.2+/5.x).
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.fps = int(job.get('fps', 18))
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.filepath = str(outputDirectory / 'frame_')
    scene.render.use_file_extension = True
    scene.render.film_transparent = False

    bpy.ops.render.render(animation=True)

    written = sorted(outputDirectory.glob('frame_*.png'))
    if not written:
        raise RuntimeError(f'Blender produced no PNG frames in {outputDirectory}')
    return outputDirectory


def main(argv: list[str] | None = None) -> int:
    args = _argvAfterDoubleDash(list(argv if argv is not None else sys.argv))
    if not args:
        print(
            'Usage: render_flyby.py <flyby_job.json>\n'
            '   or: blender --background --python render_flyby.py -- <flyby_job.json>',
            file=sys.stderr,
        )
        return 2

    jobPath = Path(args[0])
    job = loadJob(jobPath)
    print(
        f'Flyby job: {job["body"]["name"]} theme={job["theme"]} '
        f'frames={len(job["frames"])} resolution={job["resolution"]}'
    )

    try:
        import bpy  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        print('bpy not available — dry-run validation only.')
        return 0

    outputDirectory = applyFlybyJobInBlender(job)
    print(f'Rendered PNG frames → {outputDirectory}')
    return 0


if __name__ == '__main__':
    exitCode = main()
    try:
        import bpy  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        raise SystemExit(exitCode) from None
    # Background mode exits on its own after the script; GUI should not SystemExit.
    if '--background' in sys.argv or '-b' in sys.argv:
        raise SystemExit(exitCode)
