"""Blender renderer for SOLSYS planet flyby jobs.

Host writes a flyby-job JSON; this script (stdlib + bpy) builds a body-centered
close-up and writes PNG frames::

    blender --background --python animate/scenes/blender/render_flyby.py -- \\
        output/animate/blender/planets/earth/earth_flyby_dark_job.json
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

JOB_SCHEMA_ID = 'solsys.blender_flyby_job/v1'


def _bpyAvailable() -> bool:
    """True when running inside Blender (avoid unused ``import bpy`` for CodeQL)."""
    return importlib.util.find_spec('bpy') is not None


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
    for image in list(bpy.data.images):
        bpy.data.images.remove(image)
    for camera in list(bpy.data.cameras):
        bpy.data.cameras.remove(camera)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)
    for world in list(bpy.data.worlds):
        bpy.data.worlds.remove(world)


def _assignSphericalUvs(builder: Any) -> None:
    """Equirectangular UVs from vertex directions (shared by planets/moons/asteroids)."""
    uvLayer = builder.loops.layers.uv.new('UVMap')
    for face in builder.faces:
        for loop in face.loops:
            direction = loop.vert.co.normalized()
            u = 0.5 + math.atan2(direction.y, direction.x) / (2.0 * math.pi)
            v = 0.5 + math.asin(max(-1.0, min(1.0, direction.z))) / math.pi
            loop[uvLayer].uv = (u, v)


def _createUvSphere(bpy: Any, name: str, radius: float) -> Any:
    import bmesh  # type: ignore[import-not-found]

    mesh = bpy.data.meshes.new(f'{name}Mesh')
    builder = bmesh.new()
    bmesh.ops.create_uvsphere(builder, u_segments=96, v_segments=48, radius=radius)
    bmesh.ops.recalc_face_normals(builder, faces=builder.faces)
    _assignSphericalUvs(builder)
    builder.to_mesh(mesh)
    builder.free()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _loadImageTexture(bpy: Any, nodeTree: Any, imagePath: Path, *, label: str) -> Any | None:
    if not imagePath.is_file():
        print(f'Warning: texture missing ({label}): {imagePath}', file=sys.stderr)
        return None
    image = bpy.data.images.load(str(imagePath), check_existing=True)
    textureNode = nodeTree.nodes.new('ShaderNodeTexImage')
    textureNode.image = image
    textureNode.label = label
    textureNode.interpolation = 'Smart'
    return textureNode


def _mixCloudLayer(
    bpy: Any,
    nodeTree: Any,
    surfaceColorSocket: Any,
    cloudsPath: Path,
) -> Any:
    """Mix optional cloud coverage over a surface color socket.

    Keep coverage modest so continents/oceans stay readable (not a snowball).
    """
    cloudsNode = _loadImageTexture(bpy, nodeTree, cloudsPath, label='clouds')
    if cloudsNode is None:
        return surfaceColorSocket
    cloudsNode.location = (-620, -40)
    # Prefer alpha when present; otherwise use luminance of the color map.
    rawFac = (
        cloudsNode.outputs['Alpha']
        if 'Alpha' in cloudsNode.outputs
        else cloudsNode.outputs['Color']
    )
    # Extra global attenuation in-shader (texture alpha already thinned).
    scale = nodeTree.nodes.new('ShaderNodeMath')
    scale.operation = 'MULTIPLY'
    scale.location = (-460, 40)
    scale.inputs[1].default_value = 0.75
    nodeTree.links.new(rawFac, scale.inputs[0])
    facSocket = scale.outputs['Value']

    cloudTint = nodeTree.nodes.new('ShaderNodeRGB')
    cloudTint.location = (-480, -160)
    # Soft warm-white; pure chalk white reads as ice/snow in EEVEE.
    cloudTint.outputs[0].default_value = (0.86, 0.88, 0.90, 1.0)
    # Blender 4+/5: ShaderNodeMix (RGBA). Older: MixRGB.
    mix = nodeTree.nodes.new('ShaderNodeMix')
    if hasattr(mix, 'data_type'):
        mix.data_type = 'RGBA'
        mix.blend_type = 'MIX'
        mix.location = (-300, 160)
        nodeTree.links.new(facSocket, mix.inputs['Factor'])
        nodeTree.links.new(surfaceColorSocket, mix.inputs['A'])
        nodeTree.links.new(cloudTint.outputs[0], mix.inputs['B'])
        return mix.outputs['Result']
    nodeTree.nodes.remove(mix)
    mixRgb = nodeTree.nodes.new('ShaderNodeMixRGB')
    mixRgb.blend_type = 'MIX'
    mixRgb.location = (-300, 160)
    nodeTree.links.new(facSocket, mixRgb.inputs['Fac'])
    nodeTree.links.new(surfaceColorSocket, mixRgb.inputs['Color1'])
    nodeTree.links.new(cloudTint.outputs[0], mixRgb.inputs['Color2'])
    return mixRgb.outputs['Color']


def _applyBodyMaterial(
    bpy: Any,
    material: Any,
    *,
    color: list[float],
    appearance: dict[str, Any] | None,
    theme: str,
) -> None:
    nodeTree = getattr(material, 'node_tree', None)
    if nodeTree is None:
        return
    principled = nodeTree.nodes.get('Principled BSDF')
    if principled is None:
        return

    roughness = float((appearance or {}).get('roughness', 0.55))
    specular = float((appearance or {}).get('specular', 0.22))
    if theme == 'light':
        # Keep matte enough to avoid ocean glare, but not so dull the day side dies.
        roughness = max(roughness, 0.60)
        specular = min(specular, 0.12)
    principled.inputs['Base Color'].default_value = color
    if 'Roughness' in principled.inputs:
        principled.inputs['Roughness'].default_value = roughness
    for inputName in ('Specular IOR Level', 'Specular'):
        if inputName in principled.inputs:
            principled.inputs[inputName].default_value = specular
            break

    textures = (appearance or {}).get('textures') or {}
    colorPath = textures.get('color')
    if not colorPath:
        return

    colorNode = _loadImageTexture(bpy, nodeTree, Path(colorPath), label='color')
    if colorNode is None:
        return
    colorNode.location = (-820, 200)
    # Slight saturation lift so oceans/vegetation don't wash out under clouds + fill.
    hueSat = nodeTree.nodes.new('ShaderNodeHueSaturation')
    hueSat.location = (-620, 200)
    hueSat.inputs['Saturation'].default_value = 1.22 if theme == 'light' else 1.18
    hueSat.inputs['Value'].default_value = 1.28 if theme == 'light' else 0.98
    nodeTree.links.new(colorNode.outputs['Color'], hueSat.inputs['Color'])
    surfaceColorSocket = hueSat.outputs['Color']
    cloudsPath = textures.get('clouds')
    if cloudsPath:
        surfaceColorSocket = _mixCloudLayer(bpy, nodeTree, surfaceColorSocket, Path(cloudsPath))
    nodeTree.links.new(surfaceColorSocket, principled.inputs['Base Color'])

    specularPath = textures.get('specular')
    if specularPath and 'Roughness' in principled.inputs:
        specularNode = _loadImageTexture(bpy, nodeTree, Path(specularPath), label='specular')
        if specularNode is not None:
            # Bright specular mask → lower roughness (oceans).
            invert = nodeTree.nodes.new('ShaderNodeInvert')
            invert.location = (-180, 40)
            specularNode.location = (-400, 0)
            nodeTree.links.new(specularNode.outputs['Color'], invert.inputs['Color'])
            nodeTree.links.new(invert.outputs['Color'], principled.inputs['Roughness'])


def _buildAtmosphereMaterial(
    bpy: Any,
    name: str,
    *,
    color: list[float],
    strength: float,
    fresnelBlend: float,
    theme: str,
) -> Any:
    """Fresnel limb haze — shared by any body with atmosphere.enabled."""
    material = bpy.data.materials.new(name=name)
    nodeTree = material.node_tree
    if nodeTree is None:
        return material
    nodes = nodeTree.nodes
    links = nodeTree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (360, 0)
    mix = nodes.new('ShaderNodeMixShader')
    mix.location = (160, 0)
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    transparent.location = (-40, 80)
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (-40, -80)
    layerWeight = nodes.new('ShaderNodeLayerWeight')
    layerWeight.location = (-280, 0)
    layerWeight.inputs['Blend'].default_value = max(0.01, min(fresnelBlend, 0.95))
    power = nodes.new('ShaderNodeMath')
    power.location = (-120, 0)
    power.operation = 'POWER'
    power.inputs[1].default_value = 2.4

    emission.inputs['Color'].default_value = (
        float(color[0]),
        float(color[1]),
        float(color[2]),
        1.0,
    )
    themeScale = 1.15 if theme == 'dark' else 0.55
    emission.inputs['Strength'].default_value = max(0.05, strength * themeScale)

    links.new(layerWeight.outputs['Fresnel'], power.inputs[0])
    links.new(power.outputs['Value'], mix.inputs['Fac'])
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])

    # EEVEE transparency (Blender 4.2+ / 5.x).
    if hasattr(material, 'surface_render_method'):
        material.surface_render_method = 'BLENDED'
    elif hasattr(material, 'blend_method'):
        material.blend_method = 'BLEND'
    if hasattr(material, 'use_backface_culling'):
        material.use_backface_culling = True
    return material


def _addAtmosphereShell(
    bpy: Any,
    planet: Any,
    *,
    radius: float,
    atmosphere: dict[str, Any],
    theme: str,
) -> Any | None:
    if not atmosphere.get('enabled'):
        return None
    scale = float(atmosphere.get('scale', 1.04))
    color = [float(channel) for channel in atmosphere.get('colorRgba', (0.45, 0.72, 1.0, 1.0))]
    strength = float(atmosphere.get('strength', 1.0))
    fresnelBlend = float(atmosphere.get('fresnelBlend', 0.18))
    shell = _createUvSphere(bpy, f'{planet.name}Atmosphere', radius)
    shell.scale = (scale, scale, scale)
    shell.parent = planet
    material = _buildAtmosphereMaterial(
        bpy,
        f'{planet.name}AtmosphereMaterial',
        color=color,
        strength=strength,
        fresnelBlend=fresnelBlend,
        theme=theme,
    )
    if shell.data.materials:
        shell.data.materials[0] = material
    else:
        shell.data.materials.append(material)
    print(f'Atmosphere shell: scale={scale:.3f} strength={strength:.2f}')
    return shell


def _configureWorld(bpy: Any, theme: str) -> None:
    """Theme backdrop + sparse ambient fill (not a second lamp).

    Light theme: camera sees a pale backdrop, but lighting uses a dim world so
    the night side still reads. Dark theme: near-black world with tiny fill.
    """
    world = bpy.data.worlds.new('FlybyWorld')
    bpy.context.scene.world = world
    nodeTree = getattr(world, 'node_tree', None)
    if nodeTree is None:
        world.color = (0.9, 0.92, 0.95) if theme == 'light' else (0.02, 0.025, 0.035)
        return

    nodes = nodeTree.nodes
    links = nodeTree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (360, 0)

    if theme == 'light':
        # Camera ray → pale backdrop. Other rays → dim fill (real night side).
        cameraBg = nodes.new('ShaderNodeBackground')
        cameraBg.location = (40, 80)
        cameraBg.inputs['Color'].default_value = (0.90, 0.92, 0.95, 1.0)
        cameraBg.inputs['Strength'].default_value = 1.0
        lightBg = nodes.new('ShaderNodeBackground')
        lightBg.location = (40, -80)
        lightBg.inputs['Color'].default_value = (0.62, 0.68, 0.78, 1.0)
        # Enough fill to read continents on the night side without flattening day/night.
        lightBg.inputs['Strength'].default_value = 0.14
        lightPath = nodes.new('ShaderNodeLightPath')
        lightPath.location = (-220, 0)
        mix = nodes.new('ShaderNodeMixShader')
        mix.location = (220, 0)
        links.new(lightPath.outputs['Is Camera Ray'], mix.inputs['Fac'])
        links.new(lightBg.outputs['Background'], mix.inputs[1])
        links.new(cameraBg.outputs['Background'], mix.inputs[2])
        links.new(mix.outputs['Shader'], output.inputs['Surface'])
        return

    background = nodes.new('ShaderNodeBackground')
    background.location = (40, 0)
    background.inputs['Color'].default_value = (0.02, 0.025, 0.035, 1.0)
    background.inputs['Strength'].default_value = 0.12
    links.new(background.outputs['Background'], output.inputs['Surface'])


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
    appearance = job.get('appearance')
    appearanceDict = appearance if isinstance(appearance, dict) else None
    if appearanceDict and appearanceDict.get('bodyId'):
        print(
            f'Appearance: bodyId={appearanceDict.get("bodyId")} '
            f'textures={sorted((appearanceDict.get("textures") or {}).keys())} '
            f'atmosphere={bool((appearanceDict.get("atmosphere") or {}).get("enabled"))}'
        )
    _applyBodyMaterial(
        bpy,
        material,
        color=color,
        appearance=appearanceDict,
        theme=theme,
    )
    if planet.data.materials:
        planet.data.materials[0] = material
    else:
        planet.data.materials.append(material)

    if appearanceDict and isinstance(appearanceDict.get('atmosphere'), dict):
        _addAtmosphereShell(
            bpy,
            planet,
            radius=radius,
            atmosphere=appearanceDict['atmosphere'],
            theme=theme,
        )

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
    if theme == 'dark':
        lightData.energy = 2.8
        lightData.angle = math.radians(5.0)
        # Bias the sun off camera-forward so a clear night crescent stays in frame.
        sunLocation = (radius * 14.0, -radius * 8.0, radius * 6.0)
    else:
        # Punchy day side on a pale backdrop; specular stays matte in the material.
        lightData.energy = 4.2
        lightData.angle = math.radians(3.0)
        # More camera-facing key so the lit hemisphere dominates the frame.
        sunLocation = (radius * 10.0, -radius * 3.5, radius * 9.0)
    light = bpy.data.objects.new(f'{name}KeySun', lightData)
    light.location = sunLocation
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
    filmTransparent = bool(job.get('filmTransparent', False))
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA' if filmTransparent else 'RGB'
    scene.render.filepath = str(outputDirectory / 'frame_')
    scene.render.use_file_extension = True
    scene.render.film_transparent = filmTransparent
    # Light theme was reading muddy under Filmic — lift exposure for day-side punch.
    viewSettings = getattr(scene, 'view_settings', None)
    if viewSettings is not None:
        if theme == 'light':
            viewSettings.exposure = 0.55
            viewSettings.gamma = 1.05
        else:
            viewSettings.exposure = 0.0
            viewSettings.gamma = 1.0

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

    if not _bpyAvailable():
        print('bpy not available — dry-run validation only.')
        return 0

    outputDirectory = applyFlybyJobInBlender(job)
    print(f'Rendered PNG frames → {outputDirectory}')
    return 0


if __name__ == '__main__':
    exitCode = main()
    if not _bpyAvailable():
        raise SystemExit(exitCode)
    # Background mode exits on its own after the script; GUI should not SystemExit.
    if '--background' in sys.argv or '-b' in sys.argv:
        raise SystemExit(exitCode)
