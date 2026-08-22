"""Blender renderer for the K–Pg Earth-impact job.

Host writes a kpg-job JSON; this script (stdlib + bpy) places Earth, a true-scale
impactor, a diving camera and a Hollywood-adjacent schematic contact
(ember fireball, soot, 45° curtain, global ejecta, fallout shell)::

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
    for key in (
        'theme',
        'body',
        'impactor',
        'contact',
        'frames',
        'outputDirectory',
        'resolution',
    ):
        if key not in payload:
            raise ValueError(f'K–Pg job missing required key: {key}')
    if not payload['frames']:
        raise ValueError('K–Pg job must include frames')
    return payload


def _keyLocation(obj: Any, location: tuple[float, float, float], frame: int) -> None:
    obj.location = location
    obj.keyframe_insert(data_path='location', frame=frame)


def _keyScale(obj: Any, scale: float, frame: int, *, zScale: float = 1.0) -> None:
    obj.scale = (scale, scale, scale * zScale)
    obj.keyframe_insert(data_path='scale', frame=frame)
    obj.hide_render = scale < 1e-4
    obj.keyframe_insert(data_path='hide_render', frame=frame)


def _alignPlusZ(obj: Any, direction: tuple[float, float, float]) -> None:
    import mathutils  # type: ignore[import-not-found]

    target = mathutils.Vector(direction)
    if target.length < 1e-12:
        return
    target.normalize()
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = mathutils.Vector((0.0, 0.0, 1.0)).rotation_difference(target)


def _sootMaterial(
    bpy: Any,
    name: str,
    color: tuple[float, float, float],
    alpha: float,
) -> tuple[Any, Any]:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodeTree = material.node_tree
    nodes = nodeTree.nodes
    links = nodeTree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    mix = nodes.new('ShaderNodeMixShader')
    principled.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1.0)
    if 'Roughness' in principled.inputs:
        principled.inputs['Roughness'].default_value = 1.0
    if 'Metallic' in principled.inputs:
        principled.inputs['Metallic'].default_value = 0.0
    if 'Specular IOR Level' in principled.inputs:
        principled.inputs['Specular IOR Level'].default_value = 0.0
    elif 'Specular' in principled.inputs:
        principled.inputs['Specular'].default_value = 0.0
    mix.inputs['Fac'].default_value = 1.0 - alpha
    links.new(principled.outputs['BSDF'], mix.inputs[1])
    links.new(transparent.outputs['BSDF'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    if hasattr(material, 'surface_render_method'):
        material.surface_render_method = 'BLENDED'
    elif hasattr(material, 'blend_method'):
        material.blend_method = 'BLEND'
    return material, mix


def _emissionMaterial(
    bpy: Any,
    name: str,
    color: tuple[float, float, float],
    strength: float,
    alpha: float,
) -> Any:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodeTree = material.node_tree
    nodes = nodeTree.nodes
    links = nodeTree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (color[0], color[1], color[2], 1.0)
    emission.inputs['Strength'].default_value = strength
    if alpha >= 0.999:
        links.new(emission.outputs['Emission'], output.inputs['Surface'])
        return material
    mix = nodes.new('ShaderNodeMixShader')
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    mix.inputs['Fac'].default_value = 1.0 - alpha
    links.new(emission.outputs['Emission'], mix.inputs[1])
    links.new(transparent.outputs['BSDF'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    if hasattr(material, 'surface_render_method'):
        material.surface_render_method = 'BLENDED'
    elif hasattr(material, 'blend_method'):
        material.blend_method = 'BLEND'
    return material


def _createEjectaCone(
    bpy: Any,
    name: str,
    *,
    baseRadius: float,
    tipRadius: float,
    height: float,
) -> Any:
    import bmesh  # type: ignore[import-not-found]

    mesh = bpy.data.meshes.new(name)
    builder = bmesh.new()
    bmesh.ops.create_cone(
        builder,
        cap_ends=False,
        cap_tris=False,
        segments=12,
        radius1=max(baseRadius, 1e-6),
        radius2=max(tipRadius, 1e-6),
        depth=height,
    )
    builder.to_mesh(mesh)
    builder.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


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


def _offsetAlong(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    distance: float,
) -> tuple[float, float, float]:
    return (
        origin[0] + direction[0] * distance,
        origin[1] + direction[1] * distance,
        origin[2] + direction[2] * distance,
    )


def _buildEjectaStreaks(
    bpy: Any,
    contact: dict[str, Any],
    earthRadius: float,
    surface: tuple[float, float, float],
) -> list[Any]:
    height = earthRadius * float(contact['maxEjectaRadii'])
    thickness = height * 0.07
    material = _emissionMaterial(bpy, 'KpgEjectaMat', (0.95, 0.16, 0.02), 3.4, 0.86)
    streaks: list[Any] = []
    for index, raw in enumerate(contact['ejectaDirections']):
        direction = tuple(float(value) for value in raw)
        spike = _createEjectaCone(
            bpy,
            f'KpgEjecta{index:02d}',
            baseRadius=thickness,
            tipRadius=thickness * 0.14,
            height=height,
        )
        _alignPlusZ(spike, direction)
        spike.location = _offsetAlong(surface, direction, height * 0.5)
        spike.data.materials.append(material)
        streaks.append(spike)
    return streaks


def _buildContactDrawings(
    bpy: Any,
    flyby: ModuleType,
    job: dict[str, Any],
    earthRadius: float,
) -> tuple[Any, Any, list[Any], Any, Any, tuple[float, float, float]]:
    contact = job['contact']
    normal = tuple(float(value) for value in contact['normal'])
    surface = tuple(component * earthRadius for component in normal)
    fireballRadius = earthRadius * float(contact['maxFireballRadii'])
    plumeRadius = earthRadius * float(contact['maxPlumeRadii'])

    fireball = flyby._createBodySphere(bpy, 'KpgFireball', fireballRadius)
    fireball.location = _offsetAlong(surface, normal, fireballRadius * 0.55)
    fireball.data.materials.append(
        _emissionMaterial(bpy, 'KpgFireballMat', (0.95, 0.18, 0.02), 3.1, 0.9)
    )
    core = flyby._createBodySphere(bpy, 'KpgFireballCore', fireballRadius * 0.34)
    core.location = fireball.location
    core.data.materials.append(
        _emissionMaterial(bpy, 'KpgFireballCoreMat', (1.0, 0.40, 0.04), 5.2, 1.0)
    )

    streaks = _buildEjectaStreaks(bpy, contact, earthRadius, surface)

    plume = flyby._createBodySphere(bpy, 'KpgPlume', plumeRadius)
    plume.scale = (1.0, 1.0, 0.62)
    plume.location = _offsetAlong(surface, normal, plumeRadius * 0.28)
    soot, _mix = _sootMaterial(bpy, 'KpgPlumeMat', (0.04, 0.03, 0.022), 0.82)
    plume.data.materials.append(soot)
    stem = flyby._createBodySphere(bpy, 'KpgSmokeStem', plumeRadius * 0.38)
    stem.location = _offsetAlong(surface, normal, plumeRadius * 0.55)
    stemSoot, _stemMix = _sootMaterial(bpy, 'KpgStemMat', (0.03, 0.022, 0.016), 0.84)
    stem.data.materials.append(stemSoot)
    return fireball, core, streaks, plume, stem, surface


def _buildFalloutShell(
    bpy: Any,
    flyby: ModuleType,
    job: dict[str, Any],
    earthRadius: float,
    surface: tuple[float, float, float],
) -> tuple[Any, Any]:
    capRadius = earthRadius * float(job['contact']['falloutShell'])
    cap = flyby._createBodySphere(bpy, 'KpgFallout', capRadius)
    cap.location = surface
    material, mix = _sootMaterial(bpy, 'KpgFalloutMat', (0.07, 0.045, 0.03), 0.7)
    cap.data.materials.append(material)
    return cap, mix


def _buildProjectiles(
    bpy: Any, flyby: ModuleType, job: dict[str, Any], earthRadius: float
) -> list[Any]:
    count = int(job['contact']['projectileCount'])
    glowing = _emissionMaterial(bpy, 'KpgProjectileMat', (0.88, 0.18, 0.03), 2.4, 1.0)
    rocks: list[Any] = []
    for index in range(count):
        radius = earthRadius * (0.014 + 0.018 * ((index * 0.37) % 1.0))
        rock = flyby._createBodySphere(bpy, f'KpgProjectile{index:02d}', radius)
        rock.data.materials.append(glowing)
        rocks.append(rock)
    return rocks


def _keyMixFac(mix: Any, hiddenAmount: float, frame: int) -> None:
    mix.inputs['Fac'].default_value = hiddenAmount
    mix.inputs['Fac'].keyframe_insert(data_path='default_value', frame=frame)


def _keyframeShot(
    camera: Any,
    lookAt: Any,
    impactor: Any,
    flash: Any,
    fireball: Any,
    core: Any,
    streaks: list[Any],
    plume: Any,
    stem: Any,
    fallout: Any,
    falloutMix: Any,
    projectiles: list[Any],
    lightData: Any,
    flashData: Any,
    sunEnergy: float,
    flashEnergy: float,
    frames: list[dict[str, Any]],
) -> None:
    for sample in frames:
        frame = int(sample['frame'])
        _keyLocation(camera, tuple(float(value) for value in sample['cameraAu']), frame)
        _keyLocation(lookAt, tuple(float(value) for value in sample['lookAtAu']), frame)
        _keyLocation(impactor, tuple(float(value) for value in sample['impactorAu']), frame)
        _keyLocation(flash, tuple(float(value) for value in sample['impactorAu']), frame)
        _keyScale(fireball, float(sample['fireballScale']), frame)
        _keyScale(core, float(sample['fireballScale']), frame)
        for spike in streaks:
            _keyScale(spike, float(sample['ejectaScale']), frame)
        _keyScale(plume, float(sample['plumeScale']), frame, zScale=0.62)
        _keyScale(stem, float(sample['plumeScale']), frame, zScale=1.8)
        _keyScale(fallout, float(sample['falloutScale']), frame)
        _keyMixFac(falloutMix, 1.0 - 0.7 * float(sample['falloutScale']), frame)
        for rock, position, scale in zip(
            projectiles,
            sample['projectileAu'],
            sample['projectileScale'],
            strict=True,
        ):
            _keyLocation(rock, tuple(float(value) for value in position), frame)
            _keyScale(rock, float(scale), frame)
        rockVisible = float(sample['fireballScale']) < 1e-4
        _keyScale(impactor, 1.0 if rockVisible else 0.0, frame)
        lightData.energy = sunEnergy * float(sample['sunScale'])
        lightData.keyframe_insert(data_path='energy', frame=frame)
        flashData.energy = flashEnergy * float(sample['flashScale'])
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
    fireball, core, streaks, plume, stem, surface = _buildContactDrawings(bpy, flyby, job, radius)
    fallout, falloutMix = _buildFalloutShell(bpy, flyby, job, radius, surface)
    projectiles = _buildProjectiles(bpy, flyby, job, radius)

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
    siteLength = math.sqrt(sum(component * component for component in surface)) or 1.0
    sun.location = tuple(component / siteLength * radius * 16.0 for component in surface)
    scene.collection.objects.link(sun)
    sunTrack = sun.constraints.new(type='TRACK_TO')
    sunTrack.target = earth
    sunTrack.track_axis = 'TRACK_NEGATIVE_Z'
    sunTrack.up_axis = 'UP_Y'

    flashData = bpy.data.lights.new('KpgFlash', type='POINT')
    flashData.energy = 0.0
    flashData.color = (1.0, 0.28, 0.04)
    flash = bpy.data.objects.new('KpgFlash', flashData)
    scene.collection.objects.link(flash)
    flashEnergy = 1.8 if theme == 'dark' else 1.4
    _keyframeShot(
        camera,
        lookAt,
        impactor,
        flash,
        fireball,
        core,
        streaks,
        plume,
        stem,
        fallout,
        falloutMix,
        projectiles,
        lightData,
        flashData,
        sunEnergy,
        flashEnergy,
        frames,
    )

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
    viewSettings = getattr(scene, 'view_settings', None)
    if viewSettings is not None and theme == 'light':
        viewSettings.exposure = 0.12
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
