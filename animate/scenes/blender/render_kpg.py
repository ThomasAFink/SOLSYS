"""Blender renderer for the K–Pg Earth-impact job.

Host writes a kpg-job JSON; this script (stdlib + bpy) places Earth, a true-scale
impactor, a diving camera and a Hollywood-adjacent schematic contact
(Mantaflow fire/smoke, irregular debris, fallout cap)::

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


def _createCube(bpy: Any, name: str, size: float) -> Any:
    import bmesh  # type: ignore[import-not-found]

    mesh = bpy.data.meshes.new(name)
    builder = bmesh.new()
    bmesh.ops.create_cube(builder, size=size)
    builder.to_mesh(mesh)
    builder.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _volumeFireMaterial(bpy: Any) -> Any:
    material = bpy.data.materials.new(name='KpgFireSmoke')
    material.use_nodes = True
    nodeTree = material.node_tree
    nodes = nodeTree.nodes
    links = nodeTree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    volume = nodes.new('ShaderNodeVolumePrincipled')
    densityAttr = nodes.new('ShaderNodeAttribute')
    densityAttr.attribute_name = 'density'
    flameAttr = nodes.new('ShaderNodeAttribute')
    flameAttr.attribute_name = 'flame'
    densityScale = nodes.new('ShaderNodeMath')
    densityScale.operation = 'MULTIPLY'
    densityScale.inputs[1].default_value = 9.0
    emissionScale = nodes.new('ShaderNodeMath')
    emissionScale.operation = 'MULTIPLY'
    emissionScale.inputs[1].default_value = 3.4
    links.new(densityAttr.outputs['Fac'], densityScale.inputs[0])
    links.new(flameAttr.outputs['Fac'], emissionScale.inputs[0])
    if 'Density' in volume.inputs:
        links.new(densityScale.outputs['Value'], volume.inputs['Density'])
    if 'Emission Strength' in volume.inputs:
        links.new(emissionScale.outputs['Value'], volume.inputs['Emission Strength'])
    if 'Color' in volume.inputs:
        volume.inputs['Color'].default_value = (0.025, 0.02, 0.016, 1.0)
    if 'Absorption Color' in volume.inputs:
        volume.inputs['Absorption Color'].default_value = (0.08, 0.04, 0.02, 1.0)
    if 'Emission Color' in volume.inputs:
        volume.inputs['Emission Color'].default_value = (1.0, 0.22, 0.03, 1.0)
    if 'Anisotropy' in volume.inputs:
        volume.inputs['Anisotropy'].default_value = 0.25
    if 'Blackbody Intensity' in volume.inputs:
        volume.inputs['Blackbody Intensity'].default_value = 0.0
    links.new(volume.outputs['Volume'], output.inputs['Volume'])
    return material


def _enableEeveeVolumes(scene: Any, earthRadius: float) -> None:
    eevee = getattr(scene, 'eevee', None)
    if eevee is None:
        return
    if hasattr(eevee, 'volumetric_tile_size'):
        try:
            eevee.volumetric_tile_size = '4'
        except TypeError:
            pass
    if hasattr(eevee, 'volumetric_samples'):
        eevee.volumetric_samples = 24
    if hasattr(eevee, 'use_volumetric_shadows'):
        eevee.use_volumetric_shadows = True
    if hasattr(eevee, 'volumetric_start'):
        eevee.volumetric_start = earthRadius * 0.01
    if hasattr(eevee, 'volumetric_end'):
        eevee.volumetric_end = earthRadius * 12.0


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


def _impactFrame(frames: list[dict[str, Any]]) -> int:
    for sample in frames:
        if float(sample.get('flashScale', 0.0)) > 0.0:
            return int(sample['frame'])
    return int(frames[0]['frame'])


def _buildSmokeSim(
    bpy: Any,
    flyby: ModuleType,
    job: dict[str, Any],
    earth: Any,
    earthRadius: float,
    frames: list[dict[str, Any]],
) -> tuple[Any, Any, tuple[float, float, float]]:
    contact = job['contact']
    normal = tuple(float(value) for value in contact['normal'])
    surface = tuple(component * earthRadius for component in normal)
    domainSize = earthRadius * float(contact['smokeDomainRadii'])
    domain = _createCube(bpy, 'KpgSmokeDomain', domainSize)
    _alignPlusZ(domain, normal)
    domain.location = _offsetAlong(surface, normal, domainSize * 0.42)
    domain.hide_select = False
    fluid = domain.modifiers.new(name='Fluid', type='FLUID')
    fluid.fluid_type = 'DOMAIN'
    settings = fluid.domain_settings
    settings.domain_type = 'GAS'
    settings.resolution_max = int(contact['smokeResolution'])
    settings.cache_type = 'ALL'
    settings.cache_directory = str(Path(job['outputDirectory']) / 'fluid_cache')
    settings.cache_frame_start = int(frames[0]['frame'])
    settings.cache_frame_end = int(frames[-1]['frame'])
    settings.use_adaptive_domain = True
    settings.vorticity = 0.35
    settings.burning_rate = 0.7
    settings.flame_smoke = 1.2
    settings.use_dissolve_smoke = True
    settings.dissolve_speed = 22
    if hasattr(settings, 'gravity'):
        settings.gravity = (-normal[0] * 9.81, -normal[1] * 9.81, -normal[2] * 9.81)
    domain.data.materials.append(_volumeFireMaterial(bpy))

    emitter = flyby._createBodySphere(bpy, 'KpgSmokeFlow', earthRadius * 0.04)
    emitter.location = _offsetAlong(surface, normal, earthRadius * 0.03)
    emitter.hide_render = True
    flowMod = emitter.modifiers.new(name='Fluid', type='FLUID')
    flowMod.fluid_type = 'FLOW'
    flow = flowMod.flow_settings
    flow.flow_type = 'BOTH'
    flow.flow_behavior = 'INFLOW'
    flow.density = 1.0
    flow.fuel_amount = 1.0
    flow.temperature = 2.0
    flow.smoke_color = (0.04, 0.03, 0.025)
    flow.use_initial_velocity = True
    flow.velocity_normal = 2.4
    start = _impactFrame(frames)
    stop = start + int(contact['smokeInflowFrames'])
    flow.use_inflow = False
    flow.keyframe_insert(data_path='use_inflow', frame=start - 1)
    flow.use_inflow = True
    flow.keyframe_insert(data_path='use_inflow', frame=start)
    flow.use_inflow = True
    flow.keyframe_insert(data_path='use_inflow', frame=stop)
    flow.use_inflow = False
    flow.keyframe_insert(data_path='use_inflow', frame=stop + 1)

    collide = earth.modifiers.new(name='Fluid', type='FLUID')
    collide.fluid_type = 'EFFECTOR'
    if hasattr(collide, 'effector_settings'):
        collide.effector_settings.effector_type = 'COLLISION'
    return domain, emitter, surface


def _bakeGasDomain(bpy: Any, domain: Any) -> None:
    viewLayer = bpy.context.view_layer
    viewLayer.objects.active = domain
    domain.select_set(True)
    override = {'active_object': domain, 'object': domain, 'selected_objects': [domain]}
    try:
        with bpy.context.temp_override(**override):
            bpy.ops.fluid.bake_all()
    except (TypeError, RuntimeError, AttributeError):
        bpy.ops.fluid.bake_all()


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


def _createDebrisChunk(bpy: Any, name: str, radius: float, seed: int) -> Any:
    import bmesh  # type: ignore[import-not-found]

    mesh = bpy.data.meshes.new(name)
    builder = bmesh.new()
    bmesh.ops.create_icosphere(builder, subdivisions=2, radius=radius)
    state = seed * 997 + 13
    for vert in builder.verts:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        vert.co *= 1.0 + 0.45 * ((state / 0x7FFFFFFF) - 0.5)
    builder.to_mesh(mesh)
    builder.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    stretch = 0.35 + 0.8 * ((seed * 0.37) % 1.0)
    obj.scale = (1.0, stretch, 0.45 + 0.4 * ((seed * 0.19) % 1.0))
    return obj


def _buildProjectiles(bpy: Any, job: dict[str, Any], earthRadius: float) -> list[Any]:
    count = int(job['contact']['projectileCount'])
    glowing = _emissionMaterial(bpy, 'KpgDebrisMat', (0.22, 0.09, 0.04), 0.9, 1.0)
    rocks: list[Any] = []
    for index in range(count):
        radius = earthRadius * (0.011 + 0.016 * ((index * 0.41) % 1.0))
        rock = _createDebrisChunk(bpy, f'KpgDebris{index:02d}', radius, index + 1)
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
        _keyScale(fallout, float(sample['falloutScale']), frame)
        _keyMixFac(falloutMix, 1.0 - 0.7 * float(sample['falloutScale']), frame)
        for index, (rock, position, scale) in enumerate(
            zip(projectiles, sample['projectileAu'], sample['projectileScale'], strict=True)
        ):
            _keyLocation(rock, tuple(float(value) for value in position), frame)
            _keyScale(rock, float(scale), frame)
            rock.rotation_euler = (
                0.21 * index + 0.11 * frame,
                0.17 * index + 0.07 * frame,
                0.09 * frame,
            )
            rock.keyframe_insert(data_path='rotation_euler', frame=frame)
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
    domain, _emitter, surface = _buildSmokeSim(bpy, flyby, job, earth, radius, frames)
    fallout, falloutMix = _buildFalloutShell(bpy, flyby, job, radius, surface)
    projectiles = _buildProjectiles(bpy, job, radius)

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
    _enableEeveeVolumes(scene, radius)
    impactStart = _impactFrame(frames)
    for sample in frames:
        domain.hide_render = int(sample['frame']) < impactStart
        domain.keyframe_insert(data_path='hide_render', frame=int(sample['frame']))
    print('Baking K–Pg fire/smoke cache...')
    _bakeGasDomain(bpy, domain)
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
