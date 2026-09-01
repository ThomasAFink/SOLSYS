"""Blender renderer for the K–Pg Earth-impact job.

Host writes a kpg-job JSON; this script (stdlib + bpy) places Earth and a
Yucatán-facing camera. Contact fire is a 3D burst on the globe, not a
stills cutaway::

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
_VEIL_START_DELAY = 16
_VEIL_START_RAD = 0.035
_VEIL_GROW_FRAMES = 150.0


def _smooth01(progress: float) -> float:
    clamped = max(0.0, min(1.0, progress))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _veilAngleRad(frame: int, impactFrame: int) -> float:
    """Dark crawl. Starts after the site cloud, not at the bang."""
    if frame < impactFrame + _VEIL_START_DELAY:
        return 0.0
    age = (frame - impactFrame - _VEIL_START_DELAY) / _VEIL_GROW_FRAMES
    return _VEIL_START_RAD + (math.pi * 1.12 - _VEIL_START_RAD) * _smooth01(age)


def _siteCloudRad(frame: int, impactFrame: int) -> float:
    """Local fire disk. Grows behind the shock, then dies. Not a hemisphere."""
    if frame < impactFrame:
        return 0.0
    age = float(frame - impactFrame)
    grown = 0.155 * _smooth01(age / 32.0)
    fade = _smooth01((age - 18.0) / 32.0)
    return grown * (1.0 - fade)


def _falloutAngleRad(frame: int, impactFrame: int) -> float:
    """Molten rain on the globe. Hours compressed."""
    if frame < impactFrame + 18:
        return 0.0
    return 0.05 + (math.pi * 0.72) * _smooth01((frame - impactFrame - 18) / 240.0)


def _blastLampScale(frame: int, flashScale: float, impactFrame: int) -> float:
    """Keep the locked bang at contact. Do not keep lighting half the globe after."""
    if frame <= impactFrame:
        return flashScale
    return flashScale * math.exp(-0.55 * (frame - impactFrame))


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


def _iterActionFcurves(action: Any) -> Any:
    curves = getattr(action, 'fcurves', None)
    if curves is not None:
        yield from curves
        return
    for layer in getattr(action, 'layers', []):
        for strip in getattr(layer, 'strips', []):
            bag = getattr(strip, 'channelbag', None)
            if bag is None:
                continue
            yield from getattr(bag, 'fcurves', [])


def _linearizeEarthSpin(earth: Any) -> None:
    action = getattr(getattr(earth, 'animation_data', None), 'action', None)
    if action is None:
        return
    for curve in _iterActionFcurves(action):
        if getattr(curve, 'data_path', '') != 'rotation_euler':
            continue
        for key in curve.keyframe_points:
            key.interpolation = 'LINEAR'


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


def _alignCardUpright(
    obj: Any,
    up: tuple[float, float, float],
    camera: tuple[float, float, float],
    location: tuple[float, float, float],
) -> None:
    import mathutils  # type: ignore[import-not-found]

    upAxis = mathutils.Vector(up)
    if upAxis.length < 1e-12:
        return
    upAxis.normalize()
    toward = mathutils.Vector(camera) - mathutils.Vector(location)
    if toward.length < 1e-12:
        return
    flat = toward - upAxis * toward.dot(upAxis)
    if flat.length < 1e-8:
        _alignPlusZ(obj, (toward.x, toward.y, toward.z))
        return
    zAxis = flat.normalized()
    xAxis = upAxis.cross(zAxis)
    if xAxis.length < 1e-8:
        _alignPlusZ(obj, (toward.x, toward.y, toward.z))
        return
    xAxis.normalize()
    zAxis = xAxis.cross(upAxis).normalized()
    rotation = mathutils.Matrix((xAxis, upAxis, zAxis)).transposed()
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = rotation.to_quaternion()


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
    densityScale.inputs[1].default_value = 6.2
    emissionScale = nodes.new('ShaderNodeMath')
    emissionScale.operation = 'MULTIPLY'
    emissionScale.inputs[1].default_value = 6.8
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
        volume.inputs['Emission Color'].default_value = (1.0, 0.30, 0.04, 1.0)
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
            eevee.volumetric_tile_size = '8'
        except TypeError:
            pass
    if hasattr(eevee, 'volumetric_samples'):
        eevee.volumetric_samples = 16
    if hasattr(eevee, 'use_volumetric_shadows'):
        eevee.use_volumetric_shadows = True
    if hasattr(eevee, 'volumetric_start'):
        eevee.volumetric_start = earthRadius * 0.01
    if hasattr(eevee, 'volumetric_end'):
        eevee.volumetric_end = earthRadius * 3.2


def _softenCloseupTextures(material: Any) -> None:
    """Flyby uses Closest sampling to hide the date line; the dive needs filtering."""
    nodeTree = getattr(material, 'node_tree', None)
    if nodeTree is None:
        return
    for node in nodeTree.nodes:
        if getattr(node, 'type', '') == 'TEX_IMAGE' and hasattr(node, 'interpolation'):
            node.interpolation = 'Cubic'


def _valueNode(nodes: Any, name: str, value: float) -> Any:
    node = nodes.new('ShaderNodeValue')
    node.label = name
    node.outputs[0].default_value = value
    return node


def _impactAngleSocket(
    nodes: Any,
    links: Any,
    normal: tuple[float, float, float],
    inbound: tuple[float, float, float] | None = None,
) -> Any:
    del inbound
    texcoord = nodes.new('ShaderNodeTexCoord')
    length = nodes.new('ShaderNodeVectorMath')
    length.operation = 'NORMALIZE'
    impact = nodes.new('ShaderNodeCombineXYZ')
    impact.inputs[0].default_value = normal[0]
    impact.inputs[1].default_value = normal[1]
    impact.inputs[2].default_value = normal[2]
    dot = nodes.new('ShaderNodeVectorMath')
    dot.operation = 'DOT_PRODUCT'
    clampDot = nodes.new('ShaderNodeClamp')
    clampDot.inputs['Min'].default_value = -1.0
    clampDot.inputs['Max'].default_value = 1.0
    angle = nodes.new('ShaderNodeMath')
    angle.operation = 'ARCCOSINE'
    links.new(texcoord.outputs['Object'], length.inputs[0])
    links.new(length.outputs['Vector'], dot.inputs[0])
    links.new(impact.outputs['Vector'], dot.inputs[1])
    links.new(dot.outputs['Value'], clampDot.inputs['Value'])
    links.new(clampDot.outputs['Result'], angle.inputs[0])
    return angle.outputs['Value']


def _colorMix(nodes: Any, *, multiply: bool = False) -> Any:
    mix = nodes.new('ShaderNodeMix')
    if hasattr(mix, 'data_type'):
        mix.data_type = 'RGBA'
    if multiply and hasattr(mix, 'blend_type'):
        mix.blend_type = 'MULTIPLY'
    return mix


def _mixOut(mix: Any) -> Any:
    return mix.outputs['Result'] if 'Result' in mix.outputs else mix.outputs[0]


def _shockGates(nodes: Any, links: Any, angle: Any, shock: Any) -> tuple[Any, Any]:
    inside = nodes.new('ShaderNodeMath')
    inside.operation = 'LESS_THAN'
    links.new(angle, inside.inputs[0])
    links.new(shock.outputs[0], inside.inputs[1])
    delta = nodes.new('ShaderNodeMath')
    delta.operation = 'SUBTRACT'
    links.new(angle, delta.inputs[0])
    links.new(shock.outputs[0], delta.inputs[1])
    ringAbs = nodes.new('ShaderNodeMath')
    ringAbs.operation = 'ABSOLUTE'
    links.new(delta.outputs['Value'], ringAbs.inputs[0])
    width = nodes.new('ShaderNodeMath')
    width.operation = 'DIVIDE'
    width.inputs[1].default_value = 0.034
    links.new(ringAbs.outputs['Value'], width.inputs[0])
    ring = nodes.new('ShaderNodeMath')
    ring.operation = 'LESS_THAN'
    ring.inputs[1].default_value = 1.0
    links.new(width.outputs['Value'], ring.inputs[0])
    grown = nodes.new('ShaderNodeMath')
    grown.operation = 'GREATER_THAN'
    grown.inputs[1].default_value = 0.04
    links.new(shock.outputs[0], grown.inputs[0])
    gated = nodes.new('ShaderNodeMath')
    gated.operation = 'MULTIPLY'
    links.new(ring.outputs['Value'], gated.inputs[0])
    links.new(grown.outputs['Value'], gated.inputs[1])
    return inside.outputs['Value'], gated.outputs['Value']


def _equirectFromDirection(nodes: Any, links: Any, direction: Any) -> Any:
    separate = nodes.new('ShaderNodeSeparateXYZ')
    links.new(direction, separate.inputs['Vector'])
    arctan2 = nodes.new('ShaderNodeMath')
    arctan2.operation = 'ARCTAN2'
    links.new(separate.outputs['Y'], arctan2.inputs[0])
    links.new(separate.outputs['X'], arctan2.inputs[1])
    divU = nodes.new('ShaderNodeMath')
    divU.operation = 'DIVIDE'
    divU.inputs[1].default_value = 2.0 * math.pi
    links.new(arctan2.outputs['Value'], divU.inputs[0])
    addU = nodes.new('ShaderNodeMath')
    addU.operation = 'ADD'
    addU.inputs[1].default_value = 0.5
    links.new(divU.outputs['Value'], addU.inputs[0])
    arcsin = nodes.new('ShaderNodeMath')
    arcsin.operation = 'ARCSINE'
    links.new(separate.outputs['Z'], arcsin.inputs[0])
    divV = nodes.new('ShaderNodeMath')
    divV.operation = 'DIVIDE'
    divV.inputs[1].default_value = math.pi
    links.new(arcsin.outputs['Value'], divV.inputs[0])
    addV = nodes.new('ShaderNodeMath')
    addV.operation = 'ADD'
    addV.inputs[1].default_value = 0.5
    links.new(divV.outputs['Value'], addV.inputs[0])
    combine = nodes.new('ShaderNodeCombineXYZ')
    links.new(addU.outputs['Value'], combine.inputs['X'])
    links.new(addV.outputs['Value'], combine.inputs['Y'])
    return combine.outputs['Vector']


def _raggedLead(nodes: Any, links: Any, shock: Any) -> Any:
    """Break the first-wave radius so the cloud bank is not a perfect circle."""
    chop = _generatedNoise(nodes, links, 20.0, 3.5)
    wobble = _mathMulConst(nodes, links, chop, 0.16)
    scale = _mathAdd(nodes, links, wobble, 0.92)
    return _mathMul(nodes, links, shock.outputs[0], scale)


def _angleTent(nodes: Any, links: Any, angle: Any, radius: Any, width: float) -> Any:
    delta = nodes.new('ShaderNodeMath')
    delta.operation = 'SUBTRACT'
    links.new(angle, delta.inputs[0])
    links.new(radius, delta.inputs[1])
    ringAbs = nodes.new('ShaderNodeMath')
    ringAbs.operation = 'ABSOLUTE'
    links.new(delta.outputs['Value'], ringAbs.inputs[0])
    norm = nodes.new('ShaderNodeMath')
    norm.operation = 'DIVIDE'
    norm.inputs[1].default_value = width
    links.new(ringAbs.outputs['Value'], norm.inputs[0])
    tent = nodes.new('ShaderNodeMath')
    tent.operation = 'SUBTRACT'
    tent.inputs[0].default_value = 1.0
    links.new(norm.outputs['Value'], tent.inputs[1])
    clip = nodes.new('ShaderNodeClamp')
    clip.inputs['Min'].default_value = 0.0
    clip.inputs['Max'].default_value = 1.0
    links.new(tent.outputs['Value'], clip.inputs['Value'])
    return clip.outputs['Result']


def _shockActive(nodes: Any, links: Any, shock: Any) -> Any:
    gate = nodes.new('ShaderNodeMath')
    gate.operation = 'GREATER_THAN'
    gate.inputs[1].default_value = 0.02
    links.new(shock.outputs[0], gate.inputs[0])
    return gate.outputs['Value']


def _scaleBy(nodes: Any, links: Any, value: Any, factor: Any) -> Any:
    mul = nodes.new('ShaderNodeMath')
    mul.operation = 'MULTIPLY'
    links.new(value, mul.inputs[0])
    links.new(factor, mul.inputs[1])
    return mul.outputs['Value']


def _cloudCoverageScale(nodes: Any, links: Any, front: Any, inside: Any, armed: Any) -> Any:
    """Thin weather behind the front and pile it on the front. Do not punch to land."""
    thin = nodes.new('ShaderNodeMath')
    thin.operation = 'MULTIPLY'
    thin.inputs[1].default_value = 0.28
    links.new(inside, thin.inputs[0])
    kept = nodes.new('ShaderNodeMath')
    kept.operation = 'SUBTRACT'
    kept.inputs[0].default_value = 1.0
    links.new(thin.outputs['Value'], kept.inputs[1])
    pile = nodes.new('ShaderNodeMath')
    pile.operation = 'MULTIPLY'
    pile.inputs[1].default_value = 0.0
    links.new(front, pile.inputs[0])
    piled = nodes.new('ShaderNodeMath')
    piled.operation = 'ADD'
    piled.inputs[0].default_value = 1.0
    links.new(pile.outputs['Value'], piled.inputs[1])
    scaled = nodes.new('ShaderNodeMath')
    scaled.operation = 'MULTIPLY'
    links.new(kept.outputs['Value'], scaled.inputs[0])
    links.new(piled.outputs['Value'], scaled.inputs[1])
    off = nodes.new('ShaderNodeMath')
    off.operation = 'SUBTRACT'
    off.inputs[0].default_value = 1.0
    links.new(armed, off.inputs[1])
    active = _scaleBy(nodes, links, scaled.outputs['Value'], armed)
    idle = nodes.new('ShaderNodeMath')
    idle.operation = 'ADD'
    links.new(active, idle.inputs[0])
    links.new(off.outputs['Value'], idle.inputs[1])
    return idle.outputs['Value']


def _vectorScale(nodes: Any, links: Any, vector: Any, scalar: Any) -> Any:
    scaled = nodes.new('ShaderNodeVectorMath')
    scaled.operation = 'SCALE'
    vecIn = scaled.inputs['Vector'] if 'Vector' in scaled.inputs else scaled.inputs[0]
    scaleIn = scaled.inputs['Scale'] if 'Scale' in scaled.inputs else scaled.inputs[1]
    links.new(vector, vecIn)
    links.new(scalar, scaleIn)
    return scaled.outputs['Vector']


def _cloudPullTowardImpact(
    nodes: Any, links: Any, angle: Any, lead: Any, inside: Any, armed: Any
) -> Any:
    """How far to sample toward the impact so interior weather reads at the front."""
    span = nodes.new('ShaderNodeMath')
    span.operation = 'MAXIMUM'
    span.inputs[1].default_value = 0.08
    links.new(lead, span.inputs[0])
    gap = nodes.new('ShaderNodeMath')
    gap.operation = 'SUBTRACT'
    links.new(lead, gap.inputs[0])
    links.new(angle, gap.inputs[1])
    behind = nodes.new('ShaderNodeMath')
    behind.operation = 'DIVIDE'
    links.new(gap.outputs['Value'], behind.inputs[0])
    links.new(span.outputs['Value'], behind.inputs[1])
    behindClip = nodes.new('ShaderNodeClamp')
    behindClip.inputs['Min'].default_value = 0.0
    behindClip.inputs['Max'].default_value = 1.0
    links.new(behind.outputs['Value'], behindClip.inputs['Value'])
    extra = nodes.new('ShaderNodeMath')
    extra.operation = 'MULTIPLY'
    extra.inputs[1].default_value = 0.18
    links.new(behindClip.outputs['Result'], extra.inputs[0])
    pull = nodes.new('ShaderNodeMath')
    pull.operation = 'ADD'
    pull.inputs[0].default_value = 0.38
    links.new(extra.outputs['Value'], pull.inputs[1])
    gated = _scaleBy(nodes, links, pull.outputs['Value'], inside)
    return _scaleBy(nodes, links, gated, armed)


def _pushedCloudVector(
    nodes: Any, links: Any, normal: tuple[float, float, float], pull: Any
) -> Any:
    """Sample clouds from closer to the impact so they read as shoved outward."""
    texcoord = nodes.new('ShaderNodeTexCoord')
    direction = nodes.new('ShaderNodeVectorMath')
    direction.operation = 'NORMALIZE'
    links.new(texcoord.outputs['Normal'], direction.inputs[0])
    impact = nodes.new('ShaderNodeCombineXYZ')
    impact.inputs[0].default_value = normal[0]
    impact.inputs[1].default_value = normal[1]
    impact.inputs[2].default_value = normal[2]
    stay = nodes.new('ShaderNodeMath')
    stay.operation = 'SUBTRACT'
    stay.inputs[0].default_value = 1.0
    links.new(pull, stay.inputs[1])
    fromHere = _vectorScale(nodes, links, direction.outputs['Vector'], stay.outputs['Value'])
    fromImpact = _vectorScale(nodes, links, impact.outputs['Vector'], pull)
    mixed = nodes.new('ShaderNodeVectorMath')
    mixed.operation = 'ADD'
    links.new(fromHere, mixed.inputs[0])
    links.new(fromImpact, mixed.inputs[1])
    heading = nodes.new('ShaderNodeVectorMath')
    heading.operation = 'NORMALIZE'
    links.new(mixed.outputs['Vector'], heading.inputs[0])
    return _equirectFromDirection(nodes, links, heading.outputs['Vector'])


def _cloudTextureNode(nodes: Any) -> Any | None:
    for node in nodes:
        if getattr(node, 'type', '') != 'TEX_IMAGE':
            continue
        tag = f'{getattr(node, "label", "")} {node.name}'.lower()
        if 'cloud' in tag:
            return node
    return None


def _cloudAlphaSocket(texture: Any) -> Any | None:
    if texture is None:
        return None
    if 'Alpha' in texture.outputs:
        return texture.outputs['Alpha']
    if 'Color' in texture.outputs:
        return texture.outputs['Color']
    return None


def _cloudMixNode(nodes: Any, links: Any) -> Any | None:
    for node in nodes:
        if getattr(node, 'type', '') != 'RGB':
            continue
        tint = tuple(float(channel) for channel in node.outputs[0].default_value[:3])
        if abs(tint[0] - 0.86) > 0.12 or abs(tint[1] - 0.88) > 0.12:
            continue
        for link in links:
            if link.from_socket != node.outputs[0]:
                continue
            return link.to_socket.node
    return None


def _cloudSurfaceSocket(nodes: Any, links: Any) -> Any | None:
    mix = _cloudMixNode(nodes, links)
    if mix is None:
        return None
    dest = mix.inputs['A'] if 'A' in mix.inputs else mix.inputs.get('Color1')
    if dest is None:
        return None
    incoming = next((link for link in links if link.to_socket == dest), None)
    return incoming.from_socket if incoming is not None else None


def _revealLandUnderWave(nodes: Any, links: Any, cloudy: Any, fade: Any) -> Any:
    """Swap cloudy color for bare surface inside the first wave."""
    surface = _cloudSurfaceSocket(nodes, links)
    if surface is None:
        print('K–Pg cloud push: no surface socket; fading mix factor')
        _insertCoverageFade(nodes, links, fade)
        return cloudy
    print('K–Pg cloud push: revealing land behind first wave')
    reveal = _colorMix(nodes)
    _linkMix(links, reveal, surface, cloudy, fade)
    return _mixOut(reveal)


def _cloudScaleNode(nodes: Any) -> Any | None:
    for target in (0.75, 0.95):
        for node in nodes:
            if (
                getattr(node, 'type', '') == 'MATH'
                and getattr(node, 'operation', '') == 'MULTIPLY'
                and not node.inputs[1].is_linked
                and abs(float(node.inputs[1].default_value) - target) < 1e-6
            ):
                return node
    return None


def _spliceFadeOntoSocket(
    links: Any, source: Any, targets: list[Any], fade: Any, nodes: Any
) -> None:
    mul = nodes.new('ShaderNodeMath')
    mul.operation = 'MULTIPLY'
    links.new(source, mul.inputs[0])
    links.new(fade, mul.inputs[1])
    for target in targets:
        links.new(mul.outputs['Value'], target)


def _insertCoverageFade(nodes: Any, links: Any, fade: Any) -> None:
    scale = _cloudScaleNode(nodes)
    if scale is not None:
        outgoing = [link for link in links if link.from_socket == scale.outputs['Value']]
        if outgoing:
            targets = [link.to_socket for link in outgoing]
            for link in outgoing:
                links.remove(link)
            _spliceFadeOntoSocket(links, scale.outputs['Value'], targets, fade, nodes)
            return
    _insertCoverageFadeOnCloudMix(nodes, links, fade)


def _insertCoverageFadeOnCloudMix(nodes: Any, links: Any, fade: Any) -> None:
    for node in nodes:
        if getattr(node, 'type', '') != 'RGB':
            continue
        tint = tuple(float(channel) for channel in node.outputs[0].default_value[:3])
        if abs(tint[0] - 0.86) > 0.09 or abs(tint[1] - 0.88) > 0.09:
            continue
        for link in list(links):
            if link.from_socket != node.outputs[0]:
                continue
            mix = link.to_socket.node
            fac = mix.inputs['Factor'] if 'Factor' in mix.inputs else mix.inputs.get('Fac')
            if fac is None:
                continue
            incoming = next((item for item in links if item.to_socket == fac), None)
            if incoming is None:
                continue
            source = incoming.from_socket
            links.remove(incoming)
            _spliceFadeOntoSocket(links, source, [fac], fade, nodes)
            return


def _insertCloudPushWarp(
    nodes: Any,
    links: Any,
    normal: tuple[float, float, float],
    pull: Any,
    mixFac: Any,
) -> None:
    """Slide cloud UVs toward the impact so weather bunches at the first wave."""
    cloudTex = _cloudTextureNode(nodes)
    if cloudTex is None or 'Vector' not in cloudTex.inputs:
        return
    incoming = next(
        (link for link in links if link.to_socket == cloudTex.inputs['Vector']),
        None,
    )
    if incoming is None:
        return
    original = incoming.from_socket
    links.remove(incoming)
    warped = _pushedCloudVector(nodes, links, normal, pull)
    mix = nodes.new('ShaderNodeMix')
    if hasattr(mix, 'data_type'):
        mix.data_type = 'VECTOR'
        factor = mix.inputs['Factor'] if 'Factor' in mix.inputs else mix.inputs[0]
        links.new(mixFac, factor)
        links.new(original, mix.inputs['A'] if 'A' in mix.inputs else mix.inputs[1])
        links.new(warped, mix.inputs['B'] if 'B' in mix.inputs else mix.inputs[2])
        result = mix.outputs['Result'] if 'Result' in mix.outputs else mix.outputs[0]
        links.new(result, cloudTex.inputs['Vector'])
        return
    links.new(warped, cloudTex.inputs['Vector'])


def _driveCloudBlast(
    nodes: Any,
    links: Any,
    shock: Any,
    angle: Any,
    normal: tuple[float, float, float],
) -> Any:
    armed = _shockActive(nodes, links, shock)
    lead = _raggedLead(nodes, links, shock)
    swept = nodes.new('ShaderNodeMath')
    swept.operation = 'LESS_THAN'
    links.new(angle, swept.inputs[0])
    links.new(lead, swept.inputs[1])
    behind = _scaleBy(nodes, links, swept.outputs['Value'], armed)
    front = _angleTent(nodes, links, angle, lead, 0.055)
    pull = _cloudPullTowardImpact(nodes, links, angle, lead, behind, armed)
    _insertCloudPushWarp(nodes, links, normal, pull, behind)
    fade = _cloudCoverageScale(nodes, links, front, behind, armed)
    _insertCoverageFade(nodes, links, fade)
    _boostKpgCloudReadout(nodes)
    return front


def _boostKpgCloudReadout(nodes: Any) -> None:
    """Let a painted cyclone read as white weather, not a 75% grey oval."""
    for node in nodes:
        if (
            getattr(node, 'type', '') == 'MATH'
            and getattr(node, 'operation', '') == 'MULTIPLY'
            and abs(float(node.inputs[1].default_value) - 0.75) < 1e-6
        ):
            node.inputs[1].default_value = 0.95
        if getattr(node, 'type', '') == 'RGB':
            tint = tuple(float(channel) for channel in node.outputs[0].default_value[:3])
            if abs(tint[0] - 0.86) < 0.02 and abs(tint[1] - 0.88) < 0.02:
                node.outputs[0].default_value = (0.94, 0.95, 0.96, 1.0)


def _cloudMask(nodes: Any, links: Any, incoming: Any) -> Any:
    luma = nodes.new('ShaderNodeRGBToBW')
    colorIn = luma.inputs['Color'] if 'Color' in luma.inputs else luma.inputs[0]
    links.new(incoming, colorIn)
    bright = nodes.new('ShaderNodeMath')
    bright.operation = 'GREATER_THAN'
    bright.inputs[1].default_value = 0.48
    lumaOut = luma.outputs['Val'] if 'Val' in luma.outputs else luma.outputs[0]
    links.new(lumaOut, bright.inputs[0])
    return bright.outputs['Value']


def _mixWaveCloudPush(
    nodes: Any, links: Any, color: Any, angle: Any, tsunami: Any, front: Any
) -> Any:
    """Clouds stay warped. Do not paint a shock banner on the lead."""
    del nodes, links, angle, tsunami, front
    return color


def _landCoverMask(nodes: Any, links: Any, incoming: Any, cloud: Any) -> Any:
    try:
        land = nodes.new('ShaderNodeSeparateColor')
    except RuntimeError:
        land = nodes.new('ShaderNodeSeparateRGB')
    colorIn = land.inputs['Color'] if 'Color' in land.inputs else land.inputs[0]
    links.new(incoming, colorIn)
    red = land.outputs['Red'] if 'Red' in land.outputs else land.outputs[0]
    green = land.outputs['Green'] if 'Green' in land.outputs else land.outputs[1]
    blue = land.outputs['Blue'] if 'Blue' in land.outputs else land.outputs[2]
    veg = nodes.new('ShaderNodeMath')
    veg.operation = 'SUBTRACT'
    links.new(green, veg.inputs[0])
    links.new(blue, veg.inputs[1])
    dry = nodes.new('ShaderNodeMath')
    dry.operation = 'SUBTRACT'
    links.new(red, dry.inputs[0])
    links.new(blue, dry.inputs[1])
    dryBias = nodes.new('ShaderNodeMath')
    dryBias.operation = 'SUBTRACT'
    dryBias.inputs[1].default_value = 0.05
    links.new(dry.outputs['Value'], dryBias.inputs[0])
    cover = nodes.new('ShaderNodeMath')
    cover.operation = 'ADD'
    links.new(veg.outputs['Value'], cover.inputs[0])
    links.new(dryBias.outputs['Value'], cover.inputs[1])
    clip = nodes.new('ShaderNodeClamp')
    clip.inputs['Min'].default_value = 0.0
    clip.inputs['Max'].default_value = 1.0
    links.new(cover.outputs['Value'], clip.inputs['Value'])
    clear = nodes.new('ShaderNodeMath')
    clear.operation = 'SUBTRACT'
    clear.inputs[0].default_value = 1.0
    links.new(cloud, clear.inputs[1])
    exposed = nodes.new('ShaderNodeMath')
    exposed.operation = 'MULTIPLY'
    links.new(clip.outputs['Result'], exposed.inputs[0])
    links.new(clear.outputs['Value'], exposed.inputs[1])
    return exposed.outputs['Value']


def _landFireMask(nodes: Any, links: Any, incoming: Any, angle: Any, fire: Any, cloud: Any) -> Any:
    inside = nodes.new('ShaderNodeMath')
    inside.operation = 'LESS_THAN'
    links.new(angle, inside.inputs[0])
    links.new(fire.outputs[0], inside.inputs[1])
    cover = _landCoverMask(nodes, links, incoming, cloud)
    mask = nodes.new('ShaderNodeMath')
    mask.operation = 'MULTIPLY'
    links.new(inside.outputs['Value'], mask.inputs[0])
    links.new(cover, mask.inputs[1])
    return mask.outputs['Value']


def _mixDieback(nodes: Any, links: Any, color: Any, incoming: Any, cloud: Any, dieback: Any) -> Any:
    """Hold land brown after the soot lifts. Green only returns with the dieback envelope."""
    del cloud
    ocean = _oceanMask(nodes, links, incoming)
    land = nodes.new('ShaderNodeMath')
    land.operation = 'SUBTRACT'
    land.inputs[0].default_value = 1.0
    links.new(ocean, land.inputs[1])
    amount = nodes.new('ShaderNodeMath')
    amount.operation = 'MULTIPLY'
    links.new(land.outputs['Value'], amount.inputs[0])
    links.new(dieback.outputs[0], amount.inputs[1])
    dead = _colorMix(nodes)
    _linkMix(links, dead, color, (0.17, 0.11, 0.05, 1.0), amount.outputs['Value'])
    return _mixOut(dead)


def _attachImpactWeather(
    material: Any,
    normal: tuple[float, float, float],
    inbound: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    nodeTree = material.node_tree
    if nodeTree is None:
        return {}
    nodes = nodeTree.nodes
    links = nodeTree.links
    principled = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
    if principled is None or 'Base Color' not in principled.inputs:
        return {}
    incoming = next(
        (link.from_socket for link in links if link.to_socket == principled.inputs['Base Color']),
        None,
    )
    if incoming is None:
        return {}
    for link in list(links):
        if link.to_socket == principled.inputs['Base Color']:
            links.remove(link)
    shock = _valueNode(nodes, 'KpgShock', 0.0)
    fire = _valueNode(nodes, 'KpgFireFront', 0.0)
    soot = _valueNode(nodes, 'KpgSoot', 0.0)
    veil = _valueNode(nodes, 'KpgVeil', 0.0)
    site = _valueNode(nodes, 'KpgSite', 0.0)
    fallout = _valueNode(nodes, 'KpgFallout', 0.0)
    tsunami = _valueNode(nodes, 'KpgTsunami', 0.0)
    dieback = _valueNode(nodes, 'KpgDieback', 0.0)
    angle = _impactAngleSocket(nodes, links, normal, inbound)
    _, ring = _shockGates(nodes, links, angle, veil)
    front = _driveCloudBlast(nodes, links, tsunami, angle, normal)
    pushed = _mixWaveCloudPush(nodes, links, incoming, angle, tsunami, front)
    cloud = _cloudMask(nodes, links, incoming)
    rim = _colorMix(nodes)
    burned = _colorMix(nodes)
    sooted = _colorMix(nodes)
    rimAmt = nodes.new('ShaderNodeMath')
    rimAmt.operation = 'MULTIPLY'
    rimAmt.inputs[1].default_value = 0.0
    links.new(ring, rimAmt.inputs[0])
    _linkMix(links, rim, pushed, (1.0, 0.58, 0.20, 1.0), rimAmt.outputs['Value'])
    landMask = _landFireMask(nodes, links, incoming, angle, veil, cloud)
    burnAmt = nodes.new('ShaderNodeMath')
    burnAmt.operation = 'MULTIPLY'
    burnAmt.inputs[1].default_value = 0.88
    links.new(landMask, burnAmt.inputs[0])
    _linkMix(links, burned, _mixOut(rim), (0.10, 0.045, 0.02, 1.0), burnAmt.outputs['Value'])
    _linkMix(links, sooted, _mixOut(burned), (0.11, 0.09, 0.08, 1.0), soot.outputs[0])
    wilted = _mixDieback(nodes, links, _mixOut(sooted), incoming, cloud, dieback)
    crater = _valueNode(nodes, 'KpgCrater', 0.0)
    smolder = _valueNode(nodes, 'KpgSmolder', 0.0)
    scar = _mixSmolder(nodes, links, wilted, angle, smolder, veil)
    umbrella = _mixTongaUmbrella(nodes, links, scar, angle, site, normal)
    rain = _mixMoltenRain(nodes, links, umbrella, angle, fallout)
    foam = _mixTsunamiFoam(nodes, links, rain, angle, tsunami, incoming)
    cratered = _mixImpactCrater(nodes, links, foam, angle, crater)
    links.new(cratered, principled.inputs['Base Color'])
    flash = _valueNode(nodes, 'KpgFlash', 0.0)
    glow = _valueNode(nodes, 'KpgSiteGlow', 0.0)
    _wireImpactEmission(nodes, links, principled, angle, landMask, flash, shock, glow)
    return {
        'shock': shock,
        'fire': fire,
        'soot': soot,
        'flash': flash,
        'tsunami': tsunami,
        'smolder': smolder,
        'veil': veil,
        'site': site,
        'fallout': fallout,
        'glow': glow,
        'dieback': dieback,
        'crater': crater,
    }


def _oceanMask(nodes: Any, links: Any, incoming: Any) -> Any:
    try:
        split = nodes.new('ShaderNodeSeparateColor')
    except RuntimeError:
        split = nodes.new('ShaderNodeSeparateRGB')
    colorIn = split.inputs['Color'] if 'Color' in split.inputs else split.inputs[0]
    links.new(incoming, colorIn)
    red = split.outputs['Red'] if 'Red' in split.outputs else split.outputs[0]
    green = split.outputs['Green'] if 'Green' in split.outputs else split.outputs[1]
    blue = split.outputs['Blue'] if 'Blue' in split.outputs else split.outputs[2]
    veg = nodes.new('ShaderNodeMath')
    veg.operation = 'SUBTRACT'
    links.new(green, veg.inputs[0])
    links.new(blue, veg.inputs[1])
    dry = nodes.new('ShaderNodeMath')
    dry.operation = 'SUBTRACT'
    links.new(red, dry.inputs[0])
    links.new(blue, dry.inputs[1])
    land = nodes.new('ShaderNodeMath')
    land.operation = 'ADD'
    links.new(veg.outputs['Value'], land.inputs[0])
    links.new(dry.outputs['Value'], land.inputs[1])
    landClip = nodes.new('ShaderNodeClamp')
    landClip.inputs['Min'].default_value = 0.0
    landClip.inputs['Max'].default_value = 1.0
    links.new(land.outputs['Value'], landClip.inputs['Value'])
    ocean = nodes.new('ShaderNodeMath')
    ocean.operation = 'SUBTRACT'
    ocean.inputs[0].default_value = 1.0
    links.new(landClip.outputs['Result'], ocean.inputs[1])
    return ocean.outputs['Value']


def _angleRing(nodes: Any, links: Any, angle: Any, radius: Any, width: float) -> Any:
    delta = nodes.new('ShaderNodeMath')
    delta.operation = 'SUBTRACT'
    links.new(angle, delta.inputs[0])
    links.new(radius, delta.inputs[1])
    ringAbs = nodes.new('ShaderNodeMath')
    ringAbs.operation = 'ABSOLUTE'
    links.new(delta.outputs['Value'], ringAbs.inputs[0])
    norm = nodes.new('ShaderNodeMath')
    norm.operation = 'DIVIDE'
    norm.inputs[1].default_value = width
    links.new(ringAbs.outputs['Value'], norm.inputs[0])
    ring = nodes.new('ShaderNodeMath')
    ring.operation = 'LESS_THAN'
    ring.inputs[1].default_value = 1.0
    links.new(norm.outputs['Value'], ring.inputs[0])
    grown = nodes.new('ShaderNodeMath')
    grown.operation = 'GREATER_THAN'
    grown.inputs[1].default_value = 0.03
    links.new(radius, grown.inputs[0])
    return _mathMul(nodes, links, ring.outputs['Value'], grown.outputs['Value'])


def _mixTsunamiFoam(
    nodes: Any, links: Any, color: Any, angle: Any, tsunami: Any, incoming: Any
) -> Any:
    """One thin ocean ring. Mixed last. Leaves the far side instead of parking."""
    ocean = _oceanMask(nodes, links, incoming)
    lead = tsunami.outputs[0]
    rings = _angleRing(nodes, links, angle, lead, 0.012)
    cover = _mathMul(nodes, links, rings, ocean)
    fade = nodes.new('ShaderNodeMapRange')
    fade.inputs['From Min'].default_value = 2.20
    fade.inputs['From Max'].default_value = math.pi
    fade.inputs['To Min'].default_value = 1.0
    fade.inputs['To Max'].default_value = 0.0
    if hasattr(fade, 'clamp'):
        fade.clamp = True
    links.new(lead, fade.inputs['Value'])
    cover = _mathMul(nodes, links, cover, fade.outputs['Result'])
    gone = nodes.new('ShaderNodeMath')
    gone.operation = 'GREATER_THAN'
    gone.inputs[1].default_value = math.pi
    links.new(lead, gone.inputs[0])
    keep = nodes.new('ShaderNodeMath')
    keep.operation = 'SUBTRACT'
    keep.inputs[0].default_value = 1.0
    links.new(gone.outputs['Value'], keep.inputs[1])
    cover = _mathMul(nodes, links, cover, keep.outputs['Value'])
    amount = nodes.new('ShaderNodeMath')
    amount.operation = 'MULTIPLY'
    amount.inputs[1].default_value = 0.90
    links.new(cover, amount.inputs[0])
    amountClip = nodes.new('ShaderNodeClamp')
    amountClip.inputs['Min'].default_value = 0.0
    amountClip.inputs['Max'].default_value = 1.0
    links.new(amount.outputs['Value'], amountClip.inputs['Value'])
    foam = _colorMix(nodes)
    _linkMix(links, foam, color, (0.82, 0.86, 0.90, 1.0), amountClip.outputs['Result'])
    return _mixOut(foam)


def _mixMoltenRain(nodes: Any, links: Any, color: Any, angle: Any, fallout: Any) -> Any:
    """Glowing fallback on the surface. Not flying pearls."""
    inside = nodes.new('ShaderNodeMath')
    inside.operation = 'LESS_THAN'
    links.new(angle, inside.inputs[0])
    links.new(fallout.outputs[0], inside.inputs[1])
    armed = nodes.new('ShaderNodeMath')
    armed.operation = 'GREATER_THAN'
    armed.inputs[1].default_value = 0.04
    links.new(fallout.outputs[0], armed.inputs[0])
    grain = _generatedNoise(nodes, links, 150.0, 5.0)
    hot = nodes.new('ShaderNodeMath')
    hot.operation = 'SUBTRACT'
    hot.inputs[1].default_value = 0.78
    links.new(grain, hot.inputs[0])
    hot = _mathMulConst(nodes, links, hot.outputs['Value'], 6.5)
    clip = nodes.new('ShaderNodeClamp')
    clip.inputs['Min'].default_value = 0.0
    clip.inputs['Max'].default_value = 1.0
    links.new(hot, clip.inputs['Value'])
    spots = _mathMul(nodes, links, clip.outputs['Result'], inside.outputs['Value'])
    spots = _mathMul(nodes, links, spots, armed.outputs['Value'])
    amount = nodes.new('ShaderNodeMath')
    amount.operation = 'MULTIPLY'
    amount.inputs[1].default_value = 0.82
    links.new(spots, amount.inputs[0])
    rain = _colorMix(nodes)
    _linkMix(links, rain, color, (1.0, 0.32, 0.05, 1.0), amount.outputs['Value'])
    return _mixOut(rain)


def _mixSmolder(nodes: Any, links: Any, color: Any, angle: Any, smolder: Any, veil: Any) -> Any:
    falloff = nodes.new('ShaderNodeMapRange')
    falloff.inputs['From Min'].default_value = 0.02
    falloff.inputs['From Max'].default_value = 0.08
    links.new(veil.outputs[0], falloff.inputs['From Max'])
    falloff.inputs['To Min'].default_value = 1.0
    falloff.inputs['To Max'].default_value = 0.0
    if hasattr(falloff, 'clamp'):
        falloff.clamp = True
    links.new(angle, falloff.inputs['Value'])
    disk = nodes.new('ShaderNodeMath')
    disk.operation = 'MULTIPLY'
    links.new(falloff.outputs['Result'], disk.inputs[0])
    links.new(smolder.outputs[0], disk.inputs[1])
    try:
        split = nodes.new('ShaderNodeSeparateColor')
    except RuntimeError:
        split = nodes.new('ShaderNodeSeparateRGB')
    colorIn = split.inputs['Color'] if 'Color' in split.inputs else split.inputs[0]
    links.new(color, colorIn)
    red = split.outputs['Red'] if 'Red' in split.outputs else split.outputs[0]
    green = split.outputs['Green'] if 'Green' in split.outputs else split.outputs[1]
    blue = split.outputs['Blue'] if 'Blue' in split.outputs else split.outputs[2]
    warm = nodes.new('ShaderNodeMath')
    warm.operation = 'ADD'
    links.new(red, warm.inputs[0])
    links.new(green, warm.inputs[1])
    half = nodes.new('ShaderNodeMath')
    half.operation = 'MULTIPLY'
    half.inputs[1].default_value = 0.5
    links.new(warm.outputs['Value'], half.inputs[0])
    land = nodes.new('ShaderNodeMath')
    land.operation = 'SUBTRACT'
    links.new(half.outputs['Value'], land.inputs[0])
    links.new(blue, land.inputs[1])
    landClip = nodes.new('ShaderNodeClamp')
    landClip.inputs['Min'].default_value = 0.0
    landClip.inputs['Max'].default_value = 1.0
    links.new(land.outputs['Value'], landClip.inputs['Value'])
    landAmt = nodes.new('ShaderNodeMath')
    landAmt.operation = 'MULTIPLY'
    landAmt.inputs[1].default_value = 0.58
    links.new(landClip.outputs['Result'], landAmt.inputs[0])
    weight = nodes.new('ShaderNodeMath')
    weight.operation = 'ADD'
    weight.inputs[0].default_value = 0.38
    links.new(landAmt.outputs['Value'], weight.inputs[1])
    amount = nodes.new('ShaderNodeMath')
    amount.operation = 'MULTIPLY'
    links.new(disk.outputs['Value'], amount.inputs[0])
    links.new(weight.outputs['Value'], amount.inputs[1])
    scar = _colorMix(nodes)
    _linkMix(links, scar, color, (0.05, 0.03, 0.018, 1.0), amount.outputs['Value'])
    return _mixOut(scar)


def _mixImpactCrater(nodes: Any, links: Any, color: Any, angle: Any, armed: Any) -> Any:
    """Dark Chicxulub bowl with an ashy lip. Cinema-scale, still a local hole."""
    floor = nodes.new('ShaderNodeMapRange')
    floor.inputs['From Min'].default_value = 0.0
    floor.inputs['From Max'].default_value = 0.062
    floor.inputs['To Min'].default_value = 1.0
    floor.inputs['To Max'].default_value = 0.0
    if hasattr(floor, 'clamp'):
        floor.clamp = True
    links.new(angle, floor.inputs['Value'])
    outer = nodes.new('ShaderNodeMapRange')
    outer.inputs['From Min'].default_value = 0.0
    outer.inputs['From Max'].default_value = 0.092
    outer.inputs['To Min'].default_value = 1.0
    outer.inputs['To Max'].default_value = 0.0
    if hasattr(outer, 'clamp'):
        outer.clamp = True
    links.new(angle, outer.inputs['Value'])
    rim = nodes.new('ShaderNodeMath')
    rim.operation = 'SUBTRACT'
    links.new(outer.outputs['Result'], rim.inputs[0])
    links.new(floor.outputs['Result'], rim.inputs[1])
    floorAmt = nodes.new('ShaderNodeMath')
    floorAmt.operation = 'MULTIPLY'
    links.new(floor.outputs['Result'], floorAmt.inputs[0])
    links.new(armed.outputs[0], floorAmt.inputs[1])
    rimAmt = nodes.new('ShaderNodeMath')
    rimAmt.operation = 'MULTIPLY'
    links.new(rim.outputs['Value'], rimAmt.inputs[0])
    links.new(armed.outputs[0], rimAmt.inputs[1])
    bowl = _colorMix(nodes)
    _linkMix(links, bowl, color, (0.025, 0.018, 0.012, 1.0), floorAmt.outputs['Value'])
    lip = _colorMix(nodes)
    _linkMix(links, lip, _mixOut(bowl), (0.22, 0.16, 0.10, 1.0), rimAmt.outputs['Value'])
    return _mixOut(lip)


def _sphereNoise(nodes: Any, links: Any, scale: float, detail: float) -> Any:
    texcoord = nodes.new('ShaderNodeTexCoord')
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = scale
    if 'Detail' in noise.inputs:
        noise.inputs['Detail'].default_value = detail
    if 'Roughness' in noise.inputs:
        noise.inputs['Roughness'].default_value = 0.62
    links.new(texcoord.outputs['Normal'], noise.inputs['Vector'])
    return noise.outputs['Fac']


def _polarNoise(
    nodes: Any,
    links: Any,
    angle: Any,
    azimuth: Any,
    scale: float,
    detail: float,
) -> Any:
    radial = nodes.new('ShaderNodeMath')
    radial.operation = 'MULTIPLY'
    radial.inputs[1].default_value = 22.0
    links.new(angle, radial.inputs[0])
    local = nodes.new('ShaderNodeCombineXYZ')
    links.new(radial.outputs['Value'], local.inputs['X'])
    links.new(azimuth, local.inputs['Y'])
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = scale
    if 'Detail' in noise.inputs:
        noise.inputs['Detail'].default_value = detail
    if 'Roughness' in noise.inputs:
        noise.inputs['Roughness'].default_value = 0.70
    links.new(local.outputs['Vector'], noise.inputs['Vector'])
    return noise.outputs['Fac']


def _generatedNoise(nodes: Any, links: Any, scale: float, detail: float) -> Any:
    texcoord = nodes.new('ShaderNodeTexCoord')
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = scale
    if 'Detail' in noise.inputs:
        noise.inputs['Detail'].default_value = detail
    if 'Roughness' in noise.inputs:
        noise.inputs['Roughness'].default_value = 0.70
    links.new(texcoord.outputs['Generated'], noise.inputs['Vector'])
    return noise.outputs['Fac']


def _tangentNoise(
    nodes: Any,
    links: Any,
    eastDot: Any,
    northDot: Any,
    scale: float,
    detail: float,
) -> Any:
    local = nodes.new('ShaderNodeCombineXYZ')
    links.new(eastDot, local.inputs['X'])
    links.new(northDot, local.inputs['Y'])
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = scale
    if 'Detail' in noise.inputs:
        noise.inputs['Detail'].default_value = detail
    if 'Roughness' in noise.inputs:
        noise.inputs['Roughness'].default_value = 0.68
    links.new(local.outputs['Vector'], noise.inputs['Vector'])
    return noise.outputs['Fac']


def _mixTongaUmbrella(
    nodes: Any,
    links: Any,
    color: Any,
    angle: Any,
    shock: Any,
    normal: tuple[float, float, float],
) -> Any:
    """Fiery gray explosion cloud: broken billows, hot core, radial ejecta."""
    east, north = _tangentPair(normal)
    radius = shock.outputs[0]
    texcoord = nodes.new('ShaderNodeTexCoord')
    heading = nodes.new('ShaderNodeVectorMath')
    heading.operation = 'NORMALIZE'
    links.new(texcoord.outputs['Normal'], heading.inputs[0])
    eastNode = nodes.new('ShaderNodeCombineXYZ')
    eastNode.inputs[0].default_value = east[0]
    eastNode.inputs[1].default_value = east[1]
    eastNode.inputs[2].default_value = east[2]
    northNode = nodes.new('ShaderNodeCombineXYZ')
    northNode.inputs[0].default_value = north[0]
    northNode.inputs[1].default_value = north[1]
    northNode.inputs[2].default_value = north[2]
    alongEast = nodes.new('ShaderNodeVectorMath')
    alongEast.operation = 'DOT_PRODUCT'
    links.new(heading.outputs['Vector'], alongEast.inputs[0])
    links.new(eastNode.outputs['Vector'], alongEast.inputs[1])
    alongNorth = nodes.new('ShaderNodeVectorMath')
    alongNorth.operation = 'DOT_PRODUCT'
    links.new(heading.outputs['Vector'], alongNorth.inputs[0])
    links.new(northNode.outputs['Vector'], alongNorth.inputs[1])
    azimuth = nodes.new('ShaderNodeMath')
    azimuth.operation = 'ARCTAN2'
    links.new(alongEast.outputs['Value'], azimuth.inputs[0])
    links.new(alongNorth.outputs['Value'], azimuth.inputs[1])
    wind = nodes.new('ShaderNodeMath')
    wind.operation = 'SUBTRACT'
    wind.inputs[1].default_value = 0.85
    links.new(azimuth.outputs['Value'], wind.inputs[0])
    oval = nodes.new('ShaderNodeMath')
    oval.operation = 'COSINE'
    links.new(wind.outputs['Value'], oval.inputs[0])
    ovalAmt = nodes.new('ShaderNodeMath')
    ovalAmt.operation = 'MULTIPLY'
    ovalAmt.inputs[1].default_value = 0.22
    links.new(oval.outputs['Value'], ovalAmt.inputs[0])
    ovalBias = nodes.new('ShaderNodeMath')
    ovalBias.operation = 'ADD'
    ovalBias.inputs[0].default_value = 1.0
    links.new(ovalAmt.outputs['Value'], ovalBias.inputs[1])
    rag = _generatedNoise(nodes, links, 55.0, 3.0)
    ragAmt = nodes.new('ShaderNodeMath')
    ragAmt.operation = 'MULTIPLY'
    ragAmt.inputs[1].default_value = 0.32
    links.new(rag, ragAmt.inputs[0])
    ragBias = nodes.new('ShaderNodeMath')
    ragBias.operation = 'ADD'
    ragBias.inputs[0].default_value = 0.90
    links.new(ragAmt.outputs['Value'], ragBias.inputs[1])
    reach = nodes.new('ShaderNodeMath')
    reach.operation = 'MULTIPLY'
    links.new(radius, reach.inputs[0])
    links.new(ovalBias.outputs['Value'], reach.inputs[1])
    reach = _mathMul(nodes, links, reach.outputs['Value'], ragBias.outputs['Value'])
    gap = nodes.new('ShaderNodeMath')
    gap.operation = 'SUBTRACT'
    links.new(reach, gap.inputs[0])
    links.new(angle, gap.inputs[1])
    edge = nodes.new('ShaderNodeMath')
    edge.operation = 'DIVIDE'
    edge.inputs[1].default_value = 0.018
    links.new(gap.outputs['Value'], edge.inputs[0])
    inside = nodes.new('ShaderNodeClamp')
    inside.inputs['Min'].default_value = 0.0
    inside.inputs['Max'].default_value = 1.0
    links.new(edge.outputs['Value'], inside.inputs['Value'])
    armed = nodes.new('ShaderNodeMath')
    armed.operation = 'GREATER_THAN'
    armed.inputs[1].default_value = 0.02
    links.new(shock.outputs[0], armed.inputs[0])
    grain = _generatedNoise(nodes, links, 220.0, 6.0)
    body = _generatedNoise(nodes, links, 85.0, 5.0)
    puff = nodes.new('ShaderNodeMath')
    puff.operation = 'SUBTRACT'
    puff.inputs[1].default_value = 0.18
    links.new(body, puff.inputs[0])
    puff = _mathMulConst(nodes, links, puff.outputs['Value'], 2.6)
    puffClip = nodes.new('ShaderNodeClamp')
    puffClip.inputs['Min'].default_value = 0.0
    puffClip.inputs['Max'].default_value = 1.0
    links.new(puff, puffClip.inputs['Value'])
    coreSpan = nodes.new('ShaderNodeValue')
    coreSpan.outputs[0].default_value = 0.05
    coreRatio = nodes.new('ShaderNodeMath')
    coreRatio.operation = 'DIVIDE'
    links.new(angle, coreRatio.inputs[0])
    links.new(coreSpan.outputs['Value'], coreRatio.inputs[1])
    coreFall = nodes.new('ShaderNodeMath')
    coreFall.operation = 'SUBTRACT'
    coreFall.inputs[0].default_value = 1.0
    links.new(coreRatio.outputs['Value'], coreFall.inputs[1])
    coreClip = nodes.new('ShaderNodeClamp')
    coreClip.inputs['Min'].default_value = 0.0
    coreClip.inputs['Max'].default_value = 1.0
    links.new(coreFall.outputs['Value'], coreClip.inputs['Value'])
    fill = nodes.new('ShaderNodeMath')
    fill.operation = 'MAXIMUM'
    links.new(puffClip.outputs['Result'], fill.inputs[0])
    links.new(coreClip.outputs['Result'], fill.inputs[1])
    sheet = nodes.new('ShaderNodeMath')
    sheet.operation = 'MULTIPLY'
    links.new(inside.outputs['Result'], sheet.inputs[0])
    links.new(armed.outputs['Value'], sheet.inputs[1])
    sheet = _mathMul(nodes, links, sheet.outputs['Value'], fill.outputs['Value'])
    chunks = _generatedNoise(nodes, links, 130.0, 4.0)
    chunk = nodes.new('ShaderNodeMath')
    chunk.operation = 'SUBTRACT'
    chunk.inputs[1].default_value = 0.58
    links.new(chunks, chunk.inputs[0])
    chunk = _mathMulConst(nodes, links, chunk.outputs['Value'], 3.0)
    chunkClip = nodes.new('ShaderNodeClamp')
    chunkClip.inputs['Min'].default_value = 0.0
    chunkClip.inputs['Max'].default_value = 1.0
    links.new(chunk, chunkClip.inputs['Value'])
    streak = _mathMul(nodes, links, chunkClip.outputs['Result'], sheet)
    cover = sheet
    grit = nodes.new('ShaderNodeMath')
    grit.operation = 'MULTIPLY'
    grit.inputs[1].default_value = 0.16
    links.new(grain, grit.inputs[0])
    density = _mathAdd(nodes, links, grit.outputs['Value'], 0.86)
    amount = _mathMul(nodes, links, cover, density)
    amountClip = nodes.new('ShaderNodeClamp')
    amountClip.inputs['Min'].default_value = 0.0
    amountClip.inputs['Max'].default_value = 1.0
    links.new(amount, amountClip.inputs['Value'])
    heat = nodes.new('ShaderNodeMath')
    heat.operation = 'MULTIPLY'
    heat.inputs[1].default_value = 0.25
    links.new(body, heat.inputs[0])
    heat = _mathAdd(nodes, links, heat.outputs['Value'], 0.75)
    heat = _mathMul(nodes, links, coreClip.outputs['Result'], heat)
    ash = _colorMix(nodes)
    _linkMix(links, ash, (0.14, 0.12, 0.10, 1.0), (0.38, 0.28, 0.18, 1.0), grain)
    fire = _colorMix(nodes)
    _linkMix(links, fire, _mixOut(ash), (1.0, 0.48, 0.10, 1.0), heat)
    sooty = _colorMix(nodes)
    _linkMix(links, sooty, _mixOut(fire), (0.07, 0.05, 0.04, 1.0), streak)
    umbrella = _colorMix(nodes)
    _linkMix(links, umbrella, color, _mixOut(sooty), amountClip.outputs['Result'])
    return _mixOut(umbrella)


def _mathMul(nodes: Any, links: Any, left: Any, right: Any) -> Any:
    mul = nodes.new('ShaderNodeMath')
    mul.operation = 'MULTIPLY'
    links.new(left, mul.inputs[0])
    links.new(right, mul.inputs[1])
    return mul.outputs['Value']


def _mathMulConst(nodes: Any, links: Any, value: Any, constant: float) -> Any:
    mul = nodes.new('ShaderNodeMath')
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = constant
    links.new(value, mul.inputs[0])
    return mul.outputs['Value']


def _mathAddSocket(nodes: Any, links: Any, left: Any, right: Any) -> Any:
    added = nodes.new('ShaderNodeMath')
    added.operation = 'ADD'
    links.new(left, added.inputs[0])
    links.new(right, added.inputs[1])
    return added.outputs['Value']


def _mathAdd(nodes: Any, links: Any, value: Any, constant: float) -> Any:
    added = nodes.new('ShaderNodeMath')
    added.operation = 'ADD'
    added.inputs[0].default_value = constant
    links.new(value, added.inputs[1])
    return added.outputs['Value']


def _wireImpactEmission(
    nodes: Any,
    links: Any,
    principled: Any,
    angle: Any,
    landMask: Any,
    flash: Any,
    shock: Any | None = None,
    glow: Any | None = None,
) -> None:
    if 'Emission Color' in principled.inputs:
        ember = nodes.new('ShaderNodeRGB')
        ember.outputs[0].default_value = (1.0, 0.55, 0.12, 1.0)
        links.new(ember.outputs[0], principled.inputs['Emission Color'])
    if 'Emission Strength' not in principled.inputs:
        return
    falloff = nodes.new('ShaderNodeMapRange')
    falloff.inputs['From Min'].default_value = 0.0
    falloff.inputs['From Max'].default_value = 0.065
    falloff.inputs['To Min'].default_value = 1.0
    falloff.inputs['To Max'].default_value = 0.0
    if hasattr(falloff, 'clamp'):
        falloff.clamp = True
    links.new(angle, falloff.inputs['Value'])
    flashHot = nodes.new('ShaderNodeMath')
    flashHot.operation = 'MULTIPLY'
    links.new(falloff.outputs['Result'], flashHot.inputs[0])
    links.new(flash.outputs[0], flashHot.inputs[1])
    flashAmt = nodes.new('ShaderNodeMath')
    flashAmt.operation = 'MULTIPLY'
    flashAmt.inputs[1].default_value = 0.0
    links.new(flashHot.outputs['Value'], flashAmt.inputs[0])
    landGlow = nodes.new('ShaderNodeMath')
    landGlow.operation = 'MULTIPLY'
    landGlow.inputs[1].default_value = 0.0
    links.new(landMask, landGlow.inputs[0])
    core = nodes.new('ShaderNodeMapRange')
    core.inputs['From Min'].default_value = 0.0
    core.inputs['From Max'].default_value = 0.036
    core.inputs['To Min'].default_value = 1.0
    core.inputs['To Max'].default_value = 0.0
    if hasattr(core, 'clamp'):
        core.clamp = True
    links.new(angle, core.inputs['Value'])
    armed = nodes.new('ShaderNodeMath')
    armed.operation = 'GREATER_THAN'
    armed.inputs[1].default_value = 0.02
    if shock is not None:
        links.new(shock.outputs[0], armed.inputs[0])
    else:
        armed.inputs[0].default_value = 0.0
    flicker = _generatedNoise(nodes, links, 90.0, 4.0)
    flickerAmt = nodes.new('ShaderNodeMath')
    flickerAmt.operation = 'MULTIPLY'
    flickerAmt.inputs[1].default_value = 0.20
    links.new(flicker, flickerAmt.inputs[0])
    flickerAmt = _mathAdd(nodes, links, flickerAmt.outputs['Value'], 0.80)
    emberAmt = nodes.new('ShaderNodeMath')
    emberAmt.operation = 'MULTIPLY'
    emberAmt.inputs[1].default_value = 4.2
    links.new(core.outputs['Result'], emberAmt.inputs[0])
    emberAmt = _mathMul(nodes, links, emberAmt.outputs['Value'], armed.outputs['Value'])
    emberAmt = _mathMul(nodes, links, emberAmt, flickerAmt)
    punch = nodes.new('ShaderNodeMath')
    punch.operation = 'MULTIPLY'
    punch.inputs[1].default_value = 2.0
    links.new(flash.outputs[0], punch.inputs[0])
    punch = _mathAdd(nodes, links, punch.outputs['Value'], 1.0)
    emberAmt = _mathMul(nodes, links, emberAmt, punch)
    if glow is not None:
        emberAmt = _mathMul(nodes, links, emberAmt, glow.outputs[0])
    total = nodes.new('ShaderNodeMath')
    total.operation = 'ADD'
    links.new(flashAmt.outputs['Value'], total.inputs[0])
    links.new(landGlow.outputs['Value'], total.inputs[1])
    total = _mathAddSocket(nodes, links, total.outputs['Value'], emberAmt)
    links.new(total, principled.inputs['Emission Strength'])


def _linkMix(links: Any, mix: Any, colorA: Any, colorB: Any, factor: Any) -> None:
    if 'A' in mix.inputs:
        destA, destB, destFac = 'A', 'B', 'Factor'
    else:
        destA, destB, destFac = 'Color1', 'Color2', 'Fac'
    if hasattr(colorA, 'id_data'):
        links.new(colorA, mix.inputs[destA])
    else:
        mix.inputs[destA].default_value = colorA
    if hasattr(colorB, 'id_data'):
        links.new(colorB, mix.inputs[destB])
    else:
        mix.inputs[destB].default_value = colorB
    links.new(factor, mix.inputs[destFac])


def _buildEarth(
    bpy: Any, flyby: ModuleType, job: dict[str, Any]
) -> tuple[Any, float, dict[str, Any]]:
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
    normal = tuple(float(value) for value in job['contact']['normal'])
    inbound = tuple(float(value) for value in job['contact']['inbound'])
    weather = _attachImpactWeather(material, normal, inbound)
    _addEarthLimb(material)
    if earth.data.materials:
        earth.data.materials[0] = material
    else:
        earth.data.materials.append(material)
    return earth, radius, weather


def _addEarthLimb(material: Any) -> None:
    """Blue limb on the globe itself — not a second transparent shell."""
    nodeTree = getattr(material, 'node_tree', None)
    if nodeTree is None:
        return
    nodes = nodeTree.nodes
    links = nodeTree.links
    output = next((node for node in nodes if node.type == 'OUTPUT_MATERIAL'), None)
    if output is None or 'Surface' not in output.inputs:
        return
    incoming = next(
        (link.from_socket for link in links if link.to_socket == output.inputs['Surface']),
        None,
    )
    if incoming is None:
        return
    for link in list(links):
        if link.to_socket == output.inputs['Surface']:
            links.remove(link)
    weight = nodes.new('ShaderNodeLayerWeight')
    weight.inputs['Blend'].default_value = 0.22
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (0.42, 0.70, 1.0, 1.0)
    emission.inputs['Strength'].default_value = 0.78
    mix = nodes.new('ShaderNodeMixShader')
    links.new(incoming, mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(weight.outputs['Fresnel'], mix.inputs['Fac'])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])


def _createImpactorRock(bpy: Any, name: str, radius: float) -> Any:
    return _createLumpyRock(bpy, name, radius, seed=4, subdivisions=3, squash=0.16)


def _buildImpactor(bpy: Any, flyby: ModuleType, job: dict[str, Any], earthRadius: float) -> Any:
    del flyby
    scale = float(job['impactor']['radiusScale'])
    impactor = _createImpactorRock(bpy, 'Impactor', max(earthRadius * scale, earthRadius * 1e-4))
    inbound = tuple(float(value) for value in job['contact']['inbound'])
    _alignPlusZ(impactor, inbound)
    rock = bpy.data.materials.new(name='ImpactorRock')
    nodeTree = getattr(rock, 'node_tree', None)
    if nodeTree is None:
        rock.use_nodes = True
        nodeTree = rock.node_tree
    principled = next(node for node in nodeTree.nodes if node.type == 'BSDF_PRINCIPLED')
    color = [float(channel) for channel in job['impactor']['colorRgba']]
    principled.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1.0)
    texturePath = str(job['impactor'].get('colorTexture') or '')
    resolved = _jobFile(texturePath) if texturePath else None
    if resolved is not None and resolved.is_file():
        image = bpy.data.images.load(str(resolved.resolve()), check_existing=True)
        tex = nodeTree.nodes.new('ShaderNodeTexImage')
        tex.image = image
        if hasattr(tex, 'interpolation'):
            tex.interpolation = 'Cubic'
        texcoord = nodeTree.nodes.new('ShaderNodeTexCoord')
        nodeTree.links.new(texcoord.outputs['UV'], tex.inputs['Vector'])
        nodeTree.links.new(tex.outputs['Color'], principled.inputs['Base Color'])
    if 'Roughness' in principled.inputs:
        principled.inputs['Roughness'].default_value = 0.94
    if 'Emission Strength' in principled.inputs:
        principled.inputs['Emission Strength'].default_value = 0.0
    impactor.data.materials.append(rock)
    return impactor


def _buildSiteMarker(bpy: Any, flyby: ModuleType, job: dict[str, Any], earthRadius: float) -> Any:
    del flyby
    size = earthRadius * float(job['contact']['maxFireballRadii'])
    material = _emissionMaterial(bpy, 'KpgSiteMark', (0.95, 0.28, 0.06), 2.8, 1.0)
    return _buildPolyline(
        bpy,
        'KpgSiteMark',
        _circlePoints(max(size, earthRadius * 1e-4)),
        earthRadius * 0.004,
        material,
    )


def _buildPolyline(
    bpy: Any,
    name: str,
    points: list[tuple[float, float, float]],
    bevel: float,
    material: Any,
) -> Any:
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.bevel_depth = bevel
    if hasattr(curveData, 'use_fill_caps'):
        curveData.use_fill_caps = True
    spline = curveData.splines.new('POLY')
    spline.points.add(max(len(points) - 1, 0))
    for index, point in enumerate(points):
        spline.points[index].co = (point[0], point[1], point[2], 1.0)
    obj = bpy.data.objects.new(name, curveData)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    if hasattr(obj, 'visible_shadow'):
        obj.visible_shadow = False
    return obj


def _circlePoints(radius: float, count: int = 64) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        points.append((radius * math.cos(angle), radius * math.sin(angle), 0.0))
    points.append(points[0])
    return points


def _buildRod(
    bpy: Any,
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    material: Any,
) -> Any:
    import bmesh  # type: ignore[import-not-found]

    delta = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
    length = math.sqrt(sum(component * component for component in delta)) or 1.0
    mesh = bpy.data.meshes.new(name)
    builder = bmesh.new()
    bmesh.ops.create_cone(
        builder,
        cap_ends=True,
        segments=14,
        radius1=radius,
        radius2=radius,
        depth=length,
    )
    builder.to_mesh(mesh)
    builder.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (
        (start[0] + end[0]) * 0.5,
        (start[1] + end[1]) * 0.5,
        (start[2] + end[2]) * 0.5,
    )
    _alignPlusZ(obj, delta)
    obj.data.materials.append(material)
    if hasattr(obj, 'visible_shadow'):
        obj.visible_shadow = False
    return obj


def _buildInboundPath(bpy: Any, job: dict[str, Any], earthRadius: float) -> Any:
    inbound = [
        sample for sample in job['frames'] if int(sample['frame']) < _impactFrame(job['frames'])
    ]
    start = (
        tuple(float(value) for value in inbound[0]['impactorAu']) if inbound else (0.0, 0.0, 0.0)
    )
    end = tuple(component * earthRadius for component in job['contact']['positionRadii'])
    material = _emissionMaterial(bpy, 'KpgPath', (1.0, 0.52, 0.12), 4.2, 1.0)
    return _buildRod(bpy, 'KpgInboundPath', start, end, earthRadius * 0.028, material)


def _buildPlayhead(bpy: Any, flyby: ModuleType, earthRadius: float) -> Any:
    bead = flyby._createBodySphere(bpy, 'KpgPlayhead', earthRadius * 0.055)
    bead.data.materials.append(_emissionMaterial(bpy, 'KpgPlayhead', (1.0, 0.78, 0.22), 4.5, 1.0))
    if hasattr(bead, 'visible_shadow'):
        bead.visible_shadow = False
    return bead


def _buildShockRing(bpy: Any, earthRadius: float) -> Any:
    material = _emissionMaterial(bpy, 'KpgShock', (1.0, 0.62, 0.18), 2.6, 1.0)
    return _buildPolyline(bpy, 'KpgShockRing', _circlePoints(1.0), earthRadius * 0.010, material)


def _buildEjectaRays(bpy: Any, job: dict[str, Any], earthRadius: float) -> list[Any]:
    material = _emissionMaterial(bpy, 'KpgEjecta', (1.0, 0.38, 0.08), 2.1, 1.0)
    length = earthRadius * 0.42
    rays: list[Any] = []
    for index, direction in enumerate(job['contact']['ejectaDirections'][::2]):
        vector = tuple(float(value) * length for value in direction)
        rays.append(
            _buildPolyline(
                bpy,
                f'KpgEjecta{index:02d}',
                [(0.0, 0.0, 0.0), vector],
                earthRadius * 0.0038,
                material,
            )
        )
    return rays


def _keyShockRing(
    ring: Any,
    sample: dict[str, Any],
    frame: int,
    normal: tuple[float, float, float],
    earthRadius: float,
) -> None:
    angle = float(sample.get('shockAngle', 0.0))
    visible = 1.0 if angle > 0.03 else 0.0
    sine = max(math.sin(angle), 1e-4)
    cosine = math.cos(angle)
    _keyLocation(ring, tuple(axis * earthRadius * cosine for axis in normal), frame)
    _keyScale(ring, earthRadius * sine * visible, frame)
    _alignPlusZ(ring, normal)
    ring.keyframe_insert(data_path='rotation_quaternion', frame=frame)


def _keyEjectaRays(
    rays: list[Any],
    sample: dict[str, Any],
    frame: int,
    normal: tuple[float, float, float],
    earthRadius: float,
) -> None:
    site = tuple(axis * earthRadius * 1.01 for axis in normal)
    scale = float(sample.get('ejectaScale', 0.0))
    for ray in rays:
        _keyLocation(ray, site, frame)
        _keyScale(ray, scale, frame)


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
    domain.scale = (0.82, 0.82, 1.95)
    domain.location = _offsetAlong(surface, normal, domainSize * 0.55)
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
    settings.use_adaptive_domain = False
    settings.use_noise = True
    settings.noise_scale = 2
    settings.noise_strength = 0.85
    settings.vorticity = 0.85
    settings.burning_rate = 0.72
    settings.flame_smoke = 1.15
    settings.use_dissolve_smoke = True
    settings.dissolve_speed = 32
    settings.use_collision_border_bottom = True
    settings.use_collision_border_top = False
    settings.use_collision_border_front = False
    settings.use_collision_border_back = False
    settings.use_collision_border_left = False
    settings.use_collision_border_right = False
    if hasattr(settings, 'gravity'):
        settings.gravity = (-normal[0] * 9.81, -normal[1] * 9.81, -normal[2] * 9.81)
    domain.data.materials.append(_volumeFireMaterial(bpy))

    emitter = flyby._createBodySphere(bpy, 'KpgSmokeFlow', earthRadius * 0.055)
    emitter.location = _offsetAlong(surface, normal, earthRadius * 0.04)
    emitter.hide_render = True
    flowMod = emitter.modifiers.new(name='Fluid', type='FLUID')
    flowMod.fluid_type = 'FLOW'
    flow = flowMod.flow_settings
    flow.flow_type = 'BOTH'
    flow.flow_behavior = 'INFLOW'
    flow.density = 1.0
    flow.fuel_amount = 1.0
    flow.temperature = 2.2
    flow.smoke_color = (0.04, 0.03, 0.025)
    flow.use_initial_velocity = True
    flow.velocity_normal = 5.4
    flow.velocity_random = 0.7
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


def _softPuffMaterial(
    bpy: Any,
    name: str,
    color: tuple[float, float, float],
    strength: float,
    cover: float,
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
    texcoord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (2.8, 2.8, 2.8)
    links.new(texcoord.outputs['Object'], mapping.inputs['Vector'])
    noise = _explosionNoise(nodes, links, mapping)
    holes = nodes.new('ShaderNodeMath')
    holes.operation = 'SUBTRACT'
    holes.inputs[0].default_value = 1.0
    links.new(noise.outputs['Fac'], holes.inputs[1])
    fade = nodes.new('ShaderNodeMath')
    fade.operation = 'MULTIPLY'
    fade.inputs[1].default_value = 0.55
    links.new(holes.outputs['Value'], fade.inputs[0])
    clear = nodes.new('ShaderNodeMath')
    clear.operation = 'ADD'
    clear.inputs[0].default_value = max(0.08, 1.0 - cover)
    links.new(fade.outputs['Value'], clear.inputs[1])
    clip = nodes.new('ShaderNodeClamp')
    clip.inputs['Min'].default_value = 0.0
    clip.inputs['Max'].default_value = 1.0
    links.new(clear.outputs['Value'], clip.inputs['Value'])
    mix = nodes.new('ShaderNodeMixShader')
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    links.new(emission.outputs['Emission'], mix.inputs[1])
    links.new(transparent.outputs['BSDF'], mix.inputs[2])
    factor = mix.inputs['Fac'] if 'Fac' in mix.inputs else mix.inputs['Factor']
    links.new(clip.outputs['Result'], factor)
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    _markAlphaBlend(material)
    if hasattr(material, 'use_backface_culling'):
        material.use_backface_culling = False
    return material


def _contactTexturePath(job: dict[str, Any], name: str) -> str:
    textures = job['contact'].get('textures')
    if not isinstance(textures, dict) or name not in textures:
        raise RuntimeError(f'K–Pg job missing contact texture {name!r}')
    return str(textures[name])


def _markAlphaBlend(material: Any) -> None:
    if hasattr(material, 'surface_render_method'):
        material.surface_render_method = 'BLENDED'
    elif hasattr(material, 'blend_method'):
        material.blend_method = 'BLEND'


def _driveDefault(socket: Any, expression: str, *, index: int | None = None) -> None:
    try:
        driver = (
            socket.driver_add('default_value', index)
            if index is not None
            else socket.driver_add('default_value')
        )
        driver.driver.expression = expression
    except (TypeError, AttributeError):
        return


def _boxImageNode(bpy: Any, nodes: Any, imagePath: str) -> Any:
    image = nodes.new('ShaderNodeTexImage')
    image.image = bpy.data.images.load(imagePath, check_existing=True)
    image.interpolation = 'Cubic'
    image.extension = 'REPEAT'
    image.projection = 'BOX'
    if hasattr(image, 'projection_blend'):
        image.projection_blend = 0.4
    return image


def _explosionNoise(nodes: Any, links: Any, mapping: Any) -> Any:
    noise = nodes.new('ShaderNodeTexNoise')
    if hasattr(noise, 'noise_dimensions'):
        try:
            noise.noise_dimensions = '4D'
        except TypeError:
            pass
    noise.inputs['Scale'].default_value = 6.5
    if 'Detail' in noise.inputs:
        noise.inputs['Detail'].default_value = 8.0
    if 'Roughness' in noise.inputs:
        noise.inputs['Roughness'].default_value = 0.55
    links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
    if 'W' in noise.inputs:
        _driveDefault(noise.inputs['W'], 'frame * 0.06')
    return noise


def _texturedExplosionMaterial(
    bpy: Any,
    name: str,
    imagePath: str,
    *,
    strength: float,
    crawl: float,
) -> Any:
    path = Path(imagePath)
    if not path.is_file():
        raise RuntimeError(f'K–Pg contact texture missing: {path}')
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodeTree = material.node_tree
    nodes = nodeTree.nodes
    links = nodeTree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    mix = nodes.new('ShaderNodeMixShader')
    texCoord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (2.4, 2.4, 2.4)
    image = _boxImageNode(bpy, nodes, str(path))
    links.new(texCoord.outputs['Object'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], image.inputs['Vector'])
    _driveDefault(mapping.inputs['Location'], f'frame * {crawl}', index=2)
    noise = _explosionNoise(nodes, links, mapping)
    holes = nodes.new('ShaderNodeMath')
    holes.operation = 'MULTIPLY'
    links.new(image.outputs['Alpha'], holes.inputs[0])
    links.new(noise.outputs['Fac'], holes.inputs[1])
    invert = nodes.new('ShaderNodeMath')
    invert.operation = 'SUBTRACT'
    invert.inputs[0].default_value = 1.0
    links.new(holes.outputs['Value'], invert.inputs[1])
    glow = nodes.new('ShaderNodeMath')
    glow.operation = 'ADD'
    glow.inputs[1].default_value = 0.4
    links.new(noise.outputs['Fac'], glow.inputs[0])
    wattage = nodes.new('ShaderNodeMath')
    wattage.operation = 'MULTIPLY'
    wattage.inputs[1].default_value = strength
    links.new(glow.outputs['Value'], wattage.inputs[0])
    links.new(image.outputs['Color'], emission.inputs['Color'])
    links.new(wattage.outputs['Value'], emission.inputs['Strength'])
    links.new(emission.outputs['Emission'], mix.inputs[1])
    links.new(transparent.outputs['BSDF'], mix.inputs[2])
    links.new(invert.outputs['Value'], mix.inputs['Fac'])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    _markAlphaBlend(material)
    return material


def _jobFile(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    return Path(__file__).resolve().parents[3] / path


def _rockMaterial(bpy: Any, texturePath: str = '') -> Any:
    material = bpy.data.materials.new(name='KpgDebrisRock')
    material.use_nodes = True
    nodeTree = material.node_tree
    nodes = nodeTree.nodes
    links = nodeTree.links
    principled = next(node for node in nodes if node.type == 'BSDF_PRINCIPLED')
    principled.inputs['Base Color'].default_value = (0.16, 0.10, 0.07, 1.0)
    resolved = _jobFile(texturePath) if texturePath else None
    if resolved is not None and resolved.is_file():
        image = bpy.data.images.load(str(resolved.resolve()), check_existing=True)
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = image
        if hasattr(tex, 'interpolation'):
            tex.interpolation = 'Cubic'
        texcoord = nodes.new('ShaderNodeTexCoord')
        links.new(texcoord.outputs['UV'], tex.inputs['Vector'])
        links.new(tex.outputs['Color'], principled.inputs['Base Color'])
    if 'Emission Strength' in principled.inputs:
        principled.inputs['Emission Strength'].default_value = 0.0
    if 'Roughness' in principled.inputs:
        principled.inputs['Roughness'].default_value = 0.94
    if 'Metallic' in principled.inputs:
        principled.inputs['Metallic'].default_value = 0.0
    if 'Specular IOR Level' in principled.inputs:
        principled.inputs['Specular IOR Level'].default_value = 0.05
    return material


def _darkRockSkins(bpy: Any) -> tuple[Any, ...]:
    colors = ((0.07, 0.06, 0.05), (0.10, 0.08, 0.06), (0.05, 0.045, 0.04))
    skins = []
    for color in colors:
        material = _rockMaterial(bpy, '')
        principled = next(
            node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED'
        )
        principled.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1.0)
        skins.append(material)
    return tuple(skins)


def _asUnit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector)) or 1.0
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _tangentPair(
    normal: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    east = _asUnit((-normal[1], normal[0], 0.0))
    if abs(east[0]) + abs(east[1]) + abs(east[2]) < 1e-6:
        east = (0.0, 1.0, 0.0)
    north = _asUnit(
        (
            normal[1] * east[2] - normal[2] * east[1],
            normal[2] * east[0] - normal[0] * east[2],
            normal[0] * east[1] - normal[1] * east[0],
        )
    )
    return east, north


def _createCloudChunk(
    bpy: Any,
    name: str,
    radius: float,
    seed: int,
    *,
    stretchZ: float = 1.0,
    wild: float = 0.95,
) -> Any:
    import bmesh  # type: ignore[import-not-found]

    mesh = bpy.data.meshes.new(name)
    builder = bmesh.new()
    bmesh.ops.create_icosphere(builder, subdivisions=2, radius=radius)
    state = seed * 997 + 13
    for vert in builder.verts:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        vert.co *= 0.16 + wild * (state / 0x7FFFFFFF)
        vert.co.z *= stretchZ
    builder.to_mesh(mesh)
    builder.free()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _volumeCloudMaterial(
    bpy: Any,
    name: str,
    *,
    density: float,
    emission: float,
    color: tuple[float, float, float],
    crawl: float,
) -> Any:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodeTree = material.node_tree
    nodes = nodeTree.nodes
    links = nodeTree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    volume = nodes.new('ShaderNodeVolumePrincipled')
    texCoord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (3.4, 3.4, 3.4)
    links.new(texCoord.outputs['Object'], mapping.inputs['Vector'])
    _driveDefault(mapping.inputs['Location'], f'frame * {crawl}', index=2)
    noise = _explosionNoise(nodes, links, mapping)
    dens = nodes.new('ShaderNodeMath')
    dens.operation = 'MULTIPLY'
    dens.inputs[1].default_value = density
    glow = nodes.new('ShaderNodeMath')
    glow.operation = 'MULTIPLY'
    glow.inputs[1].default_value = emission
    links.new(noise.outputs['Fac'], dens.inputs[0])
    links.new(noise.outputs['Fac'], glow.inputs[0])
    if 'Density' in volume.inputs:
        links.new(dens.outputs['Value'], volume.inputs['Density'])
    if 'Emission Strength' in volume.inputs:
        links.new(glow.outputs['Value'], volume.inputs['Emission Strength'])
    if 'Color' in volume.inputs:
        volume.inputs['Color'].default_value = (0.03, 0.022, 0.018, 1.0)
    if 'Absorption Color' in volume.inputs:
        volume.inputs['Absorption Color'].default_value = (0.09, 0.04, 0.02, 1.0)
    if 'Emission Color' in volume.inputs:
        volume.inputs['Emission Color'].default_value = (color[0], color[1], color[2], 1.0)
    if 'Blackbody Intensity' in volume.inputs:
        volume.inputs['Blackbody Intensity'].default_value = 0.0
    links.new(volume.outputs['Volume'], output.inputs['Volume'])
    return material


def _applySphereUv(builder: Any) -> None:
    uvLayer = builder.loops.layers.uv.new('UVMap')
    for face in builder.faces:
        for loop in face.loops:
            x, y, z = loop.vert.co
            length = math.sqrt(x * x + y * y + z * z) or 1.0
            nx, ny, nz = x / length, y / length, z / length
            u = 0.5 + math.atan2(ny, nx) / (2.0 * math.pi)
            v = 0.5 + math.asin(max(-1.0, min(1.0, nz))) / math.pi
            loop[uvLayer].uv = (u, v)


def _createLumpyRock(
    bpy: Any,
    name: str,
    radius: float,
    seed: int,
    *,
    subdivisions: int = 2,
    squash: float = 0.22,
) -> Any:
    import bmesh  # type: ignore[import-not-found]

    mesh = bpy.data.meshes.new(name)
    builder = bmesh.new()
    bmesh.ops.create_icosphere(builder, subdivisions=subdivisions, radius=radius)
    state = seed * 997 + 4_112_009
    stretch = (
        1.0 + squash * (0.55 - ((state >> 3) & 0xFF) / 255.0),
        1.0 + squash * (0.35 - ((state >> 11) & 0xFF) / 255.0),
        1.0 + squash * (((state >> 19) & 0xFF) / 255.0 - 0.45),
    )
    for vert in builder.verts:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        lump = 0.78 + 0.32 * (state / 0x7FFFFFFF)
        vert.co.x *= stretch[0] * lump
        vert.co.y *= stretch[1] * lump
        vert.co.z *= stretch[2] * lump
    _applySphereUv(builder)
    builder.to_mesh(mesh)
    builder.free()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _createDebrisChunk(bpy: Any, name: str, radius: float, seed: int) -> Any:
    return _createLumpyRock(bpy, name, radius, seed, subdivisions=2, squash=0.28)


def _buildProjectiles(bpy: Any, job: dict[str, Any], earthRadius: float) -> list[Any]:
    count = int(job['contact']['projectileCount'])
    skins = _darkRockSkins(bpy)
    chunks: list[Any] = []
    for index in range(count):
        sizeMix = (index * 0.53 + 0.11) % 1.0
        visual = earthRadius * (0.00035 + 0.0012 * (sizeMix**1.4))
        chunk = _createLumpyRock(
            bpy,
            f'KpgEjecta{index:02d}',
            visual,
            index * 5 + 3,
            subdivisions=2,
            squash=0.18 + 0.20 * ((index * 0.37) % 1.0),
        )
        chunk.data.materials.append(skins[index % len(skins)])
        chunks.append(chunk)
    return chunks


def _buildDebrisTrails(bpy: Any, job: dict[str, Any], earthRadius: float) -> list[Any]:
    count = int(job['contact']['projectileCount'])
    fire = _texturedExplosionMaterial(
        bpy, 'KpgTrailFire', _contactTexturePath(job, 'fire'), strength=6.8, crawl=0.028
    )
    soot = _texturedExplosionMaterial(
        bpy, 'KpgTrailSoot', _contactTexturePath(job, 'smoke'), strength=1.8, crawl=0.016
    )
    trails: list[Any] = []
    for index in range(count):
        height = earthRadius * 0.09
        flame = _createCloudChunk(
            bpy, f'KpgTrailFire{index:02d}', earthRadius * 0.011, 80 + index, stretchZ=2.3
        )
        smoke = _createCloudChunk(
            bpy, f'KpgTrailSoot{index:02d}', earthRadius * 0.016, 110 + index, stretchZ=2.7
        )
        flame.data.materials.append(fire)
        smoke.data.materials.append(soot)
        trails.append((flame, smoke, height))
    return trails


def _buildLandingStrikes(bpy: Any, job: dict[str, Any], earthRadius: float) -> list[Any]:
    count = int(job['contact']['projectileCount'])
    flash = _texturedExplosionMaterial(
        bpy, 'KpgStrikeFlash', _contactTexturePath(job, 'fire'), strength=8.4, crawl=0.02
    )
    puff = _texturedExplosionMaterial(
        bpy, 'KpgStrikePuff', _contactTexturePath(job, 'smoke'), strength=2.4, crawl=0.014
    )
    strikes: list[Any] = []
    for index in range(count):
        core = _createCloudChunk(
            bpy, f'KpgStrikeCore{index:02d}', earthRadius * 0.028, 140 + index, stretchZ=1.2
        )
        cloud = _createCloudChunk(
            bpy, f'KpgStrikePuff{index:02d}', earthRadius * 0.055, 170 + index, stretchZ=1.35
        )
        core.data.materials.append(flash)
        cloud.data.materials.append(puff)
        strikes.append((core, cloud))
    return strikes


def _createCard(bpy: Any, name: str, size: float) -> Any:
    import bmesh  # type: ignore[import-not-found]

    mesh = bpy.data.meshes.new(name)
    builder = bmesh.new()
    bmesh.ops.create_grid(builder, x_segments=1, y_segments=1, size=size)
    uvLayer = builder.loops.layers.uv.new('UVMap')
    for face in builder.faces:
        for loop in face.loops:
            u = 0.5 + 0.5 * float(loop.vert.co.x) / max(size, 1e-8)
            v = 0.5 + 0.5 * float(loop.vert.co.y) / max(size, 1e-8)
            loop[uvLayer].uv = (u, v)
    builder.to_mesh(mesh)
    builder.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _explosionSequence(bpy: Any, paths: list[str]) -> Any:
    name = 'KpgFireSequence'
    loaded = bpy.data.images.get(name)
    if loaded is not None:
        return loaded
    loaded = bpy.data.images.load(str(Path(paths[0]).resolve()), check_existing=True)
    loaded.name = name
    loaded.source = 'SEQUENCE'
    if hasattr(loaded, 'alpha_mode'):
        loaded.alpha_mode = 'STRAIGHT'
    loaded.reload()
    return loaded


def _explosionCardMaterial(
    bpy: Any,
    name: str,
    paths: list[str],
    *,
    strength: float,
    start: int,
    offset: int,
) -> Any:
    if not paths or not Path(paths[0]).is_file():
        raise RuntimeError('K–Pg explosion flipbook is missing')
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodeTree = material.node_tree
    nodes = nodeTree.nodes
    links = nodeTree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    mix = nodes.new('ShaderNodeMixShader')
    image = nodes.new('ShaderNodeTexImage')
    loaded = _explosionSequence(bpy, paths)
    image.image = loaded
    image.interpolation = 'Cubic'
    image.extension = 'CLIP'
    user = image.image_user
    user.frame_duration = len(paths)
    user.frame_start = start
    user.frame_offset = offset
    user.use_auto_refresh = True
    if hasattr(user, 'use_cyclic'):
        user.use_cyclic = True
    texcoord = nodes.new('ShaderNodeTexCoord')
    emission.inputs['Strength'].default_value = strength
    links.new(texcoord.outputs['UV'], image.inputs['Vector'])
    links.new(image.outputs['Color'], emission.inputs['Color'])
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(image.outputs['Alpha'], mix.inputs['Fac'])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    _markAlphaBlend(material)
    return material


def _aimAtCamera(obj: Any, camera: Any) -> None:
    track = obj.constraints.new(type='TRACK_TO')
    track.target = camera
    track.track_axis = 'TRACK_Z'
    track.up_axis = 'UP_Y'


def _tryEeveeBloom(scene: Any) -> None:
    eevee = getattr(scene, 'eevee', None)
    if eevee is None or not hasattr(eevee, 'use_bloom'):
        return
    eevee.use_bloom = True
    if hasattr(eevee, 'bloom_intensity'):
        eevee.bloom_intensity = 0.08
    if hasattr(eevee, 'bloom_threshold'):
        eevee.bloom_threshold = 1.15
    if hasattr(eevee, 'bloom_radius'):
        eevee.bloom_radius = 4.0


def _insertGlare(tree: Any, renderLayers: Any, composite: Any) -> None:
    glare = tree.nodes.new('CompositorNodeGlare')
    if hasattr(glare, 'glare_type'):
        try:
            glare.glare_type = 'BLOOM'
        except TypeError:
            glare.glare_type = 'FOG_GLOW'
    if hasattr(glare, 'mix'):
        glare.mix = 0.22
    if hasattr(glare, 'threshold'):
        glare.threshold = 0.65
    if hasattr(glare, 'size'):
        glare.size = 7
    for link in list(tree.links):
        if link.to_node == composite:
            tree.links.remove(link)
    tree.links.new(renderLayers.outputs['Image'], glare.inputs['Image'])
    tree.links.new(glare.outputs['Image'], composite.inputs['Image'])


def _enableExplosionBloom(scene: Any) -> None:
    _tryEeveeBloom(scene)
    if hasattr(scene, 'use_nodes'):
        scene.use_nodes = True
    tree = getattr(scene, 'node_tree', None)
    if tree is None:
        return
    renderLayers = next((node for node in tree.nodes if node.type == 'R_LAYERS'), None)
    composite = next((node for node in tree.nodes if node.type == 'COMPOSITE'), None)
    if renderLayers is None or composite is None:
        return
    _insertGlare(tree, renderLayers, composite)


def _rotateAround(
    vector: tuple[float, float, float],
    axis: tuple[float, float, float],
    angle: float,
) -> tuple[float, float, float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    length = math.sqrt(sum(component * component for component in axis)) or 1.0
    unit = (axis[0] / length, axis[1] / length, axis[2] / length)
    cross = (
        unit[1] * vector[2] - unit[2] * vector[1],
        unit[2] * vector[0] - unit[0] * vector[2],
        unit[0] * vector[1] - unit[1] * vector[0],
    )
    along = unit[0] * vector[0] + unit[1] * vector[1] + unit[2] * vector[2]
    return (
        vector[0] * cosine + cross[0] * sine + unit[0] * along * (1.0 - cosine),
        vector[1] * cosine + cross[1] * sine + unit[1] * along * (1.0 - cosine),
        vector[2] * cosine + cross[2] * sine + unit[2] * along * (1.0 - cosine),
    )


def _swarmFlight(index: int, *, mode: str) -> tuple[float, float, float, float, float]:
    spin = (index * 0.6180339887) % 1.0
    azimuth = 2.0 * math.pi * spin + 1.15 * math.sin(index * 5.7) + 0.7 * math.cos(index * 2.3)
    mixA = (index * 0.137 + 0.41) % 1.0
    mixB = (index * 0.811 + 0.2) % 1.0
    mixC = (index * 0.419 + 0.08) % 1.0
    delayMix = (index * 0.6180339887 * 13.0) % 1.0
    if mode == 'escape':
        rangeDeg = 22.0 + 118.0 * (mixA**0.72)
        loft = 0.40 + 0.70 * (mixB**0.68)
        seconds = 2.2 + 3.6 * mixC
        delay = 0.10 * delayMix
    elif mode == 'core':
        rangeDeg = 0.12 + 5.2 * (mixA**0.65)
        loft = 0.05 + 0.24 * (mixB**0.80)
        seconds = 1.0 + 1.5 * mixC
        delay = 0.03 * delayMix
    elif mode == 'fountain':
        rangeDeg = 0.4 + 18.0 * (mixA**0.80)
        loft = 0.22 + 0.55 * (mixB**0.85)
        seconds = 1.5 + 2.4 * mixC
        delay = 0.06 * delayMix
    else:
        rangeDeg = 12.0 + 145.0 * (mixA**1.05)
        loft = 0.008 + 0.12 * (mixB**1.10)
        seconds = 1.6 + 4.0 * mixC
        delay = 0.22 * delayMix
    return azimuth, rangeDeg, loft, seconds, delay


def _swarmPosition(
    normal: tuple[float, float, float],
    inbound: tuple[float, float, float],
    flight: tuple[float, float, float, float, float],
    frame: int,
    impact: int,
    fps: float,
    earthRadius: float,
) -> tuple[float, float, float]:
    if frame < impact:
        return (normal[0] * earthRadius, normal[1] * earthRadius, normal[2] * earthRadius)
    azimuth, rangeDeg, loft, seconds, delay = flight
    age = (frame - impact) / fps - delay
    if age <= 0.0:
        return (normal[0] * earthRadius, normal[1] * earthRadius, normal[2] * earthRadius)
    progress = min(age / seconds, 1.0)
    align = inbound[0] * normal[0] + inbound[1] * normal[1] + inbound[2] * normal[2]
    incoming = (
        inbound[0] - normal[0] * align,
        inbound[1] - normal[1] * align,
        inbound[2] - normal[2] * align,
    )
    down = _asUnit((-incoming[0], -incoming[1], -incoming[2]))
    cross = _asUnit(
        (
            normal[1] * down[2] - normal[2] * down[1],
            normal[2] * down[0] - normal[0] * down[2],
            normal[0] * down[1] - normal[1] * down[0],
        )
    )
    heading = (
        down[0] * math.cos(azimuth) + cross[0] * math.sin(azimuth),
        down[1] * math.cos(azimuth) + cross[1] * math.sin(azimuth),
        down[2] * math.cos(azimuth) + cross[2] * math.sin(azimuth),
    )
    axis = (
        normal[1] * heading[2] - normal[2] * heading[1],
        normal[2] * heading[0] - normal[0] * heading[2],
        normal[0] * heading[1] - normal[1] * heading[0],
    )
    along = _rotateAround(normal, axis, math.radians(rangeDeg) * progress)
    if loft > 0.14:
        climb = min(progress * 1.55, 1.0)
        height = 1.0 + loft * math.sin(0.5 * math.pi * climb)
        if progress > 0.55:
            fall = (progress - 0.55) / 0.45
            height = 1.0 + loft * max(0.12, 1.0 - fall**1.15)
    else:
        height = 1.0 + loft * math.sin(math.pi * progress)
    return (
        along[0] * earthRadius * height,
        along[1] * earthRadius * height,
        along[2] * earthRadius * height,
    )


def _appendSwarmGroup(
    bpy: Any,
    swarm: list[tuple[Any, tuple[float, float, float, float, float], bool]],
    skins: tuple[Any, ...],
    *,
    count: int,
    kind: str,
    mode: str,
    small: float,
    large: float,
    earthRadius: float,
    start: int,
) -> int:
    cloudy = kind != 'rock'
    for index in range(count):
        mix = (index * 0.53 + 0.11) % 1.0
        visual = earthRadius * (small + (large - small) * (mix**1.15))
        name = f'KpgSwarm{kind[0].upper()}{start + index:04d}'
        if cloudy:
            chunk = _createCloudChunk(
                bpy,
                name,
                visual,
                start + index,
                stretchZ=1.25 + 0.70 * mix,
                wild=0.40,
            )
            if hasattr(chunk, 'visible_shadow'):
                chunk.visible_shadow = False
        else:
            chunk = _createLumpyRock(
                bpy,
                name,
                visual,
                (start + index) * 3 + 5,
                subdivisions=2,
                squash=0.16 + 0.22 * mix,
            )
        chunk.data.materials.append(skins[index % len(skins)])
        swarm.append((chunk, _swarmFlight(start + index, mode=mode), kind != 'rock'))
    return start + count


def _buildEmberBurst(
    bpy: Any, job: dict[str, Any], earthRadius: float, frames: list[dict[str, Any]]
) -> list[tuple[Any, tuple[float, float, float, float, float], bool]]:
    del job
    smokeSkins = (_softPuffMaterial(bpy, 'KpgSmoke', (0.22, 0.20, 0.17), 0.22, 0.58),)
    sootSkins = (_softPuffMaterial(bpy, 'KpgSoot', (0.13, 0.11, 0.10), 0.12, 0.62),)
    emberSkins = (_softPuffMaterial(bpy, 'KpgEmber', (0.90, 0.30, 0.06), 0.55, 0.42),)
    swarm: list[tuple[Any, tuple[float, float, float, float, float], bool]] = []
    groups = (
        (48, 'smoke', 'fountain', smokeSkins, 0.0070, 0.0160),
        (32, 'smoke', 'core', smokeSkins, 0.0080, 0.0180),
        (36, 'soot', 'fountain', sootSkins, 0.0060, 0.0130),
        (16, 'ember', 'fountain', emberSkins, 0.0020, 0.0045),
    )
    drawn = 0
    for count, kind, mode, skins, small, large in groups:
        drawn = _appendSwarmGroup(
            bpy,
            swarm,
            skins,
            count=count,
            kind=kind,
            mode=mode,
            small=small,
            large=large,
            earthRadius=earthRadius,
            start=drawn,
        )
    del frames
    return swarm


def _syncSwarm(
    swarm: list[tuple[Any, tuple[float, float, float, float, float], bool]],
    frame: int,
    normal: tuple[float, float, float],
    inbound: tuple[float, float, float],
    earthRadius: float,
    impact: int,
    fps: float,
) -> None:
    for rock, flight, weather in swarm:
        age = (frame - impact) / fps - flight[4]
        seconds = flight[3]
        rangeDeg = flight[1]
        landedFar = (not weather) and age > seconds and rangeDeg > 36.0
        weatherGone = weather and age > seconds + 7.5
        visible = frame >= impact and age > 0.0 and not landedFar and not weatherGone
        rock.location = _swarmPosition(normal, inbound, flight, frame, impact, fps, earthRadius)
        rock.scale = (1.0, 1.0, 1.0) if visible else (0.0, 0.0, 0.0)
        rock.hide_render = not visible


def _buildContactDrawings(
    bpy: Any, job: dict[str, Any], earthRadius: float, impactStart: int
) -> tuple[list[Any], list[Any], list[Any]]:
    contact = job['contact']
    paths = [str(Path(path).resolve()) for path in contact['textures']['explosion']]
    if not paths:
        raise RuntimeError('K–Pg job is missing the impact fire plate')
    normal = tuple(float(value) for value in contact['normal'])
    site = tuple(component * earthRadius * 1.045 for component in normal)
    layers = (
        ('KpgFireCore', 0.13, 5.4, 0),
        ('KpgFirePlate', 0.28, 3.1, 1),
        ('KpgFireSpray', 0.38, 1.15, 4),
    )
    cards: list[Any] = []
    for name, size, strength, offset in layers:
        plate = _explosionCardMaterial(
            bpy, name, paths, strength=strength, start=impactStart, offset=offset
        )
        card = _createCard(bpy, name, earthRadius * size)
        card.location = site
        if hasattr(card, 'visible_shadow'):
            card.visible_shadow = False
        card.data.materials.append(plate)
        cards.append(card)
    return cards, [], []


def _buildSecondaryPlates(
    bpy: Any, job: dict[str, Any], earthRadius: float, impactStart: int
) -> list[Any]:
    paths = [str(Path(path).resolve()) for path in job['contact']['textures']['explosion']]
    cards: list[Any] = []
    for index in range(int(job['contact']['projectileCount'])):
        material = _explosionCardMaterial(
            bpy,
            f'KpgSecondary{index:02d}',
            paths,
            strength=2.6,
            start=impactStart,
            offset=3 + index,
        )
        card = _createCard(bpy, f'KpgSecondary{index:02d}', earthRadius * 0.08)
        if hasattr(card, 'visible_shadow'):
            card.visible_shadow = False
        card.data.materials.append(material)
        cards.append(card)
    return cards


def _paintedBurstMaterial(bpy: Any, name: str, imagePath: str, strength: float) -> Any:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    mix = nodes.new('ShaderNodeMixShader')
    texcoord = nodes.new('ShaderNodeTexCoord')
    image = nodes.new('ShaderNodeTexImage')
    image.image = bpy.data.images.load(imagePath, check_existing=True)
    image.interpolation = 'Cubic'
    image.extension = 'CLIP'
    rgb = nodes.new('ShaderNodeSeparateColor')
    hottest = nodes.new('ShaderNodeMath')
    hottest.operation = 'MAXIMUM'
    hotter = nodes.new('ShaderNodeMath')
    hotter.operation = 'MAXIMUM'
    floor = nodes.new('ShaderNodeMath')
    floor.operation = 'SUBTRACT'
    floor.inputs[1].default_value = 0.012
    gate = nodes.new('ShaderNodeMath')
    gate.operation = 'DIVIDE'
    gate.inputs[1].default_value = 0.055
    clip = nodes.new('ShaderNodeClamp')
    clip.inputs['Min'].default_value = 0.0
    clip.inputs['Max'].default_value = 1.0
    cover = nodes.new('ShaderNodeMath')
    cover.operation = 'MULTIPLY'
    emission.inputs['Strength'].default_value = strength
    links.new(texcoord.outputs['UV'], image.inputs['Vector'])
    links.new(image.outputs['Color'], emission.inputs['Color'])
    links.new(image.outputs['Color'], rgb.inputs['Color'])
    links.new(rgb.outputs['Red'], hottest.inputs[0])
    links.new(rgb.outputs['Green'], hottest.inputs[1])
    links.new(hottest.outputs['Value'], hotter.inputs[0])
    links.new(rgb.outputs['Blue'], hotter.inputs[1])
    links.new(hotter.outputs['Value'], floor.inputs[0])
    links.new(floor.outputs['Value'], gate.inputs[0])
    links.new(gate.outputs['Value'], clip.inputs['Value'])
    links.new(clip.outputs['Result'], cover.inputs[0])
    links.new(image.outputs['Alpha'], cover.inputs[1])
    factor = mix.inputs['Fac'] if 'Fac' in mix.inputs else mix.inputs['Factor']
    links.new(cover.outputs['Value'], factor)
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    _markAlphaBlend(material)
    if hasattr(material, 'use_backface_culling'):
        material.use_backface_culling = False
    if hasattr(material, 'shadow_method'):
        material.shadow_method = 'NONE'
    return material


def _buildExplosionPlates(
    bpy: Any, job: dict[str, Any], earthRadius: float
) -> list[tuple[Any, str, float]]:
    textures = job['contact'].get('textures') or {}
    cards: list[tuple[Any, str, float]] = []
    layers = (
        ('mushroomPlate', 'mushroom', 0.12, 4.2),
        ('latePlate', 'late', 0.16, 2.8),
    )
    for key, kind, size, strength in layers:
        path = textures.get(key)
        if not path or not Path(path).is_file():
            continue
        material = _paintedBurstMaterial(bpy, f'Kpg{kind.title()}Card', str(path), strength)
        card = _createCard(bpy, f'Kpg{kind.title()}Card', earthRadius * size)
        if hasattr(card, 'visible_shadow'):
            card.visible_shadow = False
        card.data.materials.append(material)
        cards.append((card, kind, size))
    return cards


def _keyExplosionPlates(
    cards: list[tuple[Any, str, float]],
    sample: dict[str, Any],
    frame: int,
    normal: tuple[float, float, float],
    earthRadius: float,
) -> None:
    flash = float(sample['flashScale'])
    fire = float(sample['fireballScale'])
    plume = float(sample.get('plumeScale', 0.0))
    camera = tuple(float(value) for value in sample['cameraAu'])
    surface = (normal[0] * earthRadius, normal[1] * earthRadius, normal[2] * earthRadius)
    toward = (camera[0] - surface[0], camera[1] - surface[1], camera[2] - surface[2])
    length = math.sqrt(sum(component * component for component in toward)) or 1.0
    standoff = earthRadius * 0.08
    site = (
        surface[0] + toward[0] / length * standoff,
        surface[1] + toward[1] / length * standoff,
        surface[2] + toward[2] / length * standoff,
    )
    amounts = {
        'flash': flash,
        'mushroom': min(1.15, 0.90 * flash + 0.75 * fire + 0.50 * plume),
        'late': max(0.0, 0.18 * fire + 0.95 * plume - 0.40 * flash),
    }
    for card, kind, _size in cards:
        amount = amounts.get(kind, 0.0)
        _keyLocation(card, site, frame)
        _keyScale(card, amount, frame)
        _keyBillboard(card, camera, frame)


def _keyBillboard(
    obj: Any,
    camera: tuple[float, float, float],
    frame: int,
    up: tuple[float, float, float] | None = None,
) -> None:
    location = (float(obj.location[0]), float(obj.location[1]), float(obj.location[2]))
    if up is None:
        toward = (camera[0] - location[0], camera[1] - location[1], camera[2] - location[2])
        _alignPlusZ(obj, toward)
    else:
        _alignCardUpright(obj, up, camera, location)
    obj.keyframe_insert(data_path='rotation_quaternion', frame=frame)


def _keyRocks(rocks: list[Any], sample: dict[str, Any], frame: int) -> None:
    if not rocks:
        return
    for index, (rock, position, scale) in enumerate(
        zip(rocks, sample['projectileAu'], sample['projectileScale'], strict=True)
    ):
        _keyLocation(rock, tuple(float(value) for value in position), frame)
        _keyScale(rock, float(scale), frame)
        rock.rotation_euler = (
            (0.18 + 0.09 * index) * frame,
            (0.11 + 0.07 * index) * frame,
            (0.05 + 0.13 * index) * frame,
        )
        rock.keyframe_insert(data_path='rotation_euler', frame=frame)


def _keySecondaryPlates(
    cards: list[Any], sample: dict[str, Any], frame: int, earthRadius: float
) -> None:
    for card, position, strike in zip(
        cards, sample['projectileAu'], sample['projectileStrike'], strict=True
    ):
        site = tuple(float(value) for value in position)
        length = math.sqrt(sum(component * component for component in site)) or 1.0
        lifted = tuple(component / length * (length + earthRadius * 0.03) for component in site)
        _keyLocation(card, lifted, frame)
        _keyScale(card, float(strike), frame)
        _keyBillboard(card, tuple(float(value) for value in sample['cameraAu']), frame)


def _keyWeather(
    weather: dict[str, Any], sample: dict[str, Any], frame: int, impactFrame: int
) -> None:
    derived = {
        'veil': _veilAngleRad(frame, impactFrame),
        'site': _siteCloudRad(frame, impactFrame),
        'fallout': _falloutAngleRad(frame, impactFrame),
        'crater': 1.0 if frame >= impactFrame else 0.0,
    }
    sampled = {
        'shock': 'shockAngle',
        'fire': 'wildfireAngle',
        'soot': 'soot',
        'flash': 'flashScale',
        'tsunami': 'tsunamiAngle',
        'smolder': 'smolder',
        'glow': 'siteGlow',
        'dieback': 'dieback',
    }
    for key, value in derived.items():
        node = weather.get(key)
        if node is None:
            continue
        node.outputs[0].default_value = value
        node.outputs[0].keyframe_insert(data_path='default_value', frame=frame)
    for key, sampleKey in sampled.items():
        node = weather.get(key)
        if node is None:
            continue
        node.outputs[0].default_value = float(sample.get(sampleKey, 0.0))
        node.outputs[0].keyframe_insert(data_path='default_value', frame=frame)


def _buildInboundTrail(bpy: Any, job: dict[str, Any], earthRadius: float) -> tuple[Any, float]:
    inbound = tuple(float(value) for value in job['contact']['inbound'])
    back = (-inbound[0], -inbound[1], -inbound[2])
    height = earthRadius * 0.08
    trail = _createCloudChunk(bpy, 'KpgInboundTrail', earthRadius * 0.02, 7, stretchZ=2.2)
    trail.data.materials.append(
        _volumeCloudMaterial(
            bpy, 'KpgInboundFire', density=5.2, emission=3.2, color=(1.0, 0.32, 0.06), crawl=0.03
        )
    )
    _alignPlusZ(trail, back)
    return trail, height


def _keyContactMeshes(
    fireChunks: list[Any],
    plumes: list[Any],
    smokeChunks: list[Any],
    sample: dict[str, Any],
    frame: int,
    normal: tuple[float, float, float],
    earthRadius: float,
) -> None:
    fire = float(sample['fireballScale'])
    site = tuple(component * earthRadius * 1.045 for component in normal)
    camera = tuple(float(value) for value in sample['cameraAu'])
    toward = (
        camera[0] - site[0],
        camera[1] - site[1],
        camera[2] - site[2],
    )
    length = math.sqrt(sum(component * component for component in toward)) or 1.0
    flash = float(sample.get('flashScale', 0.0))
    plume = float(sample.get('plumeScale', 0.0))
    for chunk in fireChunks:
        name = str(getattr(chunk, 'name', ''))
        scale = fire
        lift = 0.055
        if 'Core' in name:
            scale = fire * (0.42 + 0.35 * flash)
            lift = 0.04
        elif 'Spray' in name:
            scale = fire * (0.85 + 0.45 * plume)
            lift = 0.08
        plantedLayer = tuple(
            site[index] + toward[index] / length * earthRadius * lift for index in range(3)
        )
        _keyLocation(chunk, plantedLayer, frame)
        _keyScale(chunk, scale, frame)
        _keyBillboard(chunk, camera, frame)
    ejecta = float(sample['ejectaScale'])
    for chunk in plumes:
        _keyScale(chunk, ejecta, frame)
    smoke = float(sample['plumeScale'])
    for chunk in smokeChunks:
        _keyScale(chunk, smoke, frame)


def _keyInboundTrail(
    trail: Any,
    height: float,
    inbound: tuple[float, float, float],
    sample: dict[str, Any],
    frame: int,
) -> None:
    site = tuple(float(value) for value in sample['impactorAu'])
    back = (-inbound[0], -inbound[1], -inbound[2])
    _alignPlusZ(trail, back)
    trail.keyframe_insert(data_path='rotation_quaternion', frame=frame)
    _keyLocation(trail, _offsetAlong(site, back, height * 0.45), frame)
    _keyScale(trail, float(sample['inboundTrail']), frame)


def _keyProjectiles(
    projectiles: list[Any],
    trails: list[Any],
    strikes: list[Any],
    sample: dict[str, Any],
    frame: int,
) -> None:
    for index, (rock, position, scale, direction, trailScale, _strikeScale) in enumerate(
        zip(
            projectiles,
            sample['projectileAu'],
            sample['projectileScale'],
            sample['projectileDir'],
            sample['projectileTrail'],
            sample['projectileStrike'],
            strict=True,
        )
    ):
        site = tuple(float(value) for value in position)
        heading = tuple(float(value) for value in direction)
        _keyLocation(rock, site, frame)
        _keyScale(rock, float(scale), frame)
        rock.rotation_euler = (
            0.21 * index + 0.11 * frame,
            0.17 * index + 0.07 * frame,
            0.09 * frame,
        )
        rock.keyframe_insert(data_path='rotation_euler', frame=frame)
        flame, smoke, height = trails[index]
        back = (-heading[0], -heading[1], -heading[2])
        _alignPlusZ(flame, back)
        _alignPlusZ(smoke, back)
        flame.keyframe_insert(data_path='rotation_quaternion', frame=frame)
        smoke.keyframe_insert(data_path='rotation_quaternion', frame=frame)
        _keyLocation(flame, _offsetAlong(site, back, height * 0.45), frame)
        _keyLocation(smoke, _offsetAlong(site, back, height * 0.55), frame)
        trail = float(trailScale)
        _keyScale(flame, trail, frame)
        _keyScale(smoke, 0.0, frame)
        core, cloud = strikes[index]
        _keyLocation(core, site, frame)
        _keyLocation(cloud, site, frame)
        _keyScale(core, 0.0, frame)
        _keyScale(cloud, 0.0, frame)


def _keyframeShot(
    camera: Any,
    lookAt: Any,
    impactor: Any,
    flash: Any,
    fireChunks: list[Any],
    plumes: list[Any],
    smokeChunks: list[Any],
    inboundTrail: Any,
    inboundHeight: float,
    inbound: tuple[float, float, float],
    contactNormal: tuple[float, float, float],
    earthRadius: float,
    projectiles: list[Any],
    trails: list[Any],
    strikes: list[Any],
    lightData: Any,
    flashData: Any,
    sunEnergy: float,
    flashEnergy: float,
    frames: list[dict[str, Any]],
) -> None:
    impactFrame = _impactFrame(frames)
    for sample in frames:
        frame = int(sample['frame'])
        _keyLocation(camera, tuple(float(value) for value in sample['cameraAu']), frame)
        _keyLocation(lookAt, tuple(float(value) for value in sample['lookAtAu']), frame)
        _keyLocation(impactor, tuple(float(value) for value in sample['impactorAu']), frame)
        _keyLocation(flash, tuple(float(value) for value in sample['impactorAu']), frame)
        _keyContactMeshes(
            fireChunks, plumes, smokeChunks, sample, frame, contactNormal, earthRadius
        )
        _keyInboundTrail(inboundTrail, inboundHeight, inbound, sample, frame)
        _keyProjectiles(projectiles, trails, strikes, sample, frame)
        rockVisible = bool(sample.get('impactorVisible', float(sample['fireballScale']) < 1e-4))
        _keyScale(impactor, 1.0 if rockVisible else 0.0, frame)
        lightData.energy = sunEnergy * float(sample['sunScale'])
        lightData.keyframe_insert(data_path='energy', frame=frame)
        flashData.energy = flashEnergy * _blastLampScale(
            frame, float(sample['flashScale']), impactFrame
        )
        flashData.keyframe_insert(data_path='energy', frame=frame)


def _createIco(bpy: Any, name: str, radius: float, subdivisions: int = 3, *, seed: int = 0) -> Any:
    import bmesh  # type: ignore[import-not-found]

    mesh = bpy.data.meshes.new(name)
    builder = bmesh.new()
    bmesh.ops.create_icosphere(builder, subdivisions=subdivisions, radius=radius)
    state = seed * 997 + 4_112_009
    for vertex in builder.verts:
        x, y, z = vertex.co.x, vertex.co.y, vertex.co.z
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        lump = (
            0.38
            + 0.42 * (state / 0x7FFFFFFF)
            + 0.28 * math.sin(x * 31.0 + y * 19.0 + seed)
            + 0.22 * math.cos(z * 27.0 + x * 14.0)
            + 0.18 * math.sin(y * 41.0 - z * 23.0)
        )
        vertex.co *= max(lump, 0.22)
        vertex.co.x *= 0.70 + 0.55 * (((state >> 4) & 0xFF) / 255.0)
        vertex.co.y *= 0.65 + 0.60 * (((state >> 12) & 0xFF) / 255.0)
        vertex.co.z *= 0.72 + 0.48 * (((state >> 20) & 0xFF) / 255.0)
    _applySphereUv(builder)
    builder.to_mesh(mesh)
    builder.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _noiseVolumeMaterial(
    bpy: Any,
    name: str,
    *,
    density: float,
    emission: float,
    color: tuple[float, float, float],
    absorption: tuple[float, float, float],
    scale: float,
    crawl: float,
) -> Any:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodeTree = material.node_tree
    nodes = nodeTree.nodes
    links = nodeTree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    volume = nodes.new('ShaderNodeVolumePrincipled')
    texcoord = nodes.new('ShaderNodeTexCoord')
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = scale
    if 'Detail' in noise.inputs:
        noise.inputs['Detail'].default_value = 8.0
    if 'Roughness' in noise.inputs:
        noise.inputs['Roughness'].default_value = 0.42
    if hasattr(noise, 'noise_dimensions'):
        try:
            noise.noise_dimensions = '4D'
        except TypeError:
            pass
    if 'W' in noise.inputs:
        _driveDefault(noise.inputs['W'], f'frame * {crawl:.4f}')
    cut = nodes.new('ShaderNodeMath')
    cut.operation = 'SUBTRACT'
    cut.inputs[1].default_value = 0.40
    clamp0 = nodes.new('ShaderNodeMath')
    clamp0.operation = 'MAXIMUM'
    clamp0.inputs[1].default_value = 0.0
    clump = nodes.new('ShaderNodeMath')
    clump.operation = 'POWER'
    clump.inputs[1].default_value = 2.2
    densMul = nodes.new('ShaderNodeMath')
    densMul.operation = 'MULTIPLY'
    densMul.inputs[1].default_value = density
    emitMul = nodes.new('ShaderNodeMath')
    emitMul.operation = 'MULTIPLY'
    emitMul.inputs[1].default_value = emission
    links.new(texcoord.outputs['Object'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], cut.inputs[0])
    links.new(cut.outputs['Value'], clamp0.inputs[0])
    links.new(clamp0.outputs['Value'], clump.inputs[0])
    links.new(clump.outputs['Value'], densMul.inputs[0])
    links.new(clump.outputs['Value'], emitMul.inputs[0])
    if 'Density' in volume.inputs:
        links.new(densMul.outputs['Value'], volume.inputs['Density'])
    if 'Emission Strength' in volume.inputs:
        links.new(emitMul.outputs['Value'], volume.inputs['Emission Strength'])
    if 'Emission Color' in volume.inputs:
        volume.inputs['Emission Color'].default_value = (color[0], color[1], color[2], 1.0)
    if 'Color' in volume.inputs:
        volume.inputs['Color'].default_value = (
            absorption[0],
            absorption[1],
            absorption[2],
            1.0,
        )
    if 'Blackbody Intensity' in volume.inputs:
        volume.inputs['Blackbody Intensity'].default_value = 0.0
    links.new(volume.outputs['Volume'], output.inputs['Volume'])
    return material


def _addStarDome(bpy: Any, earthRadius: float) -> None:
    dome = _createIco(bpy, 'KpgStars', earthRadius * 36.0, subdivisions=2)
    material = bpy.data.materials.new(name='KpgStarfield')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    mix = nodes.new('ShaderNodeMixShader')
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 220.0
    if 'Detail' in noise.inputs:
        noise.inputs['Detail'].default_value = 2.0
    gate = nodes.new('ShaderNodeMath')
    gate.operation = 'GREATER_THAN'
    gate.inputs[1].default_value = 0.78
    emission.inputs['Color'].default_value = (0.85, 0.90, 1.0, 1.0)
    emission.inputs['Strength'].default_value = 2.4
    links.new(noise.outputs['Fac'], gate.inputs[0])
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(gate.outputs['Value'], mix.inputs['Fac'])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    _markAlphaBlend(material)
    dome.data.materials.append(material)
    if hasattr(dome, 'visible_shadow'):
        dome.visible_shadow = False


def _spaceWorld(bpy: Any) -> None:
    world = bpy.data.worlds.new('KpgSpace')
    bpy.context.scene.world = world
    nodeTree = getattr(world, 'node_tree', None)
    if nodeTree is None:
        world.color = (0.004, 0.005, 0.01)
        return
    nodes = nodeTree.nodes
    links = nodeTree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputWorld')
    space = nodes.new('ShaderNodeBackground')
    space.inputs['Color'].default_value = (0.004, 0.005, 0.012, 1.0)
    space.inputs['Strength'].default_value = 0.06
    stars = nodes.new('ShaderNodeBackground')
    stars.inputs['Color'].default_value = (0.80, 0.86, 1.0, 1.0)
    stars.inputs['Strength'].default_value = 0.85
    texcoord = nodes.new('ShaderNodeTexCoord')
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 140.0
    if 'Detail' in noise.inputs:
        noise.inputs['Detail'].default_value = 0.0
    gate = nodes.new('ShaderNodeMath')
    gate.operation = 'GREATER_THAN'
    gate.inputs[1].default_value = 0.93
    mix = nodes.new('ShaderNodeMixShader')
    links.new(texcoord.outputs['Generated'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], gate.inputs[0])
    links.new(space.outputs['Background'], mix.inputs[1])
    links.new(stars.outputs['Background'], mix.inputs[2])
    links.new(gate.outputs['Value'], mix.inputs['Fac'])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])


def _texturedFireMaterial(
    bpy: Any,
    name: str,
    imagePath: str,
    *,
    emission: float,
    repeat: float,
) -> Any:
    path = Path(imagePath)
    if not path.is_file():
        raise RuntimeError(f'K–Pg fire texture missing: {path}')
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emissionNode = nodes.new('ShaderNodeEmission')
    texcoord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (repeat, repeat, repeat)
    image = nodes.new('ShaderNodeTexImage')
    loaded = bpy.data.images.load(str(path.resolve()), check_existing=True)
    image.image = loaded
    image.interpolation = 'Cubic'
    image.extension = 'REPEAT'
    toBw = nodes.new('ShaderNodeRGBToBW')
    lift = nodes.new('ShaderNodeMath')
    lift.operation = 'MULTIPLY_ADD'
    lift.inputs[1].default_value = emission
    lift.inputs[2].default_value = 1.1
    tint = nodes.new('ShaderNodeMixRGB')
    if hasattr(tint, 'blend_type'):
        tint.blend_type = 'MULTIPLY'
    tint.inputs['Fac'].default_value = 0.55
    tint.inputs['Color2'].default_value = (1.0, 0.32, 0.04, 1.0)
    links.new(texcoord.outputs['UV'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], image.inputs['Vector'])
    links.new(image.outputs['Color'], tint.inputs['Color1'])
    links.new(tint.outputs['Color'], emissionNode.inputs['Color'])
    links.new(image.outputs['Color'], toBw.inputs['Color'])
    links.new(toBw.outputs['Val'], lift.inputs[0])
    links.new(lift.outputs['Value'], emissionNode.inputs['Strength'])
    links.new(emissionNode.outputs['Emission'], output.inputs['Surface'])
    return material


def _buildFireball(
    bpy: Any, earthRadius: float, normal: tuple[float, float, float], imagePath: str
) -> Any:
    east, north = _tangentPair(normal)
    parent = bpy.data.objects.new('KpgFireball', None)
    bpy.context.scene.collection.objects.link(parent)
    material = _texturedFireMaterial(bpy, 'KpgFireSurf', imagePath, emission=2.2, repeat=2.6)
    clumps = (
        ((0.0, 0.0, 0.0), 0.040),
        ((0.022, 0.012, 0.004), 0.028),
        ((-0.018, 0.016, -0.006), 0.024),
        ((0.008, -0.020, 0.010), 0.022),
    )
    for index, (offset, size) in enumerate(clumps):
        chunk = _createIco(bpy, f'KpgFire{index:02d}', earthRadius * size, subdivisions=3)
        chunk.parent = parent
        chunk.location = (
            (east[0] * offset[0] + north[0] * offset[1] + normal[0] * offset[2]) * earthRadius,
            (east[1] * offset[0] + north[1] * offset[1] + normal[1] * offset[2]) * earthRadius,
            (east[2] * offset[0] + north[2] * offset[1] + normal[2] * offset[2]) * earthRadius,
        )
        chunk.data.materials.append(material)
        if hasattr(chunk, 'visible_shadow'):
            chunk.visible_shadow = False
    return parent


def _buildSmokeColumn(
    bpy: Any, earthRadius: float, normal: tuple[float, float, float], imagePath: str
) -> Any:
    smoke = _createIco(bpy, 'KpgSmoke', earthRadius * 0.075, subdivisions=3)
    _alignPlusZ(smoke, normal)
    smoke.data.materials.append(
        _texturedFireMaterial(bpy, 'KpgSmokeSurf', imagePath, emission=0.7, repeat=1.8)
    )
    if hasattr(smoke, 'visible_shadow'):
        smoke.visible_shadow = False
    return smoke


def _buildHeatTrail(
    bpy: Any, earthRadius: float, inbound: tuple[float, float, float], imagePath: str
) -> Any:
    trail = _createIco(bpy, 'KpgHeatTrail', earthRadius * 0.016, subdivisions=3)
    back = (-inbound[0], -inbound[1], -inbound[2])
    _alignPlusZ(trail, back)
    trail.scale = (0.55, 0.55, 3.4)
    trail.data.materials.append(
        _texturedFireMaterial(bpy, 'KpgHeatSurf', imagePath, emission=1.4, repeat=3.0)
    )
    if hasattr(trail, 'visible_shadow'):
        trail.visible_shadow = False
    return trail


def _addFillLight(
    bpy: Any, scene: Any, earth: Any, sunLocation: tuple[float, float, float], theme: str
) -> Any:
    fillData = bpy.data.lights.new('KpgFill', type='SUN')
    fillData.energy = 0.32 if theme == 'dark' else 0.58
    fillData.color = (0.55, 0.68, 1.0)
    fillData.angle = math.radians(14.0)
    fill = bpy.data.objects.new('KpgFill', fillData)
    fill.location = (-sunLocation[0], -sunLocation[1], -sunLocation[2])
    scene.collection.objects.link(fill)
    track = fill.constraints.new(type='TRACK_TO')
    track.target = earth
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    return fill


def _keyCinematicVolumes(
    fire: Any,
    smoke: Any,
    trail: Any,
    flash: Any,
    sample: dict[str, Any],
    frame: int,
    normal: tuple[float, float, float],
    inbound: tuple[float, float, float],
    earthRadius: float,
) -> None:
    site = tuple(float(value) for value in sample['impactorAu'])
    _fireScale = float(sample['fireballScale'])
    _smokeScale = float(sample['plumeScale'])
    firePos = _offsetAlong(site, normal, earthRadius * 0.07)
    smokePos = _offsetAlong(site, normal, earthRadius * 0.08)
    _keyLocation(fire, firePos, frame)
    _keyScale(fire, 0.0, frame)
    _keyLocation(smoke, smokePos, frame)
    _keyScale(smoke, 0.0, frame)
    back = (-inbound[0], -inbound[1], -inbound[2])
    _keyLocation(trail, _offsetAlong(site, back, earthRadius * 0.04), frame)
    _keyScale(trail, 0.0, frame)
    _keyLocation(flash, firePos, frame)


def _burstImageMaterial(bpy: Any, name: str, imagePath: str, strength: float) -> Any:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    image = bpy.data.images.load(imagePath)
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = image
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = strength
    links.new(tex.outputs['Color'], emission.inputs['Color'])
    rgb = nodes.new('ShaderNodeSeparateColor')
    links.new(tex.outputs['Color'], rgb.inputs['Color'])
    luma = nodes.new('ShaderNodeMath')
    luma.operation = 'ADD'
    links.new(rgb.outputs['Red'], luma.inputs[0])
    links.new(rgb.outputs['Green'], luma.inputs[1])
    clip = nodes.new('ShaderNodeMath')
    clip.operation = 'GREATER_THAN'
    clip.inputs[1].default_value = 0.04
    links.new(luma.outputs['Value'], clip.inputs[0])
    mix = nodes.new('ShaderNodeMixShader')
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    links.new(clip.outputs['Value'], mix.inputs['Fac'])
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    _markAlphaBlend(material)
    return material


def _burstSkinMaterial(
    bpy: Any,
    name: str,
    color: tuple[float, float, float],
    emission: float,
) -> Any:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodeTree = material.node_tree
    nodes = nodeTree.nodes
    links = nodeTree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1.0)
    if 'Roughness' in principled.inputs:
        principled.inputs['Roughness'].default_value = 0.64
    if 'Emission Color' in principled.inputs:
        principled.inputs['Emission Color'].default_value = (1.0, 0.34, 0.06, 1.0)
    if 'Emission Strength' in principled.inputs:
        principled.inputs['Emission Strength'].default_value = emission
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    return material


def _createSmoothOrb(bpy: Any, name: str, radius: float) -> Any:
    import bmesh  # type: ignore[import-not-found]

    mesh = bpy.data.meshes.new(name)
    builder = bmesh.new()
    bmesh.ops.create_icosphere(builder, subdivisions=3, radius=radius)
    builder.to_mesh(mesh)
    builder.free()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _fireOrbMaterial(
    bpy: Any, name: str, color: tuple[float, float, float], strength: float
) -> Any:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    mix = nodes.new('ShaderNodeMixShader')
    texcoord = nodes.new('ShaderNodeTexCoord')
    center = nodes.new('ShaderNodeVectorMath')
    center.operation = 'SUBTRACT'
    center.inputs[1].default_value = (0.5, 0.5, 0.5)
    length = nodes.new('ShaderNodeVectorMath')
    length.operation = 'LENGTH'
    edge = nodes.new('ShaderNodeMath')
    edge.operation = 'MULTIPLY'
    edge.inputs[1].default_value = 2.0
    falloff = nodes.new('ShaderNodeMath')
    falloff.operation = 'SUBTRACT'
    falloff.inputs[0].default_value = 1.0
    soft = nodes.new('ShaderNodeMath')
    soft.operation = 'POWER'
    soft.inputs[1].default_value = 1.65
    clip = nodes.new('ShaderNodeClamp')
    clip.inputs['Min'].default_value = 0.0
    clip.inputs['Max'].default_value = 1.0
    emission.inputs['Color'].default_value = (color[0], color[1], color[2], 1.0)
    emission.inputs['Strength'].default_value = strength
    links.new(texcoord.outputs['Generated'], center.inputs[0])
    links.new(center.outputs['Vector'], length.inputs[0])
    links.new(length.outputs['Value'], edge.inputs[0])
    links.new(edge.outputs['Value'], falloff.inputs[1])
    links.new(falloff.outputs['Value'], soft.inputs[0])
    links.new(soft.outputs['Value'], clip.inputs['Value'])
    factor = mix.inputs['Fac'] if 'Fac' in mix.inputs else mix.inputs['Factor']
    links.new(clip.outputs['Result'], factor)
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    _markAlphaBlend(material)
    if hasattr(material, 'use_backface_culling'):
        material.use_backface_culling = False
    if hasattr(material, 'shadow_method'):
        material.shadow_method = 'NONE'
    return material


def _buildImpactBurst(
    bpy: Any,
    job: dict[str, Any],
    earthRadius: float,
    normal: tuple[float, float, float],
) -> list[tuple[Any, tuple[float, float, float], float]]:
    del job, normal
    lumps: list[tuple[Any, tuple[float, float, float], float]] = []
    orbs = (
        ((1.00, 0.55, 0.10), 1.55, 0.024, 0.000, 0.000, 0.016, 1.00),
        ((1.00, 0.32, 0.05), 0.95, 0.050, 0.000, 0.000, 0.018, 1.00),
    )
    for index, (color, strength, size, east, north, up, scaleMul) in enumerate(orbs):
        chunk = _createSmoothOrb(bpy, f'KpgFireOrb{index}', earthRadius * size)
        chunk.data.materials.append(_fireOrbMaterial(bpy, f'KpgFireOrb{index}', color, strength))
        if hasattr(chunk, 'visible_shadow'):
            chunk.visible_shadow = False
        lumps.append((chunk, (east, north, up), scaleMul))
    return lumps


def _keyImpactBurst(
    lumps: list[tuple[Any, tuple[float, float, float], float]],
    sample: dict[str, Any],
    frame: int,
    normal: tuple[float, float, float],
    earthRadius: float,
) -> None:
    flash = float(sample['flashScale'])
    fire = float(sample['fireballScale'])
    grow = 0.62 * flash + 0.82 * fire
    east, north = _tangentPair(normal)
    surface = (normal[0] * earthRadius, normal[1] * earthRadius, normal[2] * earthRadius)
    for lump, (alongEast, alongNorth, alongNormal), scaleMul in lumps:
        lift = earthRadius * alongNormal * (0.70 + 0.40 * fire)
        _keyLocation(
            lump,
            (
                surface[0]
                + east[0] * alongEast * earthRadius
                + north[0] * alongNorth * earthRadius
                + normal[0] * lift,
                surface[1]
                + east[1] * alongEast * earthRadius
                + north[1] * alongNorth * earthRadius
                + normal[1] * lift,
                surface[2]
                + east[2] * alongEast * earthRadius
                + north[2] * alongNorth * earthRadius
                + normal[2] * lift,
            ),
            frame,
        )
        _keyScale(lump, grow * scaleMul, frame)


def _keyRockLook(impactor: Any, sample: dict[str, Any], frame: int) -> None:
    materials = getattr(impactor.data, 'materials', None)
    if not materials:
        return
    material = materials[0]
    nodeTree = getattr(material, 'node_tree', None)
    if nodeTree is None:
        return
    principled = next((node for node in nodeTree.nodes if node.type == 'BSDF_PRINCIPLED'), None)
    if principled is None:
        return
    heat = float(sample.get('rockHeat', 0.0))
    if 'Emission Color' in principled.inputs:
        principled.inputs['Emission Color'].default_value = (1.0, 0.38 + 0.28 * heat, 0.06, 1.0)
        principled.inputs['Emission Color'].keyframe_insert(data_path='default_value', frame=frame)
    if 'Emission Strength' in principled.inputs:
        principled.inputs['Emission Strength'].default_value = 0.55 * heat if heat > 1e-3 else 0.0
        principled.inputs['Emission Strength'].keyframe_insert(
            data_path='default_value', frame=frame
        )


def _buildEntryTrail(
    bpy: Any, earthRadius: float, inbound: tuple[float, float, float]
) -> tuple[Any, float]:
    back = (-inbound[0], -inbound[1], -inbound[2])
    trail = _createCloudChunk(bpy, 'KpgEntryTrail', earthRadius * 0.006, 9, stretchZ=1.6)
    trail.data.materials.append(
        _emissionMaterial(bpy, 'KpgTrailSkin', (1.0, 0.28, 0.05), 0.55, 0.85)
    )
    _alignPlusZ(trail, back)
    return trail, earthRadius * 0.03


def _buildCrustPlates(
    bpy: Any, earthRadius: float, normal: tuple[float, float, float], count: int
) -> list[tuple[Any, tuple[float, float, float], float]]:
    east, north = _tangentPair(normal)
    plates: list[tuple[Any, tuple[float, float, float], float]] = []
    skin = _burstSkinMaterial(bpy, 'KpgCrustSkin', (0.22, 0.10, 0.05), 0.04)
    azimuths = (0.18, 1.02, 1.88, 2.70, 3.85, 5.25)
    sizes = (0.010, 0.016, 0.008, 0.014, 0.011, 0.018)
    reaches = (0.72, 1.28, 0.88, 1.18, 0.64, 1.42)
    for index in range(max(count, 1)):
        azimuth = azimuths[index % len(azimuths)]
        offset = (
            east[0] * math.cos(azimuth) + north[0] * math.sin(azimuth),
            east[1] * math.cos(azimuth) + north[1] * math.sin(azimuth),
            east[2] * math.cos(azimuth) + north[2] * math.sin(azimuth),
        )
        plate = _createDebrisChunk(
            bpy,
            f'KpgCrust{index:02d}',
            earthRadius * sizes[index % len(sizes)],
            40 + index * 7,
        )
        _alignPlusZ(plate, normal)
        plate.data.materials.append(skin)
        plates.append((plate, offset, reaches[index % len(reaches)]))
    return plates


def _keyCrustPlates(
    plates: list[tuple[Any, tuple[float, float, float], float]],
    sample: dict[str, Any],
    frame: int,
    normal: tuple[float, float, float],
    earthRadius: float,
) -> None:
    tear = float(sample.get('crustTear', 0.0))
    surface = (normal[0] * earthRadius, normal[1] * earthRadius, normal[2] * earthRadius)
    for index, (plate, offset, reachMul) in enumerate(plates):
        reach = earthRadius * (0.028 + 0.07 * tear) * reachMul
        lift = earthRadius * (0.012 + 0.018 * (index % 3)) * tear
        _keyLocation(
            plate,
            (
                surface[0] + offset[0] * reach + normal[0] * lift,
                surface[1] + offset[1] * reach + normal[1] * lift,
                surface[2] + offset[2] * reach + normal[2] * lift,
            ),
            frame,
        )
        _keyScale(plate, tear * (0.75 + 0.35 * ((index * 0.37) % 1.0)), frame, zScale=0.88)


def applyKpgJobInBlender(job: dict[str, Any]) -> Path:
    import bpy  # type: ignore[import-not-found]

    flyby = _flybyModule()
    theme = str(job['theme'])
    frames = job['frames']
    outputDirectory = Path(job['outputDirectory'])
    outputDirectory.mkdir(parents=True, exist_ok=True)

    contact = job['contact']
    normal = tuple(float(value) for value in contact['normal'])
    flyby._clearSceneObjects(bpy)
    earth, radius, weather = _buildEarth(bpy, flyby, job)
    impactor = _buildImpactor(bpy, flyby, job, radius)

    lookAt = bpy.data.objects.new('KpgLookAt', None)
    bpy.context.scene.collection.objects.link(lookAt)

    scene = bpy.context.scene
    scene.frame_start = int(frames[0]['frame'])
    scene.frame_end = int(frames[-1]['frame'])
    scene.frame_current = scene.frame_start

    cameraData = bpy.data.cameras.new('KpgCameraData')
    cameraData.lens = 35
    cameraData.clip_start = max(radius * 1e-5, 1e-8)
    cameraData.clip_end = max(radius * 80.0, 10.0)
    camera = bpy.data.objects.new('KpgCamera', cameraData)
    scene.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new(type='TRACK_TO')
    track.target = lookAt
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    inbound = tuple(float(value) for value in contact['inbound'])
    trail, trailHeight = _buildEntryTrail(bpy, radius, inbound)
    bursts: list[tuple[Any, tuple[float, float, float], float]] = []
    plates: list[tuple[Any, tuple[float, float, float], float]] = []
    explosionCards: list[tuple[Any, str, float]] = []
    rocks: list[Any] = []
    swarm: list[tuple[Any, tuple[float, float, float, float, float], bool]] = []
    impact = _impactFrame(frames)
    fps = float(job.get('fps', 20))

    def _onFrame(scene: Any) -> None:
        _syncSwarm(swarm, int(scene.frame_current), normal, inbound, radius, impact, fps)

    bpy.app.handlers.frame_change_pre.append(_onFrame)
    _onFrame(scene)
    flashData = bpy.data.lights.new('KpgBlast', type='POINT')
    flashData.energy = 0.0
    flashData.color = (1.0, 0.52, 0.16)
    if hasattr(flashData, 'shadow_soft_size'):
        flashData.shadow_soft_size = radius * 0.12
    if hasattr(flashData, 'use_shadow'):
        flashData.use_shadow = False
    blast = bpy.data.objects.new('KpgBlast', flashData)
    scene.collection.objects.link(blast)
    flashEnergy = 0.08 if theme == 'dark' else 0.055

    sunEnergy = 2.15 if theme == 'dark' else 4.2
    lightData = bpy.data.lights.new('KpgSun', type='SUN')
    lightData.energy = sunEnergy
    lightData.angle = math.radians(4.8)
    sun = bpy.data.objects.new('KpgSun', lightData)
    east, north = _tangentPair(normal)
    sun.location = (
        (normal[0] + 0.85 * east[0] + 0.25 * north[0]) * radius * 22.0,
        (normal[1] + 0.85 * east[1] + 0.25 * north[1]) * radius * 22.0,
        (normal[2] + 0.85 * east[2] + 0.25 * north[2]) * radius * 22.0,
    )
    scene.collection.objects.link(sun)
    sunTrack = sun.constraints.new(type='TRACK_TO')
    sunTrack.target = earth
    sunTrack.track_axis = 'TRACK_NEGATIVE_Z'
    sunTrack.up_axis = 'UP_Y'
    fill = _addFillLight(bpy, scene, earth, sun.location, theme)
    fillData = fill.data
    fillEnergy = float(fillData.energy)
    earth.rotation_mode = 'XYZ'
    impactor.parent = earth
    trail.parent = earth
    blast.parent = earth
    sun.parent = earth
    fill.parent = earth
    tools = getattr(scene, 'tool_settings', None)
    if tools is not None and hasattr(tools, 'keyframe_interpolation'):
        tools.keyframe_interpolation = 'LINEAR'

    impactFrame = _impactFrame(frames)
    for sample in frames:
        frame = int(sample['frame'])
        earth.rotation_euler = (0.0, 0.0, float(sample.get('earthSpin', 0.0)))
        earth.keyframe_insert(data_path='rotation_euler', frame=frame)
        _keyLocation(camera, tuple(float(value) for value in sample['cameraAu']), frame)
        _keyLocation(lookAt, tuple(float(value) for value in sample['lookAtAu']), frame)
        cameraData.lens = float(sample.get('lens', 35.0))
        cameraData.keyframe_insert(data_path='lens', frame=frame)
        _keyLocation(impactor, tuple(float(value) for value in sample['impactorAu']), frame)
        _keyScale(
            impactor,
            float(sample.get('slamScale', 1.0 if sample.get('impactorVisible') else 0.0)),
            frame,
        )
        _keyRockLook(impactor, sample, frame)
        _keyWeather(weather, sample, frame, impactFrame)
        _keyInboundTrail(trail, trailHeight, inbound, sample, frame)
        _keyImpactBurst(bursts, sample, frame, normal, radius)
        _keyCrustPlates(plates, sample, frame, normal, radius)
        _keyExplosionPlates(explosionCards, sample, frame, normal, radius)
        _keyRocks(rocks, sample, frame)
        surface = (normal[0] * radius, normal[1] * radius, normal[2] * radius)
        _keyLocation(blast, _offsetAlong(surface, normal, radius * 0.08), frame)
        flashData.energy = flashEnergy * _blastLampScale(
            frame, float(sample['flashScale']), impactFrame
        )
        flashData.keyframe_insert(data_path='energy', frame=frame)
        lightData.energy = sunEnergy * float(sample['sunScale'])
        lightData.keyframe_insert(data_path='energy', frame=frame)
        fillData.energy = fillEnergy * float(sample['sunScale'])
        fillData.keyframe_insert(data_path='energy', frame=frame)
    _linearizeEarthSpin(earth)

    _spaceWorld(bpy)
    flyby._configureFlybyRenderer(
        scene,
        ringsEnabled=False,
        resolution=int(job['resolution']),
        fps=int(job.get('fps', 20)),
        filmTransparent=False,
        theme=theme,
        outputDirectory=outputDirectory,
        isStar=False,
    )
    _applyCinemaLook(scene)
    _enableExplosionBloom(scene)
    stills = [int(frame) for frame in job.get('stillFrames') or []]
    if stills:
        print(f'Rendering {len(stills)} contact stills...')
        for frame in stills:
            scene.frame_set(frame)
            scene.render.filepath = str(outputDirectory / f'frame_{frame:04d}')
            bpy.ops.render.render(write_still=True)
    else:
        print('Rendering K–Pg full event...')
        bpy.ops.render.render(animation=True)
    written = sorted(outputDirectory.glob('frame_*.png'))
    if not written:
        raise RuntimeError(f'Blender produced no PNG frames in {outputDirectory}')
    return outputDirectory


def _applyCinemaLook(scene: Any) -> None:
    # Bloom turns the ash sheet into a glowing beige dome. Tonga is matte.
    if hasattr(scene.render, 'use_motion_blur'):
        scene.render.use_motion_blur = False
    eevee = getattr(scene, 'eevee', None)
    if eevee is not None and hasattr(eevee, 'use_motion_blur'):
        eevee.use_motion_blur = False
    viewSettings = getattr(scene, 'view_settings', None)
    if viewSettings is None:
        return
    viewSettings.exposure = 0.35
    for name in ('AgX', 'Filmic', 'Standard'):
        try:
            viewSettings.view_transform = name
            break
        except TypeError:
            continue
    if hasattr(viewSettings, 'look'):
        try:
            viewSettings.look = 'None'
        except TypeError:
            pass


def _heroVolumeMaterial(bpy: Any) -> Any:
    material = bpy.data.materials.new(name='CyclesPlateFire')
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
    densityScale.inputs[1].default_value = 12.0
    emissionScale = nodes.new('ShaderNodeMath')
    emissionScale.operation = 'MULTIPLY'
    emissionScale.inputs[1].default_value = 22.0
    heat = nodes.new('ShaderNodeMath')
    heat.operation = 'MULTIPLY'
    heat.inputs[1].default_value = 2800.0
    heatBias = nodes.new('ShaderNodeMath')
    heatBias.operation = 'ADD'
    heatBias.inputs[1].default_value = 900.0
    links.new(densityAttr.outputs['Fac'], densityScale.inputs[0])
    links.new(flameAttr.outputs['Fac'], emissionScale.inputs[0])
    links.new(flameAttr.outputs['Fac'], heat.inputs[0])
    links.new(heat.outputs['Value'], heatBias.inputs[0])
    if 'Density' in volume.inputs:
        links.new(densityScale.outputs['Value'], volume.inputs['Density'])
    if 'Emission Strength' in volume.inputs:
        links.new(emissionScale.outputs['Value'], volume.inputs['Emission Strength'])
    if 'Color' in volume.inputs:
        volume.inputs['Color'].default_value = (0.018, 0.012, 0.008, 1.0)
    if 'Absorption Color' in volume.inputs:
        volume.inputs['Absorption Color'].default_value = (0.04, 0.02, 0.01, 1.0)
    if 'Blackbody Intensity' in volume.inputs:
        volume.inputs['Blackbody Intensity'].default_value = 1.0
    if 'Temperature' in volume.inputs:
        links.new(heatBias.outputs['Value'], volume.inputs['Temperature'])
    try:
        blackbody = nodes.new('ShaderNodeBlackbody')
        links.new(heatBias.outputs['Value'], blackbody.inputs['Temperature'])
        if 'Emission Color' in volume.inputs:
            links.new(blackbody.outputs['Color'], volume.inputs['Emission Color'])
    except RuntimeError:
        if 'Emission Color' in volume.inputs:
            volume.inputs['Emission Color'].default_value = (1.0, 0.38, 0.06, 1.0)
    links.new(volume.outputs['Volume'], output.inputs['Volume'])
    return material


def _addHeroInflow(
    bpy: Any,
    flyby: ModuleType,
    name: str,
    location: tuple[float, float, float],
    radius: float,
    *,
    velocity: float,
    stop: int,
) -> Any:
    inflow = flyby._createBodySphere(bpy, name, radius)
    inflow.location = location
    inflow.hide_render = True
    modifier = inflow.modifiers.new(name='Fluid', type='FLUID')
    modifier.fluid_type = 'FLOW'
    flow = modifier.flow_settings
    flow.flow_type = 'BOTH'
    flow.flow_behavior = 'INFLOW'
    flow.density = 1.0
    flow.fuel_amount = 1.0
    flow.temperature = 3.2
    flow.smoke_color = (0.04, 0.025, 0.015)
    flow.use_initial_velocity = True
    flow.velocity_normal = velocity
    flow.velocity_random = 0.75
    flow.use_inflow = False
    flow.keyframe_insert(data_path='use_inflow', frame=-1)
    flow.use_inflow = True
    flow.keyframe_insert(data_path='use_inflow', frame=1)
    flow.use_inflow = True
    flow.keyframe_insert(data_path='use_inflow', frame=stop)
    flow.use_inflow = False
    flow.keyframe_insert(data_path='use_inflow', frame=stop + 1)
    return inflow


def _buildHeroDomain(bpy: Any, flyby: ModuleType, outputDirectory: Path, frameCount: int) -> Any:
    domain = _createCube(bpy, 'HeroDomain', 2.8)
    domain.scale = (1.0, 1.0, 1.55)
    domain.location = (0.0, 0.0, 1.45)
    fluid = domain.modifiers.new(name='Fluid', type='FLUID')
    fluid.fluid_type = 'DOMAIN'
    settings = fluid.domain_settings
    settings.domain_type = 'GAS'
    settings.resolution_max = 96
    settings.cache_type = 'ALL'
    settings.cache_directory = str(outputDirectory / 'cycles_fluid_cache')
    settings.cache_frame_start = 1
    settings.cache_frame_end = frameCount
    settings.use_adaptive_domain = False
    settings.use_noise = True
    settings.noise_scale = 3
    settings.noise_strength = 1.15
    settings.vorticity = 1.15
    settings.burning_rate = 0.48
    settings.flame_smoke = 1.45
    settings.use_dissolve_smoke = False
    settings.dissolve_speed = 80
    settings.use_collision_border_bottom = True
    settings.use_collision_border_top = False
    settings.use_collision_border_front = False
    settings.use_collision_border_back = False
    settings.use_collision_border_left = False
    settings.use_collision_border_right = False
    if hasattr(settings, 'gravity'):
        settings.gravity = (0.0, 0.0, -9.81)
    domain.data.materials.append(_heroVolumeMaterial(bpy))
    _addHeroInflow(bpy, flyby, 'HeroInflow', (0.0, 0.0, 0.42), 0.62, velocity=5.8, stop=8)
    _addHeroInflow(bpy, flyby, 'HeroInflowB', (0.18, -0.12, 0.38), 0.32, velocity=4.4, stop=6)
    ground = _createCube(bpy, 'HeroGround', 4.4)
    ground.scale = (1.0, 1.0, 0.04)
    ground.location = (0.0, 0.0, -0.10)
    ground.hide_render = True
    collide = ground.modifiers.new(name='Fluid', type='FLUID')
    collide.fluid_type = 'EFFECTOR'
    if hasattr(collide, 'effector_settings'):
        collide.effector_settings.effector_type = 'COLLISION'
    return domain


def _configureHeroCycles(
    scene: Any, outputDirectory: Path, frameCount: int, resolution: int
) -> None:
    scene.frame_start = 1
    scene.frame_end = frameCount
    scene.frame_current = 1
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.filepath = str(outputDirectory / 'hero_')
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.fps = 20
    cycles = scene.cycles
    cycles.samples = 28
    cycles.use_denoising = True
    if hasattr(cycles, 'volume_bounces'):
        cycles.volume_bounces = 2
    if hasattr(cycles, 'device'):
        try:
            cycles.device = 'GPU'
        except TypeError:
            pass
    viewSettings = getattr(scene, 'view_settings', None)
    if viewSettings is not None:
        try:
            viewSettings.view_transform = 'Standard'
        except TypeError:
            pass
        viewSettings.exposure = 0.25
    world = scene.world
    if world is not None:
        world.use_nodes = True
        background = next(
            (node for node in world.node_tree.nodes if node.type == 'BACKGROUND'), None
        )
        if background is not None and 'Color' in background.inputs:
            background.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        if background is not None and 'Strength' in background.inputs:
            background.inputs['Strength'].default_value = 0.0


def renderHeroExplosion(
    outputDirectory: Path, *, frameCount: int = 36, resolution: int = 512
) -> Path:
    import bpy  # type: ignore[import-not-found]

    flyby = _flybyModule()
    outputDirectory.mkdir(parents=True, exist_ok=True)
    flyby._clearSceneObjects(bpy)
    scene = bpy.context.scene
    _configureHeroCycles(scene, outputDirectory, frameCount, resolution)
    domain = _buildHeroDomain(bpy, flyby, outputDirectory, frameCount)
    lookAt = bpy.data.objects.new('HeroLook', None)
    lookAt.location = (0.0, 0.0, 0.95)
    scene.collection.objects.link(lookAt)
    cameraData = bpy.data.cameras.new('HeroCamData')
    cameraData.lens = 32
    cameraData.clip_start = 0.05
    cameraData.clip_end = 40.0
    camera = bpy.data.objects.new('HeroCam', cameraData)
    camera.location = (2.35, -3.15, 1.85)
    scene.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new(type='TRACK_TO')
    track.target = lookAt
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    _enableExplosionBloom(scene)
    print('Baking Cycles fire plate...')
    _bakeGasDomain(bpy, domain)
    bpy.ops.render.render(animation=True)
    written = sorted(outputDirectory.glob('hero_*.png'))
    if not written:
        raise RuntimeError(f'Hero explosion produced no PNGs in {outputDirectory}')
    print(f'Cycles fire plate → {outputDirectory} ({len(written)} frames)')
    return outputDirectory


def main(argv: list[str] | None = None) -> int:
    args = _argvAfterDoubleDash(list(argv if argv is not None else sys.argv))
    if not args:
        print(
            'Usage: render_kpg.py <kpg_job.json>\n'
            '   or: blender --background --python render_kpg.py -- <kpg_job.json>\n'
            '   or: blender --background --python render_kpg.py -- --hero-explosion <dir>',
            file=sys.stderr,
        )
        return 2
    flyby = _flybyModule()
    if args[0] == '--hero-explosion':
        if len(args) < 2:
            print('Usage: --hero-explosion <output-directory>', file=sys.stderr)
            return 2
        if not flyby._bpyAvailable():
            print('bpy not available — dry-run validation only.')
            return 0
        renderHeroExplosion(Path(args[1]))
        return 0
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
