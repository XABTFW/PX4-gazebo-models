"""Reduced-order litter-capture net: tension-only ropes and sphere contact.

Simulation parameters, not specifications for a real launcher. Gazebo integrates
the particle masses; this module only computes forces. Triangle contact uses an
effective solid cloth envelope, with dense central body-contact samples and a
coarse outer perimeter for large nets. It does not simulate individual filament
contact, plastic-bag folding, rotor entanglement or tearing.
"""
from dataclasses import dataclass
import math
import re
import time
import xml.etree.ElementTree as ET

import numpy as np


class CameraRangeGate:
    """Identity-checked, exposure- and wall-time-bounded camera distance input."""
    def __init__(self, observer, target):
        self.observer, self.target = observer, target
        self.record, self.received = None, None

    def observe(self, record):
        self.record, self.received = None, None
        if not isinstance(record, dict) or record.get('valid') is not True:
            return
        if (record.get('schema_version') != 1 or record.get('source') != 'gazebo_ground_truth'
                or record.get('observer_model') != self.observer or record.get('target_model') != self.target):
            return
        distance, stamp, identity = record.get('distance_m'), record.get('stamp_ns'), record.get('tracking_target_id')
        if (not isinstance(distance, (int, float)) or isinstance(distance, bool)
                or not math.isfinite(distance) or distance <= 0
                or not isinstance(stamp, int) or isinstance(stamp, bool) or stamp <= 0
                or not isinstance(identity, int) or isinstance(identity, bool) or identity < 1):
            return
        self.record, self.received = record.copy(), time.monotonic()

    def measurement(self, sim_time):
        if self.record is None:
            return {'valid': False, 'status': 'camera_range_missing_or_invalid', 'distance_m': None}
        age = sim_time-self.record['stamp_ns']/1e9
        wall_age = time.monotonic()-self.received
        if not (0 <= age <= .2 and 0 <= wall_age <= .2):
            return {'valid': False, 'status': 'camera_range_stale_or_future', 'distance_m': None}
        return {'valid': True, 'status': 'camera_range_fresh', 'distance_m': self.record['distance_m'],
                'sample_age_sim_s': age, 'tracking_target_id': self.record['tracking_target_id']}


@dataclass
class NetConfig:
    resolution: int = 7
    width: float = 5.0
    folded_width: float = .24
    mass: float = .25
    spring: float = 12.
    damping: float = .04
    drag_area: float = .035
    launch_speed: float = 8.
    spread_speed: float = 5.
    target_radius: float = .35
    contact_spring: float = 15.
    contact_damping: float = .12
    attachment_spring: float = 10.
    break_force: float = 3.
    capture_hold: float = .04
    lifetime: float = 10.

    def validate(self):
        if not isinstance(self.resolution, int) or isinstance(self.resolution, bool) or not 3 <= self.resolution <= 9 or self.resolution % 2 != 1:
            raise ValueError('resolution must be an odd integer between 3 and 9')
        for key, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError('invalid net parameter: '+key)
        if self.folded_width >= self.width:
            raise ValueError('folded width must be smaller than deployed width')
        if self.target_radius < min(self.width, 2.)/(self.resolution-1)/2:
            raise ValueError('target envelope is too small for central contact resolution')
        if self.launch_speed > 20 or self.spread_speed > 12 or self.mass > 2:
            raise ValueError('parameters exceed the supported lightweight simulation range')
        minimum_mass = self.mass*.6/(self.resolution**2-4)
        if .004*math.sqrt(16*self.spring/minimum_mass) >= 1.8:
            raise ValueError('net too light/stiff for 4 ms physics; increase mass or reduce spring stiffness')
        return self


def grid(config):
    n = config.resolution
    coordinates = np.linspace(-1, 1, n)
    folded_y, folded_z = np.meshgrid(coordinates.copy(), coordinates.copy())
    folded = np.column_stack((np.zeros(n*n), folded_y.ravel(), folded_z.ravel()))*config.folded_width/2
    if config.width > 2:
        # Keep dense body-contact samples in the central two metres while
        # extending the perimeter. A uniform 10m/7 grid would leave too few
        # particles around the target to resolve wrapping. Faces remain a
        # continuous cloth envelope, not literal coarse filament openings.
        coordinates[1:-1] *= 2/config.width
    ys, zs = np.meshgrid(coordinates, coordinates)
    directions = np.column_stack((np.zeros(n*n), ys.ravel(), zs.ravel()))
    edges, visible, triangles = [], [], []
    for row in range(n):
        for col in range(n):
            a = row*n+col
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if row+dr < n and 0 <= col+dc < n:
                    edge = (a, (row+dr)*n+col+dc)
                    edges.append(edge)
                    if dr*dc == 0:
                        visible.append(edge)
            if row < n-1 and col < n-1:
                triangles.extend(((a, a+1, a+n), (a+1, a+n+1, a+n)))
    edges = np.array(edges, dtype=int)
    deployed = directions*config.width/2
    rest = np.linalg.norm(deployed[edges[:, 1]]-deployed[edges[:, 0]], axis=1)
    masses = np.full(n*n, config.mass/(n*n))
    corners = [0, n-1, n*(n-1), n*n-1]
    # Corner weights carry 40% of the net mass; preserve total mass exactly.
    masses[:] = config.mass*.6/(n*n-4)
    masses[corners] = config.mass*.1
    return folded, directions, edges, rest, np.array(triangles), np.array(visible), masses


def rope_forces(positions, velocities, edges, rest, stiffness, damping):
    delta = positions[edges[:, 1]]-positions[edges[:, 0]]
    lengths = np.linalg.norm(delta, axis=1)
    directions = delta/np.maximum(lengths[:, None], 1e-9)
    speeds = np.sum((velocities[edges[:, 1]]-velocities[edges[:, 0]])*directions, axis=1)
    tension = np.maximum(0., stiffness*(lengths-rest)+damping*speeds)
    tension[lengths <= rest] = 0  # Slack ropes cannot push particles apart.
    forces = np.zeros_like(positions)
    np.add.at(forces, edges[:, 0], directions*tension[:, None])
    np.add.at(forces, edges[:, 1], -directions*tension[:, None])
    return forces, float(np.max(tension, initial=0))


def closest_triangles(point, vertices):
    """Vectorised closest points and barycentric coordinates, including edges."""
    a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    ab, ac, ap = b-a, c-a, point-a
    aa, bb, cc = np.sum(ab*ab, axis=1), np.sum(ab*ac, axis=1), np.sum(ac*ac, axis=1)
    d, e = np.sum(ap*ab, axis=1), np.sum(ap*ac, axis=1)
    denominator = aa*cc-bb*bb
    safe = np.maximum(denominator, 1e-18)
    u, v = (cc*d-bb*e)/safe, (aa*e-bb*d)/safe
    weights = [np.column_stack((1-u-v, u, v))]
    candidates = [a+u[:, None]*ab+v[:, None]*ac]
    inside = (u >= 0) & (v >= 0) & (u+v <= 1) & (denominator > 1e-18)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        start, end = vertices[:, i], vertices[:, j]
        edge = end-start
        t = np.clip(np.sum((point-start)*edge, axis=1)/np.maximum(np.sum(edge*edge, axis=1), 1e-18), 0, 1)
        candidates.append(start+t[:, None]*edge)
        w = np.zeros_like(weights[0]); w[:, i] = 1-t; w[:, j] = t
        weights.append(w)
    candidates, weights = np.stack(candidates, axis=1), np.stack(weights, axis=1)
    distances = np.sum((candidates-point)**2, axis=2)
    distances[~inside, 0] = np.inf
    index = np.argmin(distances, axis=1)
    rows = np.arange(len(vertices))
    return candidates[rows, index], weights[rows, index]


def sphere_contact(positions, velocities, triangles, center, velocity, radius, stiffness, damping):
    points, weights = closest_triangles(center, positions[triangles])
    delta = points-center
    distances = np.linalg.norm(delta, axis=1)
    normals = delta/np.maximum(distances[:, None], 1e-9)
    local_velocity = np.sum(velocities[triangles]*weights[:, :, None], axis=1)
    singular = distances < 1e-9
    if np.any(singular):
        face = np.cross(positions[triangles[:, 1]]-positions[triangles[:, 0]],
                        positions[triangles[:, 2]]-positions[triangles[:, 0]])
        face /= np.maximum(np.linalg.norm(face, axis=1)[:, None], 1e-9)
        sign = np.where(np.sum((local_velocity-velocity)*face, axis=1) > 0, -1., 1.)
        normals[singular] = (face*sign[:, None])[singular]
    relative_speed = np.sum((local_velocity-velocity)*normals, axis=1)
    magnitude = np.maximum(0, stiffness*(radius-distances)-damping*relative_speed)
    magnitude[distances >= radius] = 0
    contact = normals*magnitude[:, None]
    forces = np.zeros_like(positions)
    for vertex in range(3):
        np.add.at(forces, triangles[:, vertex], contact*weights[:, vertex, None])
    # Reaction force and torque at the target centre: no momentum is fabricated.
    reaction = -contact.sum(axis=0)
    torque = -np.cross(points-center, contact).sum(axis=0)
    return forces, reaction, torque, int(np.count_nonzero(magnitude > 0))


def wrapping_nodes(positions, center, radius, shot_axis):
    relative = positions-center
    distance = np.linalg.norm(relative, axis=1)
    nearby = np.flatnonzero((distance >= radius*.6) & (distance <= radius*1.35))
    if len(nearby) < 4:
        return np.array([], dtype=int)
    axial = relative[nearby] @ shot_axis
    if np.min(axial) > -radius*.08 or np.max(axial) < radius*.08:
        return np.array([], dtype=int)  # Touching only one side is not capture.
    if np.max(np.linalg.norm(relative[nearby, None]-relative[None, nearby], axis=2)) < radius*1.2:
        return np.array([], dtype=int)
    return nearby


def capture_confirmed(contacts, wrapped_count, contact_time, wrap_time, hold):
    """Require current physical contact AND sustained two-sided body coverage."""
    return contacts > 0 and wrapped_count >= 4 and contact_time >= hold and wrap_time >= hold


def dense_net_segments(positions, resolution, width, spacing=.15):
    """Massless rendered strands interpolated on the deforming physical cloth.

    Physical nodes remain unchanged. Rest-space spacing determines strand count
    so the picture cannot gain/lose strands as the net folds or stretches.
    """
    points = np.asarray(positions, dtype=float).reshape(resolution, resolution, 3)
    if not np.all(np.isfinite(points)) or not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('invalid net display geometry')
    coordinates = np.linspace(-1, 1, resolution)
    if width > 2:
        coordinates[1:-1] *= 2/width
    divisions = np.maximum(1, np.ceil(np.diff(coordinates)*width/(2*spacing)).astype(int))
    segments = []
    for row in range(resolution-1):
        for col in range(resolution-1):
            a, b = points[row, col], points[row, col+1]
            c, d = points[row+1, col], points[row+1, col+1]
            # Avoid duplicate strands on shared cell boundaries.
            us = np.linspace(0, 1, divisions[col]+1)
            vs = np.linspace(0, 1, divisions[row]+1)
            if col: us = us[1:]
            if row: vs = vs[1:]
            segments.extend(np.stack((a+(b-a)*us[:, None], c+(d-c)*us[:, None]), axis=1))
            segments.extend(np.stack((a+(c-a)*vs[:, None], b+(d-b)*vs[:, None]), axis=1))
    return np.asarray(segments)


def valid_name(value):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', value):
        raise ValueError('model names must use letters, digits and underscores')
    return value


def model_sdf(name, config, observer, target, auto_fire=False, axis=(0, 0, 1), offset=(.18, 0, .78),
              trigger_distance=5.0, require_lock=True, demo=False, drop_on_capture=False, visible=False,
              range_source='camera'):
    """Generate independent Gazebo links: no edits to the aircraft's SDF."""
    config.validate()
    if range_source not in ('camera', 'model'):
        raise ValueError('range source must be camera or model')
    for value in (name, observer, target):
        valid_name(value)
    folded, _, _, _, _, _, masses = grid(config)
    root = ET.Element('sdf', version='1.9'); model = ET.SubElement(root, 'model', name=name)
    ET.SubElement(model, 'self_collide').text = 'false'
    for i, (position, mass) in enumerate(zip(folded, masses)):
        link = ET.SubElement(model, 'link', name=f'node_{i}')
        ET.SubElement(link, 'pose').text = ' '.join(map(str, [*position, 0, 0, 0]))
        inertia = ET.SubElement(link, 'inertial'); ET.SubElement(inertia, 'mass').text = str(mass)
        values = ET.SubElement(inertia, 'inertia')
        for key in ('ixx', 'iyy', 'izz'):
            ET.SubElement(values, key).text = str(mass*.00016)
        if not visible:
            # Omit render entities completely, not physical particle links.
            # Their inertia and Python rope/contact dynamics are unchanged.
            continue
        # Contact is analytical in the Python system, not double-counted by DART.
        visual = ET.SubElement(link, 'visual', name='particle')
        geometry = ET.SubElement(visual, 'geometry'); sphere = ET.SubElement(geometry, 'sphere')
        ET.SubElement(sphere, 'radius').text = '.025' if mass > config.mass/len(masses) else '.009'
        material = ET.SubElement(visual, 'material')
        for key in ('ambient', 'diffuse'):
            ET.SubElement(material, key).text = '1 .45 .03 1'
    plugin = ET.SubElement(model, 'plugin', filename='gz-sim-python-system-loader-system',
                           name='gz::sim::systems::PythonSystemLoader')
    values = dict(module_name='gz_net_system', observer=observer, target=target,
                  mount=name+'_launcher', auto_fire=str(auto_fire).lower(),
                  require_lock=str(require_lock).lower(), demo=str(demo).lower(),
                  drop_on_capture=str(drop_on_capture).lower(),
                  range_source=range_source,
                  axis=' '.join(map(str, axis)), offset=' '.join(map(str, offset)),
                  trigger_distance=trigger_distance, **vars(config))
    for key, value in values.items():
        ET.SubElement(plugin, key).text = str(value)
    return ET.tostring(root, encoding='unicode')


def launcher_sdf(name, visible=False):
    valid_name(name)
    sdf = f'''<sdf version="1.9"><model name="{name}"><static>true</static>
    <link name="mount"><visual name="tube"><pose>-.12 0 0 0 1.57079632679 0</pose>
    <geometry><cylinder><radius>.09</radius><length>.24</length></cylinder></geometry>
    <material><ambient>.12 .3 .18 1</ambient><diffuse>.12 .3 .18 1</diffuse></material>
    </visual><visual name="rim"><pose>.005 0 0 0 1.57079632679 0</pose>
    <geometry><cylinder><radius>.105</radius><length>.025</length></cylinder></geometry>
    <material><ambient>.8 .75 .1 1</ambient><diffuse>.8 .75 .1 1</diffuse></material>
    </visual></link></model></sdf>'''
    if visible:
        return sdf
    root = ET.fromstring(sdf)
    link = root.find('model/link')
    for visual in list(link.findall('visual')):
        link.remove(visual)
    return ET.tostring(root, encoding='unicode')
