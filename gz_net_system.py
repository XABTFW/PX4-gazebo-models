"""Gazebo 8 Python system for a lightweight elastic environmental-capture net.

No PX4 commands. Net particles are real independent Gazebo rigid bodies.
Stowed payload placement and the decorative mount are kinematic; released net
motion is force-driven. Environmental debris uses a spherical contact envelope.
"""
import json
import math
import threading
import time

import numpy as np
from gz.math7 import Vector3d, Vector2d, Pose3d, Quaterniond
from gz.sim8 import Model, Link, Joint, World, world_entity
from gz.transport13 import Node
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.float_v_pb2 import Float_V
from gz.msgs10.stringmsg_pb2 import StringMsg

from gz_net_physics import (NetConfig, CameraRangeGate, grid, rope_forces, sphere_contact,
                            wrapping_nodes, capture_confirmed)


def array(vector):
    return np.array([vector.x(), vector.y(), vector.z()])


def vector(values):
    return Vector3d(*map(float, values))


class NetSystem:
    def configure(self, entity, sdf, ecm, event_mgr):
        self.model, self.world = Model(entity), World(world_entity(ecm))
        self.name = self.model.name(ecm)
        self.config = NetConfig(**{key: (sdf.get_int(key) if key == 'resolution' else sdf.get_double(key))
                                  for key in vars(NetConfig())}).validate()
        self.folded, self.directions, self.edges, self.rest, self.triangles, self.visible, self.masses = grid(self.config)
        self.links = [Link(self.model.link_by_name(ecm, f'node_{i}')) for i in range(len(self.masses))]
        for link in self.links:
            link.enable_velocity_checks(ecm)
        self.observer_name, self.target_name = sdf.get_string('observer'), sdf.get_string('target')
        self.mount_name = sdf.get_string('mount')
        self.axis = np.array(list(map(float, sdf.get_string('axis').split())))
        self.axis /= np.linalg.norm(self.axis)
        self.offset = np.array(list(map(float, sdf.get_string('offset').split())))
        self.axis_rotation = Quaterniond(0, -math.atan2(self.axis[2], math.hypot(*self.axis[:2])),
                                         math.atan2(self.axis[1], self.axis[0]))
        self.auto, self.require_lock, self.demo = (sdf.get_bool(k) for k in ('auto_fire', 'require_lock', 'demo'))
        self.drop_on_capture = sdf.get_bool('drop_on_capture')
        self.drop_latched, self.drop_rotors = False, []
        self.captured_at, self.capture_height = None, None
        self.distance_gate = sdf.get_double('trigger_distance')
        self.range_source = sdf.get_string('range_source')
        self.range_gate = CameraRangeGate(self.observer_name, self.target_name)
        self.node = Node(); self.guard = threading.Lock()
        self.trigger, self.visual_sample = False, None
        self.visual_received = None
        self.enable_received = None
        self.node.subscribe(Boolean, f'/model/{self.name}/fire', self.fire)
        self.node.subscribe(Boolean, f'/model/{self.name}/enable', self.enable)
        self.node.subscribe(Float_V, f'/model/{self.observer_name}/dyt/target', self.visual)
        self.node.subscribe(StringMsg, f'/model/{self.observer_name}/dyt/sim_range', self.range_input)
        self.publisher = self.node.advertise(f'/model/{self.name}/state', StringMsg)
        self.observer, self.target, self.mount = None, None, None
        self.state, self.release_time, self.last_report = 'waiting_for_models', None, -1.
        self.contact_time, self.wrap_time = 0., 0.
        self.last_contact_at = None
        self.anchors, self.broken_time = {}, 0.
        self.contacts, self.tension = 0, 0.
        self.positions = None
        self.target_position = None
        self.target_velocity = None
        self.wrapped_count = 0
        self.error = None
        self.trigger_status = {}
        print(f'[eco_net] {self.name}: Python physics system loaded', flush=True)

    def fire(self, message):
        if message.data:
            with self.guard:
                self.trigger = True

    def enable(self, message):
        with self.guard:
            self.enable_received = time.monotonic() if message.data else None

    def visual(self, packet):
        with self.guard:
            if len(packet.data) == 12 and packet.data[0] == 1 and packet.data[1] == 1:
                self.visual_sample = packet.header.stamp.sec+packet.header.stamp.nsec/1e9
                self.visual_received = time.monotonic()
            else:
                self.visual_sample = None
                self.visual_received = None

    def range_input(self, message):
        try:
            record = json.loads(message.data)
        except (ValueError, TypeError):
            record = None
        with self.guard:
            self.range_gate.observe(record)

    def resolve(self, ecm):
        if self.observer is None or not self.observer.valid(ecm) or not self.target.valid(ecm):
            observer = self.world.model_by_name(ecm, self.observer_name)
            target = self.world.model_by_name(ecm, self.target_name)
            if not observer or not target:
                return False
            self.observer, self.target = Link(Model(observer).canonical_link(ecm)), Link(Model(target).canonical_link(ecm))
            self.observer.enable_velocity_checks(ecm); self.target.enable_velocity_checks(ecm)
            if self.drop_on_capture:
                # Explicit x500-as-litter proxy mode, not a universal prediction
                # that a real net will stop an aircraft's propulsion.
                ids = [Model(target).joint_by_name(ecm, f'rotor_{i}_joint') for i in range(4)]
                if not all(ids):
                    self.state, self.error = 'fault', 'drop-on-capture requires all four x500 rotor joints'
                    return False
                self.drop_rotors = [Joint(identifier) for identifier in ids]
                for joint in self.drop_rotors:
                    joint.enable_velocity_check(ecm)
        mount = self.world.model_by_name(ecm, self.mount_name)
        self.mount = Model(mount) if mount else None
        return True

    def lock_captured_proxy(self, ecm):
        if not self.drop_latched:
            return
        for joint in self.drop_rotors:
            # Simulation-only persistent constraint: PX4 can keep publishing
            # motor commands, but these target rotor joints cannot rotate.
            # Gazebo's native motor model derives thrust from measured rotation.
            # No disarm/MAVLink messages, target pose or body velocity commands.
            joint.set_velocity_limits(ecm, [Vector2d(0, 0)])
            joint.set_velocity(ecm, [0.])

    def pre_update(self, info, ecm):
        if info.paused:
            return
        now, dt = info.sim_time.total_seconds(), info.dt.total_seconds()
        if not 0 < dt <= .0041:
            self.state, self.error = 'fault', 'requires physics step <= 0.004 seconds; reload on world reset'
            return
        if self.state == 'fault' or not self.resolve(ecm):
            return
        op, tp = self.observer.world_pose(ecm), self.target.world_pose(ecm)
        ov, tv = self.observer.world_linear_velocity(ecm), self.target.world_linear_velocity(ecm)
        if any(value is None for value in (op, tp, ov, tv)):
            return
        gravity = array(self.world.gravity(ecm))
        nozzle = op.pos()+op.rot().rotate_vector(vector(self.offset))
        orientation = op.rot()*self.axis_rotation
        mount_pose = Pose3d(nozzle, orientation)
        if self.mount is not None:
            self.mount.set_world_pose_cmd(ecm, mount_pose)
        center, target_velocity = array(tp.pos()), array(tv)
        self.target_position = center
        self.target_velocity = target_velocity
        if self.demo:
            # Demo balloon only: buoyancy balances weight, smooth small gusts
            # move it through physics. Never enabled for a user's x500.
            self.target.add_world_force(ecm, vector(-gravity*.08-target_velocity*.08+
                                                     np.array([0, .012*math.sin(now*2), .004*math.cos(now)])))
        if self.release_time is None:
            self.model.set_world_pose_cmd(ecm, mount_pose)
            for link, mass in zip(self.links, self.masses):
                link.add_world_force(ecm, vector(-mass*gravity))
            self.observer.add_world_force(ecm, vector(self.config.mass*gravity), vector(self.offset))
            direction = array(orientation.rotate_vector(Vector3d(1, 0, 0)))
            delta = center-array(nozzle); distance = np.linalg.norm(delta)
            aligned = distance > .8 and np.dot(delta/max(distance, 1e-9), direction) >= math.cos(math.radians(12))
            with self.guard:
                manual = self.trigger; self.trigger = False
                lock = (self.visual_sample is not None and 0 <= now-self.visual_sample <= .2
                        and self.visual_received is not None and time.monotonic()-self.visual_received <= .2)
                enabled = (self.auto and self.enable_received is not None
                           and time.monotonic()-self.enable_received < .5)
                measured = self.range_gate.measurement(now) if self.range_source == 'camera' else {
                    'valid': True, 'status': 'model_nozzle_distance', 'distance_m': float(distance)}
            self.state = 'armed' if enabled else 'ready'
            # Camera mode uses exactly the aircraft-to-target distance shown
            # above the image ROI. Geometry still checks the actual nozzle axis.
            range_ok = measured['valid'] and measured['distance_m'] <= self.distance_gate
            allowed = aligned and range_ok and (not self.require_lock or lock)
            self.trigger_status = dict(distance_m=measured['distance_m'], nozzle_distance_m=float(distance),
                                       range_source=self.range_source, range_status=measured['status'],
                                       range_fresh=measured['valid'], aligned=bool(aligned),
                                       visual_lock_fresh=bool(lock), auto_enabled=bool(enabled),
                                       ready_to_fire=bool(allowed))
            if manual and not allowed:
                self.error = 'manual shot blocked by range freshness/distance/alignment/visual-lock gate'
            if (manual or enabled) and allowed and now >= .3:
                self.error = None
                self.release_time, self.shot_axis, self.state = now, direction, 'released'
                poses = [link.world_pose(ecm) for link in self.links]
                velocities = [link.world_linear_velocity(ecm) for link in self.links]
                spread = np.array([array(orientation.rotate_vector(vector(v))) for v in self.directions])*self.config.spread_speed
                desired = array(ov)+direction*self.config.launch_speed+spread
                current = np.array([array(v) for v in velocities])
                # Force impulse over 40 ms: no persistent velocity commands
                # that would override subsequent gravity/contact forces.
                self.launch_force = self.masses[:, None]*(desired-current)/.04
                self.recoil = -self.config.mass*direction*self.config.launch_speed/.04
                self.positions = np.array([array(p.pos()) for p in poses])
                for link, force in zip(self.links, self.launch_force):
                    link.add_world_force(ecm, vector(force))
                self.observer.add_world_force(ecm, vector(self.recoil), vector(self.offset))
            return

        positions = np.array([array(link.world_pose(ecm).pos()) for link in self.links])
        velocities = np.array([array(link.world_linear_velocity(ecm)) for link in self.links])
        forces, self.tension = rope_forces(positions, velocities, self.edges, self.rest,
                                          self.config.spring, self.config.damping)
        age = now-self.release_time
        if age < .04:
            # Scale the last integration slice to preserve the prescribed impulse.
            fraction = min(dt, .04-age)/dt
            forces += self.launch_force*fraction
            self.observer.add_world_force(ecm, vector(self.recoil*fraction), vector(self.offset))
        speed = np.linalg.norm(velocities, axis=1)
        forces -= .5*1.225*self.config.drag_area/len(self.links)*velocities*speed[:, None]
        # Ground plane is also an analytical penalty surface for these nodes.
        penetration = np.maximum(0, .015-positions[:, 2])
        forces[:, 2] += np.where(penetration > 0, np.maximum(0, penetration*80-velocities[:, 2]*.3), 0)
        contact, reaction, torque, self.contacts = sphere_contact(
            positions, velocities, self.triangles, center, target_velocity,
            self.config.target_radius, self.config.contact_spring, self.config.contact_damping)
        forces += contact
        self.contact_time = self.contact_time+dt if self.contacts else 0.
        if self.contacts:
            self.last_contact_at = now
        if self.contacts and not self.anchors and self.state not in ('escaped', 'missed'):
            self.state = 'contact'
        elif not self.anchors and self.state == 'contact':
            self.state = 'released'
        wrapped = wrapping_nodes(positions, center, self.config.target_radius, self.shot_axis)
        self.wrapped_count = len(wrapped)
        # Penalty contact can momentarily open while the net surrounds the
        # sphere. Require a recent actual impact AND continuing two-sided
        # geometry; proximity alone and a historical hit cannot latch it.
        recent_impact = self.last_contact_at is not None and now-self.last_contact_at <= .15
        self.wrap_time = self.wrap_time+dt if len(wrapped) and recent_impact else 0.
        confirmed = capture_confirmed(self.contacts, len(wrapped), self.contact_time,
                                      self.wrap_time, self.config.capture_hold)
        if not self.anchors and confirmed and self.state not in ('escaped', 'missed'):
            # Approximate entanglement only after sustained two-sided wrapping.
            # Elastic connections, not fixed joints/teleportation. They can break.
            self.anchors = {int(i): tp.rot().rotate_vector_reverse(vector(positions[i]-center)) for i in wrapped}
            self.state = 'captured'
            self.captured_at, self.capture_height = now, float(center[2])
            self.drop_latched = self.drop_on_capture
        self.lock_captured_proxy(ecm)
        overloaded = False
        angular = self.target.world_angular_velocity(ecm)
        angular = array(angular) if angular is not None else np.zeros(3)
        for i, local in self.anchors.items():
            offset = array(tp.rot().rotate_vector(local))
            relative_speed = velocities[i]-target_velocity-np.cross(angular, offset)
            force = -self.config.attachment_spring*(positions[i]-center-offset)-.08*relative_speed
            overloaded |= np.linalg.norm(force) > self.config.break_force
            forces[i] += force; reaction -= force; torque -= np.cross(offset, force)
        self.broken_time = self.broken_time+dt if overloaded else 0.
        if self.broken_time >= .04:
            self.anchors.clear(); self.state = 'escaped'
        if age > self.config.lifetime and not self.anchors:
            self.state = 'missed' if self.state != 'escaped' else 'escaped'
        if not np.all(np.isfinite(forces)) or np.max(np.linalg.norm(forces, axis=1)) > 1000:
            self.state, self.error = 'fault', 'unstable force; no further net/target forces applied'
            return
        for link, force in zip(self.links, forces):
            link.add_world_force(ecm, vector(force))
        self.target.add_world_wrench(ecm, vector(reaction), vector(torque))
        self.positions = positions

    def post_update(self, info, ecm):
        now = info.sim_time.total_seconds()
        if now-self.last_report < 1/30:
            return
        self.last_report = now
        if self.release_time is None:
            poses = [link.world_pose(ecm) for link in self.links]
            if all(pose is not None for pose in poses):
                self.positions = np.array([array(pose.pos()) for pose in poses])
        record = dict(name=self.name, state=self.state, sim_time_s=now,
                      net_width_m=self.config.width,
                      capture_required_contact_s=self.config.capture_hold,
                      drop_on_capture=self.drop_on_capture, target_passive_proxy=self.drop_latched,
                      captured_at_s=self.captured_at, capture_height_m=self.capture_height,
                      target_falling=(self.target_velocity is not None and self.state in ('captured', 'escaped')
                                      and bool(self.target_velocity[2] < -.2)),
                      descent_since_capture_m=(self.capture_height-float(self.target_position[2])
                                               if self.capture_height is not None else None),
                      released_at_s=self.release_time, contacts=self.contacts,
                      wrap_hold_s=self.wrap_time, attachments=len(self.anchors),
                      wrapping_nodes=self.wrapped_count, contact_hold_s=self.contact_time,
                      max_rope_tension_n=self.tension, error=self.error,
                      trigger=self.trigger_status,
                      node_positions=self.positions.tolist() if self.positions is not None else [],
                      visual_edges=self.visible.tolist(), target=self.target_name,
                      target_position=self.target_position.tolist() if self.target_position is not None else None,
                      target_velocity=self.target_velocity.tolist() if self.target_velocity is not None else None,
                      target_envelope_radius_m=self.config.target_radius)
        self.publisher.publish(StringMsg(data=json.dumps(record, allow_nan=False)))


def get_system():
    return NetSystem()
