"""Opt-in SITL camera tracking telemetry and fpv_quadtailsitter_gimbal following."""

import math
import time
from collections import deque

import numpy as np
from gz.msgs10.double_pb2 import Double
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.float_v_pb2 import Float_V
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.scene_pb2 import Scene

from gimbal_pose_inspector import (DEFAULT_SDF, PoseStream, compose, inverse,
                                   from_rpy, load_camera, multiply, rotate, scene_index,
                                   to_rpy_deg, world_pose)


class CameraPoseStream(PoseStream):
    """Match camera orientation to image exposure instead of using the latest pose."""
    def __init__(self):
        super().__init__()
        self.history = deque(maxlen=120)

    def receive(self, frame):
        stamp = frame.header.stamp.sec*1_000_000_000+frame.header.stamp.nsec
        with self.lock:
            if self.stamp is not None and stamp < self.stamp:
                self.history.clear()
            self.history.append((stamp, frame))
        super().receive(frame)

    def at_image(self, stamp):
        self.snapshot(0.25)
        with self.lock:
            latest = self.stamp
            nearest_stamp, frame = min(self.history, key=lambda entry: abs(entry[0]-stamp))
        if latest-stamp > 250_000_000:
            raise ValueError("dyt_image_exceeds_max_age_0.25s")
        if abs(nearest_stamp-stamp) > 20_000_000:
            raise ValueError("dyt_image_pose_time_mismatch")
        return frame


class TrackingLossGrace:
    """Delay loss notifications, never replay or retimestamp a valid observation.

    Both exposure time and wall time bound the grace period. The 0.2 s maximum
    stays below the current PX4 DYTG_MAXAGE default of 0.25 s; PX4's own freshness
    check remains authoritative if the user configures a shorter timeout.
    """

    def __init__(self, seconds=0.2):
        if not math.isfinite(seconds) or not 0 <= seconds <= 0.2:
            raise ValueError('sim_loss_grace_must_be_between_0_and_0.2s')
        self.seconds = seconds
        self.reset()

    def reset(self):
        self.last_good_stamp = None
        self.last_good_wall = None
        self.last_seen_stamp = None

    def observe(self, stamp):
        if self.last_seen_stamp is not None and stamp < self.last_seen_stamp:
            self.reset()
        self.last_seen_stamp = stamp

    def accept(self, stamp):
        self.observe(stamp)
        self.last_good_stamp = stamp
        self.last_good_wall = time.monotonic()

    def holding(self):
        if self.last_good_stamp is None or self.seconds == 0:
            return False
        age_sim = (self.last_seen_stamp-self.last_good_stamp)/1e9
        age_wall = time.monotonic()-self.last_good_wall
        return 0 <= age_sim < self.seconds and 0 <= age_wall < self.seconds

    def status(self):
        return {'measurement_published': False, 'loss_grace_active': self.holding(),
                'loss_age_sim_s': ((self.last_seen_stamp-self.last_good_stamp)/1e9
                                   if self.last_good_stamp is not None else None)}


def camera_rpy_frd(relative):
    """Equivalent camera Euler angles with yaw in [-pi/2, pi/2].

    The upward camera crosses the Euler pitch singularity. Standard extraction
    can report yaw near pi even with a centred target, beyond DYT's yaw check.
    Use the equivalent Euler branch, preserving the full measured rotation.
    """
    w, x, qy, qz = relative
    frd = (w, x, -qy, -qz)  # FLU -> FRD on both sides
    axis_x, axis_y, axis_z = [rotate(frd, axis) for axis in ((1,0,0), (0,1,0), (0,0,1))]
    horizontal = math.hypot(axis_x[0], axis_x[1])
    p = math.atan2(-axis_x[2], horizontal)
    if horizontal < 1e-8:
        y = 0.0
        r = math.atan2(-axis_z[1], axis_y[1])
    else:
        r = math.atan2(axis_y[2], axis_z[2])
        y = math.atan2(axis_x[1], axis_x[0])
    if abs(y) > math.pi/2:
        wrap = lambda angle: (angle+math.pi) % (2*math.pi)-math.pi
        r, p, y = wrap(r+math.pi), wrap(math.pi-p), wrap(y+math.pi)
    return r, p, y


def tracking_measurement(target, width, height, hfov):
    """Pinhole angles, positive image right/down, matching DYT FRD LOS."""
    if not target["valid"]:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy = target["center_px"]
    dx, dy = cx-(width-1)/2, cy-(height-1)/2
    focal = width/(2*math.tan(hfov/2))
    _, _, bw, bh = target["bbox_xywh"]
    return [math.atan2(dx, focal), math.atan2(dy, focal), bw, bh, dx, dy]


def transport_geometry(relative, los_x, los_y, virtual_frame=True):
    """Express the *same measured body ray* in a PX4-compatible SITL frame.

    Near vertical, the yaw-bounded Euler branch has pitch > 90 degrees,
    rejected by GZBridge. Use a virtual image frame with pitch bounded at
    85 degrees and reproject the real image ray into it. This is NOT angle
    clipping alone: R_virtual * ray_virtual == R_camera * ray_camera.
    Physical gimbal control must still use the original image LOS. Exposure
    time and target validity are unchanged; no prediction/replay is involved.
    Requires the existing default DYT geometry signs (+1) and offsets (0).
    """
    actual = camera_rpy_frd(relative)
    r, p, y = actual
    if virtual_frame:
        p = float(np.clip(p, -math.radians(85), math.radians(85)))
    elif abs(p) > math.pi/2+1e-7:
        raise ValueError('dyt_camera_pitch_outside_px4_packet_limits')
    changed = p != actual[1]
    if changed:
        # FRD conversion on both sides, using the measured quaternion directly.
        w, x, qy, qz = relative
        ray_body = rotate((w, x, -qy, -qz), (1, math.tan(los_x), math.tan(los_y)))
        ray_virtual = rotate(inverse(from_rpy(r, p, y)), ray_body)
        if ray_virtual[0] <= 0:
            raise ValueError('dyt_target_behind_virtual_frame')
        los_x = math.atan2(ray_virtual[1], ray_virtual[0])
        los_y = math.atan2(ray_virtual[2], ray_virtual[0])
    # Keep the real PX4 safety/geometry gates; do not disguise out-of-view data.
    if max(abs(los_x), abs(los_y)) > 0.45:
        raise ValueError('dyt_los_outside_px4_geometry_limits')
    return (r, p, y), (los_x, los_y), {
        'virtual_frame_active': changed,
        'camera_frame_rpy_deg': [math.degrees(v) for v in actual],
        'transport_frame_rpy_deg': [math.degrees(v) for v in (r, p, y)],
        'transport_los_rad': [los_x, los_y],
    }


def check_target_texture(image, target):
    if target["valid"]:
        x, y, width, height = target["bbox_xywh"]
        patch = image[y:y+height, x:x+width]
        # A constant sky/background template matches anywhere and cannot guide a gimbal.
        if max(np.std(patch, axis=(0, 1))) < 3:
            raise ValueError("dyt_target_has_no_spatial_texture")


def simulation_distance(bridge, message, target, target_model='x500_2'):
    """Read-only ground-truth ranging, not a laser raycast or DYT measurement.

    Match pose to exposure and check projected model origin near the selected
    box. This geometric association is not semantic aircraft identification.
    No old result is retained on loss, stale poses, or a different image ROI.
    """
    result = {'source': 'gazebo_ground_truth', 'target_model': target_model,
              'valid': False, 'distance_m': None, 'camera_distance_m': None}
    if not target.get('valid'):
        return {**result, 'status': 'not_locked'}
    try:
        if target_model == bridge.model:
            raise ValueError('range_target_is_observer')
        if target_model not in bridge.names:
            raise ValueError('range_target_model_unavailable_restart_reader_after_spawn')
        stamp = message.header.stamp.sec*1_000_000_000+message.header.stamp.nsec
        if hasattr(bridge.stream, 'at_image'):
            frame = bridge.stream.at_image(stamp)
        else:
            frame, _, _, _ = bridge.stream.snapshot(.25)
        pose_stamp = frame.header.stamp.sec*1_000_000_000+frame.header.stamp.nsec
        if abs(stamp-pose_stamp) > 20_000_000:
            raise ValueError('range_image_pose_time_mismatch')
        poses = {pose.id: pose for pose in frame.pose}
        observer = world_pose(bridge.names[bridge.model], bridge.parents, poses)
        target_pose = world_pose(bridge.names[target_model], bridge.parents, poses)
        camera = compose(world_pose(bridge.names[bridge.keys[-1]], bridge.parents, poses), bridge.camera.sensor_pose)
        relative = rotate(inverse(camera[1]), tuple(t-c for t, c in zip(target_pose[0], camera[0])))
        if relative[0] <= bridge.camera.near:
            raise ValueError('range_target_behind_camera')
        focal = message.width/(2*math.tan(bridge.camera.hfov/2))
        u, v = (message.width-1)/2-focal*relative[1]/relative[0], (message.height-1)/2-focal*relative[2]/relative[0]
        x, y, width, height = target['bbox_xywh']
        # The x500 origin may lie below the visible fuselage; allow a bounded
        # box-sized margin, not arbitrary association with any selected object.
        margin = max(12, min(48, max(width, height)*.5))
        if not (0 <= u < message.width and 0 <= v < message.height
                and x-margin <= u <= x+width+margin and y-margin <= v <= y+height+margin):
            raise ValueError('range_target_not_associated_with_locked_roi')
        distance = math.dist(observer[0], target_pose[0])
        camera_distance = math.dist(camera[0], target_pose[0])
        if not math.isfinite(distance) or not math.isfinite(camera_distance):
            raise ValueError('range_nonfinite_distance')
        return {**result, 'valid': True, 'status': 'measured', 'distance_m': distance,
                'camera_distance_m': camera_distance, 'stamp_ns': stamp,
                'pose_skew_sim_s': abs(stamp-pose_stamp)/1e9,
                'projected_target_center_px': [u, v]}
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        return {**result, 'status': str(error)}


def follow_step(quaternions, los_x, los_y, dt, previous_yaw=None, max_rate=0.5,
                gain=5.0):
    """Damped inverse Jacobian: yaw Z, roll X, pitch Y in their parent frames.

    Use measured link rotations, including the camera's fixed -pi/2 pitch.
    At the upward-looking yaw singularity, roll and pitch still center the image.
    Limit joint rates (default 0.5 rad/s) and SDF roll/pitch mechanical travel.
    """
    mount, yaw, roll, camera = quaternions
    forward = np.array(rotate(camera, (1, 0, 0)))
    desired = np.array(rotate(camera, (1, -math.tan(los_x), -math.tan(los_y))))
    desired /= np.linalg.norm(desired)
    axes = [rotate(mount, (0, 0, 1)), rotate(yaw, (1, 0, 0)), rotate(roll, (0, 1, 0))]
    jacobian = np.column_stack([np.cross(axis, forward) for axis in axes])
    # Do not rotate yaw around an almost vertical optical axis to chase pixel noise.
    weights = np.diag([max(0.02, 1-forward[2]**2), 1.0, 1.0])
    delta = weights @ jacobian.T @ np.linalg.solve(jacobian @ weights @ jacobian.T+0.01*np.eye(3), desired-forward)
    delta *= min(1.0, gain*dt)  # proportional velocity, independent of processing frequency
    if math.hypot(los_x, los_y) < 0.005:  # about two camera pixels
        delta[:] = 0
    delta = np.clip(delta, -max_rate*dt, max_rate*dt)
    # Fixed camera pitch is -pi/2. Remove it before extracting the pitch joint.
    yaw_angle = math.radians(to_rpy_deg(multiply(inverse(mount), yaw))[2])
    if previous_yaw is not None:
        yaw_angle += 2*math.pi*round((previous_yaw-yaw_angle)/(2*math.pi))
    roll_angle = math.radians(to_rpy_deg(multiply(inverse(yaw), roll))[0])
    relative_camera = multiply(inverse(roll), camera)
    axis_x = rotate(relative_camera, (1, 0, 0))
    fixed_pitch = math.atan2(-axis_x[2], axis_x[0])
    if fixed_pitch > 0:
        fixed_pitch -= 2*math.pi
    pitch_angle = fixed_pitch+math.pi/2
    return [yaw_angle+delta[0], float(np.clip(roll_angle+delta[1], -0.785398, 0.785398)),
            float(np.clip(pitch_angle+delta[2], -2.35619, 0.785398))]


class DytBridge:
    def __init__(self, node, model, world, image_topic, follow=False, loss_grace=0.2,
                 virtual_frame=True, gimbal_rate=1.0, gimbal_gain=10.0):
        # This trial supports the existing upward-facing tailsitter rig only.
        expected = f"/world/{world}/model/{model}/link/camera_link/sensor/camera/image"
        if image_topic != expected:
            raise ValueError("dyt_image_topic_does_not_match_model_camera:"+expected)
        self.camera = load_camera(DEFAULT_SDF, "camera_link", "camera")
        self.node, self.model, self.follow = node, model, follow
        self.virtual_frame = virtual_frame
        if not math.isfinite(gimbal_rate) or not 0 < gimbal_rate <= 2:
            raise ValueError('sim_gimbal_rate_must_be_between_0_and_2_rad_s')
        self.gimbal_rate = gimbal_rate
        if not math.isfinite(gimbal_gain) or not 0 < gimbal_gain <= 20:
            raise ValueError('sim_gimbal_gain_must_be_between_0_and_20_per_s')
        self.gimbal_gain = gimbal_gain
        self.publisher = node.advertise(f"/model/{model}/dyt/target", Float_V)
        ok, scene = node.request(f"/world/{world}/scene/info", Empty(), Empty, Scene, 500)
        if not ok:
            raise ValueError("dyt_scene_service_unavailable")
        self.parents, self.names = scene_index(scene)
        self.keys = [model, model+"::gimbal_mount_link", model+"::gimbal_yaw_link",
                     model+"::gimbal_roll_link", model+"::camera_link"]
        if any(key not in self.names for key in self.keys):
            raise ValueError("dyt_requires_fpv_quadtailsitter_gimbal_links")
        self.stream = CameraPoseStream()
        self.pose_topic = f"/world/{world}/pose/info"
        if not node.subscribe(Pose_V, self.pose_topic, self.stream.receive):
            raise ValueError("dyt_pose_subscription_failed")
        self.commands = [node.advertise(f"/model/{model}/command/gimbal_{axis}", Double)
                         for axis in ("yaw", "roll", "pitch")] if follow else []
        self.owner = node.advertise(f"/model/{model}/dyt/gimbal_owner", Boolean) if follow else None
        self.last_stamp = None
        self.last_yaw_command = None
        self.loss_gate = TrackingLossGrace(loss_grace)

    def invalidate(self, stamp_ns=None, force=False):
        # Objects constructed directly by older tests / callers retain the
        # previous immediate-invalidation behavior unless they opt into grace.
        if not hasattr(self, 'loss_gate'):
            self.loss_gate = TrackingLossGrace(0)
        if stamp_ns is not None:
            self.loss_gate.observe(stamp_ns)
            self.last_stamp = stamp_ns
        if force:
            self.loss_gate.reset()
        elif self.loss_gate.holding():
            return self.loss_gate.status()
        if self.last_stamp is not None:
            packet = Float_V(data=[1]+[0]*11)
            packet.header.stamp.sec, packet.header.stamp.nsec = divmod(self.last_stamp, 1_000_000_000)
            self.publisher.publish(packet)
        return {'measurement_published': self.last_stamp is not None, 'loss_grace_active': False}

    def update(self, message, target):
        stamp = message.header.stamp.sec*1_000_000_000+message.header.stamp.nsec
        if stamp == self.last_stamp:
            return
        sample_dt = max(0.0, (stamp-self.last_stamp)/1e9) if self.last_stamp is not None else 0.0
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.last_yaw_command = None
        dt = min(0.1, sample_dt)
        self.last_stamp = stamp
        if not hasattr(self, 'loss_gate'):
            self.loss_gate = TrackingLossGrace(0)
        self.loss_gate.observe(stamp)
        if (message.width, message.height) != (self.camera.width, self.camera.height):
            raise ValueError("dyt_image_size_does_not_match_camera_sdf")
        if getattr(self, "owner", None):
            self.owner.publish(Boolean(data=True))
        if sample_dt > 0.12:
            raise ValueError("dyt_frame_gap_exceeds_0.12s")
        if not target['valid']:
            return self.invalidate()
        if hasattr(self.stream, "at_image"):
            frame = self.stream.at_image(stamp)
        else:
            frame, _, _, _ = self.stream.snapshot(0.25)
        pose_stamp = frame.header.stamp.sec*1_000_000_000+frame.header.stamp.nsec
        # Reject old/future pose feedback instead of mixing unrelated camera orientations.
        if abs(stamp-pose_stamp) > 20_000_000:
            raise ValueError("dyt_image_pose_time_mismatch")
        poses = {pose.id: pose for pose in frame.pose}
        transforms = [world_pose(self.names[key], self.parents, poses) for key in self.keys]
        sensor = compose(transforms[-1], self.camera.sensor_pose)
        relative = multiply(inverse(transforms[0][1]), sensor[1])
        lx, ly, bw, bh, dx, dy = tracking_measurement(target, message.width, message.height, self.camera.hfov)
        # Center the physical camera using actual pixels, even when a valid
        # image is currently outside the telemetry geometry gate.
        if self.follow and dt > 0 and math.hypot(lx, ly) >= 0.005:
            angles = follow_step([transform[1] for transform in transforms[1:]], lx, ly, dt,
                                 self.last_yaw_command, max_rate=getattr(self, 'gimbal_rate', 1.0),
                                 gain=getattr(self, 'gimbal_gain', 10.0))
            self.last_yaw_command = angles[0]
            for publisher, angle in zip(self.commands, angles):
                if not publisher.publish(Double(data=angle)):
                    raise ValueError("dyt_gimbal_command_failed")
        (r, p, y), (tx, ty), geometry = transport_geometry(
            relative, lx, ly, getattr(self, 'virtual_frame', True))
        focal = message.width/(2*math.tan(self.camera.hfov/2))
        # Miss fields describe the transport frame too; real pixels remain in
        # target.center_px / target.bbox_xywh in the reader's JSON output.
        transport_dx, transport_dy = focal*math.tan(tx), focal*math.tan(ty)
        packet = Float_V()
        # Gazebo's Python modules can expose distinct Time classes per message.
        packet.header.stamp.sec = message.header.stamp.sec
        packet.header.stamp.nsec = message.header.stamp.nsec
        packet.data.extend([1, int(target["valid"]), tx, ty, r, p, y, bw, bh,
                            transport_dx, transport_dy, sample_dt])
        if not self.publisher.publish(packet):
            raise ValueError("dyt_publish_failed")
        self.loss_gate.accept(stamp)
        return {"frame_dt_sim_s": sample_dt,
                "pose_skew_sim_s": abs(stamp-pose_stamp)/1e9,
                "image_age_sim_s": max(0.0, (self.stream.stamp-stamp)/1e9),
                "measurement_published": True, "loss_grace_active": False,
                **geometry}

    def close(self):
        self.invalidate(force=True)
        if self.owner:
            self.owner.publish(Boolean(data=False))
        self.node.unsubscribe(self.pose_topic)
