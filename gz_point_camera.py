#!/usr/bin/env python3
"""Point the upward tailsitter camera at a Gazebo model; yield to visual tracking.

Only gimbal joints are commanded. This does not activate DYT or move an aircraft.
Uses measured poses, not the PX4 spawn coordinates or MAVLink system ID.
"""

import argparse
import json
import math
import sys
import threading
import time

from gimbal_pose_inspector import (DEFAULT_SDF, PoseStream, compose, inverse,
                                   load_camera, multiply, positive_float, rotate,
                                   scene_index, to_rpy_deg, world_pose)

PITCH_LIMITS = (-2.35619, 0.785398)
ROLL_LIMITS = (-0.785398, 0.785398)


def nearest_yaw(angle, reference):
    return angle + 2*math.pi*round((reference-angle)/(2*math.pi))


def measured_joints(rotations, previous_yaw=None):
    mount, yaw, roll, camera = rotations
    y = math.radians(to_rpy_deg(multiply(inverse(mount), yaw))[2])
    if previous_yaw is not None:
        y = nearest_yaw(y, previous_yaw)
    r = math.radians(to_rpy_deg(multiply(inverse(yaw), roll))[0])
    forward = rotate(multiply(inverse(roll), camera), (1, 0, 0))
    fixed_pitch = math.atan2(-forward[2], forward[0])
    if fixed_pitch > 0:
        fixed_pitch -= 2*math.pi
    return y, r, fixed_pitch+math.pi/2


def pointing_solution(direction_in_mount, current):
    """Solve yaw / zero-roll / pitch including the fixed camera -90 deg pitch.

    Two Euler branches avoid clipping a horizontal target to the +45 deg
    pitch-joint stop. Yaw is continuous even across +/- pi. If unreachable,
    choose the closest optical direction and report the remaining error.
    """
    norm = math.hypot(*direction_in_mount)
    if not math.isfinite(norm) or norm < 1e-6:
        raise ValueError('invalid_target_direction')
    x, y, z = (v/norm for v in direction_in_mount)
    horizontal = math.hypot(x, y)
    if horizontal < 1e-8:
        yaw = current[0]
    else:
        yaw = math.atan2(y, x)
    elevation = math.atan2(z, horizontal)
    candidates = []
    for desired_yaw, pitch in ((yaw, math.pi/2-elevation), (yaw+math.pi, elevation-math.pi/2)):
        desired_yaw = nearest_yaw(desired_yaw, current[0])
        pitch = max(PITCH_LIMITS[0], min(PITCH_LIMITS[1], pitch))
        optical = (math.cos(desired_yaw)*math.sin(pitch),
                   math.sin(desired_yaw)*math.sin(pitch), math.cos(pitch))
        alignment = max(-1.0, min(1.0, optical[0]*x+optical[1]*y+optical[2]*z))
        travel = abs(desired_yaw-current[0])+abs(pitch-current[2])
        candidates.append((alignment, -travel, (desired_yaw, 0.0, pitch)))
    best = max(candidates, key=lambda c: (round(c[0], 10), c[1]))
    return best[2], math.degrees(math.acos(best[0]))


def limit_step(current, desired, dt, max_rate):
    delta = max_rate*max(0.0, min(dt, 0.1))
    result = [a+max(-delta, min(delta, b-a)) for a, b in zip(current, desired)]
    result[1] = max(ROLL_LIMITS[0], min(ROLL_LIMITS[1], result[1]))
    result[2] = max(PITCH_LIMITS[0], min(PITCH_LIMITS[1], result[2]))
    return result


class VisualLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.received = None
        self.sample_ns = None
        self.valid = False

    def receive(self, packet):
        valid = (len(packet.data) == 12 and packet.data[0] == 1 and packet.data[1] == 1
                 and packet.HasField('header') and packet.header.HasField('stamp'))
        with self.lock:
            self.received = time.monotonic()
            self.valid = valid
            self.sample_ns = (packet.header.stamp.sec*1_000_000_000+packet.header.stamp.nsec) if valid else None

    def active(self, pose_stamp, max_age):
        with self.lock:
            return (self.valid and self.received is not None
                    and time.monotonic()-self.received <= max_age
                    and abs(pose_stamp-self.sample_ns) <= max_age*1e9)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', default='default')
    parser.add_argument('--observer', default='fpv_quadtailsitter_gimbal_0', help='camera aircraft Gazebo model')
    parser.add_argument('--target', default='x500_2', help='target Gazebo model, not MAV_SYS_ID')
    parser.add_argument('--rate', type=positive_float, default=30.0)
    parser.add_argument('--max-joint-rate', type=positive_float, default=1.0,
                        help='joint-command slew limit in rad/s; SDF servos also limit actual motion')
    parser.add_argument('--max-age', type=positive_float, default=0.5)
    parser.add_argument('--duration', type=positive_float, help='optional wall-time run limit in seconds')
    args = parser.parse_args(argv)
    if args.observer == args.target:
        parser.error('--observer and --target must be different')
    from gz.transport13 import Node
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.double_pb2 import Double
    from gz.msgs10.empty_pb2 import Empty
    from gz.msgs10.float_v_pb2 import Float_V
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.scene_pb2 import Scene

    node = Node()
    poses, visual = PoseStream(), VisualLock()
    pose_topic = f'/world/{args.world}/pose/info'
    visual_topic = f'/model/{args.observer}/dyt/target'
    subscribed = []
    camera = load_camera(DEFAULT_SDF, 'camera_link', 'camera')
    owner = node.advertise(f'/model/{args.observer}/dyt/gimbal_owner', Boolean)
    publishers = [node.advertise(f'/model/{args.observer}/command/gimbal_{axis}', Double)
                  for axis in ('yaw', 'roll', 'pitch')]
    keys = [args.observer+'::'+name for name in
            ('gimbal_mount_link', 'gimbal_yaw_link', 'gimbal_roll_link', 'camera_link')]
    parents, names = {}, {}
    refresh = 0.0
    previous_stamp, previous_yaw, previous_command = None, None, None
    start, last_log, last_state = time.monotonic(), 0.0, None
    try:
        for message_type, topic, callback in ((Pose_V, pose_topic, poses.receive),
                                              (Float_V, visual_topic, visual.receive)):
            if not node.subscribe(message_type, topic, callback):
                raise ValueError('subscription_failed:'+topic)
            subscribed.append(topic)
        while args.duration is None or time.monotonic()-start < args.duration:
            tick = time.monotonic()
            record = {'observer': args.observer, 'target': args.target}
            try:
                if tick >= refresh:
                    refresh = tick+1.0
                    ok, scene = node.request(f'/world/{args.world}/scene/info', Empty(), Empty, Scene, 500)
                    if not ok:
                        raise ValueError('waiting_for_scene')
                    parents, names = scene_index(scene)
                missing = [key for key in [*keys, args.observer, args.target] if key not in names]
                if missing:
                    raise ValueError('waiting_for_models_or_links:'+','.join(missing))
                frame, age, _, _ = poses.snapshot(args.max_age)
                stamp = frame.header.stamp.sec*1_000_000_000+frame.header.stamp.nsec
                if visual.active(stamp, args.max_age):
                    state = 'yielding_to_visual_tracking'
                    # Do not clear the shared ownership heartbeat: the visual
                    # controller is now publishing its own heartbeat and commands.
                    previous_stamp, previous_yaw, previous_command = None, None, None
                else:
                    state = 'pointing_at_target'
                    if previous_stamp is None or stamp != previous_stamp:
                        by_id = {pose.id: pose for pose in frame.pose}
                        if len(by_id) != len(frame.pose):
                            raise ValueError('duplicate_pose_entity')
                        transforms = [world_pose(names[key], parents, by_id) for key in keys]
                        target = world_pose(names[args.target], parents, by_id)
                        sensor = compose(transforms[-1], camera.sensor_pose)
                        direction = rotate(inverse(transforms[0][1]),
                                           tuple(t-c for t, c in zip(target[0], sensor[0])))
                        current = measured_joints([transform[1] for transform in transforms], previous_yaw)
                        desired, residual = pointing_solution(direction, current)
                        dt = (stamp-previous_stamp)/1e9 if previous_stamp is not None and stamp > previous_stamp else 1/args.rate
                        # Slew the commanded position, not measured position.
                        # Stepping only from the measured angle every frame would
                        # repeatedly shrink the servo's error and make it crawl.
                        commands = limit_step(previous_command or current, desired, dt, args.max_joint_rate)
                        if not owner.publish(Boolean(data=True)):
                            raise ValueError('owner_publish_failed')
                        for publisher, angle in zip(publishers, commands):
                            if not publisher.publish(Double(data=angle)):
                                raise ValueError('joint_publish_failed')
                        previous_stamp, previous_yaw = stamp, current[0]
                        previous_command = commands
                        record.update(distance_m=math.hypot(*direction),
                                      command_yaw_roll_pitch_deg=[math.degrees(v) for v in commands],
                                      mechanical_residual_deg=residual, pose_age_wall_s=age)
                        if residual > 1.0:
                            state = 'target_outside_gimbal_reach'
            except ValueError as error:
                state = str(error)
                previous_stamp, previous_yaw, previous_command = None, None, None
                # Stop heartbeats and commands on stale/missing poses. Existing
                # GZGimbal ownership expires automatically after 500 ms.
            if state != last_state or tick-last_log >= 1.0:
                record['state'] = state
                print(json.dumps(record, allow_nan=False), flush=True)
                last_state, last_log = state, tick
            time.sleep(max(0.0, 1/args.rate-(time.monotonic()-tick)))
        return 0
    except (ValueError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for topic in subscribed:
            node.unsubscribe(topic)
        # Do not publish False on a shared topic: it could revoke the visual
        # controller's lease. Our own lease expires within 500 ms instead.


if __name__ == '__main__':
    sys.exit(main())
