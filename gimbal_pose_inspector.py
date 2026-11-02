#!/usr/bin/env python3
"""Read-only Gazebo camera geometry inspector. No PX4 or actuator writes."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import threading
import time
import xml.etree.ElementTree as ET


IDENTITY = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
DEFAULT_SDF = Path(__file__).parent / "gz/models/fpv_quadtailsitter_gimbal/model.sdf"


def multiply(a, b):
    """Quaternion product, wxyz order; rotations map child into parent."""
    w, x, y, z = a
    v, i, j, k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def inverse(q):
    return (q[0], -q[1], -q[2], -q[3])


def rotate(q, vector):
    return multiply(multiply(q, (0.0, *vector)), inverse(q))[1:]


def compose(parent, child):
    p, q = parent
    v, r = child
    offset = rotate(q, v)
    return tuple(a+b for a, b in zip(p, offset)), multiply(q, r)


def from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)


def to_rpy_deg(q):
    w, x, y, z = q
    return [math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
            math.degrees(math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))),
            math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))]


def read_pose(msg):
    p, q = msg.position, msg.orientation
    position, quaternion = (p.x, p.y, p.z), (q.w, q.x, q.y, q.z)
    if not all(math.isfinite(v) for v in (*position, *quaternion)):
        raise ValueError("nonfinite_pose")
    norm = math.sqrt(sum(v*v for v in quaternion))
    if abs(norm-1.0) > 0.001:
        raise ValueError("invalid_quaternion")
    return position, tuple(v/norm for v in quaternion)


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    hfov: float
    near: float
    far: float
    sensor_pose: tuple

    @property
    def focal(self):
        return self.width / (2*math.tan(self.hfov/2))


def load_camera(path, link_name, sensor_name):
    sensor = ET.parse(path).find(f"./model/link[@name='{link_name}']/sensor[@name='{sensor_name}']")
    if sensor is None or sensor.get("type") != "camera":
        raise ValueError("camera sensor not found in SDF (direct model/link required)")
    # Do not silently accept optics / pose conventions we have not implemented.
    optics = sensor.find("camera")
    if optics is None:
        raise ValueError("camera optics missing in SDF")
    if any(optics.find(tag) is not None for tag in ("lens", "distortion", "intrinsics", "projection")):
        raise ValueError("only the ideal, undistorted SDF camera is supported")
    pose = sensor.find("pose")
    if pose is not None and (pose.get("relative_to") or pose.get("degrees", "false") != "false"
                             or pose.get("rotation_format", "euler_rpy") != "euler_rpy"):
        raise ValueError("sensor pose must be radians/euler_rpy relative to its link")
    values = [float(v) for v in pose.text.split()] if pose is not None else [0.0]*6
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid sensor pose")
    camera = Camera(int(optics.findtext("image/width")), int(optics.findtext("image/height")),
                    float(optics.findtext("horizontal_fov")), float(optics.findtext("clip/near")),
                    float(optics.findtext("clip/far")), (tuple(values[:3]), from_rpy(*values[3:])))
    if (camera.width <= 0 or camera.height <= 0
            or not all(math.isfinite(v) for v in (camera.hfov, camera.near, camera.far))
            or not 0 < camera.hfov < math.pi or not 0 < camera.near < camera.far):
        raise ValueError("invalid camera optics")
    return camera


def scene_index(scene):
    """Resolve duplicate link names by model hierarchy, then use entity IDs."""
    entities = {}
    names = {}

    def add(entity, parent, scope):
        name = "::".join((*scope, entity.name))
        if entity.id in entities or name in names:
            raise ValueError("ambiguous_scene_entity")
        entities[entity.id] = parent
        names[name] = entity.id
        return name

    def model_tree(model, parent=None, scope=()):
        name = add(model, parent, scope)
        for link in model.link:
            add(link, model.id, tuple(name.split("::")))
        for child in model.model:
            model_tree(child, model.id, tuple(name.split("::")))

    for model in scene.model:
        model_tree(model)
    return entities, names


def world_pose(entity_id, parents, poses):
    chain = []
    seen = set()
    while entity_id is not None:
        if entity_id in seen:
            raise ValueError("scene_cycle")
        seen.add(entity_id)
        if entity_id not in poses or entity_id not in parents:
            raise ValueError("missing_pose_entity_" + str(entity_id))
        chain.append(read_pose(poses[entity_id]))
        entity_id = parents[entity_id]
    result = IDENTITY
    for pose in reversed(chain):
        result = compose(result, pose)
    return result


def inspect_geometry(observer, target, camera_link, camera):
    sensor = compose(camera_link, camera.sensor_pose)
    delta = tuple(t-c for t, c in zip(target[0], sensor[0]))
    x, y, z = rotate(inverse(sensor[1]), delta)  # Gazebo camera: forward/left/up.
    distance = math.hypot(*delta)
    if not all(math.isfinite(v) for v in (x, y, z, distance)):
        raise ValueError("nonfinite_geometry")
    if distance <= 1e-9:
        raise ValueError("coincident_target_and_camera")
    pixel = None
    in_frustum = False
    if x > 0:
        # Pixel-center coordinates: top-left pixel is (0, 0).
        u = (camera.width-1)/2-camera.focal*y/x
        v = (camera.height-1)/2-camera.focal*z/x
        if not all(math.isfinite(a) for a in (u, v)):
            raise ValueError("nonfinite_projection")
        pixel = [u, v]
        in_frustum = (camera.near <= x <= camera.far
                      and -0.5 <= u < camera.width-0.5 and -0.5 <= v < camera.height-0.5)
    relative_rotation = multiply(inverse(observer[1]), sensor[1])
    return {
        "distance_m": distance,
        "target_in_camera_flu_m": [x, y, z],
        "azimuth_right_deg": math.degrees(math.atan2(-y, x)),
        "elevation_up_deg": math.degrees(math.atan2(z, math.hypot(x, y))),
        "target_origin_pixel_uv": pixel,
        "target_origin_in_frustum": in_frustum,
        "camera_in_observer_rpy_flu_deg": to_rpy_deg(relative_rotation),
        "camera_world_position_m": list(sensor[0]),
        "camera_world_quaternion_wxyz": list(sensor[1]),
    }


class PoseStream:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.received = None
        self.advanced = None
        self.stamp = None
        self.sequence = 0
        self.epoch = 0

    def receive(self, frame):
        stamp = frame.header.stamp.sec*1_000_000_000+frame.header.stamp.nsec
        now = time.monotonic()
        with self.lock:
            if self.stamp is None or stamp != self.stamp:
                if self.stamp is not None and stamp < self.stamp:
                    self.epoch += 1
                self.advanced = now
            self.frame, self.received, self.stamp = frame, now, stamp
            self.sequence += 1

    def snapshot(self, max_age):
        with self.lock:
            if self.frame is None:
                raise ValueError("waiting_for_pose")
            if (not self.frame.HasField("header") or not self.frame.header.HasField("stamp")
                    or self.frame.header.stamp.sec < 0
                    or not 0 <= self.frame.header.stamp.nsec < 1_000_000_000):
                raise ValueError("invalid_sim_timestamp")
            age = time.monotonic()-self.received
            if age > max_age:
                raise ValueError("stale_pose")
            if time.monotonic()-self.advanced > max_age:
                raise ValueError("simulation_time_not_advancing")
            return self.frame, age, self.sequence, self.epoch


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="default")
    parser.add_argument("--observer", help="exact Gazebo model name, including instance suffix")
    parser.add_argument("--target", help="exact Gazebo model name; target origin is the observed point")
    parser.add_argument("--camera-link", default="camera_link")
    parser.add_argument("--camera-sensor", default="camera")
    parser.add_argument("--camera-sdf", type=Path, default=DEFAULT_SDF)
    parser.add_argument("--list", action="store_true", help="list models and links, then exit")
    parser.add_argument("--output", type=Path, help="new JSONL file; existing files are never overwritten")
    parser.add_argument("--rate", type=positive_float, default=10.0, help="log rate in wall Hz (default 10)")
    parser.add_argument("--max-age", type=positive_float, default=0.5, help="wall freshness limit in seconds")
    parser.add_argument("--duration", type=positive_float, help="stop after this many wall seconds")
    args = parser.parse_args()
    if not args.list and (not args.observer or not args.target or args.observer == args.target):
        parser.error("--observer and --target must be specified and different")
    try:
        from gz.transport13 import Node
        from gz.msgs10.empty_pb2 import Empty
        from gz.msgs10.scene_pb2 import Scene
        from gz.msgs10.pose_v_pb2 import Pose_V
    except ImportError as error:
        parser.error(f"Gazebo Transport 13 / Msgs 10 bindings required; use /usr/bin/python3: {error}")
    node = Node()
    service = f"/world/{args.world}/scene/info"

    def fetch_scene():
        ok, scene = node.request(service, Empty(), Empty, Scene, 500)
        if not ok:
            raise ValueError("scene_service_unavailable")
        return scene_index(scene)

    output = None
    try:
        if args.list:
            _, names = fetch_scene()
            print("\n".join(sorted(names)))
            return 0
        camera = load_camera(args.camera_sdf, args.camera_link, args.camera_sensor)
        stream = PoseStream()
        topic = f"/world/{args.world}/pose/info"
        if not node.subscribe(Pose_V, topic, stream.receive):
            raise ValueError("pose_subscription_failed")
        output = args.output.open("x", encoding="utf-8") if args.output else sys.stdout

        def emit(record):
            output.write(json.dumps(record, allow_nan=False, ensure_ascii=False)+"\n")
            output.flush()

        emit({"type": "metadata", "schema_version": 1, "read_only": True,
              "observer": args.observer, "target": args.target, "pose_topic": topic,
              "camera_sdf": str(args.camera_sdf.resolve()), "image_size": [camera.width, camera.height],
              "horizontal_fov_rad": camera.hfov, "clip_m": [camera.near, camera.far],
              "focal_length_px": camera.focal, "sensor_pose_in_link": camera.sensor_pose,
              "projection": "ideal_pinhole_square_pixels_pixel_centers",
              "observed_point": "target_model_origin", "occlusion_checked": False})
        parents, names = {}, {}
        refresh_at = 0.0
        start = time.monotonic()
        last_reason = None
        while args.duration is None or time.monotonic()-start < args.duration:
            tick = time.monotonic()
            record = {"type": "sample", "wall_elapsed_s": tick-start, "valid": False}
            try:
                if tick >= refresh_at:
                    # Clear stale scene bindings when a refresh fails or the world resets.
                    parents, names = {}, {}
                    refresh_at = tick+2.0
                    parents, names = fetch_scene()
                selected = [args.observer, args.target, args.observer+"::"+args.camera_link]
                for name in selected:
                    if name not in names:
                        raise ValueError("missing_scene_entity:"+name)
                frame, age, sequence, epoch = stream.snapshot(args.max_age)
                if len({p.id for p in frame.pose}) != len(frame.pose):
                    raise ValueError("duplicate_pose_id")
                poses = {p.id: p for p in frame.pose}
                # Every transform comes from the same stamped Pose_V; no cached pose mixing.
                transforms = [world_pose(names[name], parents, poses) for name in selected]
                record.update(inspect_geometry(*transforms, camera))
                record.update(valid=True, sim_time_s=frame.header.stamp.sec+frame.header.stamp.nsec*1e-9,
                              pose_age_wall_s=age, source_sequence=sequence, sim_epoch=epoch)
            except ValueError as error:
                record["reason"] = str(error)
            emit(record)
            reason = record.get("reason", "receiving_valid_geometry")
            if reason != last_reason:
                print(reason, file=sys.stderr)
                last_reason = reason
            time.sleep(max(0.0, 1/args.rate-(time.monotonic()-tick)))
        return 0
    except (ValueError, OSError, ET.ParseError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        if output is not None and output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    sys.exit(main())
