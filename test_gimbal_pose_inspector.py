#!/usr/bin/env python3
"""Geometry and data-integrity checks, independent of a running simulator."""

import math
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.scene_pb2 import Scene

import gimbal_pose_inspector as inspector


def pose(entity_id, position=(0, 0, 0), quaternion=(1, 0, 0, 0)):
    result = Pose(id=entity_id)
    result.position.x, result.position.y, result.position.z = position
    result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z = quaternion
    return result


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.camera = inspector.Camera(1280, 720, 2.0, 0.05, 15000, inspector.IDENTITY)

    def inspect(self, target):
        return inspector.inspect_geometry(inspector.IDENTITY, (target, inspector.IDENTITY[1]),
                                          inspector.IDENTITY, self.camera)

    def test_projection_center_right_up_and_rear(self):
        center = self.inspect((10, 0, 0))
        self.assertEqual(center["target_origin_pixel_uv"], [639.5, 359.5])
        self.assertTrue(center["target_origin_in_frustum"])
        right = self.inspect((10, -1, 0))
        self.assertGreater(right["target_origin_pixel_uv"][0], 639.5)
        self.assertAlmostEqual(right["azimuth_right_deg"], math.degrees(math.atan(0.1)))
        up = self.inspect((10, 0, 1))
        self.assertLess(up["target_origin_pixel_uv"][1], 359.5)
        self.assertGreater(up["elevation_up_deg"], 0)
        rear = self.inspect((-10, 0, 0))
        self.assertIsNone(rear["target_origin_pixel_uv"])
        self.assertFalse(rear["target_origin_in_frustum"])

    def test_frustum_is_geometry_not_visibility(self):
        for point in ((0.01, 0, 0), (15001, 0, 0), (10, 100, 0), (10, 0, 100)):
            self.assertFalse(self.inspect(point)["target_origin_in_frustum"])
        with self.assertRaisesRegex(ValueError, "coincident"):
            self.inspect((0, 0, 0))

    def test_hierarchy_and_duplicate_link_names(self):
        scene = Scene()
        for entity_id, name in ((1, "own"), (2, "target")):
            model = scene.model.add(id=entity_id, name=name)
            model.link.add(id=entity_id+10, name="camera_link")
        parents, names = inspector.scene_index(scene)
        self.assertEqual(names["own::camera_link"], 11)
        self.assertEqual(names["target::camera_link"], 12)
        poses = {1: pose(1, (10, 20, 30), inspector.from_rpy(0, 0, math.pi/2)),
                 11: pose(11, (2, 0, 0))}
        camera_world = inspector.world_pose(11, parents, poses)
        for actual, expected in zip(camera_world[0], (10, 22, 30)):
            self.assertAlmostEqual(actual, expected)
        with self.assertRaisesRegex(ValueError, "missing_pose_entity"):
            inspector.world_pose(11, parents, {1: poses[1]})

    def test_nested_model_transform(self):
        scene = Scene()
        root = scene.model.add(id=1, name="root")
        child = root.model.add(id=2, name="child")
        child.link.add(id=3, name="link")
        parents, names = inspector.scene_index(scene)
        self.assertEqual(names["root::child::link"], 3)
        result = inspector.world_pose(3, parents, {i: pose(i, (i, 0, 0)) for i in (1, 2, 3)})
        self.assertEqual(result[0], (6, 0, 0))

    def test_actual_sdf_upward_camera_and_sensor_offset(self):
        camera = inspector.load_camera(inspector.DEFAULT_SDF, "camera_link", "camera")
        self.assertEqual((camera.width, camera.height, camera.hfov), (1280, 720, 2.0))
        link = ((0, 0, 0.678), inspector.from_rpy(0, -math.pi/2, 0))
        geometry = inspector.inspect_geometry(inspector.IDENTITY, ((0, 0, 10), inspector.IDENTITY[1]), link, camera)
        self.assertAlmostEqual(geometry["distance_m"], 10-0.678-0.031)
        self.assertAlmostEqual(geometry["target_origin_pixel_uv"][0], 639.5)
        self.assertAlmostEqual(geometry["target_origin_pixel_uv"][1], 359.5)
        self.assertTrue(geometry["target_origin_in_frustum"])

    def test_invalid_quaternion_and_nan_fail_closed(self):
        for q in ((0, 0, 0, 0), (2, 0, 0, 0), (math.nan, 0, 0, 0)):
            with self.assertRaises(ValueError):
                inspector.read_pose(pose(1, quaternion=q))
        with self.assertRaisesRegex(ValueError, "nonfinite_pose"):
            inspector.read_pose(pose(1, (math.inf, 0, 0)))


class StreamTest(unittest.TestCase):
    def test_absent_timestamp_is_not_fabricated(self):
        stream = inspector.PoseStream()
        stream.receive(Pose_V())
        with self.assertRaisesRegex(ValueError, "invalid_sim_timestamp"):
            stream.snapshot(0.5)

    def test_missing_stale_frozen_and_reset_time(self):
        stream = inspector.PoseStream()
        with self.assertRaisesRegex(ValueError, "waiting"):
            stream.snapshot(0.5)
        frame = Pose_V()
        frame.header.stamp.sec = 10
        with patch.object(inspector.time, "monotonic", return_value=100):
            stream.receive(frame)
            self.assertEqual(stream.snapshot(0.5)[3], 0)
        with patch.object(inspector.time, "monotonic", return_value=101):
            with self.assertRaisesRegex(ValueError, "stale_pose"):
                stream.snapshot(0.5)
            stream.receive(frame)  # Fresh transport data with frozen sim time.
            with self.assertRaisesRegex(ValueError, "not_advancing"):
                stream.snapshot(0.5)
            reset = Pose_V()
            reset.header.stamp.sec = 0
            stream.receive(reset)
            self.assertEqual(stream.snapshot(0.5)[3], 1)


class CommandLineTest(unittest.TestCase):
    def run_inspector(self, output, omit_target=False):
        scene = Scene()
        scene.model.add(id=1, name="own").link.add(id=11, name="camera_link")
        scene.model.add(id=2, name="target")
        frame = Pose_V()
        frame.header.stamp.sec = 1
        frame.pose.extend([pose(1), pose(11, (0, 0, 0.678), inspector.from_rpy(0, -math.pi/2, 0))])
        if not omit_target:
            frame.pose.append(pose(2, (0, 0, 10)))
        calls = []

        class ReadOnlyNode:
            # No publication or control method exists on this test node.
            def request(self, service, *_):
                calls.append(service)
                return True, scene

            def subscribe(self, message_type, topic, callback):
                calls.append(topic)
                callback(frame)
                return True

        arguments = ["inspector", "--observer", "own", "--target", "target",
                     "--output", str(output), "--duration", "0.02", "--rate", "100"]
        with patch("gz.transport13.Node", ReadOnlyNode), patch.object(inspector.sys, "argv", arguments), \
                patch.object(inspector.sys, "stderr", io.StringIO()):
            result = inspector.main()
        self.assertTrue(all(call in ("/world/default/scene/info", "/world/default/pose/info") for call in calls))
        return result

    def test_jsonl_complete_path_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"samples.jsonl"
            self.assertEqual(self.run_inspector(output), 0)
            original = output.read_text()
            records = [json.loads(line) for line in original.splitlines()]
            self.assertTrue(records[0]["read_only"])
            self.assertGreater(len(records), 1)
            self.assertTrue(all(record["valid"] for record in records[1:]))
            self.assertEqual(self.run_inspector(output), 1)
            self.assertEqual(output.read_text(), original)

    def test_incomplete_frame_produces_no_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"missing.jsonl"
            self.assertEqual(self.run_inspector(output, omit_target=True), 0)
            records = [json.loads(line) for line in output.read_text().splitlines()][1:]
            self.assertGreater(len(records), 0)
            for record in records:
                self.assertFalse(record["valid"])
                self.assertIn("missing_pose_entity", record["reason"])
                self.assertNotIn("distance_m", record)


if __name__ == "__main__":
    unittest.main()
