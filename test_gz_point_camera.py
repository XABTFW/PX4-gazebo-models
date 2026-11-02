"""Geometry, mechanical limits, and non-competing ownership for geo pointing."""

import contextlib
import io
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gz.msgs10.float_v_pb2 import Float_V
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.scene_pb2 import Scene

import gz_point_camera as pointing
from gimbal_pose_inspector import from_rpy, multiply, rotate


def rig(yaw=0, roll=0, pitch=0, mount=None):
    mount = mount or from_rpy(0, 0, 0)
    yaw_q = multiply(mount, from_rpy(0, 0, yaw))
    roll_q = multiply(yaw_q, from_rpy(roll, 0, 0))
    camera_q = multiply(roll_q, from_rpy(0, pitch-math.pi/2, 0))
    return [mount, yaw_q, roll_q, camera_q]


class GeometryTest(unittest.TestCase):
    def test_pointing_forward_side_behind_and_above(self):
        for direction in [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (1, -2, 4), (0, 0, 1), (1, 0, -.5)]:
            with self.subTest(direction=direction):
                command, residual = pointing.pointing_solution(direction, (0, 0, 0))
                forward = rotate(rig(*command)[-1], (1, 0, 0))
                norm = math.hypot(*direction)
                for actual, expected in zip(forward, direction):
                    self.assertAlmostEqual(actual, expected/norm, places=5)
                self.assertLess(residual, .01)
                self.assertTrue(pointing.PITCH_LIMITS[0] <= command[2] <= pointing.PITCH_LIMITS[1])

    def test_below_mechanical_reach_reports_error(self):
        command, residual = pointing.pointing_solution((0, 0, -1), (0, 0, 0))
        self.assertAlmostEqual(command[2], pointing.PITCH_LIMITS[0])
        self.assertGreater(residual, 44)

    def test_measured_angles_include_vehicle_rotation_and_fixed_pitch(self):
        for pitch in (-2.2, -1.8, -.6, .4):
            rotations = rig(3.3, .25, pitch, from_rpy(.2, -.5, .7))
            angles = pointing.measured_joints(rotations, 3.3)
            for a, b in zip(angles, (3.3, .25, pitch)):
                self.assertAlmostEqual(a, b, places=6)

    def test_yaw_wrap_is_continuous(self):
        command, _ = pointing.pointing_solution((1, .01, 1), (6.28, 0, .7))
        self.assertLess(abs(command[0]-6.28), math.pi+.02)

    def test_slew_limits_and_accumulates_command(self):
        command = (0, 0, 0)
        for _ in range(30):
            updated = pointing.limit_step(command, (1.2, 0, -1.2), 1/30, 1)
            self.assertLessEqual(max(abs(a-b) for a, b in zip(command, updated)), 1/30+1e-9)
            command = updated
        self.assertAlmostEqual(command[0], 1)
        self.assertAlmostEqual(command[2], -1)

    def test_invalid_direction_is_rejected(self):
        for direction in ((0, 0, 0), (math.nan, 0, 1)):
            with self.assertRaises(ValueError):
                pointing.pointing_solution(direction, (0, 0, 0))


class OwnershipTest(unittest.TestCase):
    def test_valid_visual_lock_yields_invalid_and_stale_resume(self):
        visual = pointing.VisualLock()
        packet = Float_V(data=[1, 1]+[0]*10)
        packet.header.stamp.sec = 1
        with patch.object(pointing.time, 'monotonic', return_value=10):
            visual.receive(packet)
            self.assertTrue(visual.active(1_000_000_000, .5))
            self.assertFalse(visual.active(2_000_000_000, .5))
        with patch.object(pointing.time, 'monotonic', return_value=11):
            self.assertFalse(visual.active(1_000_000_000, .5))
        packet.data[1] = 0
        visual.receive(packet)
        self.assertFalse(visual.active(1_000_000_000, .5))

    def run_cli(self, visual_locked=False, supply_poses=True):
        scene = Scene()
        observer = scene.model.add(id=1, name='own')
        for i, name in enumerate(('gimbal_mount_link', 'gimbal_yaw_link', 'gimbal_roll_link', 'camera_link'), 2):
            observer.link.add(id=i, name=name)
        scene.model.add(id=6, name='target')
        frame = Pose_V()
        frame.header.stamp.sec = 1
        for i, q in enumerate([from_rpy(0, 0, 0)]+rig()+[from_rpy(0, 0, 0)], 1):
            pose = frame.pose.add(id=i)
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = q
            if i == 6:
                pose.position.x, pose.position.z = 5, 2
        publications = []
        unsubscriptions = []

        class Node:
            def advertise(self, topic, _):
                return SimpleNamespace(publish=lambda message: publications.append((topic, message)) or True)

            def subscribe(self, message_type, topic, callback):
                if topic.endswith('/pose/info') and supply_poses:
                    callback(frame)
                if topic.endswith('/dyt/target') and visual_locked:
                    packet = Float_V(data=[1, 1]+[0]*10)
                    packet.header.stamp.sec = 1
                    callback(packet)
                return True

            def request(self, *_):
                return True, scene

            def unsubscribe(self, topic):
                unsubscriptions.append(topic)

        with patch('gz.transport13.Node', Node), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pointing.main(['--observer', 'own', '--target', 'target',
                                           '--duration', '.02', '--rate', '100']), 0)
        self.assertEqual(len(unsubscriptions), 2)
        return publications

    def test_geo_points_and_heartbeats_without_aircraft_commands(self):
        messages = self.run_cli()
        self.assertEqual(len(messages), 4)
        self.assertTrue(messages[0][1].data)
        self.assertTrue(all('/command/gimbal_' in topic for topic, _ in messages[1:]))

    def test_visual_tracking_prevents_joint_commands_and_lease_revocation(self):
        self.assertEqual(self.run_cli(visual_locked=True), [])

    def test_missing_pose_prevents_commands(self):
        self.assertEqual(self.run_cli(supply_poses=False), [])


if __name__ == '__main__':
    unittest.main()
