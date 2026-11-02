#!/usr/bin/env python3
"""Check image signs, upward rig singularity and joint limits without Gazebo."""
import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_v_pb2 import Pose_V

from gimbal_pose_inspector import Camera, PoseStream, from_rpy, multiply, rotate, inverse
from gz_dyt_bridge import (CameraPoseStream, DytBridge, TrackingLossGrace, camera_rpy_frd,
                          check_target_texture, follow_step, tracking_measurement,
                          transport_geometry, simulation_distance)


def rig(yaw=0, roll=0, pitch=0):
    mount = from_rpy(0, 0, 0)
    yaw_q = from_rpy(0, 0, yaw)
    roll_q = multiply(yaw_q, from_rpy(roll, 0, 0))
    camera_q = multiply(roll_q, from_rpy(0, pitch-math.pi/2, 0))
    return [mount, yaw_q, roll_q, camera_q]


class SimulationRangeTest(unittest.TestCase):
    def setUp(self):
        self.frame = Pose_V(); self.frame.header.stamp.sec = 12
        self.image = Image(width=1280, height=720); self.image.header.CopyFrom(self.frame.header)
        self.stream = PoseStream()
        self.bridge = SimpleNamespace(model='observer', keys=['observer', 'camera'],
                    names={'observer': 1, 'camera': 2, 'x500_2': 3}, parents={1: None, 2: None, 3: None},
                    stream=self.stream, camera=Camera(1280, 720, 2, .05, 15000, ((0, 0, 0), (1, 0, 0, 0))))
        for identifier, position, q in [(1, (0, 0, 0), from_rpy(0, 0, 0)),
                                       (2, (0, 0, .7), from_rpy(0, -math.pi/2, 0)),
                                       (3, (0, 0, 10.7), from_rpy(0, 0, 0))]:
            pose = self.frame.pose.add(id=identifier)
            pose.position.x, pose.position.y, pose.position.z = position
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = q
        self.stream.receive(self.frame)
        self.target = {'valid': True, 'bbox_xywh': [630, 350, 20, 20]}

    def test_vehicle_and_camera_distance_have_explicit_reference_points(self):
        measured = simulation_distance(self.bridge, self.image, self.target)
        self.assertTrue(measured['valid'], measured)
        self.assertAlmostEqual(measured['distance_m'], 10.7)
        self.assertAlmostEqual(measured['camera_distance_m'], 10.)
        self.assertEqual(measured['source'], 'gazebo_ground_truth')

    def test_three_dimensional_distance_not_just_height(self):
        self.frame.pose[2].position.x = 3
        self.frame.pose[2].position.y = 4
        self.frame.pose[2].position.z = 12.7
        self.stream.receive(self.frame)
        self.target['bbox_xywh'] = [490, 450, 30, 30]
        measured = simulation_distance(self.bridge, self.image, self.target)
        self.assertTrue(measured['valid'], measured)
        self.assertAlmostEqual(measured['distance_m'], math.sqrt(25+12.7**2))
        self.assertAlmostEqual(measured['camera_distance_m'], 13.)

    def test_wrong_roi_missing_model_and_loss_do_not_display_a_distance(self):
        for target, name in [(dict(valid=False), 'x500_2'),
                             ({'valid': True, 'bbox_xywh': [100, 100, 20, 20]}, 'x500_2'),
                             (self.target, 'missing'), (self.target, 'observer')]:
            result = simulation_distance(self.bridge, self.image, target, name)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['distance_m'])

    def test_pose_time_mismatch_does_not_reuse_a_previous_distance(self):
        self.assertTrue(simulation_distance(self.bridge, self.image, self.target)['valid'])
        self.image.header.stamp.nsec = 50_000_000
        result = simulation_distance(self.bridge, self.image, self.target)
        self.assertFalse(result['valid'])
        self.assertIn('time_mismatch', result['status'])

    def test_distance_updates_with_new_exposure_and_disappears_on_pose_outage(self):
        self.assertAlmostEqual(simulation_distance(self.bridge, self.image, self.target)['distance_m'], 10.7)
        self.frame.header.stamp.nsec = self.image.header.stamp.nsec = 32_000_000
        self.frame.pose[2].position.z = 20.7
        self.stream.receive(self.frame)
        measured = simulation_distance(self.bridge, self.image, self.target)
        self.assertAlmostEqual(measured['distance_m'], 20.7)
        self.assertEqual(measured['stamp_ns'], 12_032_000_000)
        self.frame.ClearField('pose')
        self.stream.receive(self.frame)
        self.assertFalse(simulation_distance(self.bridge, self.image, self.target)['valid'])


class BridgeTest(unittest.TestCase):
    def test_upward_euler_branch_preserves_all_camera_axes(self):
        for roll in (-2.9, -.2, 0, 1.7):
            for pitch in (-math.pi/2, -1.53, -1.61, .4):
                for yaw in (-3.1, -2.2, 0, 2.2, 3.1):
                    flu = from_rpy(roll, pitch, yaw)
                    frd = from_rpy(*camera_rpy_frd(flu))
                    self.assertLessEqual(abs(camera_rpy_frd(flu)[2]), math.pi/2+1e-8)
                    for axis in ((1,0,0), (0,1,0), (0,0,1)):
                        original = rotate(from_rpy(roll, -pitch, -yaw), axis)
                        np.testing.assert_allclose(rotate(frd, axis), original, atol=1e-7)

    def test_transport_packet_timestamp_frame_interval_and_invalidation(self):
        bridge = DytBridge.__new__(DytBridge)
        bridge.camera = Camera(1280, 720, 2, .05, 15000, ((0, 0, 0), (1, 0, 0, 0)))
        bridge.follow, bridge.last_stamp = False, None
        bridge.last_yaw_command = None
        bridge.keys = ["model", "mount", "yaw", "roll", "camera"]
        bridge.names = dict(zip(bridge.keys, range(1, 6)))
        bridge.parents = {1: None, 2: 1, 3: 1, 4: 1, 5: 1}
        bridge.stream = PoseStream()
        packets = []
        bridge.publisher = SimpleNamespace(publish=lambda packet: packets.append(packet) or True)
        frame = Pose_V()
        frame.header.stamp.sec = 12
        for identifier, quaternion in enumerate([from_rpy(0, 0, 0)]+rig(), 1):
            pose = frame.pose.add(id=identifier)
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = quaternion
        bridge.stream.receive(frame)
        image = Image(width=1280, height=720)
        image.header.stamp.sec = 12
        target = {"valid": True, "center_px": [639.5, 359.5], "bbox_xywh": [630, 350, 20, 20]}
        bridge.update(image, target)
        self.assertEqual(len(packets[0].data), 12)
        self.assertEqual(packets[0].header.stamp.sec, 12)
        self.assertAlmostEqual(packets[0].data[5], math.radians(85), places=6)
        self.assertAlmostEqual(packets[0].data[3], -math.radians(5), places=6)
        self.assertEqual(packets[0].data[11], 0)
        bridge.update(image, target)
        self.assertEqual(len(packets), 1)
        image.header.stamp.nsec = frame.header.stamp.nsec = 100_000_000
        bridge.stream.receive(frame)
        bridge.update(image, target)
        self.assertAlmostEqual(packets[-1].data[11], .1)
        bridge.invalidate()
        self.assertEqual(list(packets[-1].data), [1]+[0]*11)
        image.header.stamp.nsec = 300_000_000
        with self.assertRaisesRegex(ValueError, "frame_gap"):
            bridge.update(image, target)

    def test_uniform_background_cannot_guide_gimbal(self):
        image = np.full((20, 20, 3), (0, 0, 200), dtype=np.uint8)
        target = {"valid": True, "bbox_xywh": [0, 0, 20, 20]}
        with self.assertRaisesRegex(ValueError, "no_spatial_texture"):
            check_target_texture(image, target)
        image[::2, ::2] = 255
        check_target_texture(image, target)

    def test_pixel_center_and_right_down(self):
        target = {"valid": True, "center_px": [639.5, 359.5], "bbox_xywh": [630, 350, 20, 20]}
        self.assertEqual(tracking_measurement(target, 1280, 720, 2)[:2], [0, 0])
        target["center_px"] = [739.5, 409.5]
        lx, ly, *_ = tracking_measurement(target, 1280, 720, 2)
        self.assertGreater(lx, 0)
        self.assertGreater(ly, 0)
        self.assertAlmostEqual(math.tan(lx)/math.tan(ly), 2)
        self.assertEqual(tracking_measurement({"valid": False}, 1280, 720, 2), [0]*6)

    def test_follow_reduces_error_and_respects_rate(self):
        for yaw, roll, pitch in [(0, 0, 0), (0.7, 0.2, -0.8), (0, 0, -2.1)]:
            for lx, ly in [(0.15, 0), (0, 0.15), (-0.1, -0.1)]:
                with self.subTest(yaw=yaw, pitch=pitch, lx=lx, ly=ly):
                    qs = rig(yaw, roll, pitch)
                    desired = np.array(rotate(qs[-1], (1, -math.tan(lx), -math.tan(ly))))
                    desired /= np.linalg.norm(desired)
                    angles = follow_step(qs, lx, ly, 0.1)
                    old = np.array(rotate(qs[-1], (1, 0, 0)))
                    new = np.array(rotate(rig(*angles)[-1], (1, 0, 0)))
                    self.assertGreater(float(new @ desired), float(old @ desired))
                    self.assertLessEqual(max(abs(a-b) for a, b in zip(angles, [yaw, roll, pitch])), 0.050001)

    def test_center_holds_measured_angles_including_large_negative_pitch(self):
        for pitch in [0, -0.8, -1.8, -2.35619, 0.785398]:
            np.testing.assert_allclose(follow_step(rig(0.3, 0.2, pitch), 0, 0, 0.1),
                                       [0.3, 0.2, pitch], atol=1e-7)

    def test_center_pixel_noise_does_not_move_joints(self):
        angles = follow_step(rig(.4, .2, -.8), .002, -.003, 1/30)
        np.testing.assert_allclose(angles, [.4, .2, -.8], atol=1e-7)

    def test_pose_history_matches_exposure_and_rejects_old_image(self):
        stream = CameraPoseStream()
        for ns in (0, 30_000_000, 60_000_000, 100_000_000):
            frame = Pose_V()
            frame.header.stamp.sec, frame.header.stamp.nsec = 12, ns
            stream.receive(frame)
        self.assertEqual(stream.at_image(12_030_000_000).header.stamp.nsec, 30_000_000)
        with self.assertRaisesRegex(ValueError, "time_mismatch"):
            stream.at_image(12_150_000_000)
        frame.header.stamp.sec = 13
        stream.receive(frame)
        with self.assertRaisesRegex(ValueError, "max_age"):
            stream.at_image(12_030_000_000)

    def test_mechanical_limits(self):
        angles = follow_step(rig(0, 0.785398, -2.35619), -0.5, 0.5, 0.1)
        self.assertLessEqual(abs(angles[1]), 0.785398)
        self.assertGreaterEqual(angles[2], -2.35619)
        self.assertLessEqual(angles[2], 0.785398)

    def test_yaw_wrap_keeps_continuous_joint_command(self):
        angles = follow_step(rig(-math.pi+.01, 0, -.8), 0, 0, .1, math.pi+.01)
        self.assertAlmostEqual(angles[0], math.pi+.01)

    def test_faster_sim_follow_is_still_rate_and_mechanically_limited(self):
        initial = [.3, .2, -.8]
        slow = follow_step(rig(*initial), .4, -.4, .1, max_rate=.5)
        fast = follow_step(rig(*initial), .4, -.4, .1, max_rate=1.)
        self.assertLessEqual(max(abs(a-b) for a, b in zip(fast, initial)), .100001)
        self.assertGreater(np.linalg.norm(np.array(fast)-initial), np.linalg.norm(np.array(slow)-initial))
        stopped = follow_step(rig(*initial), 0, 0, .1, max_rate=1.)
        np.testing.assert_allclose(stopped, initial, atol=1e-7)
        limited = follow_step(rig(0, .785398, -2.35619), -.5, .5, .1, max_rate=1.)
        self.assertLessEqual(abs(limited[1]), .785398)
        self.assertGreaterEqual(limited[2], -2.35619)

    def test_higher_centering_gain_reduces_moving_target_lag_without_raising_rate(self):
        def run(gain):
            angles = np.array([0., 0., -.8])
            errors = []
            for i in range(180):
                wanted = rig(pitch=-.8+.35*math.sin(i/30*2))[-1]
                direction = rotate(wanted, (1, 0, 0))
                camera = rig(*angles)[-1]
                ray = rotate(inverse(camera), direction)
                lx, ly = math.atan2(-ray[1], ray[0]), math.atan2(-ray[2], ray[0])
                next_angles = np.array(follow_step(rig(*angles), lx, ly, 1/30,
                                                  max_rate=1., gain=gain))
                self.assertLessEqual(np.max(np.abs(next_angles-angles)), 1/30+1e-7)
                angles = next_angles
                if i > 30:
                    errors.append(math.hypot(lx, ly))
            return max(errors)
        self.assertLess(run(10.), run(5.)*.7)


class VirtualFrameTest(unittest.TestCase):
    def test_vertical_crossings_preserve_body_los_after_float32_transport(self):
        for pitch in np.linspace(80, 100, 81):
            for yaw in (-85, 0, 85):
                for roll in (-40, 0, 40):
                    relative = from_rpy(math.radians(roll), -math.radians(pitch), -math.radians(yaw))
                    for lx, ly in ((0, 0), (.1, -.05), (-.08, .12)):
                        angles, offsets, state = transport_geometry(relative, lx, ly)
                        # Match Float_V and the C++ Euler/tan reconstruction.
                        r, p, y, tx, ty = map(float, np.float32([*angles, *offsets]))
                        self.assertLessEqual(abs(p), math.pi/2)
                        self.assertLessEqual(abs(y), 1.92)
                        self.assertLessEqual(max(abs(tx), abs(ty)), .45)
                        original = np.array(rotate(from_rpy(math.radians(roll), math.radians(pitch), math.radians(yaw)),
                                                   (1, math.tan(lx), math.tan(ly))))
                        rebuilt = np.array(rotate(from_rpy(r, p, y), (1, math.tan(tx), math.tan(ty))))
                        np.testing.assert_allclose(rebuilt/np.linalg.norm(rebuilt),
                                                   original/np.linalg.norm(original), atol=3e-7)

    def test_native_frame_remains_available_for_comparison(self):
        relative = from_rpy(0, -math.radians(91), 0)
        with self.assertRaisesRegex(ValueError, 'pitch_outside'):
            transport_geometry(relative, 0, 0, virtual_frame=False)
        angles, offsets, state = transport_geometry(relative, 0, 0)
        self.assertTrue(state['virtual_frame_active'])
        self.assertAlmostEqual(angles[1], math.radians(85))
        self.assertAlmostEqual(offsets[1], -math.radians(6))

    def test_normal_frame_unchanged_and_out_of_view_not_fabricated(self):
        relative = from_rpy(.1, -.5, -.2)
        angles, offsets, state = transport_geometry(relative, .1, -.2)
        np.testing.assert_allclose(angles, (.1, .5, .2), atol=1e-9)
        self.assertEqual(offsets, (.1, -.2))
        self.assertFalse(state['virtual_frame_active'])
        with self.assertRaisesRegex(ValueError, 'geometry_limits'):
            transport_geometry(from_rpy(0, -math.radians(120), 0), 0, 0)

    def test_repeated_90_degree_crossings_publish_real_samples_without_loss(self):
        bridge = DytBridge.__new__(DytBridge)
        bridge.camera = Camera(1280, 720, 2, .05, 15000, ((0, 0, 0), (1, 0, 0, 0)))
        bridge.follow, bridge.last_stamp = False, None
        bridge.keys = ['model', 'mount', 'yaw', 'roll', 'camera']
        bridge.names = dict(zip(bridge.keys, range(1, 6)))
        bridge.parents = {1: None, 2: 1, 3: 1, 4: 1, 5: 1}
        bridge.stream = PoseStream()
        bridge.loss_gate = TrackingLossGrace(.2)
        packets = []
        bridge.publisher = SimpleNamespace(publish=lambda packet: packets.append(packet) or True)
        image = Image(width=1280, height=720)
        target = {'valid': True, 'center_px': [639.5, 359.5], 'bbox_xywh': [630, 350, 20, 20]}
        for i in range(600):
            stamp = 12_000_000_000+i*33_333_333
            frame = Pose_V()
            frame.header.stamp.sec, frame.header.stamp.nsec = divmod(stamp, 1_000_000_000)
            image.header.CopyFrom(frame.header)
            camera = from_rpy(0, -math.radians(90+math.sin(i*.1)), 0)
            for identifier, quaternion in enumerate([from_rpy(0, 0, 0)]*4+[camera], 1):
                pose = frame.pose.add(id=identifier)
                pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = quaternion
            bridge.stream.receive(frame)
            result = bridge.update(image, target)
            self.assertTrue(result['measurement_published'])
            self.assertFalse(result['loss_grace_active'])
            self.assertEqual(packets[-1].header.stamp, image.header.stamp)
            self.assertEqual(packets[-1].data[1], 1)
            self.assertLess(abs(packets[-1].data[5]), math.pi/2)
        self.assertEqual(len(packets), 600)

    def test_physical_follow_uses_real_pixels_not_virtual_offsets(self):
        bridge = DytBridge.__new__(DytBridge)
        bridge.camera = Camera(1280, 720, 2, .05, 15000, ((0, 0, 0), (1, 0, 0, 0)))
        bridge.follow, bridge.last_stamp = True, 11_966_666_667
        bridge.last_yaw_command = None
        bridge.keys = ['model', 'mount', 'yaw', 'roll', 'camera']
        bridge.names = dict(zip(bridge.keys, range(1, 6)))
        bridge.parents = {1: None, 2: 1, 3: 1, 4: 1, 5: 1}
        bridge.stream = PoseStream()
        bridge.publisher = SimpleNamespace(publish=lambda packet: True)
        commands = []
        bridge.commands = [SimpleNamespace(publish=lambda msg: commands.append(msg.data) or True)]*3
        frame = Pose_V()
        frame.header.stamp.sec = 12
        for identifier, quaternion in enumerate([from_rpy(0, 0, 0)]+rig(pitch=-math.radians(1)), 1):
            pose = frame.pose.add(id=identifier)
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = quaternion
        bridge.stream.receive(frame)
        image = Image(width=1280, height=720)
        image.header.CopyFrom(frame.header)
        target = {'valid': True, 'center_px': [639.5, 359.5], 'bbox_xywh': [630, 350, 20, 20]}
        result = bridge.update(image, target)
        self.assertTrue(result['virtual_frame_active'])
        self.assertEqual(commands, [])  # Virtual six-degree miss must NOT move the real camera.
        frame.header.stamp.nsec = image.header.stamp.nsec = 33_333_333
        bridge.stream.receive(frame)
        target['center_px'] = [660., 370.]
        real_los = tracking_measurement(target, 1280, 720, 2)[:2]
        with patch('gz_dyt_bridge.follow_step', return_value=[.1, .2, .3]) as follow:
            bridge.update(image, target)
            np.testing.assert_allclose(follow.call_args.args[1:3], real_los)
        np.testing.assert_allclose(commands, [.1, .2, .3])


class LossGraceTest(unittest.TestCase):
    def test_short_miss_recovers_without_an_invalid_packet_or_replayed_frame(self):
        bridge = DytBridge.__new__(DytBridge)
        bridge.camera = Camera(1280, 720, 2, .05, 15000, ((0, 0, 0), (1, 0, 0, 0)))
        bridge.follow, bridge.last_stamp = False, None
        bridge.keys = ['model', 'mount', 'yaw', 'roll', 'camera']
        bridge.names = dict(zip(bridge.keys, range(1, 6)))
        bridge.parents = {1: None, 2: 1, 3: 1, 4: 1, 5: 1}
        bridge.stream = PoseStream()
        bridge.loss_gate = TrackingLossGrace(.2)
        packets = []
        bridge.publisher = SimpleNamespace(publish=lambda packet: packets.append(packet) or True)
        frame = Pose_V()
        frame.header.stamp.sec = 12
        for identifier, quaternion in enumerate([from_rpy(0, 0, 0)]+rig(), 1):
            pose = frame.pose.add(id=identifier)
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = quaternion
        image = Image(width=1280, height=720)
        image.header.stamp.sec = 12
        good = {'valid': True, 'center_px': [639.5, 359.5], 'bbox_xywh': [630, 350, 20, 20]}
        with patch('gz_dyt_bridge.time.monotonic', return_value=100):
            bridge.stream.receive(frame)
            bridge.update(image, good)
        self.assertEqual(len(packets), 1)
        with patch('gz_dyt_bridge.time.monotonic', return_value=100.032):
            image.header.stamp.nsec = 32_000_000
            result = bridge.update(image, {'valid': False})
            self.assertTrue(result['loss_grace_active'])
        self.assertEqual(len(packets), 1)  # Not even an old observation was replayed.
        self.assertEqual(packets[0].header.stamp.nsec, 0)
        with patch('gz_dyt_bridge.time.monotonic', return_value=100.064):
            image.header.stamp.nsec = frame.header.stamp.nsec = 64_000_000
            bridge.stream.receive(frame)
            result = bridge.update(image, good)
        self.assertEqual(len(packets), 2)
        self.assertTrue(result['measurement_published'])
        self.assertAlmostEqual(packets[-1].data[11], .032)
        self.assertEqual(packets[-1].header.stamp.nsec, 64_000_000)
        with patch('gz_dyt_bridge.time.monotonic', return_value=100.265):
            bridge.invalidate(stamp_ns=12_265_000_000)
        self.assertEqual(len(packets), 3)
        self.assertEqual(list(packets[-1].data), [1]+[0]*11)

    def test_wall_time_bounds_loss_even_if_sim_time_is_frozen(self):
        gate = TrackingLossGrace(.2)
        with patch('gz_dyt_bridge.time.monotonic', return_value=100):
            gate.accept(12_000_000_000)
        with patch('gz_dyt_bridge.time.monotonic', return_value=100.1):
            self.assertTrue(gate.holding())
        with patch('gz_dyt_bridge.time.monotonic', return_value=100.201):
            self.assertFalse(gate.holding())

    def test_fast_simulation_expires_by_exposure_time(self):
        gate = TrackingLossGrace(.2)
        with patch('gz_dyt_bridge.time.monotonic', return_value=100):
            gate.accept(12_000_000_000)
            gate.observe(12_200_000_000)
            self.assertFalse(gate.holding())

    def test_time_reset_does_not_reuse_previous_lock(self):
        gate = TrackingLossGrace(.2)
        gate.accept(12_000_000_000)
        gate.observe(1_000_000_000)
        self.assertFalse(gate.holding())

    def test_force_invalidation_bypasses_grace(self):
        bridge = DytBridge.__new__(DytBridge)
        bridge.loss_gate = TrackingLossGrace(.2)
        bridge.last_stamp = 12_000_000_000
        packets = []
        bridge.publisher = SimpleNamespace(publish=lambda packet: packets.append(packet) or True)
        bridge.loss_gate.accept(bridge.last_stamp)
        bridge.invalidate(force=True)
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0].data[1], 0)
        self.assertFalse(bridge.loss_gate.holding())

    def test_grace_cannot_exceed_px4_freshness_budget(self):
        for value in (-.1, .25, math.inf, math.nan):
            with self.assertRaises(ValueError):
                TrackingLossGrace(value)
        gate = TrackingLossGrace(0)
        gate.accept(12_000_000_000)
        self.assertFalse(gate.holding())


if __name__ == "__main__":
    unittest.main()
