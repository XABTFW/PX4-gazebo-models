"""Opt-in isolated Gazebo integration: RUN_GZ_NET_RUNTIME_TESTS=1.

Does not connect to the existing PX4 partition or change any aircraft model.
"""
import json
import io
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch, MagicMock

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.float_v_pb2 import Float_V
from gz_net_system import NetSystem
from gz_net_capture import Watcher
import threading
from datetime import timedelta
from types import SimpleNamespace
from gz.math7 import Pose3d, Vector3d, Quaterniond
from gz.msgs10.stringmsg_pb2 import StringMsg
from gz_net_physics import CameraRangeGate, NetConfig


class AuthorizationTest(unittest.TestCase):
    def test_default_rope_display_hides_packed_net_but_shows_deployed_net(self):
        for state in ('ready', 'armed', 'released', 'contact', 'captured', 'escaped', 'missed'):
            with self.subTest(state=state):
                node = MagicMock(); node.request.return_value = (True, Boolean(data=True))
                watch = Watcher(node, 'eco_net', draw=True, deployed_only=True)
                watch.record = {'state': state, 'error': None,
                                'node_positions': [[1, 2, 3], [4, 5, 6]], 'visual_edges': [[0, 1]]}
                watch.received = 100.
                clock = [100.]
                def sleep(seconds): clock[0] += seconds
                with patch('gz_net_capture.time.monotonic', side_effect=lambda: clock[0]), \
                     patch('gz_net_capture.time.sleep', side_effect=sleep), \
                     patch('gz_net_capture.sys.stdout', io.StringIO()):
                    watch.run(duration=.15)
                self.assertEqual(node.request.called, state not in ('ready', 'armed'))
                if node.request.called:
                    self.assertEqual(len(node.request.call_args.args[1].marker[0].point), 2)
                watch.close()

    def test_camera_range_controls_trigger_not_nearby_geometry(self):
        system = NetSystem()
        system.guard = threading.Lock()
        system.range_source = 'camera'
        system.range_gate = CameraRangeGate('observer', 'target')
        system.config = NetConfig()
        system.resolve = lambda ecm: True
        system.state, system.release_time = 'ready', None
        system.observer, system.target, system.world, system.model = [MagicMock() for _ in range(4)]
        system.observer.world_pose.return_value = Pose3d()
        system.target.world_pose.return_value = Pose3d(1.5, 0, 0, 0, 0, 0)
        system.observer.world_linear_velocity.return_value = Vector3d()
        system.target.world_linear_velocity.return_value = Vector3d()
        system.world.gravity.return_value = Vector3d(0, 0, -9.81)
        system.offset, system.axis_rotation = (0, 0, 0), Quaterniond()
        system.mount, system.demo, system.links, system.masses = None, False, [], []
        system.trigger, system.auto, system.require_lock = False, False, True
        system.visual_sample, system.visual_received, system.enable_received = .1, 100., None
        system.distance_gate = 2.5
        info = SimpleNamespace(paused=False, sim_time=timedelta(seconds=.1), dt=timedelta(seconds=.004))
        record = dict(schema_version=1, valid=True, source='gazebo_ground_truth',
                      observer_model='observer', target_model='target', tracking_target_id=1,
                      distance_m=3., stamp_ns=100_000_000)
        with patch('gz_net_system.time.monotonic', return_value=100):
            system.pre_update(info, object())
            self.assertFalse(system.trigger_status['ready_to_fire'])
            system.range_input(StringMsg(data=json.dumps(record)))
            system.pre_update(info, object())
            self.assertFalse(system.trigger_status['ready_to_fire'])
            record['distance_m'] = 2.
            system.range_input(StringMsg(data=json.dumps(record)))
            system.pre_update(info, object())
            self.assertTrue(system.trigger_status['ready_to_fire'])
            system.visual_sample = None
            system.pre_update(info, object())
            self.assertFalse(system.trigger_status['ready_to_fire'])
            system.range_input(StringMsg(data='invalid json'))
            system.pre_update(info, object())
            self.assertFalse(system.trigger_status['range_fresh'])

    def test_target_rotors_are_only_locked_after_confirmed_capture(self):
        system = NetSystem()
        system.drop_latched = False
        system.drop_rotors = [MagicMock() for _ in range(4)]
        ecm = object()
        system.lock_captured_proxy(ecm)
        self.assertTrue(all(not joint.mock_calls for joint in system.drop_rotors))
        system.drop_latched = True
        system.lock_captured_proxy(ecm)
        for joint in system.drop_rotors:
            joint.set_velocity.assert_called_once_with(ecm, [0.])
            limits = joint.set_velocity_limits.call_args.args[1][0]
            self.assertEqual((limits.x(), limits.y()), (0, 0))

    def test_mesh_markers_preserve_world_coordinates_and_use_reply_service(self):
        node = MagicMock(); node.request.return_value = (True, Boolean(data=True))
        watch = Watcher(node, 'eco_net', draw=True)
        watch.record = {'state': 'ready', 'error': None, 'node_positions': [[1, 2, 3], [4, 5, 6]],
                        'visual_edges': [[0, 1]]}
        watch.received = 100.
        clock = [100.]
        def sleep(seconds): clock[0] += seconds
        with patch('gz_net_capture.time.monotonic', side_effect=lambda: clock[0]), \
             patch('gz_net_capture.time.sleep', side_effect=sleep), \
             patch('gz_net_capture.sys.stdout', io.StringIO()):
            watch.run(duration=.15)
        request = node.request.call_args.args
        self.assertEqual(request[0], '/marker_array')
        marker = request[1].marker[0]
        self.assertEqual((marker.scale.x, marker.scale.y, marker.scale.z), (1, 1, 1))
        self.assertEqual((marker.point[0].x, marker.point[0].y, marker.point[0].z), (1, 2, 3))
        watch.close()
        removal = node.request.call_args.args[1].marker[0]
        self.assertEqual(removal.ns, 'eco_net')
        self.assertEqual(removal.action, removal.DELETE_MARKER)

    def test_gui_demo_does_not_fire_before_renderer_is_ready(self):
        node = MagicMock(); node.request.return_value = (False, Boolean(data=False))
        watch = Watcher(node, 'eco_net', draw=True, enable_auto=True, wait_for_renderer=True)
        watch.record = {'state': 'ready', 'error': None, 'node_positions': [[1, 2, 3], [4, 5, 6]],
                        'visual_edges': [[0, 1]]}
        watch.received = 100.
        clock = [100.]
        def sleep(seconds): clock[0] += seconds
        with patch('gz_net_capture.time.monotonic', side_effect=lambda: clock[0]), \
             patch('gz_net_capture.time.sleep', side_effect=sleep), \
             patch('gz_net_capture.sys.stdout', io.StringIO()):
            watch.run(duration=.3)
        self.assertTrue(watch.enable.publish.call_args_list)
        self.assertTrue(all(not call.args[0].data for call in watch.enable.publish.call_args_list))
        watch.close()

    def test_watcher_stale_heartbeat_and_exit_revoke_auto_fire(self):
        node = MagicMock()
        watch = Watcher(node, 'eco_net', draw=False, enable_auto=True)
        watch.record, watch.received = {'state': 'ready', 'error': None}, 100.
        clock = [100.]
        def sleep(seconds): clock[0] += seconds
        with patch('gz_net_capture.time.monotonic', side_effect=lambda: clock[0]), \
             patch('gz_net_capture.time.sleep', side_effect=sleep), \
             patch('gz_net_capture.sys.stdout', io.StringIO()), \
             patch('gz_net_capture.sys.stderr', io.StringIO()):
            self.assertTrue(watch.run(duration=3))
        messages = [call.args[0].data for call in watch.enable.publish.call_args_list]
        self.assertIn(True, messages)
        self.assertFalse(messages[-1])
        watch.close()
        self.assertFalse(watch.enable.publish.call_args.args[0].data)
        node.unsubscribe.assert_called_once_with('/model/eco_net/state')

    def test_physics_fault_is_reported_and_not_left_armed_on_exit(self):
        node = MagicMock()
        watch = Watcher(node, 'eco_net', draw=False, enable_auto=True)
        watch.record, watch.received = {'state': 'fault', 'error': 'unstable force'}, 100.
        with patch('gz_net_capture.time.monotonic', return_value=100):
            with self.assertRaisesRegex(ValueError, 'unstable force'):
                watch.run(duration=1)
        watch.close()
        self.assertFalse(watch.enable.publish.call_args.args[0].data)

    def test_enabling_requires_a_recent_explicit_command(self):
        system = NetSystem()
        system.guard, system.enable_received = threading.Lock(), None
        with patch('gz_net_system.time.monotonic', return_value=100):
            system.enable(Boolean(data=True))
        self.assertEqual(system.enable_received, 100)
        system.enable(Boolean(data=False))
        self.assertIsNone(system.enable_received)

    def test_valid_visual_packet_is_recorded_without_retimestamping(self):
        system = NetSystem(); system.guard = threading.Lock()
        packet = Float_V(data=[1, 1]+[0]*10)
        packet.header.stamp.sec = 12; packet.header.stamp.nsec = 100_000_000
        with patch('gz_net_system.time.monotonic', return_value=100):
            system.visual(packet)
        self.assertEqual(system.visual_sample, 12.1)
        self.assertEqual(system.visual_received, 100)
        packet.data[1] = 0; system.visual(packet)
        self.assertIsNone(system.visual_sample)
        self.assertIsNone(system.visual_received)


@unittest.skipUnless(os.environ.get('RUN_GZ_NET_RUNTIME_TESTS') == '1' and shutil.which('gz'),
                     'set RUN_GZ_NET_RUNTIME_TESTS=1 to run isolated Gazebo simulations')
class RuntimeTest(unittest.TestCase):
    def run_demo(self, speed, duration):
        script = Path(__file__).with_name('gz_net_capture.py')
        result = subprocess.run(['/usr/bin/python3', '-s', str(script), 'demo', '--headless',
                                 '--launch-speed', str(speed), '--duration', str(duration)],
                                capture_output=True, text=True, timeout=duration+15)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        self.assertTrue(records, result.stdout)
        self.assertTrue(all(r['error'] is None for r in records), records)
        self.assertFalse(any(r['state'] == 'fault' for r in records))
        return records

    def test_deploy_contact_wrap_and_capture_with_physics_feedback(self):
        records = self.run_demo(8, 7)
        states = {r['state'] for r in records}
        self.assertTrue({'released', 'contact', 'captured'} <= states, records)
        captured = [r for r in records if r['state'] == 'captured']
        released = next(r for r in records if r['state'] == 'released')
        self.assertEqual(released['net_width_m'], 5.)
        self.assertIsNone(released['captured_at_s'])
        self.assertFalse(released['target_passive_proxy'])
        self.assertGreater(captured[0]['captured_at_s'], released['released_at_s'])
        self.assertGreaterEqual(captured[0]['attachments'], 4)
        self.assertGreater(captured[-1]['target_position'][0], captured[0]['target_position'][0]+.5)
        self.assertLess(captured[-1]['target_position'][2], captured[0]['target_position'][2])

    def test_insufficient_shot_misses_instead_of_declaring_capture(self):
        records = self.run_demo(1, 15)
        self.assertIn('released', {r['state'] for r in records})
        self.assertIn('missed', {r['state'] for r in records})
        self.assertNotIn('captured', {r['state'] for r in records})


if __name__ == '__main__':
    unittest.main()
