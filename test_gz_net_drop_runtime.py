"""Opt-in native x500 propulsion/drop check in an isolated Gazebo fixture."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
import xml.etree.ElementTree as ET


def probe():
    source = Path(__file__).resolve().parent
    os.environ['GZ_PARTITION'] = 'net_drop_probe_'+uuid.uuid4().hex
    os.environ['GZ_SIM_RESOURCE_PATH'] = str(source/'gz/models')
    from gz.sim8 import TestFixture, Model, Link, Joint, World, world_entity
    from gz.transport13 import Node
    from gz.msgs10.actuators_pb2 import Actuators
    from gz_net_system import NetSystem
    base = ET.parse(source/'gz/models/x500_base/model.sdf')
    mass = sum(float(el.text) for el in base.findall('.//inertial/mass'))
    speed = (mass*9.81/(4*8.54858e-6))**.5
    target = ET.parse(source/'gz/models/x500/model.sdf').getroot().find('model')
    target.set('name', 'drop_target')
    ET.SubElement(target, 'pose').text = '0 0 10 0 0 0'
    for plugin in target.findall('plugin'):
        if plugin.get('name') == 'gz::sim::systems::MulticopterMotorModel':
            ET.SubElement(plugin, 'robotNamespace').text = '/model/drop_target'
    world = f'''<sdf version="1.9"><world name="drop_probe">
    <physics name="p" type="ignored"><max_step_size>.004</max_step_size><real_time_factor>0</real_time_factor></physics>
    <gravity>0 0 -9.81</gravity>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    {ET.tostring(target, encoding='unicode')}
    </world></sdf>'''
    with tempfile.TemporaryDirectory(prefix='gz-net-drop-test-') as directory:
        path = Path(directory)/'world.sdf'; path.write_text(world, encoding='utf-8')
        fixture = TestFixture(str(path)); system = NetSystem()
        system.drop_latched, system.drop_rotors = False, []
        body = [None]; rows = []
        def pre(info, ecm):
            if not system.drop_rotors:
                model = Model(World(world_entity(ecm)).model_by_name(ecm, 'drop_target'))
                system.drop_rotors = [Joint(model.joint_by_name(ecm, f'rotor_{i}_joint')) for i in range(4)]
                body[0] = Link(model.canonical_link(ecm)); body[0].enable_velocity_checks(ecm)
                for joint in system.drop_rotors: joint.enable_velocity_check(ecm)
            if info.iterations >= 300:
                system.drop_latched = True
            system.lock_captured_proxy(ecm)
        def post(info, ecm):
            velocity = body[0].world_linear_velocity(ecm)
            pose = body[0].world_pose(ecm)
            rotations = [joint.velocity(ecm) for joint in system.drop_rotors]
            if velocity is not None and all(rotations):
                rows.append((info.iterations, pose.pos().z(), velocity.z(), max(abs(v[0]) for v in rotations)))
        fixture.on_pre_update(pre); fixture.on_post_update(post); fixture.finalize()
        fixture.server().run(True, 2, False)  # Instantiate motor subscriptions before publishing.
        node = Node(); pub = node.advertise('/model/drop_target/command/motor_speed', Actuators)
        time.sleep(.3); pub.publish(Actuators(velocity=[speed]*4)); time.sleep(.1)
        fixture.server().run(True, 700, False)
        before = next(row for row in reversed(rows) if row[0] < 300)
        after = rows[-1]
        print(json.dumps({'rotor_speed_before': before[3], 'rotor_speed_after': after[3],
                          'vz_before': before[2], 'vz_after': after[2], 'height_drop_m': before[1]-after[1]}))


@unittest.skipUnless(os.environ.get('RUN_GZ_NET_RUNTIME_TESTS') == '1', 'opt-in native Gazebo test')
class DropRuntimeTest(unittest.TestCase):
    def test_powered_target_becomes_gravity_driven_after_proxy_capture(self):
        result = subprocess.run(['/usr/bin/python3', '-s', str(Path(__file__).resolve()), '--probe'],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
        record = json.loads(next(line for line in reversed(result.stdout.splitlines()) if line.startswith('{')))
        self.assertGreater(record['rotor_speed_before'], 1)
        self.assertLess(record['rotor_speed_after'], .01)
        self.assertLess(record['vz_after'], record['vz_before']-5)
        self.assertGreater(record['height_drop_m'], 2)


if __name__ == '__main__':
    if '--probe' in sys.argv:
        probe()
    else:
        unittest.main()
