#!/usr/bin/env python3
"""Environmental net-capture simulation for Gazebo 8, without PX4 code changes.

Quick visual demo (isolated from the current PX4 Gazebo partition):
  /usr/bin/python3 -s Tools/simulation/gz_net_capture.py demo

Before STARTING Gazebo/PX4 for an existing world, export the Python plugin path:
  export GZ_SIM_SYSTEM_PLUGIN_PATH="$PWD/Tools/simulation:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
  export PYTHONNOUSERSITE=1
Then add the launcher/net, watch mesh lines, and explicitly enable auto fire:
  /usr/bin/python3 -s Tools/simulation/gz_net_capture.py spawn --auto-fire

Manual one-shot trigger (still requires distance, alignment and fresh DYT lock):
  /usr/bin/python3 -s Tools/simulation/gz_net_capture.py fire

GUI rope lines are diagnostic markers; particle visuals are real model visuals.
spawn hides the launcher and packed net; deployed ropes are shown as GUI markers.
--show-net additionally shows the launcher and particle visuals before firing.
spawn uses the camera script's SIM distance input by default; stale/missing
measurements inhibit firing. --range-source model selects geometry-only ranging.
Strings are massless elastic links, with an effective spherical target envelope.
Add --drop-on-capture to spawn to treat a captured x500 as passive litter:
only Gazebo rotor joints are locked after confirmed capture, not PX4 disarmed.
Without that option, the powered x500 can keep compensating for the net weight.
"""
import argparse
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from gz_net_physics import NetConfig, launcher_sdf, model_sdf, valid_name, dense_net_segments


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return result


def triple(value):
    try:
        result = tuple(map(float, value.split(',')))
        if len(result) != 3 or not all(math.isfinite(v) for v in result):
            raise ValueError()
        return result
    except ValueError:
        raise argparse.ArgumentTypeError('must be three finite numbers: x,y,z')


def demo_world(config):
    model = model_sdf('eco_net', config, 'demo_launcher', 'litter_balloon', True,
                      axis=(1, 0, 0), offset=(.7, 0, 0), require_lock=False, demo=True, visible=True,
                      range_source='model')
    model = model[model.index('<model '):model.rindex('</sdf>')]
    mount = launcher_sdf('eco_net_launcher', visible=True)
    mount = mount[mount.index('<model '):mount.rindex('</sdf>')]
    return f'''<sdf version="1.9"><world name="net_demo">
    <physics name="net_physics" type="ignored"><max_step_size>.004</max_step_size><real_time_factor>1</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <gravity>0 0 -9.81</gravity><scene><ambient>.6 .6 .6 1</ambient><background>.7 .8 .9 1</background></scene>
    <light name="sun" type="directional"><pose>0 0 10 0 0 0</pose><diffuse>1 1 1 1</diffuse><direction>-.3 -.3 -1</direction></light>
    <model name="ground"><static>true</static><link name="ground"><visual name="ground">
    <geometry><plane><normal>0 0 1</normal><size>50 50</size></plane></geometry>
    <material><ambient>.35 .45 .3 1</ambient><diffuse>.35 .45 .3 1</diffuse></material></visual></link></model>
    <model name="demo_launcher"><static>true</static><pose>0 0 3.5 0 0 0</pose><link name="base">
    <visual name="stand"><geometry><box><size>1 .5 .2</size></box></geometry>
    <material><diffuse>.1 .25 .15 1</diffuse></material></visual></link></model>
    <model name="litter_balloon"><pose>2.5 0 3.5 0 0 0</pose><link name="balloon">
    <inertial><mass>.08</mass><inertia><ixx>.00392</ixx><iyy>.00392</iyy><izz>.00392</izz></inertia></inertial>
    <visual name="balloon"><geometry><sphere><radius>{config.target_radius}</radius></sphere></geometry>
    <material><ambient>.8 .2 .3 1</ambient><diffuse>.8 .2 .3 1</diffuse></material></visual></link></model>
    {mount}{model}</world></sdf>'''


class Watcher:
    def __init__(self, node, name, draw=True, enable_auto=False, wait_for_renderer=False,
                 deployed_only=False):
        from gz.msgs10.stringmsg_pb2 import StringMsg
        self.node, self.name, self.draw = node, name, draw
        self.wait_for_renderer = wait_for_renderer
        self.deployed_only = deployed_only
        from gz.msgs10.boolean_pb2 import Boolean
        self.enable = node.advertise(f'/model/{name}/enable', Boolean) if enable_auto else None
        self.lock, self.record, self.received = threading.Lock(), None, None
        self.topic = f'/model/{name}/state'
        if not node.subscribe(StringMsg, self.topic, self.receive):
            raise ValueError('net state subscription failed')

    def receive(self, message):
        try:
            record = json.loads(message.data)
        except ValueError:
            return
        with self.lock:
            self.record, self.received = record, time.monotonic()

    def run(self, duration=None, process=None):
        from gz.msgs10.marker_pb2 import Marker
        from gz.msgs10.marker_v_pb2 import Marker_V
        from gz.msgs10.boolean_pb2 import Boolean
        start = time.monotonic(); last_state = None; last_log = last_draw = last_enable = 0.; marker_available = False
        renderer_ready_at = None
        seen, stale_reported, marker_reported = False, False, None
        while duration is None or time.monotonic()-start < duration:
            if process is not None and process.poll() is not None:
                raise ValueError('Gazebo exited: '+str(process.returncode))
            with self.lock:
                record, received = self.record, self.received
            now = time.monotonic()
            if self.enable is not None and now-last_enable >= .1:
                rendered = (not self.wait_for_renderer or
                            (renderer_ready_at is not None and marker_available and now-renderer_ready_at >= 1))
                self.enable.publish(Boolean(data=received is not None and now-received < 1. and rendered))
                last_enable = now
            if received is not None and now-received < 1.:
                seen = True
                stale_reported = False
                if record['state'] == 'fault':
                    raise ValueError('net physics fault: '+str(record['error']))
                if record['state'] != last_state or now-last_log > 1:
                    print(json.dumps({k: v for k, v in record.items() if k not in ('node_positions', 'visual_edges')},
                                     allow_nan=False), flush=True)
                    last_state, last_log = record['state'], now
                if self.draw and now-last_draw > (1/20 if marker_available else 1):
                    last_draw = now
                    deployed = record['state'] in ('released', 'contact', 'captured', 'escaped', 'missed')
                    if self.deployed_only and not deployed:
                        if marker_available:
                            removal = Marker_V()
                            removal.marker.add(ns=self.name, id=1, action=Marker.DELETE_MARKER)
                            self.node.request('/marker_array', removal, Marker_V, Boolean, 100)
                            marker_available = False
                        time.sleep(.02)
                        continue
                    marker = Marker(ns=self.name, id=1, action=Marker.ADD_MODIFY, type=Marker.LINE_LIST)
                    marker.pose.orientation.w = 1
                    # LINE_LIST points already use world metres. Scale is not
                    # line thickness: shrinking it would move the mesh to 0.
                    marker.scale.x = marker.scale.y = marker.scale.z = 1.
                    marker.material.diffuse.r, marker.material.diffuse.g = .85, .65
                    marker.material.diffuse.b, marker.material.diffuse.a = .2, 1.
                    marker.material.ambient.CopyFrom(marker.material.diffuse)
                    positions = record['node_positions']
                    resolution = math.isqrt(len(positions))
                    if resolution >= 3 and resolution**2 == len(positions):
                        segments = dense_net_segments(positions, resolution, record.get('net_width_m', 5.))
                    else:
                        # Compatibility with old/non-grid diagnostic records.
                        segments = [[positions[a], positions[b]] for a, b in record['visual_edges']] if positions else []
                    for segment in segments:
                        for point in segment:
                            p = marker.point.add(); p.x, p.y, p.z = point
                    if marker.point:
                        request = Marker_V(); request.marker.add().CopyFrom(marker)
                        # /marker is a one-way service. Python's blocking API
                        # needs the /marker_array request/reply service instead.
                        ok, response = self.node.request('/marker_array', request, Marker_V, Boolean, 100)
                        marker_available = ok and response.data
                        if marker_available and renderer_ready_at is None:
                            renderer_ready_at = now
                        if marker_available != marker_reported:
                            print(json.dumps(dict(type='net_visualization', mesh_lines_visible=marker_available)), flush=True)
                            marker_reported = marker_available
            elif now-start > 8 and not seen:
                raise ValueError('No net physics heartbeat. Restart Gazebo with GZ_SIM_SYSTEM_PLUGIN_PATH '
                                 'pointing to Tools/simulation and PYTHONNOUSERSITE=1; inspect server errors.')
            elif seen and received is not None and now-received > 2 and not stale_reported:
                print('Net physics heartbeat stale/world paused; auto-fire lease disabled while waiting.',
                      file=sys.stderr, flush=True)
                stale_reported = True
            time.sleep(.02)
        return seen

    def close(self):
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.marker_pb2 import Marker
        from gz.msgs10.marker_v_pb2 import Marker_V
        if self.enable is not None:
            self.enable.publish(Boolean(data=False))
        if self.draw:
            # Delete this watcher's marker only, never other users' markers.
            marker = Marker(ns=self.name, id=1, action=Marker.DELETE_MARKER)
            request = Marker_V(); request.marker.add().CopyFrom(marker)
            self.node.request('/marker_array', request, Marker_V, Boolean, 100)
        self.node.unsubscribe(self.topic)


def spawn(node, world, sdf):
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    from gz.msgs10.boolean_pb2 import Boolean
    ok, response = node.request(f'/world/{world}/create', EntityFactory(sdf=sdf, allow_renaming=False),
                                EntityFactory, Boolean, 2000)
    if not ok or not response.data:
        raise ValueError('model creation failed; check world name and avoid existing eco_net models')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('demo', 'spawn', 'watch', 'fire', 'print-model', 'print-world'))
    parser.add_argument('--world', default='default')
    parser.add_argument('--name', default='eco_net')
    parser.add_argument('--observer', default='fpv_quadtailsitter_gimbal_0')
    parser.add_argument('--target', default='x500_2')
    parser.add_argument('--auto-fire', action='store_true', help='explicitly enable one automatic shot')
    parser.add_argument('--drop-on-capture', action='store_true',
                        help='x500 litter-proxy mode: lock target simulation rotors only after confirmed capture')
    parser.add_argument('--no-visual-lock', action='store_true', help='simulation geometry-only trigger, e.g. demo debris')
    parser.add_argument('--axis', type=triple, default=(0, 0, 1), help='fixed launcher axis in observer FLU body axes')
    parser.add_argument('--offset', type=triple, default=(.18, 0, .78),
                        help='launcher offset in observer FLU, metres; lateral default keeps camera unobstructed')
    parser.add_argument('--trigger-distance', type=positive, default=5.0,
                        help='maximum camera-measured distance for firing, metres (default 5)')
    parser.add_argument('--range-source', choices=('camera', 'model'), default='camera',
                        help='spawn trigger distance input: camera overlay measurement (default) or model/nozzle geometry')
    parser.add_argument('--net-width', type=positive, default=5.,
                        help='deployed net side length, metres (default 5)')
    parser.add_argument('--spread-speed', type=positive,
                        help='perimeter opening speed, m/s (default equals net width, maximum 12)')
    parser.add_argument('--net-mass', type=positive, default=.25)
    parser.add_argument('--launch-speed', type=positive, default=8.)
    parser.add_argument('--target-radius', type=positive, default=.35)
    parser.add_argument('--duration', type=positive, help='watch/demo wall-time limit')
    parser.add_argument('--headless', action='store_true', help='demo server only / no GUI mesh markers')
    parser.add_argument('--show-net', action='store_true',
                        help='also show launcher and packed particles; deployed GUI ropes are visible by default')
    args = parser.parse_args(argv)
    if args.drop_on_capture and args.action in ('demo', 'print-world'):
        parser.error('demo balloon already falls physically; --drop-on-capture is for an x500 target in spawn')
    for name in (args.name, args.world, args.observer, args.target):
        valid_name(name)
    if args.observer == args.target or np_norm(args.axis) < 1e-6:
        parser.error('observer/target must differ and launcher axis must be nonzero')
    config = NetConfig(width=args.net_width, mass=args.net_mass, launch_speed=args.launch_speed,
                       spread_speed=args.spread_speed if args.spread_speed is not None else args.net_width,
                       target_radius=args.target_radius).validate()
    if args.action == 'print-model':
        print(model_sdf(args.name, config, args.observer, args.target, args.auto_fire,
                        args.axis, args.offset, args.trigger_distance, not args.no_visual_lock,
                        drop_on_capture=args.drop_on_capture, visible=args.show_net,
                        range_source=args.range_source)); return 0
    if args.action == 'print-world':
        print(demo_world(config)); return 0
    # Keep the current PX4 Gazebo partition untouched by the standalone demo.
    if args.action == 'demo':
        os.environ['GZ_PARTITION'] = 'eco_net_demo_'+uuid.uuid4().hex[:12]
    from gz.transport13 import Node
    node = Node()
    if args.action == 'fire':
        from gz.msgs10.boolean_pb2 import Boolean
        pub = node.advertise(f'/model/{args.name}/fire', Boolean)
        time.sleep(.3); pub.publish(Boolean(data=True)); time.sleep(.1)
        print('Simulation shot requested; distance/alignment/lock gates still apply.'); return 0
    draw = not args.headless
    watch = Watcher(node, 'eco_net' if args.action == 'demo' else args.name, draw,
                    args.action == 'demo' or (args.action == 'spawn' and args.auto_fire),
                    wait_for_renderer=args.action == 'demo' and not args.headless,
                    deployed_only=args.action != 'demo' and not args.show_net)
    process = None
    try:
        if args.action == 'demo':
            with tempfile.TemporaryDirectory(prefix='gz-eco-net-') as directory:
                path = Path(directory)/'net_demo.sdf'
                path.write_text(demo_world(config), encoding='utf-8')
                environment = os.environ.copy()
                environment['PYTHONNOUSERSITE'] = '1'
                source = str(Path(__file__).resolve().parent)
                environment['GZ_SIM_SYSTEM_PLUGIN_PATH'] = source+':'+environment.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
                command = ['gz', 'sim', '-r']+(['-s'] if args.headless else [])+[str(path)]
                process = subprocess.Popen(command, env=environment, start_new_session=True)
                try:
                    watch.run(args.duration, process)
                finally:
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                        process.wait()
            return 0
        if args.action == 'spawn':
            from gz.msgs10.empty_pb2 import Empty
            from gz.msgs10.scene_pb2 import Scene
            from gimbal_pose_inspector import scene_index
            ok, scene = node.request(f'/world/{args.world}/scene/info', Empty(), Empty, Scene, 1000)
            if not ok: raise ValueError('Gazebo scene is unavailable')
            _, names = scene_index(scene)
            if args.observer not in names or args.target not in names:
                raise ValueError('observer or target does not exist in the selected world')
            if args.name in names or args.name+'_launcher' in names:
                raise ValueError('net/mount already exists; use another --name or restart the simulation')
            spawn(node, args.world, launcher_sdf(args.name+'_launcher', visible=args.show_net))
            spawn(node, args.world, model_sdf(args.name, config, args.observer, args.target, args.auto_fire,
                                           args.axis, args.offset, args.trigger_distance, not args.no_visual_lock,
                                           drop_on_capture=args.drop_on_capture, visible=args.show_net,
                                           range_source=args.range_source))
        watch.run(args.duration)
        return 0
    finally:
        watch.close()


def np_norm(values):
    return math.hypot(*values)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except (ValueError, OSError) as error:
        print('error: '+str(error), file=sys.stderr); sys.exit(1)
