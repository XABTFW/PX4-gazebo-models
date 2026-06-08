#!/usr/bin/env python3
"""Apply force / torque to a Gazebo entity in a running gz world."""

import argparse
import shlex
import subprocess
import sys
import time


DIRECTION_FORCES = {
    "front": (1.0, 0.0, 0.0),
    "forward": (1.0, 0.0, 0.0),
    "fwd": (1.0, 0.0, 0.0),
    "前": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "后": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "左": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "右": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "上": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
    "下": (0.0, 0.0, -1.0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish a gz.msgs.EntityWrench message to ApplyLinkWrench."
    )
    parser.add_argument(
        "-w",
        "--world",
        help="Gazebo world name. Default: auto-detect the running world.",
    )
    parser.add_argument(
        "-e",
        "--entity",
        default="x500_gimbal_1",
        help="Target model or link name. Default: x500_gimbal_1.",
    )
    parser.add_argument(
        "--type",
        choices=("MODEL", "LINK"),
        default="MODEL",
        help="Target entity type. Default: MODEL.",
    )
    parser.add_argument(
        "--force",
        nargs=3,
        type=float,
        metavar=("FX", "FY", "FZ"),
        help="Force in world coordinates, in N.",
    )
    parser.add_argument(
        "--dir",
        choices=tuple(DIRECTION_FORCES),
        help="Convenience direction: front/back/left/right/up/down.",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=2.0,
        help="Force magnitude in N when using --dir. Default: 2.",
    )
    parser.add_argument(
        "--torque",
        nargs=3,
        type=float,
        metavar=("TX", "TY", "TZ"),
        default=(0.0, 0.0, 0.0),
        help="Torque in world coordinates, in N*m. Default: 0 0 0.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Apply for one simulation step instead of persistently.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear persistent wrench from the entity.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Publish the message this many times. Default: 5.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Seconds between repeated publishes. Default: 0.1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gz command without running it.",
    )

    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    if args.interval < 0:
        parser.error("--interval must be nonnegative")

    if args.clear:
        if args.force is not None or args.dir is not None:
            parser.error("--clear cannot be combined with force inputs")
        return args

    if args.force is not None and args.dir is not None:
        parser.error("Use either --force or --dir, not both")

    if args.force is None and args.dir is None:
        parser.error("Provide --dir DIR, --force FX FY FZ, or use --clear")

    return args


def detect_world():
    completed = subprocess.run(
        ["gz", "topic", "-l"],
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        return "default"

    worlds = []

    for line in completed.stdout.splitlines():
        if not line.startswith("/world/") or not line.endswith("/clock"):
            continue

        world = line.removeprefix("/world/").removesuffix("/clock")

        if world and world not in worlds:
            worlds.append(world)

    if len(worlds) == 1:
        return worlds[0]

    if len(worlds) > 1:
        print(
            "Multiple Gazebo worlds are running. Please choose one with --world.",
            file=sys.stderr,
        )
        return None

    return "default"


def wrench_command(world, args):
    entity_type = args.type.upper()

    if args.clear:
        topic = f"/world/{world}/wrench/clear"
        payload = f'name: "{args.entity}", type: {entity_type}'
        return [
            "gz",
            "topic",
            "-t",
            topic,
            "-m",
            "gz.msgs.Entity",
            "-p",
            payload,
        ]

    topic_suffix = "wrench" if args.once else "wrench/persistent"
    topic = f"/world/{world}/{topic_suffix}"
    fx, fy, fz = force_vector(args)
    tx, ty, tz = args.torque
    payload = (
        f'entity: {{name: "{args.entity}", type: {entity_type}}}, '
        f"wrench: {{force: {{x: {fx:.6g}, y: {fy:.6g}, z: {fz:.6g}}}, "
        f"torque: {{x: {tx:.6g}, y: {ty:.6g}, z: {tz:.6g}}}}}"
    )
    return [
        "gz",
        "topic",
        "-t",
        topic,
        "-m",
        "gz.msgs.EntityWrench",
        "-p",
        payload,
    ]


def force_vector(args):
    if args.force is not None:
        return args.force

    dx, dy, dz = DIRECTION_FORCES[args.dir]
    return dx * args.amount, dy * args.amount, dz * args.amount


def main():
    args = parse_args()
    world = args.world or detect_world()

    if world is None:
        return 2

    command = wrench_command(world, args)

    if args.dry_run:
        print(shlex.join(command))
        return 0

    for publish_index in range(args.repeat):
        completed = subprocess.run(command, check=False)

        if completed.returncode != 0:
            return completed.returncode

        if publish_index + 1 < args.repeat:
            time.sleep(args.interval)

    topic = command[command.index("-t") + 1]
    print(f"Published wrench command to {topic}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
